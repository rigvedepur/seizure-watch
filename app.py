"""Night Watch: seizure detection prototype.

    streamlit run app.py

Pipeline (one step per file):
    video frame --pose.py--> skeleton --features.py--> last ~1 s of motion
                --scorer.py--> "seizure-like frame?" --events.py--> seizure event log
Track A swaps the middle for the pretrained VSViG model (vsvig_detector.py):
    video frame --VSViG pose + 5 s clip--> seizure probability --events.py--> seizure event log
"""
import csv
import datetime as dt
import time
from pathlib import Path

import altair as alt
import cv2
import pandas as pd
import streamlit as st

from events import NOT_VISIBLE, SEIZURE, WARMING_UP, WATCHING, SeizureEventDetector
from features import PoseHistory
from pose import FRAME_WIDTH, PoseExtractor, draw_overlay, iter_video, resize, video_info
from scorer import TRAINED_MODEL, MotionRuleScorer, TrainedModelScorer

HERE = Path(__file__).resolve().parent
DEFAULT_VIDEO = HERE.parent / "Sz2PG_720p.mp4"
UPLOAD_DIR = HERE / "videos"
LOG_FILE = HERE / "logs" / "events.csv"
CHART_EVERY_S = 5        # redraw the chart every 5 s of video (redrawing every frame is slow)
LIVE_CHART_POINTS = 600  # while running, chart only the most recent frames (~2 min at 5 fps)
CHUNK_S = 60             # end the script run every 60 s of video so Streamlit frees old frames
JPEG_QUALITY = 70        # displayed frames: ~4x smaller than Streamlit's default encoding

STATE_STYLE = {  # state -> (emoji, overlay color in BGR)
    WATCHING: ("🟢", (0, 200, 0)),
    WARMING_UP: ("⚪", (200, 200, 200)),
    NOT_VISIBLE: ("⚪", (200, 200, 200)),
    SEIZURE: ("🔴", (0, 0, 255)),
}

st.set_page_config(page_title="Night Watch", page_icon="🌙", layout="wide")
st.title("🌙 Night Watch")
st.caption("Video seizure detection prototype. A student science project, **not a medical device**.")


VSVIG_NAME = "Pretrained VSViG (Track A)"


@st.cache_resource
def get_pose_extractor():
    return PoseExtractor()


@st.cache_resource
def get_vsvig_models():
    """Load VSViG's pose model and seizure model once (imported here so other modes start faster)."""
    import vsvig_detector
    return vsvig_detector, vsvig_detector.PatientPose(vsvig_detector.best_device()), vsvig_detector.VSViGModel("cpu")


def mmss(seconds):
    if seconds is None:
        return "ongoing"
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


# ---------------- Sidebar: settings ----------------
with st.sidebar:
    st.header("1. Video")
    uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    path_text = st.text_input("…or a video file on this computer", str(DEFAULT_VIDEO))
    if uploaded is not None:
        UPLOAD_DIR.mkdir(exist_ok=True)
        video_path = UPLOAD_DIR / uploaded.name
        if not video_path.exists():
            video_path.write_bytes(uploaded.getbuffer())
    else:
        video_path = Path(path_text).expanduser()

    try:
        video_fps, video_len = video_info(video_path)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    st.caption(f"{mmss(video_len)} long, {video_fps:.0f} fps")
    start_s = st.number_input("Start at (seconds)", 0, int(video_len), 0, step=10)
    run_for = st.number_input("Analyze for (seconds, 0 = until the end)", 0, int(video_len), 0, step=30)
    analysis_fps = st.slider("Frames analyzed per second", 2, 15, 5,
                             help="Must match the --fps used in train.py if you use a trained model.")
    realtime = st.toggle("Real-time playback", True,
                         help="Play at the video's true speed, like a live camera. If the computer falls "
                              "behind, frames are dropped. Off = analyze as fast as possible.")

    st.header("2. Bed region")
    st.caption("Only a person inside the blue box is tracked (ignores parents, nurses…).")
    x0, x1 = st.slider("Left / right", 0.0, 1.0, (0.28, 0.75), 0.01)
    y0, y1 = st.slider("Top / bottom", 0.0, 1.0, (0.0, 1.0), 0.01)
    bed_roi = (x0, y0, x1, y1)

    st.header("3. Frame detector")
    options = [VSVIG_NAME, MotionRuleScorer.name]
    if TRAINED_MODEL.exists():
        options.append(TrainedModelScorer.name)
    choice = st.radio("Is this frame seizure-like?", options,
                      help="Train a model with train.py to unlock the 'Trained model' option.")
    use_vsvig = choice == VSVIG_NAME
    if use_vsvig:
        st.caption("VSViG looks at 5-second clips (6 frames/s) and outputs a seizure probability "
                   "every 0.5 s. It ignores the frames-per-second slider. Works best on 1080p video.")
        threshold = st.slider("Flag frame if seizure probability ≥", 0.05, 0.95, 0.5, 0.05)
        scorer = None
    elif choice == MotionRuleScorer.name:
        threshold = st.slider("Flag frame if body moves faster than (body-lengths / s)", 0.05, 2.0, 0.3, 0.05)
        scorer = MotionRuleScorer(threshold)
    else:
        threshold = st.slider("Flag frame if model probability ≥", 0.05, 0.95, 0.5, 0.05)
        scorer = TrainedModelScorer(threshold=threshold)

    st.header("4. Event rule")
    window_s = st.slider("Look at the last (seconds)", 3, 30, 10)
    min_ratio = st.slider("Seizure if this fraction of frames is flagged", 0.3, 1.0, 0.6, 0.05)
    cooldown_s = st.slider("Wait before a new event (seconds)", 0, 120, 30, 5)


# ---------------- Main area ----------------
col_video, col_side = st.columns([3, 2])
with col_video:
    frame_box = st.empty()
with col_side:
    b1, b2 = st.columns(2)
    analyze_clicked = b1.button("▶ Analyze", type="primary", width="stretch")
    stop_clicked = b2.button("⏹ Stop", width="stretch")
    state_box = st.empty()
    metrics_box = st.empty()
    st.subheader("Event log")
    events_box = st.empty()
chart_box = st.empty()


def show_events(events):
    if not events:
        events_box.info("No seizure events yet.")
        return
    events_box.dataframe(pd.DataFrame([{
        "#": i + 1,
        "Starts": mmss(e.start_s),
        "Alarm at": mmss(e.detected_s),
        "Ends": mmss(e.end_s),
        "Peak flagged": f"{e.peak_ratio:.0%}",
    } for i, e in enumerate(events)]), hide_index=True, width="stretch")


def show_chart(rows, events, cfg):
    if not rows:
        return
    df = pd.DataFrame(rows)
    shade = alt.Chart(pd.DataFrame([{"start": e.start_s, "end": e.end_s or df.t.max()} for e in events])
                      if events else pd.DataFrame({"start": [], "end": []})
                      ).mark_rect(color="red", opacity=0.15).encode(x="start:Q", x2="end:Q")

    def panel(column, title, line_at):
        base = alt.Chart(df).mark_line().encode(
            x=alt.X("t:Q", title="video time (s)"),
            y=alt.Y(f"{column}:Q", title=title))
        rule = alt.Chart(pd.DataFrame({"y": [line_at]})).mark_rule(strokeDash=[4, 4], color="gray").encode(y="y:Q")
        return (shade + base + rule).properties(height=160)

    chart = alt.vconcat(
        panel("score", "frame score", cfg["threshold"]),
        panel("flagged_ratio", "flagged fraction", cfg["min_ratio"]),
    )
    chart_box.altair_chart(chart, width="stretch")


def log_event(event, cfg):
    LOG_FILE.parent.mkdir(exist_ok=True)
    new = not LOG_FILE.exists()
    with LOG_FILE.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["logged_at", "video", "detector", "start_s", "alarm_s", "end_s", "peak_flagged"])
        w.writerow([dt.datetime.now().isoformat(timespec="seconds"), cfg["video"].name, cfg["choice"],
                    round(event.start_s, 1), round(event.detected_s, 1), round(event.end_s, 1),
                    round(event.peak_ratio, 2)])


def show_state(state):
    emoji, _ = STATE_STYLE[state]
    if state == SEIZURE:
        state_box.error(f"## {emoji} SEIZURE DETECTED")
    else:
        state_box.markdown(f"## {emoji} {state.capitalize()}")


def show_metrics(run, score, status, speed):
    with metrics_box.container():
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Frame score", "–" if score is None else f"{score:.2f}")
        m2.metric(f"Flagged in last {run['cfg']['window_s']}s", f"{status.flagged_ratio:.0%}")
        m3.metric("Speed", "–" if speed is None else f"{speed:.1f}×",
                  help="Seconds of video processed per real second, over the last ~5 s. Below 1× = slower than real time.")
        m4.metric("Frames dropped", run["dropped"], help="Real-time mode drops frames when the computer can't keep up.")


def new_run():
    """Snapshot the sidebar settings and build fresh detectors. Settings changed mid-run apply next time."""
    cfg = dict(video=video_path, start_s=start_s, end_s=start_s + run_for if run_for else None,
               realtime=realtime, choice=choice, use_vsvig=use_vsvig, threshold=threshold,
               min_ratio=min_ratio, window_s=window_s, bed_roi=bed_roi, fps=analysis_fps)
    run = dict(cfg=cfg, t_next=float(start_s), t_last=float(start_s), rows=[], logged=set(), dropped=0,
               wall_start=time.monotonic(), done=False, last_frame=None,
               events=SeizureEventDetector(window_s, min_ratio, cooldown_s))
    if use_vsvig:
        vsvig_detector, vsvig_pose, vsvig_model = get_vsvig_models()
        run["vsvig"] = vsvig_detector.VSViGDetector(bed_roi, pose=vsvig_pose, model=vsvig_model)
        cfg["fps"] = vsvig_detector.FPS
    else:
        run["history"] = PoseHistory(getattr(scorer, "history_s", 1.0))
        run["scorer"] = scorer
    return run


def log_finished_events(run):
    for k, e in enumerate(run["events"].events):
        if e.end_s is not None and k not in run["logged"]:
            log_event(e, run["cfg"])
            run["logged"].add(k)


def finish_run(run):
    run["events"].finish(run["t_last"])
    log_finished_events(run)
    run["done"] = True


def process_chunk(run):
    """Analyze up to CHUNK_S seconds of video, then start a new script run.

    Streamlit only frees displayed frames when a script run ends, so one run per
    chunk keeps memory flat however long the video is. Everything needed to carry
    on (detectors, position in the video, results) lives in `run`, saved in
    st.session_state, and is updated *before* anything is drawn, so a rerun at any
    moment (a click, a chunk boundary) loses nothing.
    """
    cfg, events = run["cfg"], run["events"]
    use_vsvig = cfg["use_vsvig"]
    if use_vsvig:
        vsvig_detector = get_vsvig_models()[0]
    else:
        extractor = get_pose_extractor()
    chunk_end = run["t_next"] + CHUNK_S
    shown_state = shown_events = None   # redraw these only when they change
    last_chart_t = -1e9

    frames = iter_video(cfg["video"], cfg["fps"], run["t_next"], cfg["end_s"],
                        width=None if use_vsvig else FRAME_WIDTH)
    for t, frame in frames:
        run["t_next"], run["t_last"] = t + 1 / cfg["fps"], t
        if cfg["realtime"]:
            # How far behind (+) or ahead (-) of the video clock are we?
            lag = (time.monotonic() - run["wall_start"]) - (t - cfg["start_s"])
            if lag < 0:
                time.sleep(-lag)          # ahead: wait, like a camera delivering frames on time
            elif lag > 1 / cfg["fps"]:
                run["dropped"] += 1       # behind: skip this frame, like a live feed would
                continue

        if use_vsvig:
            score, kp18 = run["vsvig"].update(frame)   # full-resolution frame in
            flagged = None if score is None else score >= cfg["threshold"]
            frame, pose = resize(frame), None          # small frame for display
        else:
            pose = extractor.detect(frame, cfg["bed_roi"])
            run["history"].add(t, pose)
            score, flagged = run["scorer"].score(run["history"])
        status = events.update(t, flagged)
        wall = time.monotonic() - run["wall_start"]
        rows = run["rows"]
        rows.append({"t": t, "score": score, "flagged_ratio": status.flagged_ratio, "wall": wall})
        log_finished_events(run)

        # ---- drawing (everything above is already saved) ----
        _, color = STATE_STYLE[status.state]
        shown = draw_overlay(frame, pose, cfg["bed_roi"], color, f"{mmss(t)}  {status.state}")
        if use_vsvig:
            vsvig_detector.draw_openpose(shown, kp18, color)
        run["last_frame"] = cv2.imencode(".jpg", shown, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1].tobytes()
        frame_box.image(run["last_frame"], width="stretch")

        # Speed = video seconds per wall-clock second, measured over the last ~5 s of video.
        past = next((r for r in reversed(rows) if r["t"] <= t - 5), None)
        speed = (t - past["t"]) / max(wall - past["wall"], 1e-6) if past else None
        show_metrics(run, score, status, speed)
        if status.state != shown_state:
            show_state(status.state)
            shown_state = status.state
        events_key = [(e.start_s, e.end_s) for e in events.events]
        if events_key != shown_events:
            show_events(events.events)
            shown_events = events_key
        if t - last_chart_t >= CHART_EVERY_S:
            show_chart(rows[-LIVE_CHART_POINTS:], events.events, cfg)  # recent part only, so redraws stay fast
            last_chart_t = t

        if t >= chunk_end:
            st.rerun()

    finish_run(run)


if analyze_clicked:
    st.session_state.run = new_run()
run = st.session_state.get("run")
if stop_clicked and run and not run["done"]:
    finish_run(run)

if run and not run["done"]:
    process_chunk(run)

if run and run["done"]:
    if run["last_frame"]:
        frame_box.image(run["last_frame"], width="stretch")
    n = len(run["events"].events)
    state_box.success(f"Done: {n} event(s). Log saved to {LOG_FILE.relative_to(HERE)}")
    show_events(run["events"].events)
    show_chart(run["rows"], run["events"].events, run["cfg"])
    st.caption("Change settings and press ▶ Analyze to run again.")
else:
    preview = next(iter_video(video_path, 1, start_s), None)
    if preview is not None:
        frame_box.image(draw_overlay(preview[1], None, bed_roi, label="Preview: adjust the blue bed box"),
                        channels="BGR", width="stretch")
    state_box.markdown("## ⚪ Ready")
    show_events([])
