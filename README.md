# 🌙 Night Watch: seizure detection prototype

A Streamlit app that watches a saved video of someone in bed, tracks their skeleton,
and logs an event when the movement looks like a seizure.

Three ways to decide "is this a seizure?" (sidebar → *Frame detector*):

| Detector | Training needed? | What it is |
|---|---|---|
| **Pretrained VSViG (Track A)** | No (already trained by the authors) | The real research model from [VSViG](https://github.com/xuyankun/VSViG) (ECCV 2024) |
| **Motion rule** | No | "Is the body moving fast?": a simple baseline |
| **Trained model** (Track B) | Yes, with `train.py` | A random forest you train on labeled videos |

> Student science project. **Not a medical device.**

## How it works

```
video frame ─► pose.py ─► skeleton ─► features.py ─► motion in the last 1 s
          ─► scorer.py ─► "seizure-like frame?" ─► events.py ─► event log
```

1. **Pose** (`pose.py`): a small YOLO pose model finds 17 body keypoints. Only the person
   inside the *bed region* is used, so a parent or nurse in the room is ignored.
2. **Motion** (`features.py`): one pose can't show a seizure; *movement over time* can.
   Each frame is described by how fast and how jerkily each joint moved during the last second.
3. **Frame score** (`scorer.py`):
   - *Motion rule*: flag the frame if the body moves faster than a threshold. No training needed.
   - *Trained model*: a random forest trained by `train.py` on labeled videos.
4. **Events** (`events.py`): if at least 60% of frames in the last 10 s are flagged → **seizure event**.
   This ignores short movements like rolling over. Events are shown in the app and saved
   to `logs/events.csv`.

### Track A: pretrained VSViG (`vsvig_detector.py`)

```
video frame ─► VSViG's patient pose model ─► 15 joints ─► image patch around each joint
          ─► 5-second clip (30 frames) ─► VSViG model ─► seizure probability ─► events.py
```

- Uses the authors' own weights, unchanged: `pose.pth` (Lightweight OpenPose fine-tuned on
  patients in bed) and `VSViG-base.pth`. A new probability comes out every 0.5 s.
- Unlike our skeleton-only models, VSViG also *looks at the image* around each joint (32×32 patches).
- It was built for 1920×1080 video at 6 frames/s; the app resizes frames to match.
- Two input details weren't documented (keypoint and patch scaling). We worked them out from the
  model's own BatchNorm statistics: keypoints are `(x, y, confidence)` in 1080p pixels, patches are 0–255.
- **Fair-test warning:** the demo video `Sz2PG` comes from the VSViG authors' own dataset, so the model
  may have been trained on it. Great for a demo, but it is not proof that it works on new patients.

## Run it

```bash
cd seizure-watch
uv venv .venv && uv pip install -r requirements.txt   # or: python -m venv .venv && .venv/bin/pip install -r requirements.txt
# Track A code + weights (both Apache 2.0):
git clone --depth 1 https://github.com/xuyankun/VSViG.git vendor/VSViG
git clone --depth 1 https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch.git vendor/lwopenpose
.venv/bin/streamlit run app.py
```

In the sidebar, pick the video, a start time, and drag the blue **bed region** over the bed.
Then press **▶ Analyze**.

**Real-time playback** (on by default) plays the video at its true speed, like a live camera.
If the computer can't keep up, frames are dropped instead of falling behind ("Frames dropped" counter).
The pose model runs on the GPU automatically (Apple M-series or NVIDIA), which is ~4× faster than CPU.
Turn real-time off to analyze a video as fast as possible.

**Convert videos to 720p first.** Decoding a 1080p video every frame is wasted work: the app
shrinks frames anyway. One-time conversion (also drops the audio):

```bash
ffmpeg -i input.mp4 -vf scale=-2:720 -an -c:v libx264 -preset veryfast -crf 23 input_720p.mp4
```

The app analyzes **5 frames per second** by default. That's enough to see sustained jerking and cheap
enough to run in real time. If you train a model, use the same `--fps` in `train.py` as in the app.

## Train your own frame classifier

1. Write `labels.csv` (times in seconds, one row per seizure; a video without a seizure gets blank times).
   The times below are **examples only**; use the real onset times from the dataset's `Label.xlsx`:

   ```csv
   video,seizure_start_s,seizure_end_s,bed_roi
   ../Sz2PG_720p.mp4,410,480,"0.28,0,0.75,1"
   videos/normal_sleep_1.mp4,,,
   ```

2. `.venv/bin/python train.py labels.csv`. This saves `models/seizure_model.joblib`.
   The first run is slow because the pose model goes over every video. Poses are cached in `cache/`, so later runs are fast.
3. Restart the app. **Trained model** now appears under *Frame detector*.

With more than one video, `train.py` tests on each video after training on the others.
That's the only honest score. Testing on the same video you trained on looks great and means nothing.

## Things to try

- Tune the motion threshold: what's the lowest value that never flags quiet sleep?
- Record normal sleep movements (rolling over, scratching, sitting up). Does the app raise false alarms?
- Change the event window and fraction. How does that change detection delay vs. false alarms?
- Add features in `features.py` (e.g., only arms and legs) and retrain.

## Tests

```bash
.venv/bin/python -m pytest
```
