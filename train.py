"""Train a "seizure / not seizure" frame classifier from labeled videos.

Usage:
    python train.py labels.csv

labels.csv has one row per seizure (a video with no seizure gets one row with
blank times). bed_roi is "x0,y0,x1,y1" as fractions of the frame; blank = whole frame.
The times below are examples only. Take real onset times from the dataset's Label.xlsx.

    video,seizure_start_s,seizure_end_s,bed_roi
    ../Sz2PG_720p.mp4,410,480,"0.28,0,0.75,1"
    videos/normal_sleep_1.mp4,,,

Every analyzed frame gets the label 1 if it falls inside a seizure, else 0.
Poses are cached in cache/ so each video only goes through the pose model once.
"""
import argparse
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score

from features import PoseHistory, feature_names
from pose import Pose, PoseExtractor, iter_video
from scorer import TRAINED_MODEL

CACHE_DIR = Path(__file__).parent / "cache"


def parse_roi(text):
    if not isinstance(text, str) or not text.strip():
        return (0.0, 0.0, 1.0, 1.0)
    return tuple(float(v) for v in text.split(","))


def extract_poses(video, fps, roi, extractor=None):
    """Run the pose model over a video once and cache the result. Returns [(t, Pose | None)]."""
    key = hashlib.md5(f"{Path(video).resolve()}|{fps}|{roi}".encode()).hexdigest()[:8]
    cache = CACHE_DIR / f"{Path(video).stem}_{key}.npz"
    if cache.exists():
        d = np.load(cache)
        return [(t, Pose(k, c, b) if ok else None)
                for t, ok, k, c, b in zip(d["t"], d["ok"], d["kp"], d["conf"], d["box"])]

    extractor = extractor or PoseExtractor()
    poses = []
    for i, (t, frame) in enumerate(iter_video(video, fps)):
        poses.append((t, extractor.detect(frame, roi)))
        if i % 500 == 0:
            print(f"  {Path(video).name}: {t / 60:.1f} min", flush=True)

    CACHE_DIR.mkdir(exist_ok=True)
    blank = Pose(np.zeros((17, 2)), np.zeros(17), np.zeros(4))
    np.savez_compressed(
        cache,
        t=np.array([t for t, _ in poses]),
        ok=np.array([p is not None for _, p in poses]),
        kp=np.stack([(p or blank).keypoints for _, p in poses]),
        conf=np.stack([(p or blank).conf for _, p in poses]),
        box=np.stack([(p or blank).box for _, p in poses]),
    )
    return poses


def features_over_time(poses, history_s):
    """Replay poses through a PoseHistory, exactly like the app does. Returns times, X."""
    history = PoseHistory(history_s)
    times, rows = [], []
    for t, pose in poses:
        history.add(t, pose)
        x = history.features()
        if x is not None:
            times.append(t)
            rows.append(x)
    return np.array(times), np.array(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels")
    ap.add_argument("--fps", type=float, default=5.0, help="frames analyzed per second (use the app's value)")
    ap.add_argument("--history", type=float, default=1.0, help="seconds of pose history per frame")
    ap.add_argument("--out", default=str(TRAINED_MODEL))
    args = ap.parse_args()

    labels = pd.read_csv(args.labels)
    base = Path(args.labels).parent
    extractor = PoseExtractor()

    videos = []
    for video, rows in labels.groupby("video", sort=False):
        path = (base / video).resolve()
        roi = parse_roi(rows["bed_roi"].iloc[0] if "bed_roi" in rows else None)
        print(f"Poses for {path.name} ...")
        times, X = features_over_time(extract_poses(path, args.fps, roi, extractor), args.history)
        y = np.zeros(len(times), dtype=int)
        for _, r in rows.dropna(subset=["seizure_start_s", "seizure_end_s"]).iterrows():
            y[(times >= r.seizure_start_s) & (times <= r.seizure_end_s)] = 1
        videos.append((path.name, X, y))
        print(f"  {len(y)} frames, {y.sum()} labeled seizure")

    def new_model():
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                      class_weight="balanced", n_jobs=-1, random_state=0)

    # Honest test: train on all videos but one, test on the one left out.
    if len(videos) > 1:
        print("\nLeave-one-video-out test (frame level):")
        for i, (name, X_test, y_test) in enumerate(videos):
            X_tr = np.concatenate([v[1] for j, v in enumerate(videos) if j != i])
            y_tr = np.concatenate([v[2] for j, v in enumerate(videos) if j != i])
            if len(set(y_tr)) < 2:
                print(f"  {name}: skipped (other videos don't have both classes)")
                continue
            pred = new_model().fit(X_tr, y_tr).predict(X_test)
            print(f"  {name}: recall {recall_score(y_test, pred, zero_division=0):.2f}, "
                  f"precision {precision_score(y_test, pred, zero_division=0):.2f}")
    else:
        print("\nOnly one video: no honest test possible. Expect the model to be overconfident.")

    X = np.concatenate([v[1] for v in videos])
    y = np.concatenate([v[2] for v in videos])
    if len(set(y)) < 2:
        raise SystemExit("Need both seizure and non-seizure frames to train.")
    model = new_model().fit(X, y)

    top = sorted(zip(model.feature_importances_, feature_names()), reverse=True)[:8]
    print("\nMost useful features:")
    for imp, name in top:
        print(f"  {name:28s} {imp:.3f}")

    Path(args.out).parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "history_s": args.history, "fps": args.fps}, args.out)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
