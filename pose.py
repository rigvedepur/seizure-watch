"""Step 1: video frames -> one person's skeleton (pose).

We use a small off-the-shelf YOLO pose model (17 COCO keypoints).
Rooms often have more than one person (a parent, a nurse), so we only keep
the person whose body is inside the "bed region" the user picks.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

MODELS_DIR = Path(__file__).parent / "models"
POSE_MODEL = MODELS_DIR / "yolo11n-pose.pt"

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Pairs of keypoints joined by a bone, for drawing the skeleton.
BONES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6),
]

# Frames are shrunk to this width before anything else (faster, and plenty for pose).
FRAME_WIDTH = 720


@dataclass
class Pose:
    keypoints: np.ndarray  # (17, 2) pixel x, y
    conf: np.ndarray       # (17,) how sure the model is about each keypoint, 0..1
    box: np.ndarray        # (4,) x0, y0, x1, y1 of the person

    @property
    def scale(self) -> float:
        """Body size in pixels (box diagonal). Lets us measure motion in 'body lengths'."""
        x0, y0, x1, y1 = self.box
        return float(np.hypot(x1 - x0, y1 - y0))


def best_device():
    """Use the GPU if there is one: about 4x faster than CPU on an M-series Mac."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class PoseExtractor:
    def __init__(self, model_path=POSE_MODEL, min_person_conf=0.25, device=None):
        self.model = YOLO(str(model_path))
        self.min_person_conf = min_person_conf
        self.device = device or best_device()

    def detect(self, frame, bed_roi) -> Pose | None:
        """Return the pose of the most confident person whose box center is in bed_roi.

        bed_roi is (x0, y0, x1, y1) as fractions of the frame (0..1).
        Returns None when nobody is found in the bed region.
        """
        result = self.model(frame, verbose=False, conf=self.min_person_conf, device=self.device)[0]
        if result.keypoints is None or len(result.boxes) == 0:
            return None

        h, w = frame.shape[:2]
        rx0, ry0, rx1, ry1 = bed_roi[0] * w, bed_roi[1] * h, bed_roi[2] * w, bed_roi[3] * h
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()

        best = None
        for i, (x0, y0, x1, y1) in enumerate(boxes):
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            in_bed = rx0 <= cx <= rx1 and ry0 <= cy <= ry1
            if in_bed and (best is None or scores[i] > scores[best]):
                best = i
        if best is None:
            return None

        kp = result.keypoints
        return Pose(
            keypoints=kp.xy[best].cpu().numpy(),
            conf=kp.conf[best].cpu().numpy() if kp.conf is not None else np.ones(17),
            box=boxes[best],
        )


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, n_frames / fps


def resize(frame, width=FRAME_WIDTH):
    h, w = frame.shape[:2]
    if width is None or w <= width:
        return frame
    return cv2.resize(frame, (width, int(h * width / w)))


def iter_video(path, analysis_fps=5.0, start_s=0.0, end_s=None, width=FRAME_WIDTH):
    """Yield (time_in_seconds, frame) about `analysis_fps` times per second of video.

    Frames are shrunk to `width` pixels wide (None = keep full size).
    Skipped frames are only grabbed, not decoded, which keeps this fast.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / analysis_fps))
    start_frame = int(start_s * fps)
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    i = start_frame
    try:
        while cap.grab():
            t = i / fps
            if end_s is not None and t > end_s:
                break
            if (i - start_frame) % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                yield t, resize(frame, width)
            i += 1
    finally:
        cap.release()


def draw_overlay(frame, pose: Pose | None, bed_roi, color=(0, 200, 0), label=""):
    """Draw the bed region, the skeleton and a status label onto a copy of the frame (BGR)."""
    out = frame.copy()
    h, w = out.shape[:2]
    x0, y0, x1, y1 = (int(bed_roi[0] * w), int(bed_roi[1] * h), int(bed_roi[2] * w), int(bed_roi[3] * h))
    cv2.rectangle(out, (x0, y0), (x1, y1), (255, 200, 0), 2)

    if pose is not None:
        pts = pose.keypoints.astype(int)
        good = pose.conf >= 0.3
        for a, b in BONES:
            if good[a] and good[b]:
                cv2.line(out, tuple(pts[a]), tuple(pts[b]), color, 3)
        for (x, y), g in zip(pts, good):
            if g:
                cv2.circle(out, (x, y), 4, (255, 255, 255), -1)

    if label:
        cv2.rectangle(out, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(out, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return out
