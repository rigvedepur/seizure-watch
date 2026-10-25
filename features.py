"""Step 2: a short history of poses -> numbers that describe the motion.

A single pose can't tell a seizure from someone stretching. What gives a
seizure away is how the body moves over time: fast, jerky, back-and-forth.
So every frame is described using the last ~1 second of poses.
"""
import warnings
from collections import deque

import numpy as np

from pose import KEYPOINT_NAMES, Pose

KEYPOINT_MIN_CONF = 0.3   # ignore keypoints the pose model is unsure about
MOVING_SPEED = 0.05       # body-lengths per second; slower than this counts as "still"
MIN_POSES = 4             # need a few poses before we can talk about motion


def feature_names():
    names = []
    names += [f"speed_mean_{k}" for k in KEYPOINT_NAMES]
    names += [f"speed_std_{k}" for k in KEYPOINT_NAMES]
    names += [f"reversals_{k}" for k in KEYPOINT_NAMES]
    names += ["speed_mean_all", "visible_fraction"]
    return names


def _nan_stat(fn, x, axis=None):
    """nanmean/nanstd that gives 0 (not a warning) when every value is missing."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nan_to_num(fn(x, axis=axis), nan=0.0)


class PoseHistory:
    """Remembers the poses from the last `seconds` of video."""

    def __init__(self, seconds=1.0):
        self.seconds = seconds
        self.items: deque[tuple[float, Pose]] = deque()

    def add(self, t: float, pose: Pose | None):
        if pose is None:
            # Lost the person: start over rather than measure motion across a gap.
            self.items.clear()
            return
        self.items.append((t, pose))
        while self.items and self.items[0][0] < t - self.seconds:
            self.items.popleft()

    def _velocities(self):
        """Per-step joint velocities in body-lengths per second, plus which ones are trustworthy."""
        ts = np.array([t for t, _ in self.items])
        pts = np.stack([p.keypoints for _, p in self.items])        # (n, 17, 2)
        conf = np.stack([p.conf for _, p in self.items])             # (n, 17)
        scale = np.median([p.scale for _, p in self.items]) or 1.0

        dt = np.diff(ts)[:, None, None]
        vel = np.diff(pts, axis=0) / dt / scale                      # (n-1, 17, 2)
        ok = (conf[1:] >= KEYPOINT_MIN_CONF) & (conf[:-1] >= KEYPOINT_MIN_CONF)
        return vel, ok, ts[-1] - ts[0]

    def motion_score(self) -> float | None:
        """Average joint speed (body-lengths per second). None if not enough history yet."""
        if len(self.items) < MIN_POSES:
            return None
        vel, ok, _ = self._velocities()
        speed = np.linalg.norm(vel, axis=2)
        speed[~ok] = np.nan
        return float(_nan_stat(np.nanmean, speed))

    def features(self) -> np.ndarray | None:
        """Feature vector for a trained model (see feature_names()). None if not enough history yet."""
        if len(self.items) < MIN_POSES:
            return None
        vel, ok, duration = self._velocities()
        speed = np.linalg.norm(vel, axis=2)
        speed[~ok] = np.nan

        # Direction reversals: a joint moving left then right (or up then down).
        # Seizure jerking is rhythmic, so it reverses direction a lot.
        direction = np.sign(vel) * (np.abs(vel) > MOVING_SPEED)
        flips = (direction[1:] * direction[:-1] < 0) & ok[1:, :, None] & ok[:-1, :, None]
        reversals_per_s = flips.sum(axis=(0, 2)) / max(duration, 1e-6)

        return np.concatenate([
            _nan_stat(np.nanmean, speed, axis=0),
            _nan_stat(np.nanstd, speed, axis=0),
            reversals_per_s,
            [_nan_stat(np.nanmean, speed), ok.mean()],
        ]).astype(np.float32)
