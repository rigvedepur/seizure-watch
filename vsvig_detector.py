"""Track A: the pretrained VSViG model (Xu et al., ECCV 2024), used as-is.

    github.com/xuyankun/VSViG   (code + weights, Apache 2.0, cloned into vendor/VSViG)

How VSViG sees a video (from the paper and code):
  1. Their own pose model (Lightweight OpenPose, fine-tuned on patients in bed)
     finds 18 keypoints on a 1920x1080 frame. 15 are used (neck and ears dropped).
  2. Around each keypoint, a 128x128 Gaussian-weighted image patch is cut out
     and shrunk to 32x32. So the model sees *what the body looks like* near each joint.
  3. A clip = 30 frames covering 5 seconds (6 frames per second).
  4. The model outputs a probability 0..1 that the clip is a seizure.

Details not in the docs were worked out from the weights themselves:
keypoints go in as (x, y, confidence) in 1920x1080 pixels, patches as 0..255 values.
"""
import math
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

VENDOR = Path(__file__).parent / "vendor"
# Appended (not prepended) so their train.py files never shadow ours.
sys.path += [str(VENDOR / "lwopenpose"), str(VENDOR / "VSViG")]
from models.with_mobilenet import PoseEstimationWithMobileNet  # noqa: E402
from modules.keypoints import extract_keypoints, group_keypoints  # noqa: E402
from VSViG import STViG  # noqa: E402

from pose import best_device  # noqa: E402

POSE_WEIGHTS = VENDOR / "VSViG" / "pose.pth"
VSVIG_WEIGHTS = VENDOR / "VSViG" / "VSViG-base.pth"
PARTITION_ORDER = VENDOR / "VSViG" / "dy_point_order.pt"

FULL_W, FULL_H = 1920, 1080      # VSViG patch sizes assume this resolution
CLIP_FRAMES, CLIP_SECONDS = 30, 5.0
FPS = CLIP_FRAMES / CLIP_SECONDS  # 6 frames per second
PATCH_KERNEL, PATCH_SIGMA, PATCH_SCALE = 128, 0.3, 1 / 4

# OpenPose's 18 keypoints -> VSViG's 15, in 5 body parts of 3:
# head (nose, l_eye, r_eye), right arm, left arm, right leg, left leg.
OPENPOSE_TO_VSVIG = [0, 15, 14, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
# Bones between OpenPose keypoints, for drawing.
OPENPOSE_BONES = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
                  (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]


def _gaussian_kernel(size, sigma):
    """Same patch weighting as VSViG's extract_patches.py: 1 at the center, fading to 0."""
    k = np.fromfunction(lambda x, y: np.exp(-((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2)
                                            / (2 * sigma ** 2)), (size, size))
    k = (k - k.min()) / (k.max() - k.min())
    return np.repeat(k[:, :, None], 3, axis=2)


class PatientPose:
    """VSViG's patient-tuned Lightweight OpenPose. Returns 18 (x, y, confidence) keypoints."""

    def __init__(self, device="cpu", height=256):
        self.net = PoseEstimationWithMobileNet()
        state = torch.load(POSE_WEIGHTS, map_location="cpu")["state_dict"]
        self.net.load_state_dict(state)
        self.net.eval().to(device)
        self.device, self.height = device, height

    @torch.no_grad()
    def detect(self, frame, bed_roi):
        """frame: 1920x1080 BGR. Returns (18, 3) array or None if nobody is in the bed region."""
        stride, upsample = 8, 4
        scale = self.height / frame.shape[0]
        img = cv2.resize(frame, (0, 0), fx=scale, fy=scale).astype(np.float32)
        img = (img - 128) / 256
        h, w = img.shape[:2]
        pad_w = math.ceil(w / stride) * stride - w
        img = np.pad(img, ((0, 0), (0, pad_w), (0, 0)))

        x = torch.from_numpy(img).permute(2, 0, 1)[None].to(self.device)
        heatmaps, pafs = self.net(x)[-2:]
        heatmaps = cv2.resize(heatmaps[0].permute(1, 2, 0).cpu().numpy(), (0, 0), fx=upsample, fy=upsample,
                              interpolation=cv2.INTER_CUBIC)
        pafs = cv2.resize(pafs[0].permute(1, 2, 0).cpu().numpy(), (0, 0), fx=upsample, fy=upsample,
                          interpolation=cv2.INTER_CUBIC)

        by_type, total = [], 0
        for k in range(18):
            total += extract_keypoints(heatmaps[:, :, k], by_type, total)
        people, all_kp = group_keypoints(by_type, pafs)
        if len(people) == 0:
            return None
        all_kp[:, :2] = all_kp[:, :2] * stride / upsample / scale

        rx0, ry0, rx1, ry1 = bed_roi[0] * FULL_W, bed_roi[1] * FULL_H, bed_roi[2] * FULL_W, bed_roi[3] * FULL_H
        best, best_score = None, -1
        for person in people:
            kp = np.full((18, 3), -1.0)
            for k in range(18):
                if person[k] != -1:
                    kp[k] = all_kp[int(person[k]), :3]
            found = kp[:, 2] >= 0
            cx, cy = kp[found, 0].mean(), kp[found, 1].mean()
            if rx0 <= cx <= rx1 and ry0 <= cy <= ry1 and person[18] > best_score:
                best, best_score = kp, person[18]
        return best


class VSViGModel:
    def __init__(self, device="cpu"):
        class Opt:  # VSViG-base settings from VSViG.py
            dynamic = 1
            num_layer = [2, 2, 6, 2]
            output_channels = [24, 48, 96, 192]
            dynamic_point_order = torch.load(PARTITION_ORDER)
            expansion = 2
            pos_emb = "stem"

        self.model = STViG(Opt())
        self.model.load_state_dict(torch.load(VSVIG_WEIGHTS, map_location="cpu"))
        self.model.eval().to(device)
        self.device = device

    @torch.no_grad()
    def predict(self, patches, kpts):
        """patches: (30, 15, 3, 32, 32), kpts: (30, 15, 3). Returns seizure probability 0..1."""
        p = torch.from_numpy(patches).float()[None].to(self.device)
        k = torch.from_numpy(kpts).float()[None].to(self.device)
        return float(self.model(p, k))


class VSViGDetector:
    """Streaming wrapper: feed frames at 6 fps, get a seizure probability every `every` frames."""

    def __init__(self, bed_roi, every=3, pose=None, model=None):
        # Pose runs ~3x faster on the GPU; the VSViG model itself is just as fast on CPU.
        self.pose = pose or PatientPose(best_device())
        self.model = model or VSViGModel("cpu")
        self.bed_roi = bed_roi
        self.every = every            # 3 frames at 6 fps = a new prediction every 0.5 s
        self.kernel = _gaussian_kernel(PATCH_KERNEL, PATCH_KERNEL * PATCH_SIGMA)
        self.clip: deque = deque(maxlen=CLIP_FRAMES)
        self.last_kpts = None         # remembers where each joint was, for frames where it's hidden
        self.frames_seen = 0
        self.prob = None

    def _fill_missing(self, kp):
        """Joints the pose model can't see keep their last known position, with confidence 0."""
        kp = kp.copy()
        missing = kp[:, 2] < 0
        if self.last_kpts is not None:
            kp[missing, :2] = self.last_kpts[missing, :2]
        elif (~missing).any():
            kp[missing, :2] = kp[~missing, :2].mean(axis=0)
        kp[missing, 2] = 0.0
        self.last_kpts = kp
        return kp

    def _patches(self, frame, kp15):
        """Cut a Gaussian-weighted 128x128 patch around each joint, shrink to 32x32 (like extract_patches.py)."""
        k = PATCH_KERNEL
        padded = cv2.copyMakeBorder(frame, k, k, k, k, cv2.BORDER_CONSTANT, value=0)
        size = math.ceil(PATCH_SCALE * k)
        out = np.zeros((15, size, size, 3), np.float32)
        for i, (x, y) in enumerate(kp15[:, :2]):
            x0, y0 = int(np.clip(x + 0.5 * k, 0, FULL_W + k)), int(np.clip(y + 0.5 * k, 0, FULL_H + k))
            patch = padded[y0:y0 + k, x0:x0 + k].astype(np.float32) * self.kernel
            patch = (patch - patch.min()) / max(patch.max() - patch.min(), 1e-6) * 255
            out[i] = cv2.resize(patch, (size, size))
        return out.transpose(0, 3, 1, 2)  # (15, 3, 32, 32)

    def update(self, frame):
        """frame: BGR at any resolution. Returns (probability or None, 18 keypoints or None)."""
        if frame.shape[:2] != (FULL_H, FULL_W):
            frame = cv2.resize(frame, (FULL_W, FULL_H))
        kp18 = self.pose.detect(frame, self.bed_roi)
        if kp18 is None:
            self.clip.clear()  # lost the patient: start a fresh clip
            self.prob = None
            return None, None

        kp18 = self._fill_missing(kp18)
        kp15 = kp18[OPENPOSE_TO_VSVIG]
        self.clip.append((self._patches(frame, kp15), kp15))
        self.frames_seen += 1
        if len(self.clip) == CLIP_FRAMES and self.frames_seen % self.every == 0:
            patches = np.stack([p for p, _ in self.clip])
            kpts = np.stack([k for _, k in self.clip])
            self.prob = self.model.predict(patches, kpts)
        return self.prob, kp18


def draw_openpose(frame, kp18, color):
    """Draw VSViG's 18-keypoint skeleton. kp18 is in 1920x1080 coordinates."""
    out = frame
    s = frame.shape[1] / FULL_W
    if kp18 is None:
        return out
    pts = (kp18[:, :2] * s).astype(int)
    seen = kp18[:, 2] > 0
    for a, b in OPENPOSE_BONES:
        if seen[a] and seen[b]:
            cv2.line(out, tuple(pts[a]), tuple(pts[b]), color, 3)
    for (x, y), ok in zip(pts, seen):
        if ok:
            cv2.circle(out, (x, y), 4, (255, 255, 255), -1)
    return out
