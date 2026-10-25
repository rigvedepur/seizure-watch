"""Make a synthetic test clip from a single still: sleep, then a tonic-clonic seizure.

The person is animated with a skeleton-driven thin-plate-spline warp of the photo:
joints are moved by forward kinematics, points sampled along each bone follow
their bone rigidly, and fixed anchors on the bed keep the background still.

Timeline (10 s, 30 fps by default):
  0-3 s    asleep, slow breathing
  3-4.5 s  tonic phase: limbs stiffen/flex, head turns, fine tremor
  4.5-10 s clonic phase: rhythmic bilateral jerks slowing from ~4.5 Hz to ~2 Hz

This is a crude motion proxy for pipeline testing, not realistic seizure semiology.

    python synth_seizure.py photo.png videos/synthetic_seizure_10s.mp4
"""
import argparse
import subprocess

import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator

# Joints in the 2000x1373 source photo (YOLO pose, hand-corrected).
# "A" = the body's right side (left of image), "B" = body's left side.
JOINTS = {
    "pelvis": (1130, 650), "neck": (1430, 380), "head": (1580, 170),
    "A_sh": (1250, 330), "A_el": (930, 460), "A_wr": (700, 560),
    "B_sh": (1600, 440), "B_el": (1650, 640), "B_wr": (1600, 830),
    "A_hip": (1020, 610), "A_kn": (640, 800), "A_an": (380, 1060), "A_toe": (90, 1020),
    "B_hip": (1250, 690), "B_kn": (1140, 950), "B_an": (940, 1240), "B_toe": (880, 1340),
}
PARENT = {
    "neck": "pelvis", "head": "neck",
    "A_sh": "neck", "A_el": "A_sh", "A_wr": "A_el",
    "B_sh": "neck", "B_el": "B_sh", "B_wr": "B_el",
    "A_hip": "pelvis", "A_kn": "A_hip", "A_an": "A_kn", "A_toe": "A_an",
    "B_hip": "pelvis", "B_kn": "B_hip", "B_an": "B_kn", "B_toe": "B_an",
}
ORDER = list(PARENT)  # parents always precede children
BONES = [(PARENT[j], j) for j in ORDER]
SRC_W = 2000


def smoothstep(a, b, t):
    x = np.clip((t - a) / (b - a), 0, 1)
    return x * x * (3 - 2 * x)


def clonic_pulse(t, onset=4.5, end=10.0, f0=4.5, f1=2.0):
    """Train of sharp jerks (fast contraction, slower relaxation), slowing over time."""
    if t < onset:
        return 0.0
    # Phase of a chirp whose frequency falls linearly from f0 to f1.
    k = (f1 - f0) / (end - onset)
    tt = t - onset
    phase = f0 * tt + 0.5 * k * tt * tt
    frac = phase % 1.0
    period = 1.0 / (f0 + k * tt)
    s = frac * period  # seconds since this jerk began
    return float(np.where(s < 0.05, s / 0.05, np.exp(-(s - 0.05) / 0.09)))


def joint_angles(t, rng_phase):
    """Rotation (degrees, + = clockwise on screen) of each joint's child bone."""
    tonic = smoothstep(3.0, 3.6, t) * (1 - 0.5 * smoothstep(4.5, 5.5, t))
    clonic = smoothstep(4.3, 4.8, t) * clonic_pulse(t)
    tremor = smoothstep(3.0, 3.4, t) * (1 - smoothstep(4.5, 4.8, t)) * np.sin(2 * np.pi * 9 * t + rng_phase)
    a = {j: 0.0 for j in ORDER}
    a["pelvis"] = 1.5 * clonic                 # whole-body jerk
    a["neck"] = -2.0 * tonic - 2.0 * clonic
    a["head"] = 22 * tonic + 10 * clonic + 1.5 * tremor   # head turns and jerks
    # Arms: flex at elbow, pull in at shoulder (signs mirror for the two sides).
    a["A_sh"] = -6 * tonic - 6 * clonic + 1.5 * tremor
    a["A_el"] = 40 * tonic + 25 * clonic + 2.5 * tremor
    a["B_sh"] = -10 * tonic - 9 * clonic - 1.5 * tremor
    a["B_el"] = -30 * tonic - 22 * clonic - 2.5 * tremor
    # Legs: stiffen in tonic phase, small knee jerks in clonic phase.
    a["A_hip"] = -4 * tonic - 4 * clonic + 1.0 * tremor
    a["A_kn"] = -6 * tonic - 8 * clonic + 1.0 * tremor
    a["A_an"] = 10 * tonic + 6 * clonic
    a["B_hip"] = -2 * tonic - 2.5 * clonic - 1.0 * tremor
    a["B_kn"] = 3 * tonic + 5 * clonic - 1.0 * tremor
    a["B_an"] = 0.0
    return a


def pose_at(t, scale, rng_phase):
    """Forward kinematics: return joint positions and per-bone rigid transforms."""
    base = {j: np.array(p, float) * scale for j, p in JOINTS.items()}
    ang = joint_angles(t, rng_phase)
    breath = 0.0 if t > 3.2 else np.sin(2 * np.pi * t / 4.0)
    # Pelvis translation: breathing drift while asleep, jolts during clonic phase.
    shift = np.array([0.0, 2.0 * breath]) * scale * 2
    shift += np.array([-6.0, 4.0]) * scale * 2 * smoothstep(4.3, 4.8, t) * clonic_pulse(t)

    rot = {"pelvis": 0.0}
    pos = {"pelvis": base["pelvis"] + shift}
    xform = {}
    for j in ORDER:
        p = PARENT[j]
        # Rotation accumulates down the chain; bone p->j is turned by its parent joint's angle.
        r = rot[p] + ang[p]
        rot[j] = r
        th = np.deg2rad(r)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        pos[j] = pos[p] + R @ (base[j] - base[p])
        xform[(p, j)] = (R, base[p], pos[p])
    return base, pos, xform


def bone_samples(base, pos, xform, n=7):
    # Each bone contributes its interior and far end; its start is the parent's end.
    src, dst = [base["pelvis"]], [pos["pelvis"]]
    for (p, j), (R, b0, p0) in xform.items():
        for s in np.linspace(0, 1, n)[1:]:
            q = base[p] + s * (base[j] - base[p])
            src.append(q)
            dst.append(p0 + R @ (q - b0))
    return np.array(src), np.array(dst)


def seg_dist(pts, a, b):
    ab = b - a
    s = np.clip(((pts - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(pts - (a + s[:, None] * ab), axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("out")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=10.0)
    args = ap.parse_args()

    img = cv2.imread(args.image)
    img = img[:-12]  # drop the dark strip along the bottom edge of the photo
    scale = args.width / img.shape[1]
    img = cv2.resize(img, (args.width, 2 * round(img.shape[0] * scale / 2)), interpolation=cv2.INTER_AREA)
    H, W = img.shape[:2]
    times = np.arange(int(args.seconds * args.fps)) / args.fps
    rng_phase = 0.7

    # Fixed anchors: a grid on the bed/background, away from every pose the body takes.
    margin = 95 * scale * (SRC_W / 1000)
    gy, gx = np.mgrid[0:H:H / 14, 0:W:W / 20]
    grid = np.c_[gx.ravel(), gy.ravel()]
    border = np.r_[[[x, 0] for x in np.linspace(0, W, 9)], [[x, H] for x in np.linspace(0, W, 9)],
                   [[0, y] for y in np.linspace(0, H, 7)], [[W, y] for y in np.linspace(0, H, 7)]]
    keep = np.ones(len(grid), bool)
    for t in times[::3]:
        _, pos, _ = pose_at(t, scale, rng_phase)
        base = {j: np.array(p, float) * scale for j, p in JOINTS.items()}
        for p, j in BONES:
            keep &= seg_dist(grid, pos[p], pos[j]) > margin
            keep &= seg_dist(grid, base[p], base[j]) > margin
    anchors = np.unique(np.r_[grid[keep], border].round(1), axis=0)

    # Backward map evaluated on a coarse grid, then upsampled.
    step = 8
    cy, cx = np.mgrid[0:H:step, 0:W:step].astype(float)
    coarse = np.c_[cx.ravel(), cy.ravel()]

    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{W}x{H}", "-r", str(args.fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", args.out],
        stdin=subprocess.PIPE)
    for t in times:
        base, pos, xform = pose_at(t, scale, rng_phase)
        src, dst = bone_samples(base, pos, xform)
        tps = RBFInterpolator(np.r_[dst, anchors], np.r_[src, anchors], kernel="thin_plate_spline")
        m = tps(coarse).astype(np.float32).reshape(cy.shape + (2,))
        mx = cv2.resize(m[..., 0], (W, H), interpolation=cv2.INTER_CUBIC)
        my = cv2.resize(m[..., 1], (W, H), interpolation=cv2.INTER_CUBIC)
        frame = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"wrote {args.out}: {W}x{H}, {len(times)} frames @ {args.fps} fps")


if __name__ == "__main__":
    main()
