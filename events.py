"""Step 4: per-frame flags -> seizure events.

Single frames are noisy (rolling over looks like fast motion for a second or two).
A seizure is *sustained*, so we raise an event only when most frames in the
last `window_s` seconds were flagged.

    flagged fraction >= min_ratio      -> event starts
    flagged fraction <  min_ratio / 2  -> event ends
    then wait cooldown_s before another event can start
Frames where the person isn't visible don't count either way.
"""
from collections import deque
from dataclasses import dataclass

WATCHING = "watching"
WARMING_UP = "warming up"
NOT_VISIBLE = "person not visible"
SEIZURE = "SEIZURE"


@dataclass
class SeizureEvent:
    start_s: float              # first flagged frame in the window that set off the alarm
    detected_s: float           # when the alarm went off
    end_s: float | None = None  # None while still happening
    peak_ratio: float = 0.0


@dataclass
class Status:
    state: str
    flagged_ratio: float    # fraction of visible frames in the window that were flagged
    visible_ratio: float    # fraction of frames in the window where we had a pose score


class SeizureEventDetector:
    def __init__(self, window_s=10.0, min_ratio=0.6, cooldown_s=30.0, min_visible=0.5):
        self.window_s = window_s
        self.min_ratio = min_ratio
        self.cooldown_s = cooldown_s
        self.min_visible = min_visible
        self.frames: deque[tuple[float, bool | None]] = deque()
        self.events: list[SeizureEvent] = []
        self.active: SeizureEvent | None = None
        self._first_t = None
        self._last_end = float("-inf")

    def update(self, t: float, flagged: bool | None) -> Status:
        """Feed one frame. flagged is True/False, or None if there was no pose score."""
        if self._first_t is None:
            self._first_t = t
        self.frames.append((t, flagged))
        while self.frames[0][0] <= t - self.window_s:
            self.frames.popleft()

        known = [f for _, f in self.frames if f is not None]
        visible = len(known) / len(self.frames)
        ratio = sum(known) / len(known) if known else 0.0

        if self.active is not None:
            self.active.peak_ratio = max(self.active.peak_ratio, ratio)
            if ratio < self.min_ratio / 2:
                self.active.end_s = t
                self._last_end = t
                self.active = None
            else:
                return Status(SEIZURE, ratio, visible)

        if t - self._first_t < self.window_s:
            return Status(WARMING_UP, ratio, visible)
        if visible < self.min_visible:
            return Status(NOT_VISIBLE, ratio, visible)
        if ratio >= self.min_ratio and t - self._last_end >= self.cooldown_s:
            start = next(ft for ft, f in self.frames if f)
            self.active = SeizureEvent(start_s=start, detected_s=t, peak_ratio=ratio)
            self.events.append(self.active)
            return Status(SEIZURE, ratio, visible)
        return Status(WATCHING, ratio, visible)

    def finish(self, t: float):
        """Close an event that is still open when the video ends."""
        if self.active is not None:
            self.active.end_s = t
            self.active = None
