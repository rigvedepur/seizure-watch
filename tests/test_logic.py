import numpy as np

from events import NOT_VISIBLE, SEIZURE, WARMING_UP, WATCHING, SeizureEventDetector
from features import PoseHistory, feature_names
from pose import Pose

FPS = 10


def run(detector, flags, t0=0.0):
    statuses = []
    for i, f in enumerate(flags):
        statuses.append(detector.update(t0 + i / FPS, f))
    return statuses


def test_no_event_while_warming_up_even_if_all_flagged():
    d = SeizureEventDetector(window_s=10)
    statuses = run(d, [True] * 50)  # 5 s
    assert all(s.state == WARMING_UP for s in statuses)
    assert d.events == []


def test_short_burst_of_motion_is_not_a_seizure():
    d = SeizureEventDetector(window_s=10, min_ratio=0.6)
    run(d, [False] * 150 + [True] * 30 + [False] * 100)  # 3 s of rolling over
    assert d.events == []


def test_sustained_flags_make_one_event_with_start_and_end():
    d = SeizureEventDetector(window_s=10, min_ratio=0.6, cooldown_s=30)
    statuses = run(d, [False] * 150 + [True] * 400 + [False] * 300)
    assert len(d.events) == 1
    ev = d.events[0]
    assert ev.start_s == 15.0                 # first flagged frame
    assert 15.0 < ev.detected_s < 25.0        # alarm after the window fills up
    assert ev.end_s is not None and ev.end_s > 55.0
    assert SEIZURE in {s.state for s in statuses}
    assert statuses[-1].state == WATCHING


def test_cooldown_prevents_back_to_back_events():
    d = SeizureEventDetector(window_s=10, min_ratio=0.6, cooldown_s=60)
    run(d, [False] * 100 + [True] * 150 + [False] * 100 + [True] * 150)
    assert len(d.events) == 1


def test_missing_person_is_not_counted_as_normal_or_seizure():
    d = SeizureEventDetector(window_s=10, min_visible=0.5)
    statuses = run(d, [False] * 120 + [None] * 120)
    assert statuses[-1].state == NOT_VISIBLE
    assert d.events == []


def test_finish_closes_open_event():
    d = SeizureEventDetector(window_s=10)
    run(d, [False] * 100 + [True] * 200)
    d.finish(30.0)
    assert d.events[0].end_s == 30.0


def make_pose(offset):
    kp = np.zeros((17, 2)) + 100 + offset
    return Pose(keypoints=kp, conf=np.ones(17), box=np.array([0, 0, 300, 400]))  # diagonal 500 px


def test_still_person_has_zero_motion_and_jitter_has_reversals():
    still, shaky = PoseHistory(1.0), PoseHistory(1.0)
    for i in range(10):
        still.add(i / FPS, make_pose(0))
        shaky.add(i / FPS, make_pose(20 if i % 2 else 0))  # 20 px back and forth every frame

    assert still.motion_score() == 0.0
    # 20 px per 0.1 s on both axes = hypot(200, 200) px/s, over a 500 px body
    assert np.isclose(shaky.motion_score(), np.hypot(200, 200) / 500)

    f = dict(zip(feature_names(), shaky.features()))
    assert f["reversals_nose"] > 10
    assert f["visible_fraction"] == 1.0


def test_losing_the_person_resets_history():
    h = PoseHistory(1.0)
    for i in range(10):
        h.add(i / FPS, make_pose(0))
    h.add(1.0, None)
    assert h.motion_score() is None
