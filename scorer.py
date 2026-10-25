"""Step 3: decide, frame by frame, "does this look like a seizure?"

Two scorers with the same interface, so the app can swap them:
  - MotionRuleScorer: no training, just "is the body moving fast?"  (the baseline)
  - TrainedModelScorer: a model trained with train.py on labeled videos
Both return (score, flagged). score is None when there isn't enough pose history yet.
"""
from pathlib import Path

import joblib

from features import PoseHistory

TRAINED_MODEL = Path(__file__).parent / "models" / "seizure_model.joblib"


class MotionRuleScorer:
    name = "Motion rule (no training)"

    def __init__(self, speed_threshold=0.3):
        self.threshold = speed_threshold  # body-lengths per second

    def score(self, history: PoseHistory):
        s = history.motion_score()
        if s is None:
            return None, None
        return s, s >= self.threshold


class TrainedModelScorer:
    name = "Trained model"

    def __init__(self, path=TRAINED_MODEL, threshold=0.5):
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.history_s = bundle["history_s"]
        self.threshold = threshold  # probability of "seizure" needed to flag a frame

    def score(self, history: PoseHistory):
        x = history.features()
        if x is None:
            return None, None
        p = float(self.model.predict_proba(x[None, :])[0, 1])
        return p, p >= self.threshold
