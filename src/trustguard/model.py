"""ML risk model: TF-IDF text signals + engineered tabular signals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import generate_listings
from .features import NUMERIC_FEATURES, TEXT_FEATURE, build_feature_frame

DEFAULT_MODEL_PATH = Path("artifacts/model.joblib")


def _text_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), TEXT_FEATURE),
        ("num", StandardScaler(), NUMERIC_FEATURES),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced"))])


@dataclass
class RiskModel:
    """Blend of a linear text+tabular model and a gradient-boosted tabular model."""

    linear: Pipeline
    gbm: CalibratedClassifierCV
    blend: float = 0.5
    version: str = "1.0.0"

    @classmethod
    def train(cls, X: pd.DataFrame, y: pd.Series, seed: int = 7) -> RiskModel:
        linear = _text_pipeline().fit(X, y)
        gbm = CalibratedClassifierCV(
            HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_leaf_nodes=24,
                                           l2_regularization=1.0, random_state=seed),
            method="isotonic", cv=3,
        ).fit(X[NUMERIC_FEATURES], y)
        return cls(linear=linear, gbm=gbm)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p_lin = self.linear.predict_proba(X)[:, 1]
        p_gbm = self.gbm.predict_proba(X[NUMERIC_FEATURES])[:, 1]
        return self.blend * p_lin + (1 - self.blend) * p_gbm

    def save(self, path: Path | str = DEFAULT_MODEL_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path | str = DEFAULT_MODEL_PATH) -> RiskModel:
        return joblib.load(path)


def train_default_model(n: int = 8000, seed: int = 7, test_size: float = 0.25):
    """Generate data, split, train. Returns (model, test_listings, X_test, y_test)."""
    df = generate_listings(n=n, seed=seed)
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df["is_violation"], random_state=seed)
    model = RiskModel.train(build_feature_frame(train_df), train_df["is_violation"], seed=seed)
    X_test = build_feature_frame(test_df)
    return model, test_df, X_test, test_df["is_violation"]
