"""Gradient-boosted trees on the tabular view: one pooled model for all 336 leads."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from ..features import WEATHER_TAB_PREFIXES, Samples


def columns(s: Samples, use_weather: bool) -> list[str]:
    if use_weather:
        return list(s.tab.columns)
    return [c for c in s.tab.columns if not c.startswith(WEATHER_TAB_PREFIXES)]


class GBM:
    def __init__(self, use_weather: bool = True, seed: int = 0):
        self.use_weather = use_weather
        self.model = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.05,
            max_iter=1500,
            max_leaf_nodes=31,
            min_samples_leaf=100,
            l2_regularization=1.0,
            categorical_features=["hour", "dow"],
            early_stopping=True,
            n_iter_no_change=40,
            random_state=seed,
        )

    def fit(self, train: Samples, val: Samples) -> GBM:
        cols = columns(train, self.use_weather)
        # The model learns the correction to the 7-day profile baseline.
        self.model.fit(
            train.tab[cols],
            (train.y - train.profile).reshape(-1),
            X_val=val.tab[cols],
            y_val=(val.y - val.profile).reshape(-1),
        )
        return self

    def predict(self, s: Samples) -> np.ndarray:
        cols = columns(s, self.use_weather)
        return self.model.predict(s.tab[cols]).reshape(s.profile.shape) + s.profile
