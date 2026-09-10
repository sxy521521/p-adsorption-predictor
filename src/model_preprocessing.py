"""Leakage-safe preprocessing shared by tuning, training and ICP workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "data" / "raw" / "adsorption_source_with_missing.csv"
TARGET = "P adsorption capacity (mg/g)"

BINARY_COLUMNS = [
    "Modified or unmodified",
    "Cross-linked or uncross-linked",
]
CATEGORICAL_COLUMNS = [
    "Modified material type",
    "Cross-linking agent type",
]
CATEGORY_LEVELS = {
    "Modified material type": [str(value) for value in range(31)],
    "Cross-linking agent type": [str(value) for value in range(5)],
}
CONTINUOUS_COLUMNS = [
    "Adsorbent dosage (g/L) ",
    "Reactor temperature (℃)",
    "Initial P concentration (mg/L)",
    "Reaction time (min)",
    "Solution pH",
    "Pore volume (cm³/g)",
    "BET surface area (m²/g)",
]
IMPUTED_COLUMNS = ["Pore volume (cm³/g)", "BET surface area (m²/g)"]
MISSINGNESS_COLUMNS = [f"{column} was imputed" for column in IMPUTED_COLUMNS]


def load_source_data(path: Path = SOURCE_PATH) -> pd.DataFrame:
    data = pd.read_csv(path, dtype={column: str for column in CATEGORICAL_COLUMNS})
    for column in BINARY_COLUMNS + CONTINUOUS_COLUMNS + [TARGET]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    for column in CATEGORICAL_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0).astype(int).astype(str)
    if data[TARGET].isna().any():
        raise ValueError("The prediction target contains missing values.")
    return data


def build_dtr(random_state: int) -> DecisionTreeRegressor:
    return DecisionTreeRegressor(
        min_samples_leaf=3,
        min_samples_split=5,
        max_depth=8,
        max_features="sqrt",
        random_state=random_state,
    )


@dataclass
class LeakageSafePreprocessor:
    """Fit categorical encoding and iterative DTR imputation on training rows only."""

    random_state: int = 42
    rounds: int = 2

    def _fit_category_levels(self, train: pd.DataFrame) -> None:
        self.category_levels_ = CATEGORY_LEVELS
        for column, levels in self.category_levels_.items():
            unexpected = set(train[column].astype(str).unique()) - set(levels)
            if unexpected:
                raise ValueError(f"Unexpected values in {column}: {sorted(unexpected)}")
        self.dummy_columns_ = [
            f"{column}_{level}"
            for column in CATEGORICAL_COLUMNS
            for level in self.category_levels_[column][1:]
        ]

    def _categorical_matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        matrix = pd.DataFrame(index=frame.index)
        for column in CATEGORICAL_COLUMNS:
            levels = self.category_levels_[column]
            for level in levels[1:]:
                matrix[f"{column}_{level}"] = frame[column].astype(str).eq(level).astype(float)
        return matrix

    def _feature_matrix(self, frame: pd.DataFrame, target: str) -> pd.DataFrame:
        numeric = [column for column in BINARY_COLUMNS + CONTINUOUS_COLUMNS if column != target]
        return pd.concat(
            [frame[numeric].astype(float), self._categorical_matrix(frame)], axis=1
        )

    def fit_transform(
        self, train: pd.DataFrame, others: list[pd.DataFrame] | None = None
    ) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
        """Fit only on ``train`` and transform train plus any validation/test frames."""
        others = others or []
        self._fit_category_levels(train)
        frames = [train.copy(), *[frame.copy() for frame in others]]
        original_missing = [
            {column: frame[column].isna().copy() for column in IMPUTED_COLUMNS}
            for frame in frames
        ]
        self.models_ = {}

        for round_index in range(self.rounds):
            for target in IMPUTED_COLUMNS:
                train_observed = ~original_missing[0][target]
                x_train = self._feature_matrix(frames[0], target).loc[train_observed]
                y_train = frames[0].loc[train_observed, target]
                transformer = SimpleImputer(strategy="median")
                transformed_train = transformer.fit_transform(x_train)
                model = build_dtr(self.random_state + round_index)
                model.fit(transformed_train, y_train)

                for frame_index, frame in enumerate(frames):
                    missing = original_missing[frame_index][target]
                    if missing.any():
                        x_missing = self._feature_matrix(frame, target).loc[missing]
                        frame.loc[missing, target] = model.predict(transformer.transform(x_missing))
                self.models_[target] = (transformer, model)

        encoded = [
            self.encode(
                frame,
                {f"{column} was imputed": original_missing[index][column] for column in IMPUTED_COLUMNS},
            )
            for index, frame in enumerate(frames)
        ]
        if any(frame.isna().any().any() for frame in encoded):
            raise ValueError("Missing values remain after training-only preprocessing.")
        return encoded[0], encoded[1:]

    def encode(self, frame: pd.DataFrame, missingness: dict[str, pd.Series] | None = None) -> pd.DataFrame:
        direct = frame[BINARY_COLUMNS + CONTINUOUS_COLUMNS].astype(float)
        encoded = pd.concat([direct, self._categorical_matrix(frame)], axis=1)
        if missingness is not None:
            for column in MISSINGNESS_COLUMNS:
                encoded[column] = missingness[column].astype(int)
        encoded[TARGET] = frame[TARGET].astype(float)
        return encoded


def prepare_train_and_others(
    train: pd.DataFrame,
    others: list[pd.DataFrame] | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    return LeakageSafePreprocessor(random_state=random_state).fit_transform(train, others)
