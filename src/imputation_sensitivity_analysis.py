"""Evaluate whether the main model depends excessively on imputed structural features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from model_preprocessing import TARGET, load_source_data, prepare_train_and_others


ROOT = Path(__file__).resolve().parent.parent
PARAMETER_PATH = ROOT / "models" / "catboost_best_params.json"
OUTPUT_PATH = ROOT / "results" / "imputation_sensitivity_metrics.csv"
RANDOM_STATE = 42
STRUCTURAL_COLUMNS = [
    "Pore volume (cm³/g)",
    "BET surface area (m²/g)",
    "Pore volume (cm³/g) was imputed",
    "BET surface area (m²/g) was imputed",
]


def evaluate(name: str, train: pd.DataFrame, test: pd.DataFrame, parameters: dict, drop_columns: list[str]) -> dict:
    x_train = train.drop(columns=[TARGET, *drop_columns])
    y_train = train[TARGET]
    x_test = test.drop(columns=[TARGET, *drop_columns])
    y_test = test[TARGET]
    model = CatBoostRegressor(
        loss_function="RMSE", random_seed=RANDOM_STATE, thread_count=2,
        allow_writing_files=False, verbose=False, **parameters,
    )
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    return {
        "model_variant": name,
        "n_features": x_train.shape[1],
        "test_r2": r2_score(y_test, prediction),
        "test_rmse_mg_g": np.sqrt(mean_squared_error(y_test, prediction)),
        "test_mae_mg_g": mean_absolute_error(y_test, prediction),
    }


def main() -> None:
    source = load_source_data()
    train_source, test_source = train_test_split(source, test_size=0.20, random_state=RANDOM_STATE)
    train, [test] = prepare_train_and_others(train_source, [test_source], random_state=RANDOM_STATE)
    with PARAMETER_PATH.open(encoding="utf-8") as file:
        parameters = json.load(file)["parameters"]
    results = pd.DataFrame([
        evaluate("Full feature set", train, test, parameters, []),
        evaluate("Without pore volume and BET", train, test, parameters, STRUCTURAL_COLUMNS),
    ])
    results.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(results.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
