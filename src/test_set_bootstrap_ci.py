"""Bootstrap uncertainty intervals for final held-out test metrics."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parent.parent
TEST_PATH = ROOT / "data" / "processed" / "model_test_processed.csv"
OUTPUT_PATH = ROOT / "results" / "model_metrics_bootstrap_ci.csv"
TARGET = "P adsorption capacity (mg/g)"
RANDOM_STATE = 42
N_BOOTSTRAP = 2000


def main() -> None:
    test = pd.read_csv(TEST_PATH)
    x, y = test.drop(columns=[TARGET]), test[TARGET].to_numpy()
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for name in ["CatBoost", "XGBoost", "LightGBM"]:
        model = joblib.load(ROOT / "models" / f"best_{name}_model.pkl")
        prediction = np.asarray(model.predict(x))
        values = {"r2": [], "rmse_mg_g": [], "mae_mg_g": []}
        for _ in range(N_BOOTSTRAP):
            index = rng.integers(0, len(y), len(y))
            values["r2"].append(r2_score(y[index], prediction[index]))
            values["rmse_mg_g"].append(np.sqrt(mean_squared_error(y[index], prediction[index])))
            values["mae_mg_g"].append(mean_absolute_error(y[index], prediction[index]))
        for metric, samples in values.items():
            point = {"r2": r2_score(y, prediction), "rmse_mg_g": np.sqrt(mean_squared_error(y, prediction)), "mae_mg_g": mean_absolute_error(y, prediction)}[metric]
            low, high = np.percentile(samples, [2.5, 97.5])
            rows.append({"model": name, "metric": metric, "point_estimate": point, "bootstrap_lower_95": low, "bootstrap_upper_95": high, "n_bootstrap": N_BOOTSTRAP})
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(result.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
