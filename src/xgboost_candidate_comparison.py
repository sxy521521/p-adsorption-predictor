"""Compare a small set of XGBoost configurations using development-set CV.

The independent 20% test set is not accessed here.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "adsorption_data_processed.csv"
OUT = ROOT / "results" / "xgboost_candidate_comparison.csv"
TARGET = "P adsorption capacity (mg/g)"
RANDOM_STATE = 42

# Current manuscript configuration and the most plausible alternatives suggested
# by the one-factor sensitivity screen.
CANDIDATES = {
    "Current (0.05, depth 6, 500 trees)": dict(learning_rate=0.05, max_depth=6, n_estimators=500),
    "Conservative alternative (0.08, depth 5, 500 trees)": dict(learning_rate=0.08, max_depth=5, n_estimators=500),
    "Sensitivity-led alternative (0.10, depth 5, 500 trees)": dict(learning_rate=0.10, max_depth=5, n_estimators=500),
    "More trees alternative (0.10, depth 5, 700 trees)": dict(learning_rate=0.10, max_depth=5, n_estimators=700),
}


def main():
    data = pd.read_csv(DATA)
    x, y = data.drop(columns=[TARGET]), data[TARGET]
    x_dev, _, y_dev, _ = train_test_split(x, y, test_size=0.20, random_state=RANDOM_STATE)
    folds = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, params in CANDIDATES.items():
        fold_metrics = []
        for train_idx, valid_idx in folds.split(x_dev):
            model = XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1, **params)
            model.fit(x_dev.iloc[train_idx], y_dev.iloc[train_idx])
            prediction = model.predict(x_dev.iloc[valid_idx])
            observed = y_dev.iloc[valid_idx]
            fold_metrics.append((r2_score(observed, prediction), np.sqrt(mean_squared_error(observed, prediction)), mean_absolute_error(observed, prediction)))
        metrics = np.asarray(fold_metrics)
        rows.append({"configuration": name, **params,
                     "cv_r2_mean": metrics[:, 0].mean(), "cv_r2_sd": metrics[:, 0].std(ddof=1),
                     "cv_rmse_mean": metrics[:, 1].mean(), "cv_rmse_sd": metrics[:, 1].std(ddof=1),
                     "cv_mae_mean": metrics[:, 2].mean(), "cv_mae_sd": metrics[:, 2].std(ddof=1)})
    result = pd.DataFrame(rows).sort_values(["cv_r2_mean", "cv_rmse_mean"], ascending=[False, True])
    result.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(result.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
