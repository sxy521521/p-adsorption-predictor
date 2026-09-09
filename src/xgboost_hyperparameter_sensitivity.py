"""XGBoost 超参数敏感性分析。

在固定的 80% 开发集内进行 5 折交叉验证；预留的 20% 独立测试集
不参与本分析。采用单因素变化方式考察 learning_rate、max_depth
和 n_estimators 对模型性能的影响。
"""

from pathlib import Path

import matplotlib

# This script is run non-interactively when producing manuscript figures.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "adsorption_data_processed.csv"
RESULT_DIR = PROJECT_DIR / "results"
FIGURE_DIR = RESULT_DIR / "figures"
TARGET = "P adsorption capacity (mg/g)"
RANDOM_STATE = 42

# 当前论文中 XGBoost 的核心参数。每次只改变其中一项，以便直观看到影响。
BASELINE = {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 6}
SENSITIVITY_SETTINGS = {
    "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.10],
    "max_depth": [3, 4, 5, 6, 7, 8],
    "n_estimators": [200, 300, 500, 700, 900],
}


def evaluate_setting(x_development, y_development, parameter, value):
    """Return five-fold CV metrics for one one-factor-at-a-time setting."""
    params = BASELINE.copy()
    params[parameter] = value
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for fold, (train_index, validation_index) in enumerate(cv.split(x_development), start=1):
        model = XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **params,
        )
        model.fit(x_development.iloc[train_index], y_development.iloc[train_index])
        prediction = model.predict(x_development.iloc[validation_index])
        observed = y_development.iloc[validation_index]
        rows.append(
            {
                "fold": fold,
                "r2": r2_score(observed, prediction),
                "rmse": np.sqrt(mean_squared_error(observed, prediction)),
                "mae": mean_absolute_error(observed, prediction),
            }
        )

    result = pd.DataFrame(rows)
    return {
        "parameter": parameter,
        "value": value,
        "r2_mean": result["r2"].mean(),
        "r2_std": result["r2"].std(ddof=1),
        "rmse_mean": result["rmse"].mean(),
        "rmse_std": result["rmse"].std(ddof=1),
        "mae_mean": result["mae"].mean(),
        "mae_std": result["mae"].std(ddof=1),
    }


def draw_figure(results):
    """Create a compact manuscript-ready sensitivity figure."""
    palette = {"learning_rate": "#159a9c", "max_depth": "#e07a3f", "n_estimators": "#376f9f"}
    labels = {"learning_rate": "Learning rate", "max_depth": "Maximum depth", "n_estimators": "Number of trees"}
    metrics = [("r2_mean", "CV R² (mean)"), ("rmse_mean", "CV RMSE (mg/g)"), ("mae_mean", "CV MAE (mg/g)")]

    fig, axes = plt.subplots(3, 3, figsize=(10.2, 8.4))
    for row, (metric, ylabel) in enumerate(metrics):
        for col, parameter in enumerate(SENSITIVITY_SETTINGS):
            axis = axes[row, col]
            subset = results.loc[results["parameter"] == parameter].sort_values("value")
            error_col = metric.replace("_mean", "_std")
            axis.errorbar(
                subset["value"], subset[metric], yerr=subset[error_col],
                color=palette[parameter], marker="o", markersize=4.5,
                capsize=3, linewidth=1.4,
            )
            baseline_value = BASELINE[parameter]
            axis.axvline(baseline_value, color="#555555", linestyle="--", linewidth=1)
            axis.set_xlabel(labels[parameter])
            axis.set_ylabel(ylabel)
            axis.grid(axis="y", alpha=0.18)
            axis.spines[["top", "right"]].set_visible(False)

    fig.suptitle("XGBoost hyperparameter sensitivity on the development set (5-fold CV)", y=0.995, fontsize=12, fontweight="bold")
    fig.text(0.5, 0.005, "Dashed lines denote the parameter values used in the final XGBoost model. Error bars show ±1 SD across folds.", ha="center", fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(FIGURE_DIR / "08_xgboost_hyperparameter_sensitivity.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def main():
    RESULT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)
    data = pd.read_csv(DATA_PATH)
    x = data.drop(columns=[TARGET])
    y = data[TARGET]

    # The held-out test set is deliberately excluded from all parameter comparisons.
    x_development, _, y_development, _ = train_test_split(
        x, y, test_size=0.20, random_state=RANDOM_STATE
    )

    results = []
    for parameter, values in SENSITIVITY_SETTINGS.items():
        for value in values:
            print(f"Evaluating {parameter}={value} ...")
            results.append(evaluate_setting(x_development, y_development, parameter, value))

    result_table = pd.DataFrame(results)
    result_table.to_csv(RESULT_DIR / "xgboost_hyperparameter_sensitivity.csv", index=False, encoding="utf-8-sig")
    draw_figure(result_table)

    print("\nFive-fold CV sensitivity summary:")
    print(result_table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved: {RESULT_DIR / 'xgboost_hyperparameter_sensitivity.csv'}")
    print(f"Saved: {FIGURE_DIR / '08_xgboost_hyperparameter_sensitivity.png'}")


if __name__ == "__main__":
    main()
