"""使用归纳共形预测（ICP）为 CatBoost 的 P 吸附容量预测生成 95% 区间图。

文件名沿用历史命名，实际流程使用CatBoost：训练集拟合模型，独立校准集确定残差分位数，测试集用于最终评估与绘图。
"""
from math import ceil
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "adsorption_data_processed.csv"
MODEL_PATH = PROJECT_DIR / "models" / "catboost_icp_model.pkl"
PARAMETER_PATH = PROJECT_DIR / "models" / "catboost_best_params.json"
FIGURE_PATH = PROJECT_DIR / "results" / "figures" / "05_catboost_conformal_prediction_interval.png"
PREDICTION_PATH = PROJECT_DIR / "results" / "catboost_icp_test_predictions.csv"
METRICS_PATH = PROJECT_DIR / "results" / "catboost_icp_metrics.csv"

TARGET = "P adsorption capacity (mg/g)"
CONFIDENCE_LEVEL = 0.95
ALPHA = 1 - CONFIDENCE_LEVEL
RANDOM_STATE = 42


def conformal_quantile(nonconformity_scores: np.ndarray, alpha: float) -> float:
    """返回有限样本修正后的 split-conformal 残差分位数。"""
    n_calibration = len(nonconformity_scores)
    quantile_level = ceil((n_calibration + 1) * (1 - alpha)) / n_calibration
    # ``higher`` 保证经验覆盖率不低于目标置信水平对应的有限样本界。
    return float(np.quantile(nonconformity_scores, quantile_level, method="higher"))


def style_axis(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#2F2F2F")
        spine.set_linewidth(0.8)
    ax.grid(False)
    ax.tick_params(labelsize=8.5, width=0.7, length=3, color="#2F2F2F")


def main():
    df = pd.read_csv(DATA_PATH)
    x = df.drop(columns=[TARGET])
    y = df[TARGET]

    # 20% 测试集完全保留；其余样本再划出 20% 作为独立校准集。
    x_development, x_test, y_development, y_test = train_test_split(
        x, y, test_size=0.20, random_state=RANDOM_STATE
    )
    x_train, x_calibration, y_train, y_calibration = train_test_split(
        x_development,
        y_development,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    # 使用在独立开发集内经贝叶斯优化后冻结的 CatBoost 参数，仅在训练子集拟合。
    with PARAMETER_PATH.open(encoding="utf-8") as file:
        catboost_parameters = json.load(file)["parameters"]
    model = CatBoostRegressor(
        loss_function="RMSE", random_seed=RANDOM_STATE, thread_count=2,
        allow_writing_files=False, verbose=False, **catboost_parameters,
    )
    model.fit(x_train, y_train)

    calibration_prediction = np.asarray(model.predict(x_calibration))
    calibration_scores = np.abs(y_calibration.to_numpy() - calibration_prediction)
    q_hat = conformal_quantile(calibration_scores, ALPHA)

    test_prediction = np.asarray(model.predict(x_test))
    lower_bound = test_prediction - q_hat
    upper_bound = test_prediction + q_hat
    y_test_array = y_test.to_numpy()
    covered = (y_test_array >= lower_bound) & (y_test_array <= upper_bound)

    # 按真实观测值排序，复刻论文图 5 的“真实曲线 + 灰色预测区间带”形式。
    sorted_index = np.argsort(y_test_array)
    sample_rank = np.arange(1, len(y_test_array) + 1)
    observed_sorted = y_test_array[sorted_index]
    lower_sorted = lower_bound[sorted_index]
    upper_sorted = upper_bound[sorted_index]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "SimHei"],
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.fill_between(
        sample_rank,
        lower_sorted,
        upper_sorted,
        color="#B9B9B9",
        alpha=0.68,
        linewidth=0,
        label="95% prediction interval",
        zorder=1,
    )
    ax.plot(
        sample_rank,
        observed_sorted,
        color="#C73E1D",
        linewidth=1.25,
        label="Observed P",
        zorder=2,
    )
    ax.set_box_aspect(0.92)
    ax.set_xlim(1, len(sample_rank))
    ax.set_xlabel("Test samples (ordered by observed P)", fontsize=10)
    ax.set_ylabel("P adsorption capacity (mg/g)", fontsize=10)
    # 原文图 5 由图注交代模型与方法，图内不再放醒目的大标题。
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        [handles[1], handles[0]],
        [labels[1], labels[0]],
        loc="upper left",
        fontsize=8.5,
        frameon=True,
        facecolor="white",
        edgecolor="#B8B8B8",
        framealpha=0.96,
        borderpad=0.45,
    )
    style_axis(ax)
    fig.tight_layout(pad=0.8)

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=400, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)

    prediction_table = pd.DataFrame({
        "test_sample_rank": sample_rank,
        "observed_p_mg_g": observed_sorted,
        "catboost_prediction_mg_g": test_prediction[sorted_index],
        "lower_95_prediction_bound_mg_g": lower_sorted,
        "upper_95_prediction_bound_mg_g": upper_sorted,
        "within_95_prediction_interval": covered[sorted_index],
    })
    prediction_table.to_csv(PREDICTION_PATH, index=False, encoding="utf-8-sig")

    empirical_coverage = float(np.mean(covered))
    mean_interval_width = float(np.mean(upper_bound - lower_bound))
    metrics_table = pd.DataFrame({
        "metric": [
            "confidence_level",
            "training_samples",
            "calibration_samples",
            "test_samples",
            "conformal_residual_quantile_mg_g",
            "empirical_test_coverage",
            "mean_prediction_interval_width_mg_g",
            "test_r2",
            "test_rmse_mg_g",
        ],
        "value": [
            CONFIDENCE_LEVEL,
            len(y_train),
            len(y_calibration),
            len(y_test),
            q_hat,
            empirical_coverage,
            mean_interval_width,
            r2_score(y_test, test_prediction),
            np.sqrt(mean_squared_error(y_test, test_prediction)),
        ],
    })
    metrics_table.to_csv(METRICS_PATH, index=False, encoding="utf-8-sig")
    joblib.dump(model, MODEL_PATH)

    print(f"Saved figure: {FIGURE_PATH}")
    print(f"Saved test intervals: {PREDICTION_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved ICP model: {MODEL_PATH}")
    print(f"95% ICP interval: ±{q_hat:.2f} mg/g")
    print(f"Empirical test coverage: {empirical_coverage:.3f}")
    print(f"Mean interval width: {mean_interval_width:.2f} mg/g")


if __name__ == "__main__":
    main()
