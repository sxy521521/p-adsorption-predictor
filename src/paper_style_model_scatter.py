"""按参考论文风格绘制三种模型的联合散点图。每个模型一张图，颜色区分材料状态。"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "adsorption_data_processed.csv"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_PATH = PROJECT_DIR / "results" / "figures" / "04_model_joint_scatter.png"


def style_axis(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(0.75)
    ax.grid(False)
    ax.tick_params(labelsize=8, width=0.7, length=3)


def main():
    df = pd.read_csv(DATA_PATH)
    target = "P adsorption capacity (mg/g)"
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_files = [
        ("CatBoost", MODEL_DIR / "best_CatBoost_model.pkl"),
        ("XGBoost", MODEL_DIR / "best_XGBoost_model.pkl"),
        ("LightGBM", MODEL_DIR / "best_LightGBM_model.pkl"),
    ]
    models = [(name, joblib.load(path)) for name, path in model_files]

    predictions = [(np.asarray(model.predict(X_train)), np.asarray(model.predict(X_test))) for _, model in models]
    upper = float(max(y.max(), *(pred.max() for pair in predictions for pred in pair))) * 1.05
    lower = min(0.0, float(y.min()))

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "SimHei"],
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.4), sharex=True, sharey=True)
    # 参考原文的高对比绿色/橙色组合，分别表示训练集和测试集。
    split_colors = {"train": "#1B9E77", "test": "#D95F02"}
    ideal_color = "#333333"

    for ax, (name, _), (train_pred, test_pred) in zip(axes, models, predictions):
        ax.scatter(y_train, train_pred, s=14, alpha=0.68, color=split_colors["train"], edgecolors="white", linewidths=0.25, label="Train")
        ax.scatter(y_test, test_pred, s=16, alpha=0.72, color=split_colors["test"], edgecolors="white", linewidths=0.25, label="Test")
        ax.plot([lower, upper], [lower, upper], linestyle="--", linewidth=0.9, color=ideal_color, label="Ideal")
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_title(name, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Actual P adsorption capacity (mg/g)", fontsize=8.5)
        style_axis(ax)

        r2_train = r2_score(y_train, train_pred)
        r2_test = r2_score(y_test, test_pred)
        rmse_train = np.sqrt(mean_squared_error(y_train, train_pred))
        rmse_test = np.sqrt(mean_squared_error(y_test, test_pred))
        ax.text(
            0.97, 0.06,
            f"R²_train = {r2_train:.3f}\nR²_test = {r2_test:.3f}\n"
            f"RMSE_train = {rmse_train:.2f}\nRMSE_test = {rmse_test:.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="square,pad=0.35", facecolor="white", edgecolor="#777777", linewidth=0.6, alpha=0.92),
        )

    for ax in axes:
        ax.set_ylabel("Predicted P adsorption capacity (mg/g)", fontsize=8.5)
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=split_colors["train"],
                   markeredgecolor="white", markeredgewidth=0.25, markersize=6, alpha=0.68, label="Train"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=split_colors["test"],
                   markeredgecolor="white", markeredgewidth=0.25, markersize=6, alpha=0.72, label="Test"),
    ]
    for ax in axes:
        ax.legend(
            handles=handles, loc="upper left", fontsize=7.5,
            frameon=True, facecolor="white", edgecolor="#333333",
            framealpha=0.92, borderpad=0.45, handlelength=1.5,
            labelspacing=0.35,
        )
    fig.text(0.5, 0.012, "Observed and model-predicted P adsorption capacity", ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=[0.02, 0.07, 0.98, 0.91], w_pad=1.1)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=400, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
