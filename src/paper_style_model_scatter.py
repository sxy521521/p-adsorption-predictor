"""按参考论文风格绘制三种模型的联合散点图。每个模型一张图，颜色区分材料状态。"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score, mean_squared_error

PROJECT_DIR = Path(__file__).resolve().parent.parent
TRAIN_DATA_PATH = PROJECT_DIR / "data" / "processed" / "model_train_processed.csv"
TEST_DATA_PATH = PROJECT_DIR / "data" / "processed" / "model_test_processed.csv"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_PATH = PROJECT_DIR / "results" / "figures" / "04_model_joint_scatter.png"


def style_axis(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(0.75)
    ax.grid(False)
    ax.tick_params(labelsize=8, width=0.7, length=3)


def draw_marginal_density(top, right, observed_train, observed_test, predicted_train, predicted_test, colors):
    """采用与散点一致的颜色绘制边际核密度。"""
    def draw_kde(values, color, axis, orientation):
        values = np.asarray(values, dtype=float)
        if np.unique(values).size < 2:
            return
        grid = np.linspace(0, max(float(values.max()) * 1.08, 1.0), 240)
        density = gaussian_kde(values, bw_method=0.25)(grid)
        if orientation == "top":
            axis.fill_between(grid, density, color=color, alpha=0.18, linewidth=0)
            axis.plot(grid, density, color=color, linewidth=0.75)
        else:
            axis.fill_betweenx(grid, density, color=color, alpha=0.18, linewidth=0)
            axis.plot(density, grid, color=color, linewidth=0.75)
    for values, color in [(observed_train, colors["train"]), (observed_test, colors["test"])]:
        draw_kde(values, color, top, "top")
    for values, color in [(predicted_train, colors["train"]), (predicted_test, colors["test"])]:
        draw_kde(values, color, right, "right")
    top.set_ylim(bottom=0)
    right.set_xlim(left=0)
    for marginal in (top, right):
        marginal.tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        marginal.set_xlabel("")
        marginal.set_ylabel("")
        for spine in marginal.spines.values():
            spine.set_visible(False)
        marginal.set_facecolor("none")


def main():
    target = "P adsorption capacity (mg/g)"
    train = pd.read_csv(TRAIN_DATA_PATH)
    test = pd.read_csv(TEST_DATA_PATH)
    X_train, y_train = train.drop(columns=[target]), train[target]
    X_test, y_test = test.drop(columns=[target]), test[target]

    model_files = [
        ("CatBoost", MODEL_DIR / "best_CatBoost_model.pkl"),
        ("XGBoost", MODEL_DIR / "best_XGBoost_model.pkl"),
        ("LightGBM", MODEL_DIR / "best_LightGBM_model.pkl"),
    ]
    models = [(name, joblib.load(path)) for name, path in model_files]

    predictions = [(np.asarray(model.predict(X_train)), np.asarray(model.predict(X_test))) for _, model in models]
    # 为边际核密度曲线预留尾部空间，避免在最高值处被坐标边界截断。
    observed_upper = float(max(y_train.max(), y_test.max(), *(pred.max() for pair in predictions for pred in pair)))
    upper = max(230.0, observed_upper * 1.15)
    lower = min(0.0, float(y_train.min()), float(y_test.min()))

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "SimHei"],
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    fig = plt.figure(figsize=(11.6, 4.35))
    outer_grid = fig.add_gridspec(1, 3, wspace=0.08)
    axes, top_axes, right_axes = [], [], []
    for i in range(3):
        group_grid = outer_grid[0, i].subgridspec(
        2, 2, height_ratios=[0.80, 4.55], width_ratios=[4.55, 0.80], hspace=0.0, wspace=0.0,
        )
        main_axis = fig.add_subplot(group_grid[1, 0], sharex=axes[0] if axes else None, sharey=axes[0] if axes else None)
        axes.append(main_axis)
        top_axes.append(fig.add_subplot(group_grid[0, 0], sharex=main_axis))
        right_axes.append(fig.add_subplot(group_grid[1, 1], sharey=main_axis))
    # 参考原文的高对比绿色/橙色组合，分别表示训练集和测试集。
    split_colors = {"train": "#1B9E77", "test": "#D95F02"}
    ideal_color = "#333333"

    for ax, top, right, (name, _), (train_pred, test_pred) in zip(axes, top_axes, right_axes, models, predictions):
        ax.scatter(y_train, train_pred, s=14, alpha=0.68, color=split_colors["train"], edgecolors="white", linewidths=0.25, label="Train")
        ax.scatter(y_test, test_pred, s=16, alpha=0.72, color=split_colors["test"], edgecolors="white", linewidths=0.25, label="Test")
        ax.plot([lower, upper], [lower, upper], linestyle="--", linewidth=0.9, color=ideal_color, label="Ideal")
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_xlabel("Actual P adsorption capacity (mg/g)", fontsize=8.5)
        style_axis(ax)
        draw_marginal_density(top, right, y_train, y_test, train_pred, test_pred, split_colors)

        r2_train = r2_score(y_train, train_pred)
        r2_test = r2_score(y_test, test_pred)
        rmse_train = np.sqrt(mean_squared_error(y_train, train_pred))
        rmse_test = np.sqrt(mean_squared_error(y_test, test_pred))
        ax.text(
            0.97, 0.06,
            f"{name}\nR²_train = {r2_train:.3f}, R²_test = {r2_test:.3f}\n"
            f"RMSE_train = {rmse_train:.2f}, RMSE_test = {rmse_test:.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.25,
            bbox=dict(boxstyle="square,pad=0.30", facecolor="white", edgecolor="#888888", linewidth=0.55, alpha=0.90),
        )

    for ax in axes:
        ax.set_ylabel("Predicted P adsorption capacity (mg/g)", fontsize=8.5)
        ax.tick_params(labelleft=True, labelbottom=True)
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=split_colors["train"],
                   markeredgecolor="white", markeredgewidth=0.25, markersize=6, alpha=0.68, label="Train"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=split_colors["test"],
                   markeredgecolor="white", markeredgewidth=0.25, markersize=6, alpha=0.72, label="Test"),
    ]
    for ax in axes:
        legend = ax.legend(
            handles=handles, loc="upper left", fontsize=6.5,
            frameon=True, facecolor="white", edgecolor="#888888", framealpha=0.90,
            borderpad=0.30, handlelength=1.0, labelspacing=0.20,
        )
        legend.get_frame().set_linewidth(0.55)
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.13, top=0.93)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=400, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
