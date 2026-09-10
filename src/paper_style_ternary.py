"""按参考论文 Fig. 2 风格绘制本项目数据的三元图。"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import ternary
from matplotlib.lines import Line2D

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "adsorption_sample_data.csv"
OUTPUT_PATH = PROJECT_DIR / "results" / "figures" / "04_ternary_plot.png"


def main():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]

    # 先按变量分别做Min–Max标准化，再将每行标准化得分转换为三元比例。
    # 原始变量单位不同，不能直接按原始数值相加后解释为组成比例。
    columns = [
        "Initial P concentration (mg/L)",
        "BET surface area (m²/g)",
        "Reactor temperature (℃)",
    ]
    raw_values = df[columns].astype(float).to_numpy()
    minimum = raw_values.min(axis=0)
    span = raw_values.max(axis=0) - minimum
    scaled_values = (raw_values - minimum) / np.where(span == 0, 1, span)
    row_sum = scaled_values.sum(axis=1, keepdims=True)
    # 极少数三个变量同时处于全局最小值的样本不具有可定义的三元比例，均分处理。
    values = np.divide(
        scaled_values,
        row_sum,
        out=np.full_like(scaled_values, 1 / 3),
        where=row_sum != 0,
    ) * 100
    target = df["P adsorption capacity (mg/g)"].astype(float).to_numpy()

    norm = mpl.colors.Normalize(vmin=np.nanpercentile(target, 2), vmax=np.nanpercentile(target, 98))
    # 参考论文的蓝-蓝紫-紫红-红渐变，避免中间出现明显的黄色断层。
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "paper_blue_purple_red",
        ["#173F8A", "#315FA8", "#5A3E9B", "#9E3A8A", "#D73027"],
        N=256,
    )

    figure, tax = ternary.figure(scale=100)
    figure.set_size_inches(8.8, 7.6)
    for spine in tax.ax.spines.values():
        spine.set_visible(False)
    tax.boundary(linewidth=1.4, axes_colors={"l": "#222222", "r": "#222222", "b": "#222222"})
    tax.gridlines(multiple=10, color="#B8B8B8", linewidth=0.45, alpha=0.75)
    tax.clear_matplotlib_ticks()

    # 用颜色表示吸附容量，用点形状表示改性状态。
    for modified, marker, label in [(0, "o", "Unmodified"), (1, "^", "Modified")]:
        mask = df["Modified or unmodified"].to_numpy() == modified
        points = [tuple(p) for p in values[mask]]
        colors = cmap(norm(target[mask]))
        tax.scatter(points, marker=marker, color=colors, s=25, alpha=0.78,
                    edgecolors="white", linewidths=0.25, label=label)

    tax.left_axis_label("Initial P concentration (normalized %)", fontsize=10, offset=0.14)
    tax.right_axis_label("BET surface area (normalized %)", fontsize=10, offset=0.14)
    tax.bottom_axis_label("Reactor temperature (normalized %)", fontsize=10, offset=0.10)
    tax.ticks(axis="lbr", multiple=20, linewidth=0.6, fontsize=8, tick_formats="%.0f")
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#555555",
               markeredgecolor="white", markersize=7, label="Unmodified"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#555555",
               markeredgecolor="white", markersize=7, label="Modified"),
    ]
    tax.ax.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(1.02, 1.02),
                  frameon=True, facecolor="white", edgecolor="#333333", fontsize=8)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(target)
    cbar = figure.colorbar(sm, ax=tax.ax, fraction=0.045, pad=0.08)
    cbar.set_label("P adsorption capacity (mg/g)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    figure.text(0.5, 0.02, "Axes show row-wise proportions of Min–Max normalized variables; color indicates adsorption capacity.",
                ha="center", fontsize=8, color="#555555")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=400, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(figure)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
