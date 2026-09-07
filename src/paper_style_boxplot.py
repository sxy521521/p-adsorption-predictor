"""按参考论文 Fig. 1 风格绘制本项目数据的多子图箱线图。"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "adsorption_sample_data.csv"
OUTPUT_PATH = PROJECT_DIR / "results" / "figures" / "00_paper_boxplot.png"
# Nature 风格的 muted 配色：有区分度，但不过分鲜艳。
COLORS = ["#4C72B0", "#8172B3", "#55A868", "#CCB974", "#64B5CD", "#C44E52", "#937860"]
# 同一类变量复用同一种颜色；用珊瑚红替代紫色，整体更自然。
VARIABLE_COLORS = ["#E07A5F", "#0072BD", "#77AC30", "#EDB120", "#E07A5F", "#40E0D0", "#0072BD"]


def style_axis(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(0.8)
    ax.tick_params(axis="both", labelsize=8, width=0.7, length=3)
    ax.grid(False)


def draw_box(ax, values, color, label):
    values = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    bp = ax.boxplot(
        [values], positions=[1], widths=0.45, patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.0},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
        flierprops={"marker": "^", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 3.2},
    )
    bp["boxes"][0].set_facecolor(color)
    bp["boxes"][0].set_alpha(0.82)
    bp["boxes"][0].set_edgecolor("black")
    ax.set_xlim(0.45, 1.55)
    ax.set_xticks([1])
    ax.set_xticklabels([label])
    style_axis(ax)


def draw_target_violin(ax, df):
    target = "P adsorption capacity (mg/g)"
    groups = [
        df.loc[df["Modified or unmodified"] == 0, target].dropna(),
        df.loc[df["Modified or unmodified"] == 1, target].dropna(),
    ]
    parts = ax.violinplot(groups, positions=[1, 2], widths=0.72, showextrema=False)
    for body, color in zip(parts["bodies"], ["#C44E52", "#64B5CD"]):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_linewidth(0.6)
        body.set_alpha(0.78)
    for x, values in zip([1, 2], groups):
        ax.plot([x - 0.16, x + 0.16], [values.median(), values.median()], color="black", linewidth=1.0)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Unmodified", "Modified"], fontsize=7)
    ax.set_ylabel("P adsorption capacity\n(mg/g)", fontsize=8)
    ax.set_xlabel("Material condition", fontsize=8)
    style_axis(ax)


def main():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    variables = [
        ("Adsorbent dosage (g/L)", "Ads. dosage\n(g/L)"),
        ("Reactor temperature (℃)", "Reactor temp.\n(°C)"),
        ("Initial P concentration (mg/L)", "Initial P conc.\n(mg/L)"),
        ("Reaction time (min)", "Reaction time\n(min)"),
        ("Solution pH", "Solution pH"),
        ("Pore volume (cm³/g)", "Pore volume\n(cm³/g)"),
        ("BET surface area (m²/g)", "BET surface area\n(m²/g)"),
    ]
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans", "SimHei"], "axes.unicode_minus": False})
    # 参考论文中的近方形子图布局，避免 4×2 排列被拉成宽扁形。
    fig, axes = plt.subplots(4, 2, figsize=(8.6, 16.4))
    axes = axes.ravel()
    for ax in axes:
        ax.set_box_aspect(1)
    for i, (column, label) in enumerate(variables):
        draw_box(axes[i], df[column], VARIABLE_COLORS[i], label)
    draw_target_violin(axes[7], df)
    legend = [
        Patch(facecolor=VARIABLE_COLORS[0], edgecolor="black", label="25%-75%"),
        Line2D([0], [0], marker="^", color="white", markerfacecolor="black", markersize=5, label="Outlier"),
        Line2D([0], [0], color="black", linewidth=1.0, label="Median"),
    ]
    axes[0].legend(
        handles=legend,
        loc="upper left",
        fontsize=7.2,
        frameon=True,
        facecolor="white",
        edgecolor="#333333",
        framealpha=0.92,
        borderpad=0.45,
        handlelength=1.5,
        labelspacing=0.35,
    )
    fig.suptitle("Distribution of input variables and adsorption capacity", fontsize=12, fontweight="bold", y=0.999)
    fig.text(0.02, 0.006, "Each panel uses its own y-axis scale because the variables have different units.", fontsize=7.5, color="#555555")
    fig.tight_layout(rect=[0.02, 0.025, 0.98, 0.955], h_pad=1.5, w_pad=1.2)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=400, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
