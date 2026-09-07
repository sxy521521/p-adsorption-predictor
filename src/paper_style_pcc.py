"""按参考论文 Fig. 3 风格绘制单面板 PCC 相关性图。"""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "adsorption_sample_data.csv"
OUTPUT_PATH = PROJECT_DIR / "results" / "figures" / "03_pcc_map.png"


def main():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    columns = [
        "BET surface area (m²/g)",
        "Pore volume (cm³/g)",
        "Solution pH",
        "Reaction time (min)",
        "Initial P concentration (mg/L)",
        "Adsorbent dosage (g/L)",
        "Reactor temperature (℃)",
        "P adsorption capacity (mg/g)",
    ]
    labels = [
        "BET S. area\n(m²/g)", "P. vol\n(cm³/g)", "Sol. pH",
        "Rx. time\n(min)", "Initial P conc.\n(mg/L)",
        "Ads. dosage\n(g/L)", "Rx. temp.\n(°C)", "P adsorption\n(mg/g)",
    ]
    corr = df[columns].corr(method="pearson")
    corr.index = labels
    corr.columns = labels
    mask = pd.DataFrame(True, index=corr.index, columns=corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            if i >= j:
                mask.iloc[i, j] = False

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "SimHei"],
        "axes.unicode_minus": False,
    })
    fig, ax = plt.subplots(figsize=(7.4, 6.8))
    pcc_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "paper_pcc_warm_cool",
        ["#E9A227", "#FFE28A", "#FFF4B8", "#55B8D4", "#174A9B"],
        N=256,
    )
    sns.heatmap(
        corr, mask=mask, ax=ax, square=True,
        cmap=pcc_cmap, vmin=-0.4, vmax=1.0, center=0,
        linewidths=0.8, linecolor="white",
        cbar_kws={"shrink": 0.78, "pad": 0.04, "label": "Pearson correlation coefficient"},
    )
    ax.tick_params(axis="x", rotation=55, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("PCC map of input and target variables", fontsize=11, fontweight="bold", pad=10, loc="left")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=400, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
