"""统一的论文风格绘图设置。"""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

COLORS = {"blue":"#0072BD", "purple":"#7E57C2", "green":"#77AC30", "orange":"#EDB120", "cyan":"#40BFC1", "red":"#D9534F", "dark":"#222222"}
PALETTE = [COLORS["blue"], COLORS["purple"], COLORS["green"], COLORS["orange"], COLORS["cyan"]]

mpl.rcParams.update({
    "font.family":"sans-serif", "font.sans-serif":["Arial", "DejaVu Sans", "SimHei"],
    "axes.unicode_minus":False, "axes.linewidth":0.9, "axes.edgecolor":COLORS["dark"],
    "xtick.direction":"out", "ytick.direction":"out", "xtick.major.width":0.8,
    "ytick.major.width":0.8, "axes.titlesize":11, "axes.labelsize":10,
    "xtick.labelsize":8.5, "ytick.labelsize":8.5, "legend.fontsize":8,
    "figure.facecolor":"white", "axes.facecolor":"white", "savefig.facecolor":"white",
})

def style_axis(ax, grid=False):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9); ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(length=3, width=0.8)
    if grid:
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.8); ax.set_axisbelow(True)

def save_figure(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=400, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
