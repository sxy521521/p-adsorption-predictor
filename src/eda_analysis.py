import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ternary
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from paper_style import COLORS, PALETTE, style_axis, save_figure

script_dir = Path(__file__).parent.parent
fig_dir = script_dir / 'results' / 'figures'
df = pd.read_csv(script_dir / 'data/processed/adsorption_data_processed.csv')

print("=== 数据探索性分析 ===")
print("\n描述性统计:")
print(df.describe())

# 独热编码列是 bool 类型，不适合绘制箱线图；箱线图只展示连续数值变量。
feature_cols = list(df.columns[2:])
continuous_cols = [
    c for c in feature_cols
    if pd.api.types.is_numeric_dtype(df[c])
    and not pd.api.types.is_bool_dtype(df[c])
]
excluded_cols = [c for c in feature_cols if c not in continuous_cols]
if excluded_cols:
    print(f"\n箱线图已跳过非连续变量: {', '.join(excluded_cols)}")

fig, ax = plt.subplots(figsize=(9, 4.8))
bp = ax.boxplot([df[c].dropna() for c in continuous_cols], tick_labels=continuous_cols, patch_artist=True, widths=0.55,
                medianprops=dict(color='black', linewidth=1.2), whiskerprops=dict(color='black', linewidth=0.8),
                capprops=dict(color='black', linewidth=0.8), flierprops=dict(marker='^', markerfacecolor='black', markeredgecolor='black', markersize=3.5))
for i, box in enumerate(bp['boxes']): box.set(facecolor=PALETTE[i % len(PALETTE)], alpha=0.85, edgecolor='black', linewidth=0.8)
style_axis(ax, grid=True); ax.set_title('Distribution of input variables', loc='left'); ax.tick_params(axis='x', rotation=38)
save_figure(fig, fig_dir / '01_boxplot.png')
print("箱线图已保存")

fig, ax = plt.subplots(figsize=(8.8, 7.5))
corr_matrix = df.iloc[:, 2:].corr()
sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu', center=0, fmt='.2f', linewidths=0.35, linecolor='white', square=True, cbar_kws={'shrink': 0.75}, ax=ax)
ax.set_title('Pearson correlation map', loc='left'); save_figure(fig, fig_dir / '02_correlation_heatmap.png')
print("相关性热图已保存")

key_vars = ['P adsorption capacity (mg/g)', 'BET surface area (m²/g)', 'Solution pH', 'Initial P concentration (mg/L)']
if all(var in df.columns for var in key_vars):
    pair = sns.pairplot(df[key_vars], corner=True, plot_kws={'s': 18, 'alpha': 0.65, 'color': COLORS['blue']}, diag_kws={'color': COLORS['purple'], 'alpha': 0.75})
    pair.fig.savefig(fig_dir / '03_pairplot.png', dpi=400, bbox_inches='tight', pad_inches=0.04)
    plt.close()
    print("散点图矩阵已保存")

ternary_vars = ['BET surface area (m²/g)', 'Reactor temperature (℃)', 'Initial P concentration (mg/L)']
if all(var in df.columns for var in ternary_vars):
    ternary_data = df[ternary_vars].values
    
    # 数据归一化（三元图要求三个值之和为1）
    row_sums = ternary_data.sum(axis=1, keepdims=True)
    ternary_normalized = ternary_data / row_sums
    
    # 创建三元图
    figure, tax = ternary.figure(scale=1.0)
    tax.boundary(linewidth=2.0)
    tax.gridlines(color="#D9D9D9", multiple=0.1, linewidth=0.5)
    
    # 绘制散点
    for point in ternary_normalized:
        tax.scatter([point], color=COLORS['blue'], marker='o', s=22, alpha=0.5)
    
    # 设置标签
    tax.left_axis_label(ternary_vars[0], fontsize=12)
    tax.right_axis_label(ternary_vars[1], fontsize=12)
    tax.bottom_axis_label(ternary_vars[2], fontsize=12)
    
    tax.set_title('三元图：BET表面积 vs 反应温度 vs 初始P浓度', fontsize=14, pad=20)
    tax.ticks(axis='lbr', multiple=0.2, tick_formats="%.1f", fontsize=8)
    
    save_figure(figure, fig_dir / '04_ternary_plot.png')
    print("三元图已保存")

print("\nEDA分析完成！图表保存在 results/figures/ 目录")
