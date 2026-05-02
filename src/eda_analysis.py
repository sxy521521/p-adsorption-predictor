import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ternary
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

Path("results/figures").mkdir(exist_ok=True)

df = pd.read_csv('data/processed/adsorption_data_processed.csv')

print("=== 数据探索性分析 ===")
print("\n描述性统计:")
print(df.describe())

plt.figure(figsize=(14, 6))
df.iloc[:, 2:].boxplot(rot=45)
plt.title('各变量分布箱线图')
plt.tight_layout()
plt.savefig('results/figures/01_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("箱线图已保存")

plt.figure(figsize=(12, 10))
corr_matrix = df.iloc[:, 2:].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('变量相关性热图')
plt.tight_layout()
plt.savefig('results/figures/02_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("相关性热图已保存")

key_vars = ['P adsorption capacity (mg/g)', 'BET surface area (m²/g)', 'Solution pH', 'Initial P concentration (mg/L)']
if all(var in df.columns for var in key_vars):
    sns.pairplot(df[key_vars])
    plt.savefig('results/figures/03_pairplot.png', dpi=300, bbox_inches='tight')
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
    tax.gridlines(color="blue", multiple=0.1)
    
    # 绘制散点
    for point in ternary_normalized:
        tax.scatter([point], color='red', marker='o', s=50, alpha=0.6)
    
    # 设置标签
    tax.left_axis_label(ternary_vars[0], fontsize=12)
    tax.right_axis_label(ternary_vars[1], fontsize=12)
    tax.bottom_axis_label(ternary_vars[2], fontsize=12)
    
    tax.set_title('三元图：BET表面积 vs 反应温度 vs 初始P浓度', fontsize=14, pad=20)
    tax.ticks(axis='lbr', multiple=0.2, tick_formats="%.1f", fontsize=8)
    
    plt.savefig('results/figures/04_ternary_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("三元图已保存")

print("\nEDA分析完成！图表保存在 results/figures/ 目录")
