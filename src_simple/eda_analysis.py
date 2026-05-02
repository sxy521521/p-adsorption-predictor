import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

Path("results/figures").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

def load_csv(filepath):
    rows = []
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows

def to_float(val):
    try:
        return float(val)
    except:
        return 0.0

def plot_boxplot(rows, fieldnames, numeric_cols):
    print("生成箱线图...")
    fig, ax = plt.subplots(figsize=(14, 6))

    data_to_plot = []
    labels = []
    for col in numeric_cols:
        values = [to_float(row[col]) for row in rows]
        data_to_plot.append(values)
        labels.append(col)

    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')

    ax.set_ylabel('值')
    ax.set_title('各变量分布箱线图')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('results/figures/01_boxplot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("箱线图已保存")

def plot_violin_modified(rows):
    print("生成小提琴图...")
    fig, ax = plt.subplots(figsize=(10, 6))

    modified = [to_float(row['adsorption_capacity']) for row in rows if row['is_modified'] == '1']
    unmodified = [to_float(row['adsorption_capacity']) for row in rows if row['is_modified'] == '0']

    parts = ax.violinplot([unmodified, modified], positions=[0, 1], showmeans=True, showmedians=True)

    for pc in parts['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_alpha(0.7)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['未改性 (0)', '改性 (1)'])
    ax.set_xlabel('是否改性')
    ax.set_ylabel('吸附容量 (mg/g)')
    ax.set_title('改性/未改性材料的吸附容量分布')
    plt.tight_layout()
    plt.savefig('results/figures/02_violinplot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("小提琴图已保存")

def plot_correlation_heatmap(rows, fieldnames, numeric_cols):
    print("生成相关性热图...")
    n = len(numeric_cols)
    corr_matrix = np.zeros((n, n))

    for i, col1 in enumerate(numeric_cols):
        vals1 = [to_float(row[col1]) for row in rows]
        mean1 = sum(vals1) / len(vals1)
        std1 = (sum((v - mean1)**2 for v in vals1) / len(vals1)) ** 0.5
        if std1 == 0:
            std1 = 1

        for j, col2 in enumerate(numeric_cols):
            vals2 = [to_float(row[col2]) for row in rows]
            mean2 = sum(vals2) / len(vals2)
            std2 = (sum((v - mean2)**2 for v in vals2) / len(vals2)) ** 0.5
            if std2 == 0:
                std2 = 1

            cov = sum((vals1[k] - mean1) * (vals2[k] - mean2) for k in range(len(vals1))) / len(vals1)
            corr_matrix[i][j] = cov / (std1 * std2)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(numeric_cols, rotation=45, ha='right')
    ax.set_yticklabels(numeric_cols)

    for i in range(n):
        for j in range(n):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)

    plt.colorbar(im, ax=ax, label='相关系数')
    ax.set_title('变量相关性热图')
    plt.tight_layout()
    plt.savefig('results/figures/03_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("相关性热图已保存")

def plot_pairplot_key(rows):
    print("生成散点图矩阵...")
    key_vars = ['adsorption_capacity', 'surface_area', 'pH', 'initial_concentration']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, var in enumerate(key_vars):
        ax = axes[idx]
        other_vars = [v for v in key_vars if v != var]
        for other in other_vars[:1]:
            x_vals = [to_float(row[other]) for row in rows]
            y_vals = [to_float(row[var]) for row in rows]
            ax.scatter(x_vals, y_vals, alpha=0.5, s=20)
            ax.set_xlabel(other)
            ax.set_ylabel(var)
            ax.set_title(f'{var} vs {other}')

    plt.tight_layout()
    plt.savefig('results/figures/04_pairplot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("散点图矩阵已保存")

def main():
    print("=== 数据探索性分析 ===")

    rows = load_csv('data/processed/adsorption_data_processed.csv')
    fieldnames = list(rows[0].keys())
    numeric_cols = [col for col in fieldnames if col != 'adsorption_capacity']

    print("\n描述性统计:")
    print(f"样本数: {len(rows)}")
    print(f"字段数: {len(fieldnames)}")

    print("\n正在生成可视化图表...")
    plot_boxplot(rows, fieldnames, numeric_cols)
    plot_violin_modified(rows)
    plot_correlation_heatmap(rows, fieldnames, numeric_cols)
    plot_pairplot_key(rows)

    print("\nEDA分析完成！图表保存在 results/figures/ 目录")

if __name__ == "__main__":
    main()
