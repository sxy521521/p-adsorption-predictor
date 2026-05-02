import csv
import json
from pathlib import Path
import math

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

def calc_mean(vals):
    return sum(vals) / len(vals) if vals else 0

def calc_std(vals, mean):
    variance = sum((v - mean)**2 for v in vals) / len(vals)
    return math.sqrt(variance)

def calc_percentile(sorted_vals, p):
    n = len(sorted_vals)
    idx = int(n * p)
    if idx >= n:
        idx = n - 1
    return sorted_vals[idx]

def descriptive_stats(values):
    mean = calc_mean(values)
    std = calc_std(values, mean)
    sorted_vals = sorted(values)
    min_val = sorted_vals[0]
    max_val = sorted_vals[-1]
    p25 = calc_percentile(sorted_vals, 0.25)
    median = calc_percentile(sorted_vals, 0.50)
    p75 = calc_percentile(sorted_vals, 0.75)
    return {
        'count': len(values),
        'mean': round(mean, 4),
        'std': round(std, 4),
        'min': round(min_val, 4),
        '25%': round(p25, 4),
        '50%': round(median, 4),
        '75%': round(p75, 4),
        'max': round(max_val, 4)
    }

def correlation(x_vals, y_vals, x_mean, y_mean, x_std, y_std):
    n = len(x_vals)
    cov = sum((x_vals[i] - x_mean) * (y_vals[i] - y_mean) for i in range(n)) / n
    return cov / (x_std * y_std) if x_std > 0 and y_std > 0 else 0

def main():
    print("=== 数据探索性分析 ===\n")

    rows = load_csv('data/processed/adsorption_data_processed.csv')
    fieldnames = list(rows[0].keys())
    numeric_cols = [col for col in fieldnames if col != 'adsorption_capacity']

    print(f"样本数: {len(rows)}")
    print(f"特征数: {len(numeric_cols)}\n")

    print("=" * 60)
    print("1. 描述性统计分析")
    print("=" * 60)

    stats_all = {}
    for col in fieldnames:
        vals = [to_float(row[col]) for row in rows]
        stats = descriptive_stats(vals)
        stats_all[col] = stats
        print(f"\n【{col}】")
        print(f"  计数: {stats['count']}")
        print(f"  均值: {stats['mean']}")
        print(f"  标准差: {stats['std']}")
        print(f"  最小值: {stats['min']}")
        print(f"  25%: {stats['25%']}")
        print(f"  中位数: {stats['50%']}")
        print(f"  75%: {stats['75%']}")
        print(f"  最大值: {stats['max']}")

    print("\n" + "=" * 60)
    print("2. 相关性分析 (Pearson相关系数)")
    print("=" * 60)

    corr_results = []
    target = 'adsorption_capacity'

    print(f"\n与 {target} 的相关性:")
    target_vals = [to_float(row[target]) for row in rows]
    target_mean = calc_mean(target_vals)
    target_std = calc_std(target_vals, target_mean)

    correlations_with_target = []
    for col in numeric_cols:
        col_vals = [to_float(row[col]) for row in rows]
        col_mean = calc_mean(col_vals)
        col_std = calc_std(col_vals, col_mean)

        corr = correlation(target_vals, col_vals, target_mean, col_mean, target_std, col_std)
        correlations_with_target.append((col, corr))

    correlations_with_target.sort(key=lambda x: abs(x[1]), reverse=True)
    for col, corr in correlations_with_target:
        strength = "强" if abs(corr) > 0.5 else ("中等" if abs(corr) > 0.3 else "弱")
        direction = "正" if corr > 0 else "负"
        print(f"  {col}: {corr:.4f} ({direction}相关, {strength})")
        corr_results.append({'var1': target, 'var2': col, 'correlation': round(corr, 4)})

    print("\n变量间相关性矩阵:")
    print(f"\n{'变量1':<25} {'变量2':<25} {'相关系数':>10}")
    print("-" * 62)

    all_corr_pairs = []
    for i, col1 in enumerate(numeric_cols):
        vals1 = [to_float(row[col1]) for row in rows]
        mean1 = calc_mean(vals1)
        std1 = calc_std(vals1, mean1)

        for j, col2 in enumerate(numeric_cols):
            if i < j:
                vals2 = [to_float(row[col2]) for row in rows]
                mean2 = calc_mean(vals2)
                std2 = calc_std(vals2, mean2)

                corr = correlation(vals1, vals2, mean1, mean2, std1, std2)
                all_corr_pairs.append((col1, col2, corr))

    all_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    for col1, col2, corr in all_corr_pairs[:10]:
        print(f"{col1:<25} {col2:<25} {corr:>10.4f}")
        corr_results.append({'var1': col1, 'var2': col2, 'correlation': round(corr, 4)})

    print("\n" + "=" * 60)
    print("3. 分组统计分析 (改性 vs 未改性)")
    print("=" * 60)

    modified = [to_float(row['adsorption_capacity']) for row in rows if row['is_modified'] == '1']
    unmodified = [to_float(row['adsorption_capacity']) for row in rows if row['is_modified'] == '0']

    print(f"\n未改性材料: {len(unmodified)} 个样本")
    print(f"  平均吸附容量: {calc_mean(unmodified):.2f} mg/g")
    print(f"  标准差: {calc_std(unmodified, calc_mean(unmodified)):.2f}")

    print(f"\n改性材料: {len(modified)} 个样本")
    print(f"  平均吸附容量: {calc_mean(modified):.2f} mg/g")
    print(f"  标准差: {calc_std(modified, calc_mean(modified)):.2f}")

    improvement = ((calc_mean(modified) - calc_mean(unmodified)) / calc_mean(unmodified)) * 100
    print(f"\n改性后吸附容量提升: {improvement:.1f}%")

    print("\n" + "=" * 60)
    print("4. 材料类型分析")
    print("=" * 60)

    material_stats = {}
    for row in rows:
        mt = row['material_type']
        if mt not in material_stats:
            material_stats[mt] = []
        material_stats[mt].append(to_float(row['adsorption_capacity']))

    print(f"\n{'材料类型':<20} {'样本数':>8} {'平均吸附容量':>15} {'标准差':>10}")
    print("-" * 55)

    for mt, vals in sorted(material_stats.items()):
        print(f"{mt:<20} {len(vals):>8} {calc_mean(vals):>15.2f} {calc_std(vals, calc_mean(vals)):>10.2f}")

    print("\n" + "=" * 60)
    print("5. 数据质量检查")
    print("=" * 60)

    print("\n缺失值统计:")
    for col in fieldnames:
        missing = sum(1 for row in rows if not row[col] or row[col].strip() == '')
        pct = (missing / len(rows)) * 100 if missing > 0 else 0
        print(f"  {col}: {missing} ({pct:.1f}%)")

    print("\n异常值检测 (IQR方法):")
    outlier_info = []
    for col in numeric_cols[:5]:
        vals = [to_float(row[col]) for row in rows]
        mean = calc_mean(vals)
        sorted_vals = sorted(vals)
        q1 = calc_percentile(sorted_vals, 0.25)
        q3 = calc_percentile(sorted_vals, 0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = sum(1 for v in vals if v < lower or v > upper)
        outlier_info.append({'variable': col, 'outliers': outliers, 'percentage': (outliers/len(vals))*100})
        print(f"  {col}: {outliers} 个异常值 ({(outliers/len(vals))*100:.1f}%)")

    results = {
        'descriptive_stats': stats_all,
        'correlations': corr_results,
        'material_comparison': {
            'modified': {'mean': round(calc_mean(modified), 2), 'std': round(calc_std(modified, calc_mean(modified)), 2)},
            'unmodified': {'mean': round(calc_mean(unmodified), 2), 'std': round(calc_std(unmodified, calc_mean(unmodified)), 2)}
        },
        'material_stats': {mt: {'count': len(vals), 'mean': round(calc_mean(vals), 2)} for mt, vals in material_stats.items()},
        'outliers': outlier_info
    }

    with open('results/eda_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("✓ 分析结果已保存到 results/eda_results.json")
    print("=" * 60)

if __name__ == "__main__":
    main()
