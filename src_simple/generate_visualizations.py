import csv
from PIL import Image, ImageDraw, ImageFont
import math
from pathlib import Path

Path("results/figures").mkdir(exist_ok=True)

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

def draw_text_centered(draw, text, x, y, width, fill=(0, 0, 0)):
    text_width = len(text) * 8
    draw.text((x + (width - text_width) // 2, y), text, fill=fill)

def create_boxplot_image(rows, numeric_cols, output_path):
    width, height = 1200, 600
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)

    margin = 60
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin - 40

    n_vars = len(numeric_cols)
    box_width = plot_width / n_vars - 20

    all_values = []
    for col in numeric_cols:
        vals = [to_float(row[col]) for row in rows]
        all_values.extend(vals)

    global_min = min(all_values)
    global_max = max(all_values)
    global_range = global_max - global_min if global_max != global_min else 1

    y_scale = plot_height / global_range

    for i, col in enumerate(numeric_cols):
        vals = [to_float(row[col]) for row in rows]
        vals_sorted = sorted(vals)

        q1 = calc_percentile(vals_sorted, 0.25)
        median = calc_percentile(vals_sorted, 0.50)
        q3 = calc_percentile(vals_sorted, 0.75)
        iqr = q3 - q1
        lower_whisker = max(min(vals), q1 - 1.5 * iqr)
        upper_whisker = min(max(vals), q3 + 1.5 * iqr)

        x_center = margin + i * (plot_width / n_vars) + plot_width / n_vars / 2

        box_left = x_center - box_width / 2
        box_right = x_center + box_width / 2
        box_top = height - margin - 40 - (q3 - global_min) * y_scale
        box_bottom = height - margin - 40 - (q1 - global_min) * y_scale

        draw.rectangle([box_left, box_top, box_right, box_bottom], outline='blue', width=2)

        median_y = height - margin - 40 - (median - global_min) * y_scale
        draw.line([(box_left, median_y), (box_right, median_y)], fill='red', width=3)

        whisker_top_y = height - margin - 40 - (upper_whisker - global_min) * y_scale
        whisker_bottom_y = height - margin - 40 - (lower_whisker - global_min) * y_scale
        draw.line([(x_center, box_top), (x_center, whisker_top_y)], fill='black', width=2)
        draw.line([(x_center, box_bottom), (x_center, whisker_bottom_y)], fill='black', width=2)

        short_col = col[:10] if len(col) > 10 else col
        draw.text((x_center - len(short_col) * 4, height - margin - 20), short_col, fill='black')

    title_font_size = 20
    draw.text((width // 2 - 150, 20), "Variable Distribution Boxplot", fill='black')

    draw.text((20, height - 30), f"Min: {global_min:.2f}", fill='black')
    draw.text((width - 100, height - 30), f"Max: {global_max:.2f}", fill='black')

    img.save(output_path)
    print(f"Boxplot saved: {output_path}")

def create_correlation_heatmap_image(rows, numeric_cols, output_path):
    width, height = 900, 900
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)

    n = len(numeric_cols)
    cell_size = (min(width, height) - 120) // n
    margin = 80

    corr_matrix = []
    for i, col1 in enumerate(numeric_cols):
        vals1 = [to_float(row[col1]) for row in rows]
        mean1 = calc_mean(vals1)
        std1 = calc_std(vals1, mean1)
        if std1 == 0:
            std1 = 1
        row_corr = []
        for j, col2 in enumerate(numeric_cols):
            vals2 = [to_float(row[col2]) for row in rows]
            mean2 = calc_mean(vals2)
            std2 = calc_std(vals2, mean2)
            if std2 == 0:
                std2 = 1
            cov = sum((vals1[k] - mean1) * (vals2[k] - mean2) for k in range(len(vals1))) / len(vals1)
            corr = cov / (std1 * std2)
            row_corr.append(corr)
        corr_matrix.append(row_corr)

    for i in range(n):
        for j in range(n):
            corr = corr_matrix[i][j]
            r = int((corr + 1) / 2 * 255)
            b = int((1 - corr) / 2 * 255)
            color = (r, 100, b)

            x = margin + j * cell_size
            y = margin + i * cell_size
            draw.rectangle([x, y, x + cell_size, y + cell_size], fill=color, outline='white')

            text = f"{corr:.2f}"
            text_color = (0, 0, 0) if abs(corr) < 0.5 else (255, 255, 255)
            draw.text((x + cell_size // 2 - 15, y + cell_size // 2 - 5), text, fill=text_color)

        short_col = numeric_cols[i][:8] if len(numeric_cols[i]) > 8 else numeric_cols[i]
        draw.text((margin + i * cell_size + cell_size // 2 - 20, margin - 20), short_col, fill='black')
        draw.text((margin - 60, margin + i * cell_size + cell_size // 2 - 5), short_col, fill='black')

    draw.text((width // 2 - 150, 20), "Correlation Heatmap", fill='black')
    img.save(output_path)
    print(f"Correlation heatmap saved: {output_path}")

def create_violin_image(rows, output_path):
    width, height = 800, 500
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)

    margin = 80
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin - 40

    modified = [to_float(row['adsorption_capacity']) for row in rows if row['is_modified'] == '1']
    unmodified = [to_float(row['adsorption_capacity']) for row in rows if row['is_modified'] == '0']

    all_vals = modified + unmodified
    min_val = min(all_vals)
    max_val = max(all_vals)
    val_range = max_val - min_val if max_val != min_val else 1

    y_scale = plot_height / val_range

    def draw_violin(vals, x_center, color, label):
        sorted_vals = sorted(vals)
        n = len(sorted_vals)
        density = [0] * n
        bandwidth = (max_val - min_val) / 20

        for i, v in enumerate(sorted_vals):
            for other in vals:
                density[i] += math.exp(-0.5 * ((v - other) / bandwidth) ** 2)
            density[i] /= len(vals) * bandwidth * math.sqrt(2 * math.pi)

        max_density = max(density)
        if max_density > 0:
            density = [d / max_density * (plot_width / 4) for d in density]

        for i, (v, d) in enumerate(zip(sorted_vals, density)):
            y = height - margin - 40 - (v - min_val) * y_scale
            left_x = x_center - d
            right_x = x_center + d
            draw.line([(left_x, y), (right_x, y)], fill=color, width=3)

        mean_y = height - margin - 40 - (calc_mean(vals) - min_val) * y_scale
        draw.line([(x_center - 20, mean_y), (x_center + 20, mean_y)], fill='red', width=3)

        draw.text((x_center - len(label) * 4, height - margin - 20), label, fill='black')

    draw_violin(unmodified, margin + plot_width * 0.3, 'blue', 'Unmodified (0)')
    draw_violin(modified, margin + plot_width * 0.7, 'green', 'Modified (1)')

    draw.text((width // 2 - 150, 20), "Adsorption Capacity Distribution", fill='black')
    draw.text((20, height - 30), f"Min: {min_val:.2f}", fill='black')
    draw.text((width - 100, height - 30), f"Max: {max_val:.2f}", fill='black')

    img.save(output_path)
    print(f"Violin plot saved: {output_path}")

def create_pareto_front_image(output_path):
    try:
        rows = []
        with open('results/pareto_optimal_solutions.csv', 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except:
        print("Pareto solutions file not found, skipping...")
        return

    width, height = 800, 500
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)

    margin = 80
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin - 40

    adsorptions = [to_float(row['predicted_adsorption']) for row in rows]
    energies = [to_float(row['energy_cost']) for row in rows]

    min_ad = min(adsorptions)
    max_ad = max(adsorptions)
    min_en = min(energies)
    max_en = max(energies)

    ad_range = max_ad - min_ad if max_ad != min_ad else 1
    en_range = max_en - min_en if max_en != min_en else 1

    for row in rows:
        ad = to_float(row['predicted_adsorption'])
        en = to_float(row['energy_cost'])

        x = margin + (en - min_en) / en_range * plot_width
        y = height - margin - 40 - (ad - min_ad) / ad_range * plot_height

        draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill='blue', outline='darkblue')

    draw.text((width // 2 - 100, 20), "Pareto Front", fill='black')
    draw.text((margin, height - 30), f"Energy Min: {min_en:.2f}", fill='black')
    draw.text((margin, height - 50), f"Energy Max: {max_en:.2f}", fill='black')
    draw.text((width - 150, margin - 20), f"Ad Max: {max_ad:.2f}", fill='black')
    draw.text((width - 150, margin), f"Ad Min: {min_ad:.2f}", fill='black')

    draw.text((20, height // 2), "Adsorption\nCapacity", fill='black')
    draw.text((width // 2 - 40, height - 30), "Energy Cost", fill='black')

    img.save(output_path)
    print(f"Pareto front saved: {output_path}")

def main():
    print("=== Generating Visualization Images ===\n")

    rows = load_csv('data/processed/adsorption_data_processed.csv')
    fieldnames = list(rows[0].keys())
    numeric_cols = [col for col in fieldnames if col != 'adsorption_capacity']

    print("Creating boxplot...")
    create_boxplot_image(rows, numeric_cols, 'results/figures/01_boxplot.png')

    print("Creating correlation heatmap...")
    create_correlation_heatmap_image(rows, numeric_cols, 'results/figures/03_correlation_heatmap.png')

    print("Creating violin plot...")
    create_violin_image(rows, 'results/figures/02_violinplot.png')

    print("Creating pareto front...")
    create_pareto_front_image('results/figures/pareto_front.png')

    print("\nAll images generated successfully!")

if __name__ == "__main__":
    main()
