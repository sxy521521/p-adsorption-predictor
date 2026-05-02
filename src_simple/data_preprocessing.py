import csv
from pathlib import Path

def load_and_preprocess_data(input_file, output_file):
    print("=== 原始数据信息 ===")

    rows = []
    with open(input_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    print(f"数据形状: {len(rows)} 行, {len(fieldnames)} 列")
    print(f"字段: {fieldnames}")

    categorical_cols = ['Modified material type']
    target_col = 'P adsorption capacity (mg/g)'
    numeric_cols = [col for col in fieldnames if col not in categorical_cols]

    material_dummies = {}
    for row in rows:
        mt = row['Modified material type']
        if mt not in material_dummies:
            material_dummies[mt] = []
        material_dummies[mt].append(row)

    unique_materials = sorted(material_dummies.keys())
    print(f"材料类型: {unique_materials}")

    processed_rows = []
    new_fieldnames = []

    for col in numeric_cols:
        if col != target_col:
            if col not in new_fieldnames:
                new_fieldnames.append(col)

    for mat in unique_materials:
        col_name = f'Modified material type_{mat}'
        if col_name not in new_fieldnames:
            new_fieldnames.append(col_name)

    new_fieldnames.append(target_col)

    for i, row in enumerate(rows):
        new_row = {}

        for col in numeric_cols:
            if col != target_col:
                new_row[col] = row[col]

        for mat in unique_materials:
            col_name = f'Modified material type_{mat}'
            new_row[col_name] = 1 if row['Modified material type'] == mat else 0

        new_row[target_col] = row[target_col]
        processed_rows.append(new_row)

    print(f"\n处理后数据形状: {len(processed_rows)} 行, {len(new_fieldnames)} 列")
    print(f"新字段: {new_fieldnames}")

    with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(processed_rows)

    print("\n数据预处理完成！")
    return new_fieldnames

if __name__ == "__main__":
    input_file = 'data/raw/adsorption_sample_data.csv'
    output_file = 'data/processed/adsorption_data_processed.csv'
    load_and_preprocess_data(input_file, output_file)