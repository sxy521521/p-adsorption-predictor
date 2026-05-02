import pandas as pd
import numpy as np
from pathlib import Path

script_dir = Path(__file__).parent.parent
Path(script_dir / "data/processed").mkdir(exist_ok=True)
Path(script_dir / "data/raw").mkdir(exist_ok=True)
np.random.seed(42)

n_samples = 300

data = {
    'Modified or unmodified': np.random.choice([0, 1], n_samples),
    'Modified material type': np.random.choice(['biochar', 'MOF', 'activated_carbon', 'chitosan'], n_samples),
    'Cross-linked or uncross-linked': np.random.choice([0, 1], n_samples),
    'Adsorbent dosage (g/L)': np.random.uniform(0.1, 2.0, n_samples),
    'Reactor temperature (℃)': np.random.uniform(20, 50, n_samples),
    'Initial P concentration (mg/L)': np.random.uniform(10, 200, n_samples),
    'Reaction time (min)': np.random.uniform(30, 240, n_samples),
    'Solution pH': np.random.uniform(2, 10, n_samples),
    'Pore volume (cm³/g)': np.random.uniform(0.1, 1.5, n_samples),
    'BET surface area (m²/g)': np.random.uniform(50, 500, n_samples),
    'P adsorption capacity (mg/g)': np.zeros(n_samples)
}

df = pd.DataFrame(data)

def calculate_adsorption(row):
    base = 50
    base += row['BET surface area (m²/g)'] * 0.1
    base += row['Modified or unmodified'] * 20
    base -= abs(row['Solution pH'] - 4) * 2
    base += np.log(row['Initial P concentration (mg/L)']) * 5
    base -= row['Reactor temperature (℃)'] * 0.1
    base += np.random.normal(0, 5)
    return max(0, base)

df['P adsorption capacity (mg/g)'] = df.apply(calculate_adsorption, axis=1)

df.to_csv(script_dir / 'data/raw/adsorption_sample_data.csv', index=False, encoding='utf-8-sig')
df.to_excel(script_dir / 'data/raw/adsorption_sample_data.xlsx', index=False)

print(f"样本数据已生成，共 {len(df)} 条记录")
print("\n前5条数据预览：")
print(df.head())