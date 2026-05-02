import csv
import random
import math
from pathlib import Path

Path("data/processed").mkdir(exist_ok=True)
random.seed(42)

n_samples = 300
data = []

material_types = ['biochar', 'MOF', 'activated_carbon', 'chitosan']

for _ in range(n_samples):
    modified_or_unmodified = random.randint(0, 1)
    modified_material_type = random.choice(material_types)
    cross_linked = random.randint(0, 1)
    adsorbent_dosage = random.uniform(0.1, 2.0)
    reactor_temperature = random.uniform(20, 50)
    initial_p_concentration = random.uniform(10, 200)
    reaction_time = random.uniform(30, 240)
    solution_pH = random.uniform(2, 10)
    pore_volume = random.uniform(0.1, 1.5)
    bet_surface_area = random.uniform(50, 500)

    base = 50
    base += bet_surface_area * 0.1
    base += modified_or_unmodified * 20
    base -= abs(solution_pH - 4) * 2
    base += math.log(initial_p_concentration) * 5
    base -= reactor_temperature * 0.1
    base += random.gauss(0, 5)
    adsorption = max(0, base)

    row = {
        'Modified or unmodified': modified_or_unmodified,
        'Modified material type': modified_material_type,
        'Cross-linked or uncross-linked': cross_linked,
        'Adsorbent dosage (g/L)': adsorbent_dosage,
        'Reactor temperature (℃)': reactor_temperature,
        'Initial P concentration (mg/L)': initial_p_concentration,
        'Reaction time (min)': reaction_time,
        'Solution pH': solution_pH,
        'Pore volume (cm³/g)': pore_volume,
        'BET surface area (m²/g)': bet_surface_area,
        'P adsorption capacity (mg/g)': adsorption
    }
    data.append(row)

fieldnames = ['Modified or unmodified', 'Modified material type', 'Cross-linked or uncross-linked',
              'Adsorbent dosage (g/L)', 'Reactor temperature (℃)', 'Initial P concentration (mg/L)',
              'Reaction time (min)', 'Solution pH', 'Pore volume (cm³/g)',
              'BET surface area (m²/g)', 'P adsorption capacity (mg/g)']

with open('data/raw/adsorption_sample_data.csv', 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)

print(f"样本数据已生成，共 {len(data)} 条记录")
print("\n前5条数据预览:")
for i, row in enumerate(data[:5]):
    print(row)