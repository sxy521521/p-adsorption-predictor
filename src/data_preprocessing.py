"""Encode the 12-column DTR-imputed source data for machine learning."""

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
CATEGORICAL_COLUMNS = ["Modified material type", "Cross-linking agent type"]


def load_and_preprocess_data(filepath: Path) -> pd.DataFrame:
    if filepath.suffix.lower() == ".xlsx":
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath, encoding="utf-8-sig")
    df.columns = [str(column).replace("\n", "").strip() for column in df.columns]
    df = df.rename(columns={"Adsorbent dosage (g/L)": "Adsorbent dosage (g/L) "})
    df["Cross-linking agent type"] = df["Cross-linking agent type"].fillna("None").astype(str)

    encoded = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=True, dtype=int)
    output = PROJECT_DIR / "data" / "processed" / "adsorption_data_processed.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Raw shape: {df.shape}; processed shape: {encoded.shape}")
    print(f"Saved: {output}")
    return encoded


if __name__ == "__main__":
    load_and_preprocess_data(PROJECT_DIR / "需缺失值补充数据_DTR插补结果.xlsx")
