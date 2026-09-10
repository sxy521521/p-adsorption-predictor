"""Generate SHAP importance from the same CatBoost model used by the web ICP predictor."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import Pool


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "icp_train_processed.csv"
MODEL_PATH = ROOT / "models" / "catboost_icp_model.pkl"
OUTPUT_PATH = ROOT / "results" / "catboost_icp_shap_feature_importance.csv"
TARGET = "P adsorption capacity (mg/g)"

GROUPS = {
    "Modified material type": lambda columns: [column for column in columns if column.startswith("Modified material type_")],
    "Modified / unmodified": lambda columns: ["Modified or unmodified"],
    "Cross-linked / uncross-linked": lambda columns: ["Cross-linked or uncross-linked"],
    "Cross-linking agent type": lambda columns: [column for column in columns if column.startswith("Cross-linking agent type_")],
    "Adsorbent dosage (g/L)": lambda columns: ["Adsorbent dosage (g/L) "],
    "Reactor temperature (°C)": lambda columns: ["Reactor temperature (℃)"],
    "Initial P concentration (mg/L)": lambda columns: ["Initial P concentration (mg/L)"],
    "Reaction time (min)": lambda columns: ["Reaction time (min)"],
    "Solution pH": lambda columns: ["Solution pH"],
    "Pore volume (cm³/g)": lambda columns: ["Pore volume (cm³/g)"],
    "BET surface area (m²/g)": lambda columns: ["BET surface area (m²/g)"],
    "Pore volume missingness": lambda columns: ["Pore volume (cm³/g) was imputed"],
    "BET surface area missingness": lambda columns: ["BET surface area (m²/g) was imputed"],
}


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    x = data.drop(columns=[TARGET])
    model = joblib.load(MODEL_PATH)
    feature_names = list(getattr(model, "feature_names_in_", model.feature_names_))
    x = x.loc[:, feature_names]
    shap_values = np.asarray(model.get_feature_importance(Pool(x, feature_names=feature_names), type="ShapValues"))[:, :-1]
    absolute = pd.Series(np.abs(shap_values).mean(axis=0), index=x.columns)
    rows = []
    for name, selector in GROUPS.items():
        columns = selector(x.columns)
        rows.append({"feature": name, "mean_absolute_shap": float(absolute[columns].sum()), "source_columns": " | ".join(columns)})
    result = pd.DataFrame(rows).sort_values("mean_absolute_shap", ascending=False, ignore_index=True)
    result["percentage_contribution"] = result["mean_absolute_shap"] / result["mean_absolute_shap"].sum() * 100
    result.index += 1
    result.index.name = "rank"
    result.to_csv(OUTPUT_PATH, encoding="utf-8-sig")
    print(result[["feature", "percentage_contribution"]].round(4).to_string())


if __name__ == "__main__":
    main()
