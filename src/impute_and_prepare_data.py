"""Prepare the revised adsorption dataset and impute pore volume and BET area.

The source workbook contains '-' placeholders for missing pore-volume and BET
values. Binary status fields remain 0/1, while material type and cross-linking
agent type are one-hot encoded because their integer codes are nominal labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_COLUMNS = [
    "Modified or unmodified",
    "Modified material type",
    "Cross-linked or uncross-linked",
    "Cross-linking agent type",
    "Adsorbent dosage (g/L) ",
    "Reactor temperature (℃)",
    "Initial P concentration (mg/L)",
    "Reaction time (min)",
    "Solution pH",
    "Pore volume (cm³/g)",
    "BET surface area (m²/g)",
    "P adsorption capacity (mg/g)",
]

BINARY_COLUMNS = [
    "Modified or unmodified",
    "Cross-linked or uncross-linked",
]

CATEGORICAL_COLUMNS = [
    "Modified material type",
    "Cross-linking agent type",
]

CONTINUOUS_COLUMNS = [
    "Adsorbent dosage (g/L) ",
    "Reactor temperature (℃)",
    "Initial P concentration (mg/L)",
    "Reaction time (min)",
    "Solution pH",
    "Pore volume (cm³/g)",
    "BET surface area (m²/g)",
]

NUMERIC_COLUMNS = BINARY_COLUMNS + CONTINUOUS_COLUMNS

TARGET_COLUMNS = ["Pore volume (cm³/g)", "BET surface area (m²/g)"]


def build_dtr(random_state: int = 42) -> DecisionTreeRegressor:
    return DecisionTreeRegressor(
        min_samples_leaf=3,
        min_samples_split=5,
        max_depth=8,
        max_features="sqrt",
        random_state=random_state,
    )


def load_source(workbook: Path) -> pd.DataFrame:
    # 新上传的工作簿使用 Sheet1；不固定工作表名称，兼容后续修订版。
    df = pd.read_excel(workbook)
    df.columns = [str(column).replace("\n", "").strip() for column in df.columns]
    df = df.replace(r"^\s*$", np.nan, regex=True).replace("-", np.nan)
    df = df.rename(columns={"Adsorbent dosage (g/L)": "Adsorbent dosage (g/L) "})

    required = set(MODEL_COLUMNS) | {"Cross-linking agent type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Source workbook is missing columns: {sorted(missing)}")

    for column in NUMERIC_COLUMNS + ["P adsorption capacity (mg/g)"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    # 类别编号只表示不同类别，不具有数值大小或先后顺序。
    # 0 分别表示未改性材料的基准类型、未使用交联剂的基准类型。
    for column in CATEGORICAL_COLUMNS:
        df[column] = (
            pd.to_numeric(df[column], errors="coerce")
            .fillna(0)
            .astype(int)
            .astype(str)
        )
    return df


def make_imputation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    categorical = pd.get_dummies(
        df[CATEGORICAL_COLUMNS],
        columns=CATEGORICAL_COLUMNS,
        drop_first=True,
        dtype=float,
    )
    # 目标变量不能参与输入特征补充，否则会把结果信息泄漏到输入变量中。
    return pd.concat([df[NUMERIC_COLUMNS].astype(float), categorical], axis=1)


def dtr_impute(matrix: pd.DataFrame, random_state: int = 42, rounds: int = 2) -> pd.DataFrame:
    """Use DTR to fill only the original missing values of pore volume and BET."""
    result = matrix.copy()
    original_missing = {column: result[column].isna() for column in TARGET_COLUMNS}

    for _ in range(rounds):
        for target in TARGET_COLUMNS:
            missing_mask = original_missing[target]
            if not missing_mask.any():
                continue
            feature_columns = [column for column in result.columns if column != target]
            train_mask = ~original_missing[target]
            transformer = SimpleImputer(strategy="median")
            x_train = transformer.fit_transform(result.loc[train_mask, feature_columns])
            x_missing = transformer.transform(result.loc[missing_mask, feature_columns])
            model = build_dtr(random_state)
            model.fit(x_train, result.loc[train_mask, target])
            result.loc[missing_mask, target] = model.predict(x_missing)
    return result


def validate_imputation(matrix: pd.DataFrame, repeats: int = 5) -> pd.DataFrame:
    """Mask observed values to estimate reconstruction quality under this dataset."""
    records: list[dict[str, float | int | str]] = []
    for repeat in range(repeats):
        masked = matrix.copy()
        rng = np.random.default_rng(100 + repeat)
        for column in TARGET_COLUMNS:
            observed_index = masked.index[masked[column].notna()]
            holdout_size = max(1, round(len(observed_index) * 0.1))
            holdout_index = rng.choice(observed_index, size=holdout_size, replace=False)
            truth = matrix.loc[holdout_index, column].to_numpy()
            masked.loc[holdout_index, column] = np.nan
            predicted_frame = dtr_impute(masked, random_state=100 + repeat)
            estimate = predicted_frame.loc[holdout_index, column].to_numpy()
            records.append(
                {
                    "repeat": repeat + 1,
                    "variable": column,
                    "n_holdout": holdout_size,
                    "MAE": mean_absolute_error(truth, estimate),
                    "RMSE": mean_squared_error(truth, estimate) ** 0.5,
                    "R2": r2_score(truth, estimate),
                }
            )
    return pd.DataFrame(records)


def run(workbook: Path) -> None:
    source = load_source(workbook)
    matrix = make_imputation_matrix(source)
    missing_before = source[TARGET_COLUMNS].isna().sum().to_dict()
    validation = validate_imputation(matrix)

    completed_matrix = dtr_impute(matrix)
    completed = source[MODEL_COLUMNS].copy()
    for column in TARGET_COLUMNS:
        completed[column] = completed_matrix[column]

    if completed.isna().any().any():
        raise ValueError("Missing values remain after imputation.")
    if (completed[TARGET_COLUMNS] < 0).any().any():
        raise ValueError("Imputation generated an invalid negative pore volume or BET area.")

    source_path = PROJECT_ROOT / "data/raw/adsorption_source_with_missing.csv"
    raw_path = PROJECT_ROOT / "data/raw/adsorption_sample_data.csv"
    processed_path = PROJECT_ROOT / "data/processed/adsorption_data_processed.csv"
    validation_path = PROJECT_ROOT / "results/imputation_validation_metrics.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.parent.mkdir(parents=True, exist_ok=True)

    # 保留一份尚未插补的规范化源数据，供严格的训练集内插补流程使用。
    source.to_csv(source_path, index=False, encoding="utf-8-sig")
    completed.to_csv(raw_path, index=False, encoding="utf-8-sig")
    processed = pd.get_dummies(
        completed,
        columns=CATEGORICAL_COLUMNS,
        drop_first=True,
        dtype=int,
    )
    processed.to_csv(processed_path, index=False, encoding="utf-8-sig")
    validation.to_csv(validation_path, index=False)

    summary = validation.groupby("variable")[["MAE", "RMSE", "R2"]].mean().round(4)
    print("Rows:", len(completed))
    print("Missing values before imputation:", missing_before)
    print("Validation summary (five masked-data repeats):")
    print(summary.to_string())
    print("Saved:", source_path)
    print("Saved:", raw_path)
    print("Saved:", processed_path)
    print("Saved:", validation_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("workbook", type=Path, help="Revised source workbook (.xlsx)")
    args = parser.parse_args()
    run(args.workbook)
