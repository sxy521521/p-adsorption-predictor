"""用统一的 5 折交叉验证比较多种回归模型。"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "adsorption_data_processed.csv"
OUTPUT_PATH = PROJECT_DIR / "results" / "model_comparison_cv.csv"


def build_models():
    models = {
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=400, min_samples_leaf=1, max_features=1.0,
            random_state=42, n_jobs=1,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=400, min_samples_leaf=1, max_features=0.8,
            random_state=42, n_jobs=1,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=0.1, random_state=42,
        ),
        "SVR_RBF": make_pipeline(
            StandardScaler(), SVR(C=30, gamma="scale", epsilon=0.05, kernel="rbf")
        ),
    }
    try:
        import catboost as cb
        models["CatBoost"] = cb.CatBoostRegressor(
            iterations=400, learning_rate=0.05, depth=6,
            loss_function="RMSE", verbose=False, random_seed=42,
        )
    except Exception as exc:
        print(f"跳过 CatBoost: {exc}")
    try:
        import xgboost as xgb
        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=6,
            subsample=0.9, colsample_bytree=0.9, objective="reg:squarederror",
            random_state=42, n_jobs=1,
        )
    except Exception as exc:
        print(f"跳过 XGBoost: {exc}")
    try:
        import lightgbm as lgb
        models["LightGBM"] = lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=6,
            num_leaves=31, subsample=0.9, colsample_bytree=0.9,
            random_state=42, n_jobs=1, verbosity=-1,
        )
    except Exception as exc:
        print(f"跳过 LightGBM: {exc}")
    return models


def main():
    df = pd.read_csv(DATA_PATH)
    target = "P adsorption capacity (mg/g)"
    X = df.drop(columns=[target])
    y = df[target]
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    models = build_models()
    rows = []

    for name, model in models.items():
        print(f"评估 {name} ...")
        scores = cross_validate(
            clone(model), X, y, cv=cv,
            scoring={"r2": "r2", "rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"},
            n_jobs=1, return_train_score=False,
        )
        row = {
            "Model": name,
            "CV_R2_mean": scores["test_r2"].mean(),
            "CV_R2_std": scores["test_r2"].std(),
            "CV_RMSE_mean": -scores["test_rmse"].mean(),
            "CV_RMSE_std": scores["test_rmse"].std(),
            "CV_MAE_mean": -scores["test_mae"].mean(),
            "CV_MAE_std": scores["test_mae"].std(),
        }
        rows.append(row)
        print(f"  R²={row['CV_R2_mean']:.4f} ± {row['CV_R2_std']:.4f}; RMSE={row['CV_RMSE_mean']:.3f} ± {row['CV_RMSE_std']:.3f}")

    result = pd.DataFrame(rows).sort_values(["CV_RMSE_mean", "CV_R2_mean"], ascending=[True, False])
    result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print("\n=== 模型排名 ===")
    print(result.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n结果已保存到: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
