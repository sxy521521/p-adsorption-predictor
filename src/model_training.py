import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import catboost as cb

# XGBoost 和 LightGBM 在 macOS 上依赖 OpenMP（libomp.dylib）。
# OpenMP 未安装时不要让整个训练脚本在导入阶段退出，先跳过对应模型。
try:
    import xgboost as xgb
except Exception as exc:
    xgb = None
    print(f"警告：XGBoost 不可用，将跳过该模型（{exc.__class__.__name__}）。")

try:
    import lightgbm as lgb
except Exception as exc:
    lgb = None
    print(f"警告：LightGBM 不可用，将跳过该模型（{exc.__class__.__name__}）。")
from pathlib import Path

from model_preprocessing import TARGET, load_source_data, prepare_train_and_others

script_dir = Path(__file__).parent.parent
Path(script_dir / "models").mkdir(exist_ok=True)

def load_optimized_parameters(model_name):
    """Load parameters frozen after development-set Bayesian optimization."""
    with open(script_dir / 'models' / f'{model_name.lower()}_best_params.json', encoding='utf-8') as file:
        return json.load(file)['parameters']

def prepare_data():
    source = load_source_data()
    train_source, test_source = train_test_split(
        source, test_size=0.2, random_state=42
    )
    train, [test] = prepare_train_and_others(
        train_source, [test_source], random_state=42
    )
    processed_dir = script_dir / "data" / "processed"
    train.to_csv(processed_dir / "model_train_processed.csv", index=False, encoding="utf-8-sig")
    test.to_csv(processed_dir / "model_test_processed.csv", index=False, encoding="utf-8-sig")
    return (
        train.drop(columns=[TARGET]), train[TARGET],
        test.drop(columns=[TARGET]), test[TARGET],
    )

def train_models(X_train, y_train):
    models = {}
    
    print("\n=== 训练 CatBoost ===")
    cat_parameters = load_optimized_parameters('CatBoost')
    cat_model = cb.CatBoostRegressor(
        verbose=100,
        random_state=42,
        allow_writing_files=False,
        thread_count=2,
        **cat_parameters,
    )
    cat_model.fit(X_train, y_train)
    models['CatBoost'] = cat_model
    
    if xgb is not None:
        print("\n=== 训练 XGBoost ===")
        xgb_parameters = load_optimized_parameters('XGBoost')
        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1,
            **xgb_parameters,
        )
        xgb_model.fit(X_train, y_train, verbose=100)
        models['XGBoost'] = xgb_model

    if lgb is not None:
        print("\n=== 训练 LightGBM ===")
        lgb_parameters = load_optimized_parameters('LightGBM')
        lgb_model = lgb.LGBMRegressor(
            random_state=42,
            n_jobs=2,
            verbosity=-1,
            **lgb_parameters,
        )
        lgb_model.fit(X_train, y_train)
        models['LightGBM'] = lgb_model
    
    return models

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    metrics = {
        'Model': model_name,
        'R²': r2_score(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred)
    }
    return metrics

def main():
    print("=== 机器学习模型训练 ===")
    
    X_train, y_train, X_test, y_test = prepare_data()
    
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    models = train_models(X_train, y_train)
    
    # 不再生成基于残差标准差的“95%预测区间”图。
    # 项目中的不确定性评估统一由 paper_style_xgboost_conformal.py 的CatBoost-ICP流程完成。
    
    results = []
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)
        print(f"\n{name} 评估结果:")
        print(f"  R²: {metrics['R²']:.4f}")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  MAE: {metrics['MAE']:.4f}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(script_dir / 'results' / 'model_metrics.csv', index=False, encoding='utf-8-sig')
    
    for name, model in models.items():
        joblib.dump(model, script_dir / 'models' / f'best_{name}_model.pkl')
        print(f"{name} 模型已保存")
    
    # 后续论文分析固定使用CatBoost；不再生成含义不明确的通用 best_model.pkl。
    best_model_name = "CatBoost"
    print(f"\n后续分析模型: {best_model_name}")

if __name__ == "__main__":
    main()
