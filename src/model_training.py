import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
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
import sys
sys.path.insert(0, str(Path(__file__).parent))
from paper_style import COLORS, style_axis

script_dir = Path(__file__).parent.parent
Path(script_dir / "models").mkdir(exist_ok=True)

def load_optimized_parameters(model_name):
    """Load parameters frozen after development-set Bayesian optimization."""
    with open(script_dir / 'models' / f'{model_name.lower()}_best_params.json', encoding='utf-8') as file:
        return json.load(file)['parameters']

def load_data():
    df = pd.read_csv(script_dir / 'data/processed/adsorption_data_processed.csv')
    X = df.drop('P adsorption capacity (mg/g)', axis=1)
    y = df['P adsorption capacity (mg/g)']
    return X, y

def train_models(X_train, X_test, y_train, y_test):
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

def plot_joint_scatter(models, X_train, X_test, y_train, y_test):
    fig_dir = script_dir / 'results' / 'figures'
    
    for model_name, model in models.items():
        # 训练集预测
        y_train_pred = model.predict(X_train)
        r2_train = r2_score(y_train, y_train_pred)
        
        # 测试集预测
        y_test_pred = model.predict(X_test)
        r2_test = r2_score(y_test, y_test_pred)
        
        # 创建联合散点图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.2), sharex=True, sharey=True)
        
        # 训练集
        sns.scatterplot(x=y_train, y=y_train_pred, ax=ax1, color=COLORS['blue'], alpha=0.65, s=18, edgecolor='none')
        ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color=COLORS['dark'], linestyle='--', lw=1)
        ax1.set_xlabel('实际值 (mg/g)')
        ax1.set_ylabel('预测值 (mg/g)')
        ax1.set_title(f'(a) Training (R² = {r2_train:.3f})', loc='left')
        style_axis(ax1, grid=True)
        
        # 测试集
        sns.scatterplot(x=y_test, y=y_test_pred, ax=ax2, color=COLORS['orange'], alpha=0.8, s=22, edgecolor='none')
        ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color=COLORS['dark'], linestyle='--', lw=1)
        ax2.set_xlabel('实际值 (mg/g)')
        ax2.set_ylabel('预测值 (mg/g)')
        ax2.set_title(f'(b) Testing (R² = {r2_test:.3f})', loc='left')
        style_axis(ax2, grid=True)
        
        plt.suptitle(model_name, fontsize=12, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(fig_dir / f'model_{model_name.lower()}_jointplot.png', dpi=400, bbox_inches='tight', pad_inches=0.04)
        plt.close()
        print(f"{model_name} 联合散点图已保存")

def main():
    print("=== 机器学习模型训练 ===")
    
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    models = train_models(X_train, X_test, y_train, y_test)
    
    plot_joint_scatter(models, X_train, X_test, y_train, y_test)
    
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
    best_model = models[best_model_name]
    print(f"\n后续分析模型: {best_model_name}")
    
    print(f"\n{best_model_name} 特征重要性:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    feature_importance.to_csv(script_dir / 'results' / 'feature_importance.csv', index=False, encoding='utf-8-sig')
    print("特征重要性已保存到 results/feature_importance.csv")

if __name__ == "__main__":
    main()
