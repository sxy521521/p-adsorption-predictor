import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import catboost as cb
import xgboost as xgb
import lightgbm as lgb
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

Path("models").mkdir(exist_ok=True)

def load_data():
    df = pd.read_csv('data/processed/adsorption_data_processed.csv')
    X = df.drop('P adsorption capacity (mg/g)', axis=1)
    y = df['P adsorption capacity (mg/g)']
    return X, y

def train_models(X_train, X_test, y_train, y_test):
    models = {}
    
    print("\n=== 训练 CatBoost ===")
    cat_model = cb.CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        verbose=100,
        random_state=42,
        early_stopping_rounds=50
    )
    cat_model.fit(X_train, y_train, eval_set=(X_test, y_test))
    models['CatBoost'] = cat_model
    
    print("\n=== 训练 XGBoost ===")
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        early_stopping_rounds=50
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
    models['XGBoost'] = xgb_model
    
    print("\n=== 训练 LightGBM ===")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        early_stopping_rounds=50,
        verbose=100
    )
    lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
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
    Path("results/figures").mkdir(exist_ok=True)
    
    for model_name, model in models.items():
        # 训练集预测
        y_train_pred = model.predict(X_train)
        r2_train = r2_score(y_train, y_train_pred)
        
        # 测试集预测
        y_test_pred = model.predict(X_test)
        r2_test = r2_score(y_test, y_test_pred)
        
        # 创建联合散点图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # 训练集
        sns.scatterplot(x=y_train, y=y_train_pred, ax=ax1, color='blue', alpha=0.6)
        ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
        ax1.set_xlabel('实际值 (mg/g)')
        ax1.set_ylabel('预测值 (mg/g)')
        ax1.set_title(f'{model_name} - 训练集 (R² = {r2_train:.4f})')
        ax1.grid(True)
        
        # 测试集
        sns.scatterplot(x=y_test, y=y_test_pred, ax=ax2, color='orange', alpha=0.6)
        ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax2.set_xlabel('实际值 (mg/g)')
        ax2.set_ylabel('预测值 (mg/g)')
        ax2.set_title(f'{model_name} - 测试集 (R² = {r2_test:.4f})')
        ax2.grid(True)
        
        plt.suptitle(f'{model_name} 模型预测结果联合散点图', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(f'results/figures/model_{model_name.lower()}_jointplot.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{model_name} 联合散点图已保存")

def plot_model_uncertainty(model, X_train, X_test, y_train, y_test, model_name):
    Path("results/figures").mkdir(exist_ok=True)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_errors = y_train_pred - y_train.values
    test_errors = y_test_pred - y_test.values
    
    train_std = np.std(train_errors)
    test_std = np.std(test_errors)
    
    train_lower = y_train_pred - 1.96 * train_std
    train_upper = y_train_pred + 1.96 * train_std
    
    test_lower = y_test_pred - 1.96 * test_std
    test_upper = y_test_pred + 1.96 * test_std
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    sorted_idx_train = np.argsort(y_train.values)
    ax1.fill_between(y_train.values[sorted_idx_train], train_lower[sorted_idx_train], train_upper[sorted_idx_train], 
                     alpha=0.3, color='blue', label='95% CI')
    ax1.scatter(y_train.values[sorted_idx_train], y_train_pred[sorted_idx_train], 
                color='blue', s=30, alpha=0.7, label='Prediction')
    ax1.plot(y_train.values[sorted_idx_train], y_train.values[sorted_idx_train], 
             'r--', lw=2, label='Ideal')
    ax1.set_xlabel('Actual Value (mg/g)')
    ax1.set_ylabel('Predicted Value (mg/g)')
    ax1.set_title(f'{model_name} - Training Set Uncertainty')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    sorted_idx_test = np.argsort(y_test.values)
    ax2.fill_between(y_test.values[sorted_idx_test], test_lower[sorted_idx_test], test_upper[sorted_idx_test], 
                     alpha=0.3, color='orange', label='95% CI')
    ax2.scatter(y_test.values[sorted_idx_test], y_test_pred[sorted_idx_test], 
                color='orange', s=30, alpha=0.7, label='Prediction')
    ax2.plot(y_test.values[sorted_idx_test], y_test.values[sorted_idx_test], 
             'r--', lw=2, label='Ideal')
    ax2.set_xlabel('Actual Value (mg/g)')
    ax2.set_ylabel('Predicted Value (mg/g)')
    ax2.set_title(f'{model_name} - Test Set Uncertainty')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name} Model Prediction with 95% Confidence Interval', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'results/figures/model_{model_name.lower()}_uncertainty.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{model_name} 模型不确定性图已保存")

def plot_shap_analysis(models, X_train, model_name):
    Path("results/figures").mkdir(exist_ok=True)
    
    model = models[model_name]
    
    print(f"\n=== SHAP Analysis for {model_name} ===")
    
    if model_name == 'XGBoost':
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_train)
    elif model_name == 'LightGBM':
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_train)
    elif model_name == 'CatBoost':
        explainer = shap.Explainer(model)
        shap_values = explainer(X_train)
    else:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    shap.summary_plot(shap_values, X_train, show=False, plot_size=None)
    plt.title(f'{model_name} - SHAP Summary Plot')
    plt.tight_layout()
    plt.savefig(f'results/figures/shap_{model_name.lower()}_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{model_name} SHAP Summary Plot saved")
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
    plt.title(f'{model_name} - SHAP Feature Importance')
    plt.tight_layout()
    plt.savefig(f'results/figures/shap_{model_name.lower()}_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{model_name} SHAP Feature Importance saved")
    
    shap_df = pd.DataFrame({
        'feature': X_train.columns,
        'mean_shap': np.abs(shap_values.values).mean(axis=0)
    }).sort_values('mean_shap', ascending=False)
    shap_df.to_csv(f'results/shap_{model_name.lower()}_values.csv', index=False, encoding='utf-8-sig')
    print(f"{model_name} SHAP values saved to CSV")

def main():
    print("=== 机器学习模型训练 ===")
    
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    models = train_models(X_train, X_test, y_train, y_test)
    
    plot_joint_scatter(models, X_train, X_test, y_train, y_test)
    
    print("\n=== 生成模型不确定性图 ===")
    for name, model in models.items():
        plot_model_uncertainty(model, X_train, X_test, y_train, y_test, name)
    
    results = []
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)
        print(f"\n{name} 评估结果:")
        print(f"  R²: {metrics['R²']:.4f}")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  MAE: {metrics['MAE']:.4f}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/model_metrics.csv', index=False, encoding='utf-8-sig')
    
    for name, model in models.items():
        joblib.dump(model, f'models/best_{name}_model.pkl')
        print(f"{name} 模型已保存")
    
    best_model_name = results_df.loc[results_df['R²'].idxmax(), 'Model']
    best_model = models[best_model_name]
    joblib.dump(best_model, 'models/best_model.pkl')
    
    print(f"\n最佳模型: {best_model_name}")
    print("模型已保存到 models/ 目录")
    
    print(f"\n{best_model_name} 特征重要性:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    feature_importance.to_csv('results/feature_importance.csv', index=False, encoding='utf-8-sig')
    print("特征重要性已保存到 results/feature_importance.csv")

if __name__ == "__main__":
    main()
