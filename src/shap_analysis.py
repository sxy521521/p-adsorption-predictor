import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

Path("results/figures").mkdir(exist_ok=True)

def load_data():
    df = pd.read_csv('data/processed/adsorption_data_processed.csv')
    X = df.drop('P adsorption capacity (mg/g)', axis=1)
    y = df['P adsorption capacity (mg/g)']
    return X, y

print("Loading data and model...")
X, y = load_data()

try:
    model = joblib.load('models/best_model.pkl')
    print(f"Model loaded: {type(model)}")

    print("\nCalculating SHAP values using TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    print("\nGenerating SHAP plots...")
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X, show=False)
    plt.title('SHAP Summary Plot - Biochar Adsorption for As(III) and As(V)', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/figures/shap_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("SHAP Summary Plot saved")

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title('SHAP Feature Importance - Biochar Adsorption', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/figures/shap_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("SHAP Feature Importance saved")

    shap_df = pd.DataFrame({
        'feature': X.columns,
        'mean_shap': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_shap', ascending=False)
    shap_df.to_csv('results/shap_values.csv', index=False, encoding='utf-8-sig')
    print("SHAP values saved to CSV")

    print("\nTop 10 Important Features:")
    for idx, row in shap_df.head(10).iterrows():
        print(f"  {row['feature']}: {row['mean_shap']:.4f}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\nSHAP analysis completed!")