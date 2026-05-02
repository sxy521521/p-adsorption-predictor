import pandas as pd
import numpy as np
from pathlib import Path

script_dir = Path(__file__).parent.parent

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    
    print("=== 原始数据信息 ===")
    print(f"数据形状: {df.shape}")
    print(f"\n缺失值统计:\n{df.isnull().sum()}")
    
    categorical_cols = ['Modified material type']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    print(f"\n处理后数据形状: {df_encoded.shape}")
    
    df_encoded.to_csv(script_dir / 'data/processed/adsorption_data_processed.csv', index=False, encoding='utf-8-sig')
    
    return df_encoded

if __name__ == "__main__":
    df_processed = load_and_preprocess_data(script_dir / 'data/raw/adsorption_sample_data.csv')
    print("\n数据预处理完成！")
