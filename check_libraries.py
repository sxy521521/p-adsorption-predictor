import sys
import subprocess
import importlib

def check_library(name, import_name=None):
    if import_name is None:
        import_name = name
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✓ {name}: {version}")
        return True
    except ImportError:
        print(f"✗ {name}: 未安装")
        return False

print("=== 检查Python库 ===")
print(f"Python版本: {sys.version}")
print()

print("核心数据科学库:")
check_library("pandas", "pandas")
check_library("numpy", "numpy")
check_library("scikit-learn", "sklearn")
check_library("matplotlib", "matplotlib")
check_library("seaborn", "seaborn")

print("\n机器学习库:")
check_library("XGBoost", "xgboost")
check_library("LightGBM", "lightgbm")
check_library("CatBoost", "catboost")

print("\n优化库:")
check_library("DEAP", "deap")

print("\n其他:")
check_library("joblib", "joblib")
check_library("openpyxl", "openpyxl")
