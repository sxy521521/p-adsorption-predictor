# 网页部署说明

本项目的网页入口为 `app.py`，使用 Streamlit 运行。

## 本地运行

在项目目录中运行：

```bash
streamlit run app.py
```

## 发布到 Streamlit Community Cloud

1. 将整个项目上传到一个 GitHub 仓库。
2. 在 Streamlit Community Cloud 中选择 **Create app**。
3. 选择该 GitHub 仓库和分支，主文件填写 `app.py`。
4. 点击部署。平台会根据 `requirements.txt` 与 `packages.txt` 安装依赖。

网页会加载最终的 CatBoost 共形预测模型 `models/catboost_icp_model.pkl`，并读取 `data/processed/adsorption_data_processed.csv`、`results/catboost_icp_metrics.csv` 和 `results/catboost_shap_feature_importance.csv`。若需要重新运行图7，还需保留 `models/best_CatBoost_model.pkl`、`results/ga_optimization_summary.csv` 等优化结果文件。发布网页至少需要保留上述模型、处理后数据、指标和SHAP结果，以及 `requirements.txt` 与 `packages.txt`；不需要上传个人电脑上的原始Excel路径。
