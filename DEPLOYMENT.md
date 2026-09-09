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

网页会加载最终的 CatBoost 模型 `models/catboost_icp_model.pkl`，并读取共形预测、SHAP 和处理后数据文件。因此发布时需保留 `models/`、`data/processed/` 和 `results/` 目录，以及 `requirements.txt`。
