# 机器学习吸附材料研究系统

面向壳聚糖水凝胶磷酸盐吸附研究的数据驱动分析与预测平台。项目使用最新的缺失值补充数据，比较 CatBoost、XGBoost 和 LightGBM，并将调参后的 CatBoost 作为最终预测模型。

---

## 项目概述

本项目实现了从数据收集、探索性分析、机器学习建模到多目标优化的完整研究流程，可应用于各类吸附材料（如重金属离子吸附、有机污染物吸附等）的研究。

### 核心功能

- **数据预处理**：基于 DTR 对缺失的孔体积和 BET 比表面积进行补充；预测模型额外保留相应的缺失标记，避免把插补值误当作实测值
- **模型比较与调参**：使用 Optuna TPE 和 5 折交叉验证在 80% 开发集上优化三种梯度提升树
- **不确定性分析**：使用 64% 训练集、16% 校准集和 20% 测试集构建 95% 归纳共形预测区间
- **解释与优化**：使用 CatBoost 原生 SHAP 分析，并用 NSGA-II 遗传算法兼顾吸附容量和能耗代理
- **网页预测**：Streamlit 支持材料组合约束下的单条预测和预测区间展示

说明：`best_CatBoost_model.pkl`用于重新运行NSGA-II多目标优化；`catboost_icp_model.pkl`用于网页预测和归纳共形预测区间，两者分别服务于优化复现和不确定性评估。

---

## 数据集字段说明

本项目使用以下 12 个字段，与最新的缺失值补充工作簿保持一致；孔体积与 BET 比表面积的缺失值已使用决策树回归（DTR）补充。工作簿路径由运行命令显式传入，不将个人电脑路径写入项目。

| 字段名 | 说明 | 数据类型 |
|--------|------|----------|
| Modified or unmodified | 是否改性 | 0/1 |
| Modified material type | 改性材料类型编号 | 0–30的编码 |
| Cross-linked or uncross-linked | 是否交联 | 0/1 |
| Cross-linking agent type | 交联剂类型 | 类别编号 |
| Adsorbent dosage (g/L) | 吸附剂用量 | 数值 |
| Reactor temperature (℃) | 反应器温度 | 数值 |
| Initial P concentration (mg/L) | 初始P浓度 | 数值 |
| Reaction time (min) | 反应时间 | 数值 |
| Solution pH | 溶液pH | 数值 |
| Pore volume (cm³/g) | 孔体积 | 数值 |
| BET surface area (m²/g) | BET比表面积 | 数值 |
| P adsorption capacity (mg/g) | P吸附容量（目标变量） | 数值 |

---

## 目录结构

```
adsorption_ml_project/
├── data/
│   ├── raw/
│   │   └── adsorption_sample_data.csv
│   └── processed/
│       └── adsorption_data_processed.csv
├── models/
│   ├── best_CatBoost_model.pkl
│   ├── catboost_icp_model.pkl
│   ├── catboost_best_params.json
│   ├── xgboost_best_params.json
│   └── lightgbm_best_params.json
├── results/
│   ├── figures/
│   │   ├── 00_paper_boxplot.png
│   │   ├── 03_pcc_map.png
│   │   ├── 04_ternary_plot.png
│   │   ├── 04_model_joint_scatter.png
│   │   ├── 05_catboost_conformal_prediction_interval.png
│   │   ├── 06_catboost_shap_importance.png
│   │   └── 07_catboost_multiobjective_ga.png
│   ├── model_metrics.csv
│   ├── catboost_icp_metrics.csv
│   ├── catboost_shap_feature_importance.csv
│   ├── catboost_icp_shap_feature_importance.csv
│   ├── imputation_sensitivity_metrics.csv
│   ├── model_metrics_bootstrap_ci.csv
│   └── ga_optimization_summary.csv
├── src/
│   ├── impute_and_prepare_data.py
│   ├── model_preprocessing.py             # 训练集内DTR插补与独热编码
│   ├── tree_model_bayesian_optimization.py
│   ├── xgboost_bayesian_optimization.py
│   ├── catboost_icp_bayesian_optimization.py
│   ├── model_training.py
│   ├── paper_style_xgboost_conformal.py  # 历史文件名，实际使用CatBoost
│   ├── paper_style_xgboost_shap.py      # 历史文件名，实际使用CatBoost
│   └── xgboost_multiobjective_ga.py
│   └── imputation_sensitivity_analysis.py
│   └── test_set_bootstrap_ci.py
│   └── icp_shap_importance.py
```

---

## 安装依赖

```bash
pip install -r requirements.txt
```

---

## 使用步骤

### 步骤1：准备处理后数据

```bash
cd adsorption_ml_project
python src/impute_and_prepare_data.py /path/to/需缺失值补充数据.xlsx
```

**输出**：完整描述性数据文件，以及保留缺失值的规范化源数据
`data/raw/adsorption_source_with_missing.csv`。后者供模型训练时在训练集内部拟合DTR插补器使用。

### 步骤2：训练与比较模型

```bash
python src/xgboost_bayesian_optimization.py
python src/tree_model_bayesian_optimization.py
python src/catboost_icp_bayesian_optimization.py
python src/model_training.py
```

**输出**：三种模型、模型指标和最佳模型文件。两个调参脚本在80%开发集内部使用5折交叉验证；每一折仅用该折训练部分拟合DTR插补器与编码器，独立20%测试集不参与预处理、超参数搜索或模型选择。

### 步骤3：生成论文图

```bash
python src/paper_style_boxplot.py
python src/paper_style_ternary.py
python src/paper_style_pcc.py
python src/paper_style_model_scatter.py
python src/paper_style_xgboost_conformal.py
python src/paper_style_xgboost_shap.py
python src/icp_shap_importance.py
python src/imputation_sensitivity_analysis.py
python src/test_set_bootstrap_ci.py
python src/xgboost_multiobjective_ga.py
```

### 步骤4：启动网页

```bash
streamlit run app.py
```

**优化目标**：
- 最大化 P adsorption capacity (mg/g)
- 最小化以环境温度（25 ℃）为基准的热调节强度×反应时间代理指标（不等同于实际全过程能耗）

---

## 输出结果说明

### 特征重要性说明

当前 CatBoost 模型的 SHAP 结果显示，初始 P 浓度、改性材料类型、改性状态、吸附剂投加量和溶液 pH 是贡献较高的变量。该排序用于解释当前数据集中的模型预测，不等同于单变量因果效应。

---

## 可视化图片说明

### 00_paper_boxplot.png - 箱线图
展示所有输入变量的分布情况，包括中位数、四分位数和异常值。

### 03_pcc_map.png - 相关性热图
展示变量间的Pearson相关系数，红色表示负相关，蓝色表示正相关。

### 04_model_joint_scatter.png - 模型联合散点图
展示三种模型在训练集与测试集上的实测值、预测值和边际核密度分布。

### 05_catboost_conformal_prediction_interval.png - 归纳共形预测区间
展示CatBoost在独立测试集上的95%预测区间。

### 06_catboost_shap_importance.png - SHAP特征贡献
展示CatBoost模型中各输入变量的相对贡献。

### 07_catboost_multiobjective_ga.png - 多目标优化
展示不同初始P浓度约束下的CatBoost预测结果和NSGA-II迭代过程。

## 研究边界与结果解读

- 箱线图、PCC 图和三元图使用完整数据的 DTR 补充结果，属于描述性分析；模型比较、测试集评价和 ICP 均在外部测试集划分后，仅用训练部分拟合插补器与编码器。
- 孔体积和 BET 比表面积的缺失比例较高，模型中保留了缺失标记，并提供“剔除这两项及标记”的敏感性分析结果；预测结论应结合该分析解释。
- 当前数据未保留文献来源标识，因而无法进行按文献分组的外部验证；结果反映本汇总数据集内的泛化性能，不等同于对全新文献或全新材料体系的外部验证。
- 遗传算法仅在训练数据支持域附近搜索；用于界定该支持域的孔体积和 BET 记录均为实测值。图中多次优化形成的阴影或范围反映算法重复运行的离散性，不是实验重复或统计置信区间。
- 三元图中的三项变量先分别进行 Min–Max 标准化，再映射为三角坐标比例；它用于展示联合分布，不表示因果关系。

---

## 应用场景

本项目可应用于：

1. **重金属吸附研究**：Cr(VI)、Pb(II)、Cd(II)、As(III)/As(V) 等
2. **有机污染物吸附**：染料、抗生素、农药残留等
3. **废水处理优化**：优化吸附剂用量、预测去除效率
4. **材料设计**：预测新材料性能、指导改性方向

---

## 扩展研究

### 替换为自己的数据

1. 准备 Excel 或 CSV 格式数据文件
2. 确保包含与上述 12 个字段一致的数据
3. 修改数据文件路径
4. 重新运行完整流程

### 添加新模型

在 `model_training.py` 中添加新模型类，并沿用开发集内交叉验证的参数选择流程。

### 调整优化目标

修改 `xgboost_multiobjective_ga.py` 中的目标函数和搜索边界。

---

## 技术规格

| 项目 | 规格 |
|------|------|
| 编程语言 | Python 3.10+ |
| 核心算法 | CatBoost、XGBoost、LightGBM、Optuna、归纳共形预测、NSGA-II |
| 数据规模 | 1233 个样本，11 个输入变量 |
| 输出格式 | CSV、JSON、PNG |

---

## 参考论文

> Zhang, W., et al. (2023). "Machine Learning for As(III) and As(V) Adsorption on Diverse Materials"

---

## 版本历史

- **v1.1** (2026-05-02)
  - 更新数据集字段与Excel模板一致
  - 优化路径处理

- **v1.0** (2026-05-02)
  - 初始版本发布
