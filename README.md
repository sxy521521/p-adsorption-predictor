# 机器学习吸附材料研究系统

面向壳聚糖水凝胶磷酸盐吸附研究的数据驱动分析与预测平台。项目使用最新的缺失值补充数据，比较 CatBoost、XGBoost 和 LightGBM，并将调参后的 CatBoost 作为最终预测模型。

---

## 项目概述

本项目实现了从数据收集、探索性分析、机器学习建模到多目标优化的完整研究流程，可应用于各类吸附材料（如重金属离子吸附、有机污染物吸附等）的研究。

### 核心功能

- **数据预处理**：基于 DTR 对缺失的孔体积和 BET 比表面积进行补充，保留 12 列原始字段结构
- **模型比较与调参**：使用 Optuna TPE 和 5 折交叉验证在 80% 开发集上优化三种梯度提升树
- **不确定性分析**：使用 64% 训练集、16% 校准集和 20% 测试集构建 95% 归纳共形预测区间
- **解释与优化**：使用 CatBoost 原生 SHAP 分析，并用 NSGA-II 遗传算法兼顾吸附容量和能耗代理
- **网页预测**：Streamlit 支持材料组合约束下的单条预测和预测区间展示

---

## 数据集字段说明

本项目使用以下 12 个字段，与 `需缺失值补充数据_DTR插补结果.xlsx` 保持一致；孔体积与 BET 比表面积的缺失值已使用决策树回归（DTR）补充。

| 字段名 | 说明 | 数据类型 |
|--------|------|----------|
| Modified or unmodified | 是否改性 | 0/1 |
| Modified material type | 材料类型 | biochar/MOF/activated_carbon/chitosan |
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
│   └── model_info.json
├── results/
│   ├── figures/
│   │   ├── 01_boxplot.png
│   │   ├── 02_violinplot.png
│   │   ├── 03_correlation_heatmap.png
│   │   └── pareto_front.png
│   ├── eda_results.json
│   └── pareto_optimal_solutions.csv
├── src/
│   ├── create_sample_data.py
│   ├── data_preprocessing.py
│   ├── eda_analysis.py
│   ├── model_training.py
│   └── ga_optimization.py
└── src_simple/
    ├── create_sample_data.py
    ├── data_preprocessing.py
    ├── eda_analysis_fixed.py
    ├── model_training_fixed.py
    ├── ga_optimization.py
    └── generate_visualizations.py
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
python src/impute_and_prepare_data.py
```

**输出**：`data/processed/adsorption_data_processed.csv`

### 步骤2：训练与比较模型

```bash
python src/model_training.py
```

**输出**：三种模型、模型指标和最佳模型文件。

### 步骤3：生成论文图

```bash
python src/paper_style_model_scatter.py
python src/paper_style_xgboost_conformal.py
python src/paper_style_xgboost_shap.py
python src/xgboost_multiobjective_ga.py
```

### 步骤4：启动网页

```bash
streamlit run app.py
```

**优化目标**：
- 最大化 P adsorption capacity (mg/g)
- 最小化能耗成本

---

## 输出结果说明

### 特征重要性说明

当前 CatBoost 模型的 SHAP 结果显示，初始 P 浓度、改性材料类型、改性状态、吸附剂投加量和溶液 pH 是贡献较高的变量。该排序用于解释当前数据集中的模型预测，不等同于单变量因果效应。

---

## 可视化图片说明

### 01_boxplot.png - 箱线图
展示所有输入变量的分布情况，包括中位数、四分位数和异常值。

### 02_violinplot.png - 小提琴图
对比改性/未改性材料的吸附容量分布。

### 03_correlation_heatmap.png - 相关性热图
展示变量间的Pearson相关系数，红色表示负相关，蓝色表示正相关。

### 04_pairplot.png - 散点图矩阵
展示关键变量两两之间的关系。

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

在 `model_training_fixed.py` 中添加新模型类。

### 调整优化目标

修改 `ga_optimization.py` 中的评估函数。

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
