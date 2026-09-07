# 机器学习吸附材料研究系统

基于论文《Machine Learning for As(III) and As(V) Adsorption on Diverse Materials》的研究方法复现，采用纯Python实现的数据驱动吸附材料研究平台。

---

## 项目概述

本项目实现了从数据收集、探索性分析、机器学习建模到多目标优化的完整研究流程，可应用于各类吸附材料（如重金属离子吸附、有机污染物吸附等）的研究。

### 核心功能

- **数据生成与预处理**：自动生成模拟数据，支持one-hot编码
- **探索性数据分析(EDA)**：描述性统计、相关性分析、分组对比
- **机器学习建模**：线性回归、决策树、随机森林
- **多目标优化**：遗传算法(GA)实现吸附容量最大化和能耗最小化
- **可视化输出**：箱线图、小提琴图、相关性热图、帕累托前沿

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

### 方案一：基础环境（纯Python实现）

```bash
pip install Pillow
```

### 方案二：完整环境（推荐）

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
pip install catboost xgboost lightgbm deap joblib openpyxl
```

---

## 使用步骤

### 步骤1：生成模拟数据

```bash
cd adsorption_ml_project
python src/create_sample_data.py
```

**输出**：`data/raw/adsorption_sample_data.csv`

### 步骤2：数据预处理

```bash
python src/data_preprocessing.py
```

**输出**：`data/processed/adsorption_data_processed.csv`

**说明**：对分类变量（Modified material type）进行one-hot编码

### 步骤3：探索性数据分析

```bash
python src/eda_analysis.py
```

**输出**：
- `results/figures/01_boxplot.png` - 箱线图
- `results/figures/02_violinplot.png` - 小提琴图
- `results/figures/03_correlation_heatmap.png` - 相关性热图
- `results/figures/04_pairplot.png` - 散点图矩阵
- `results/eda_results.json` - EDA统计结果

### 步骤4：模型训练

```bash
python src/model_training.py
```

**输出**：
- `models/model_info.json` - 最佳模型信息
- `results/feature_importance.json` - 特征重要性

**训练的模型**：线性回归、决策树、随机森林

### 步骤5：遗传算法优化

```bash
python src/ga_optimization.py
```

**输出**：
- `results/pareto_optimal_solutions.csv` - 帕累托最优解

**优化目标**：
- 最大化 P adsorption capacity (mg/g)
- 最小化能耗成本

---

## 输出结果说明

### 特征重要性说明

与P吸附容量强相关的特征：
1. **BET surface area (m²/g)**：比表面积越大，吸附容量越高
2. **Modified or unmodified**：改性材料通常具有更高吸附容量
3. **Solution pH**：pH对吸附效果有显著影响
4. **Initial P concentration (mg/L)**：初始浓度影响吸附平衡

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
| 编程语言 | Python 3.6+ |
| 核心算法 | 线性回归、决策树、随机森林、遗传算法 |
| 数据规模 | 200-1000样本，5-15个变量 |
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
