"""按参考论文图 6 的形式绘制 CatBoost 的 SHAP 特征贡献图。"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import shap
except Exception:
    shap = None
from catboost import Pool


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "model_train_processed.csv"
# 图 6 与图 7 使用主模型训练脚本保存的同一份 CatBoost 模型。
# 图 5 为独立的归纳式共形预测流程，使用独立的 64%/16%/20% 数据划分。
MODEL_PATH = PROJECT_DIR / "models" / "best_CatBoost_model.pkl"
FIGURE_PATH = PROJECT_DIR / "results" / "figures" / "06_catboost_shap_importance.png"
TABLE_PATH = PROJECT_DIR / "results" / "catboost_shap_feature_importance.csv"
TARGET = "P adsorption capacity (mg/g)"


FEATURE_GROUPS = {
    "Modified material type": lambda columns: [
        column for column in columns if column.startswith("Modified material type_")
    ],
    "Modified / unmodified": lambda columns: ["Modified or unmodified"],
    "Cross-linked / uncross-linked": lambda columns: ["Cross-linked or uncross-linked"],
    "Cross-linking agent type": lambda columns: [
        column for column in columns if column.startswith("Cross-linking agent type_")
    ],
    "Adsorbent dosage (g/L)": lambda columns: ["Adsorbent dosage (g/L) "],
    "Reactor temperature (°C)": lambda columns: ["Reactor temperature (℃)"],
    "Initial P concentration (mg/L)": lambda columns: ["Initial P concentration (mg/L)"],
    "Reaction time (min)": lambda columns: ["Reaction time (min)"],
    "Solution pH": lambda columns: ["Solution pH"],
    # 将缺失标记并入对应结构变量，保证图6的贡献率覆盖最终模型的全部输入，
    # 同时避免把“数据是否插补”误解释为独立材料机制。
    "Pore volume (cm³/g)": lambda columns: [
        "Pore volume (cm³/g)", "Pore volume (cm³/g) was imputed"
    ],
    "BET surface area (m²/g)": lambda columns: [
        "BET surface area (m²/g)", "BET surface area (m²/g) was imputed"
    ],
}


def style_axis(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#303030")
        spine.set_linewidth(0.75)
    ax.grid(False)
    ax.tick_params(labelsize=8.4, width=0.7, length=3, color="#303030")


def main():
    df = pd.read_csv(DATA_PATH)
    x = df.drop(columns=[TARGET])
    model = joblib.load(MODEL_PATH)
    # 与保存模型时的特征顺序严格一致。
    model_features = getattr(model, "feature_names_in_", None)
    if model_features is None:
        model_features = model.feature_names_
    x = x.loc[:, model_features]

    if shap is not None:
        explainer = shap.TreeExplainer(model)
        shap_values = np.asarray(explainer.shap_values(x))
    else:
        # CatBoost 原生 ShapValues 在未安装 shap 时提供同等的全局分解。
        native_values = model.get_feature_importance(
            Pool(x, feature_names=list(x.columns)), type="ShapValues"
        )
        shap_values = np.asarray(native_values)[:, :-1]
    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=x.columns)

    group_rows = []
    for display_name, column_selector in FEATURE_GROUPS.items():
        members = column_selector(x.columns)
        if not members or any(member not in x.columns for member in members):
            raise ValueError(f"Feature group is missing columns: {display_name}")
        group_rows.append({
            "feature": display_name,
            "mean_absolute_shap": float(mean_abs_shap[members].sum()),
            "source_columns": " | ".join(members),
        })

    importance = pd.DataFrame(group_rows).sort_values(
        "mean_absolute_shap", ascending=False, ignore_index=True
    )
    importance["percentage_contribution"] = (
        importance["mean_absolute_shap"] / importance["mean_absolute_shap"].sum() * 100
    )
    importance.index = importance.index + 1
    importance.index.name = "rank"
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(TABLE_PATH, encoding="utf-8-sig")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "SimHei"],
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(7.0, 5.15))
    ordered = importance.iloc[::-1]
    # 与原文图 6(a) 相近的蓝绿色系，最高贡献变量颜色最深。
    teal_palette = [
        "#0B6266", "#187A7C", "#318F90", "#4FA4A2", "#6BB5B0",
        "#89C5BE", "#A4D2C9", "#BCDED3", "#D1E8DC", "#E2F1E6", "#EDF6EB",
    ]
    color_by_feature = dict(zip(importance["feature"], teal_palette))
    bars = ax.barh(
        ordered["feature"],
        ordered["percentage_contribution"],
        height=0.62,
        color=[color_by_feature[name] for name in ordered["feature"]],
        edgecolor="none",
    )
    max_contribution = float(importance["percentage_contribution"].max())
    for bar, contribution in zip(bars, ordered["percentage_contribution"]):
        ax.text(
            bar.get_width() + max_contribution * 0.018,
            bar.get_y() + bar.get_height() / 2,
            f"{contribution:.1f}%",
            va="center",
            ha="left",
            fontsize=8.2,
            color="#303030",
        )

    ax.set_xlabel("Percentage contribution (%)", fontsize=10)
    ax.set_xlim(0, max_contribution * 1.20)
    ax.set_box_aspect(0.84)
    style_axis(ax)
    fig.subplots_adjust(left=0.39, right=0.96, top=0.97, bottom=0.13)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=400, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)

    print(f"Saved figure: {FIGURE_PATH}")
    print(f"Saved SHAP importance: {TABLE_PATH}")
    print(importance[["feature", "percentage_contribution"]].to_string())


if __name__ == "__main__":
    main()
