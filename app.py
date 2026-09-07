"""P 吸附容量预测网页 - Streamlit 入口。"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "adsorption_data_processed.csv"
MODEL_PATH = PROJECT_DIR / "models" / "xgboost_icp_model.pkl"
ICP_METRICS_PATH = PROJECT_DIR / "results" / "xgboost_icp_metrics.csv"
SHAP_PATH = PROJECT_DIR / "results" / "xgboost_shap_feature_importance.csv"
TARGET = "P adsorption capacity (mg/g)"


st.set_page_config(
    page_title="P吸附容量预测平台",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_model_assets():
    model = joblib.load(MODEL_PATH)
    processed = pd.read_csv(DATA_PATH)
    feature_names = list(model.feature_names_in_)
    metrics = pd.read_csv(ICP_METRICS_PATH).set_index("metric")["value"]
    importance = pd.read_csv(SHAP_PATH)
    return model, processed, feature_names, metrics, importance


def build_model_input(values: dict, feature_names: list[str]) -> pd.DataFrame:
    """将网页输入转换成与训练模型一致的特征列。"""
    row = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
    direct_columns = [
        "Modified or unmodified",
        "Cross-linked or uncross-linked",
        "Adsorbent dosage (g/L) ",
        "Reactor temperature (℃)",
        "Initial P concentration (mg/L)",
        "Reaction time (min)",
        "Solution pH",
        "Pore volume (cm³/g)",
        "BET surface area (m²/g)",
    ]
    for column in direct_columns:
        row.loc[0, column] = values[column]

    # 预处理时材料类型 0 为基准类别，其余类别采用 one-hot 编码。
    material_type = str(values["Modified material type"])
    if material_type != "0":
        material_column = f"Modified material type_{material_type}"
        if material_column not in row.columns:
            raise ValueError(f"不支持的材料类型：{material_type}")
        row.loc[0, material_column] = 1
    agent_type = str(values["Cross-linking agent type"])
    if agent_type != "0":
        agent_column = f"Cross-linking agent type_{agent_type}"
        if agent_column not in row.columns:
            raise ValueError(f"不支持的交联剂类型：{agent_type}")
        row.loc[0, agent_column] = 1
    return row.loc[:, feature_names]


def main():
    model, processed, feature_names, metrics, importance = load_model_assets()
    numeric_defaults = processed.median(numeric_only=True)

    st.title("P 吸附容量预测平台")
    st.caption("基于 XGBoost 与归纳共形预测的研究辅助工具")
    st.info("输入材料与反应条件后，系统将输出预测 P 吸附容量及 95% 预测区间。结果用于研究筛选，仍需实验验证。")

    with st.sidebar:
        st.header("输入参数")
        st.subheader("材料信息")
        modified = st.selectbox(
            "材料状态", options=[0, 1],
            format_func=lambda value: "改性" if value == 1 else "未改性",
        )
        crosslinked = st.selectbox(
            "交联状态", options=[0, 1],
            format_func=lambda value: "交联" if value == 1 else "未交联",
        )
        material_type = st.selectbox("改性材料类型编号", options=[str(i) for i in range(31)], index=0)
        crosslink_agent_type = st.selectbox(
            "交联剂类型编号", options=["0", "1", "2", "3", "4"], index=0,
            help="0 为基准类型。",
        )

        st.subheader("反应与材料参数")
        adsorbent_dosage = st.number_input(
            "吸附剂投加量 (g/L)", min_value=0.04, max_value=50.0,
            value=float(numeric_defaults["Adsorbent dosage (g/L) "]), step=0.1,
        )
        reactor_temp = st.number_input(
            "反应温度 (°C)", min_value=4.0, max_value=60.0,
            value=float(numeric_defaults["Reactor temperature (℃)"]), step=1.0,
        )
        initial_p = st.number_input(
            "初始 P 浓度 (mg/L)", min_value=0.1, max_value=500.0,
            value=min(float(numeric_defaults["Initial P concentration (mg/L)"]), 500.0), step=1.0,
            help="网页预测范围限定为不高于 500 mg/L。",
        )
        reaction_time = st.number_input(
            "反应时间 (min)", min_value=0.0, max_value=2880.0,
            value=float(numeric_defaults["Reaction time (min)"]), step=10.0,
        )
        solution_ph = st.number_input(
            "溶液 pH", min_value=1.0, max_value=12.0,
            value=float(numeric_defaults["Solution pH"]), step=0.1,
        )
        pore_volume = st.number_input(
            "孔体积 (cm³/g)", min_value=0.0085, max_value=9.223,
            value=float(numeric_defaults["Pore volume (cm³/g)"]), step=0.01,
            format="%.4f",
        )
        bet_area = st.number_input(
            "BET 比表面积 (m²/g)", min_value=0.056, max_value=147.97,
            value=float(numeric_defaults["BET surface area (m²/g)"]), step=1.0,
        )

        predict_clicked = st.button("预测 P 吸附容量", type="primary", use_container_width=True)

    values = {
        "Modified or unmodified": modified,
        "Cross-linked or uncross-linked": crosslinked,
        "Modified material type": material_type,
        "Cross-linking agent type": crosslink_agent_type,
        "Adsorbent dosage (g/L) ": adsorbent_dosage,
        "Reactor temperature (℃)": reactor_temp,
        "Initial P concentration (mg/L)": initial_p,
        "Reaction time (min)": reaction_time,
        "Solution pH": solution_ph,
        "Pore volume (cm³/g)": pore_volume,
        "BET surface area (m²/g)": bet_area,
    }

    if predict_clicked:
        model_input = build_model_input(values, feature_names)
        prediction = float(model.predict(model_input)[0])
        interval_half_width = float(metrics["conformal_residual_quantile_mg_g"])
        lower, upper = prediction - interval_half_width, prediction + interval_half_width
        empirical_coverage = float(metrics["empirical_test_coverage"])

        result_col, interval_col, note_col = st.columns(3)
        result_col.metric("预测 P 吸附容量", f"{prediction:.2f} mg/g")
        interval_col.metric("95% 预测区间", f"{lower:.2f}–{upper:.2f} mg/g")
        note_col.metric("测试集区间覆盖率", f"{empirical_coverage:.1%}")
        st.caption("该预测区间由独立校准集计算；覆盖率用于说明模型在测试数据上的区间表现。")
    else:
        st.write("在左侧填写参数后，点击“预测 P 吸附容量”。")

    st.divider()
    left, right = st.columns([1.15, 0.85])
    with left:
        st.subheader("关键变量贡献")
        chart_data = importance.sort_values("percentage_contribution", ascending=True)
        figure = px.bar(
            chart_data,
            x="percentage_contribution",
            y="feature",
            orientation="h",
            text=chart_data["percentage_contribution"].map(lambda value: f"{value:.1f}%"),
            labels={"percentage_contribution": "贡献率 (%)", "feature": "输入变量"},
            color_discrete_sequence=["#0B6266"],
        )
        figure.update_layout(
            height=440, margin=dict(l=10, r=10, t=20, b=20),
            showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
        )
        figure.update_xaxes(showgrid=False, zeroline=False)
        figure.update_yaxes(showgrid=False)
        st.plotly_chart(figure, use_container_width=True)

    with right:
        st.subheader("模型与使用说明")
        st.markdown(
            """
            - **预测模型**：XGBoost 回归模型
            - **预测目标**：P 吸附容量（mg/g）
            - **不确定性评估**：95% 归纳共形预测区间
            - **初始 P 浓度范围**：0.1–500 mg/L
            - **适用性**：用于筛选候选实验条件，不替代实验验证
            """
        )
        with st.expander("查看本次输入参数"):
            display_values = pd.DataFrame({"参数": list(values), "输入值": list(values.values())})
            st.dataframe(display_values, hide_index=True, use_container_width=True)

    st.divider()
    st.caption("P adsorption capacity prediction platform | Research use only")


if __name__ == "__main__":
    main()
