"""P 吸附容量预测网页 - Streamlit 入口。"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "icp_train_processed.csv"
MODEL_PATH = PROJECT_DIR / "models" / "catboost_icp_model.pkl"
ICP_METRICS_PATH = PROJECT_DIR / "results" / "catboost_icp_metrics.csv"

# 编号与名称来自用户提供的“改性材料以及交联剂对应名称.docx”。
MATERIAL_NAMES = {
    "0": "Unmodified (no modifier)", "1": "ATP", "2": "La-ATP", "3": "MIL-101(Al)",
    "4": "MIL-101(Fe)", "5": "Zr-Fe", "6": "Zr", "7": "Ni", "8": "Le-Ca-LDH",
    "9": "La-Ca-LDH", "10": "La", "11": "MIL-88B(Fe)", "12": "La@PDA", "13": "Ni",
    "14": "La-Ca/ATP", "15": "Ca-ATP", "16": "La-ATP", "17": "MMT-Fe", "18": "Fe",
    "19": "Ce", "20": "Cu", "21": "Calcite", "22": "Zn(II)", "23": "Zn",
    "24": "nano-ZnO", "25": "ZnO", "26": "Ca-OMMT", "27": "TAC", "28": "Zr-GO",
    "29": "La-ZSM-5-H", "30": "La-Bentonite",
}
AGENT_NAMES = {"0": "No cross-linking agent", "1": "GA", "2": "ECH", "3": "TPP", "4": "CIT"}

st.set_page_config(page_title="P Adsorption Capacity Prediction", page_icon="🧪", layout="wide", initial_sidebar_state="collapsed")


def sort_codes(codes):
    return sorted((str(code) for code in codes), key=int)


def material_label(code):
    return f"{code} · {MATERIAL_NAMES.get(str(code), 'Unnamed material')}"


def agent_label(code):
    return f"{code} · {AGENT_NAMES.get(str(code), 'Unnamed cross-linking agent')}"


def reset_widgets(*keys):
    """上游选择变化时，清除已不适用的下游选择。"""
    for key in keys:
        st.session_state.pop(key, None)


@st.cache_resource
def load_model_assets():
    model = joblib.load(MODEL_PATH)
    processed = pd.read_csv(DATA_PATH)
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        feature_names = model.feature_names_
    feature_names = list(feature_names)
    metrics = pd.read_csv(ICP_METRICS_PATH).set_index("metric")["value"]
    material_columns = [name for name in feature_names if name.startswith("Modified material type_")]
    agent_columns = [name for name in feature_names if name.startswith("Cross-linking agent type_")]

    def category_code(frame, columns, prefix):
        codes = pd.Series("0", index=frame.index, dtype="object")
        for name in columns:
            codes.loc[frame[name].eq(1)] = name.removeprefix(prefix)
        return codes

    profile_rows = pd.DataFrame({
        "modified": processed["Modified or unmodified"].astype(int),
        "material_type": category_code(processed, material_columns, "Modified material type_"),
        "crosslinked": processed["Cross-linked or uncross-linked"].astype(int),
        "agent_type": category_code(processed, agent_columns, "Cross-linking agent type_"),
    })
    profiles = profile_rows.drop_duplicates(ignore_index=True)
    profile_counts = profile_rows.groupby(
        ["modified", "material_type", "crosslinked", "agent_type"]
    ).size().rename("n_train").reset_index()
    continuous = [
        "Adsorbent dosage (g/L) ", "Reactor temperature (℃)", "Initial P concentration (mg/L)",
        "Reaction time (min)", "Solution pH", "Pore volume (cm³/g)", "BET surface area (m²/g)",
    ]
    minimum = processed[continuous].min()
    span = (processed[continuous].max() - minimum).replace(0, 1)
    normalized = (processed[continuous] - minimum) / span
    distances = np.linalg.norm(normalized.to_numpy()[:, None, :] - normalized.to_numpy()[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    support_limit = float(np.percentile(distances.min(axis=1), 95))
    return model, processed, feature_names, metrics, profiles, profile_counts, continuous, minimum, span, support_limit


def build_model_input(values, feature_names):
    row = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
    for name in [
        "Modified or unmodified", "Cross-linked or uncross-linked", "Adsorbent dosage (g/L) ",
        "Reactor temperature (℃)", "Initial P concentration (mg/L)", "Reaction time (min)",
        "Solution pH", "Pore volume (cm³/g)", "BET surface area (m²/g)",
    ]:
        row.loc[0, name] = values[name]
    material_type = str(values["Modified material type"])
    agent_type = str(values["Cross-linking agent type"])
    if material_type != "0":
        row.loc[0, f"Modified material type_{material_type}"] = 1
    if agent_type != "0":
        row.loc[0, f"Cross-linking agent type_{agent_type}"] = 1
    return row.loc[:, feature_names]


def main():
    model, processed, feature_names, metrics, profiles, profile_counts, continuous, minimum, span, support_limit = load_model_assets()
    defaults = processed.median(numeric_only=True)

    st.title("P Adsorption Capacity Prediction")
    st.caption("Enter material properties and reaction conditions to estimate P adsorption capacity and a 95% prediction interval. Results support preliminary screening and require experimental validation.")

    catalog_col, form_col = st.columns([0.9, 1.7], gap="large")
    with catalog_col:
        st.subheader("Available Modifiers")
        st.caption("Only material combinations represented in the training data can be selected.")
        material_table = pd.DataFrame({"Code": list(MATERIAL_NAMES), "Modifier": list(MATERIAL_NAMES.values())}).iloc[1:]
        st.dataframe(material_table, hide_index=True, use_container_width=True, height=420)
        st.subheader("Available Cross-linking Agents")
        agent_table = pd.DataFrame({"Code": list(AGENT_NAMES), "Cross-linking agent": list(AGENT_NAMES.values())}).iloc[1:]
        st.dataframe(agent_table, hide_index=True, use_container_width=True)

    with form_col:
        st.subheader("Prediction Inputs")
        # 这里不能使用 st.form：材料状态变化时需要立即刷新下级选项。
        with st.container():
            material_col, condition_col = st.columns(2, gap="large")
            with material_col:
                st.markdown("**Material Information**")
                modified = st.selectbox(
                    "Modification status", [0, 1], key="modified",
                    format_func=lambda x: "0 · Unmodified" if x == 0 else "1 · Modified",
                    on_change=reset_widgets, args=("material_type", "crosslinked", "agent_type"),
                )
                if modified == 0:
                    material_type = "0"
                    st.text_input("Modifier type", "0 · Unmodified (set automatically)", key="material_disabled", disabled=True)
                    profiles_after_material = profiles.loc[profiles["modified"].eq(0) & profiles["material_type"].eq("0")]
                else:
                    material_options = sort_codes(profiles.loc[
                        profiles["modified"].eq(1) & profiles["material_type"].ne("0"), "material_type"
                    ].unique())
                    material_type = st.selectbox(
                        "Modifier type", material_options, key="material_type", format_func=material_label,
                        on_change=reset_widgets, args=("crosslinked", "agent_type"),
                    )
                    profiles_after_material = profiles.loc[
                        profiles["modified"].eq(1) & profiles["material_type"].eq(material_type)
                    ]

                crosslinked = st.selectbox(
                    "Cross-linking status", sorted(profiles_after_material["crosslinked"].unique()), key="crosslinked",
                    format_func=lambda x: "0 · Uncross-linked" if x == 0 else "1 · Cross-linked",
                    on_change=reset_widgets, args=("agent_type",),
                )
                if crosslinked == 0:
                    crosslink_agent_type = "0"
                    st.text_input("Cross-linking agent type", "0 · None (set automatically)", key="agent_disabled", disabled=True)
                else:
                    agent_options = sort_codes(profiles_after_material.loc[
                        profiles_after_material["crosslinked"].eq(1) & profiles_after_material["agent_type"].ne("0"), "agent_type"
                    ].unique())
                    crosslink_agent_type = st.selectbox("Cross-linking agent type", agent_options, key="agent_type", format_func=agent_label)

            with condition_col:
                st.markdown("**Reaction Conditions and Material Properties**")
                adsorbent_dosage = st.number_input("Adsorbent dosage (g/L)", 0.04, 50.0, float(defaults["Adsorbent dosage (g/L) "]), 0.1)
                reactor_temp = st.number_input("Reaction temperature (°C)", 4.0, 60.0, float(defaults["Reactor temperature (℃)"]), 1.0)
                initial_p = st.number_input("Initial P concentration (mg/L)", 0.1, 500.0, min(float(defaults["Initial P concentration (mg/L)"]), 500.0), 1.0)
                reaction_time = st.number_input("Reaction time (min)", 0.0, 2880.0, float(defaults["Reaction time (min)"]), 10.0)
                solution_ph = st.number_input("Solution pH", 1.0, 12.0, float(defaults["Solution pH"]), 0.1)
                pore_volume = st.number_input("Pore volume (cm³/g)", 0.0085, 9.223, float(defaults["Pore volume (cm³/g)"]), 0.01, format="%.4f")
                bet_area = st.number_input("BET surface area (m²/g)", 0.056, 147.97, float(defaults["BET surface area (m²/g)"]), 1.0)
            predict_clicked = st.button("Predict P Adsorption Capacity", type="primary", use_container_width=True)

        values = {
            "Modified or unmodified": modified, "Cross-linked or uncross-linked": crosslinked,
            "Modified material type": material_type, "Cross-linking agent type": crosslink_agent_type,
            "Adsorbent dosage (g/L) ": adsorbent_dosage, "Reactor temperature (℃)": reactor_temp,
            "Initial P concentration (mg/L)": initial_p, "Reaction time (min)": reaction_time,
            "Solution pH": solution_ph, "Pore volume (cm³/g)": pore_volume,
            "BET surface area (m²/g)": bet_area,
        }
        if predict_clicked:
            model_input = build_model_input(values, feature_names)
            normalized_input = (model_input[continuous].iloc[0] - minimum) / span
            normalized_training = (processed[continuous] - minimum) / span
            nearest_distance = float(np.linalg.norm(normalized_training.to_numpy() - normalized_input.to_numpy(), axis=1).min())
            selected_profile = profile_counts.loc[
                profile_counts["modified"].eq(modified) & profile_counts["material_type"].eq(material_type)
                & profile_counts["crosslinked"].eq(crosslinked) & profile_counts["agent_type"].eq(crosslink_agent_type)
            ]
            profile_n = int(selected_profile["n_train"].iloc[0])
            prediction = float(model.predict(model_input)[0])
            half_width = float(metrics["conformal_residual_quantile_mg_g"])
            # P 吸附容量不存在负值；仅在网页展示端截断不具物理意义的负下限。
            lower_bound = max(0.0, prediction - half_width)
            upper_bound = prediction + half_width
            st.divider()
            result_col, interval_col = st.columns(2)
            result_col.metric("Predicted P adsorption capacity", f"{prediction:.2f} mg/g")
            interval_col.metric("95% prediction interval", f"{lower_bound:.2f}–{upper_bound:.2f} mg/g")
            if profile_n <= 10:
                st.warning(f"This material/cross-linking combination has only {profile_n} training records. Results should be used for exploratory screening only.")
            if nearest_distance > support_limit:
                st.warning("The current combination of continuous inputs lies outside the typical training-data support. The prediction interval may underestimate uncertainty.")

        with st.expander("Model Information"):
            st.write("Predictions are generated by a CatBoost regression model. The 95% conformal prediction interval is calibrated using a separate calibration set. Initial P concentration can range from 0.1 to 500 mg/L. P denotes elemental phosphorus; concentrations and adsorption capacities are reported on a P mass basis. Negative interval lower bounds are displayed as 0 mg/g without changing the original model output or calibration.")

    st.divider()
    st.caption("P adsorption capacity prediction platform | Research use only")


if __name__ == "__main__":
    main()
