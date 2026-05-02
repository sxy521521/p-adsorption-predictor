import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Biochar Phosphorus Adsorption Predictor", layout="wide")

@st.cache_resource
def load_model_and_features():
    model = joblib.load('models/best_model.pkl')
    df = pd.read_csv('data/processed/adsorption_data_processed.csv')
    feature_names = df.drop('P adsorption capacity (mg/g)', axis=1).columns.tolist()
    return model, feature_names

def main():
    st.title("Biochar Phosphorus Adsorption Predictor")
    st.markdown("### Predict P (Phosphorus) Adsorption on Biochar")
    st.markdown("""
    This web application predicts the adsorption capacity of biochar for phosphorus removal.
    Enter the process conditions and material properties to estimate P adsorption.
    """)

    model, feature_names = load_model_and_features()

    with st.sidebar:
        st.header("Input Parameters")

        col1, col2 = st.columns(2)
        with col1:
            modified = st.selectbox("Modified or Unmodified", [0, 1], format_func=lambda x: "Modified" if x == 1 else "Unmodified")
            crosslinked = st.selectbox("Cross-linked or Uncross-linked", [0, 1], format_func=lambda x: "Cross-linked" if x == 1 else "Uncross-linked")
            material_type = st.selectbox("Material Type", ["biochar", "activated_carbon", "chitosan"])

        with col2:
            bet_area = st.slider("BET Surface Area (m2/g)", min_value=100.0, max_value=1000.0, value=500.0, step=10.0)
            pore_volume = st.slider("Pore Volume (cm3/g)", min_value=0.1, max_value=1.5, value=0.5, step=0.05)

        col3, col4 = st.columns(2)
        with col3:
            reactor_temp = st.slider("Reactor Temperature (C)", min_value=15.0, max_value=80.0, value=25.0, step=1.0)
            adsorbent_dosage = st.slider("Adsorbent Dosage (g/L)", min_value=0.1, max_value=20.0, value=5.0, step=0.1)

        with col4:
            initial_p_conc = st.slider("Initial P Concentration (mg/L)", min_value=10.0, max_value=200.0, value=100.0, step=5.0)
            reaction_time = st.slider("Reaction Time (min)", min_value=10.0, max_value=720.0, value=120.0, step=10.0)

        solution_ph = st.slider("Solution pH", min_value=2.0, max_value=12.0, value=7.0, step=0.1)

    input_dict = {
        'Modified or unmodified': modified,
        'Cross-linked or uncross-linked': crosslinked,
        'Adsorbent dosage (g/L)': adsorbent_dosage,
        'Reactor temperature (℃)': reactor_temp,
        'Initial P concentration (mg/L)': initial_p_conc,
        'Reaction time (min)': reaction_time,
        'Solution pH': solution_ph,
        'Pore volume (cm³/g)': pore_volume,
        'BET surface area (m²/g)': bet_area,
        'Modified material type_activated_carbon': 1 if material_type == 'activated_carbon' else 0,
        'Modified material type_biochar': 1 if material_type == 'biochar' else 0,
        'Modified material type_chitosan': 1 if material_type == 'chitosan' else 0
    }

    input_data = pd.DataFrame([input_dict])
    input_data = input_data[feature_names]

    if st.button("Predict Adsorption Capacity"):
        prediction = model.predict(input_data)[0]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Prediction Results")
            st.markdown(f"**P Adsorption Capacity:** {prediction:.2f} mg/g")

            st.markdown("""
            **Interpretation:**
            - Higher BET surface area generally increases adsorption
            - Optimal pH range for phosphorus adsorption is typically 6-8
            - Higher initial concentration may increase adsorption capacity
            - Longer reaction time typically improves adsorption
            """)

        with col2:
            st.subheader("Parameter Importance")
            shap_df = pd.read_csv('results/shap_values.csv')

            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(data=shap_df, x='mean_shap', y='feature', ax=ax)
            ax.set_title('Feature Importance (SHAP Values)')
            ax.set_xlabel('Mean |SHAP Value|')
            ax.set_ylabel('Feature')
            plt.tight_layout()
            st.pyplot(fig)

    st.markdown("---")
    st.markdown("### About This Application")
    st.markdown("""
    This application uses a machine learning model trained on biochar phosphorus adsorption data.
    The model was developed using CatBoost, XGBoost, and LightGBM algorithms,
    with hyperparameter optimization and SHAP analysis for feature importance.

    **Key Features:**
    - Predicts P (Phosphorus) adsorption capacity
    - User-friendly interface for parameter input
    - Real-time prediction with interpretation
    - Feature importance visualization

    **Disclaimer:** This is a research tool and predictions should be validated experimentally.
    """)

if __name__ == "__main__":
    main()