import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Respiratory Disease Classification Dashboard",
    page_icon="🩺",
    layout="wide"
)

@st.cache_resource
def load_model():
    model = joblib.load("respiratory_rf_pipeline_v5.pkl")
    return model

model = load_model()

st.title("Respiratory Disease Classification Dashboard")
st.markdown(
    """
    This educational prototype predicts whether an input respiratory profile is more consistent with
    **Healthy**, **Asthma**, or **Pneumonia** using a trained machine learning model.
    """
)

st.info(
    "This app is a dissertation prototype based on a medically informed synthetic dataset. "
    "It is for educational and demonstration purposes only and must not be used for real clinical diagnosis."
)

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Enter patient features")

    with st.form("prediction_form"):
        age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)

        sex = st.selectbox("Sex", ["Female", "Male"])

        smoking_status_label = st.selectbox(
            "Smoking status",
            ["Never", "Ex-smoker", "Current smoker"]
        )

        heart_rate = st.number_input("Heart rate (bpm)", min_value=40, max_value=180, value=80, step=1)
        respiratory_rate = st.number_input("Respiratory rate (breaths/min)", min_value=8, max_value=50, value=16, step=1)
        temperature = st.number_input("Temperature (°C)", min_value=34.0, max_value=42.0, value=37.0, step=0.1, format="%.1f")
        spo2 = st.number_input("SpO2 (%)", min_value=70, max_value=100, value=97, step=1)

        cough_severity = st.slider("Cough severity", min_value=0, max_value=3, value=1)
        dyspnea_severity = st.slider("Dyspnoea severity", min_value=0, max_value=3, value=1)

        fev1_fvc_ratio = st.number_input("FEV1/FVC ratio", min_value=30.0, max_value=100.0, value=80.0, step=0.1, format="%.1f")
        symptom_duration_days = st.number_input("Symptom duration (days)", min_value=0, max_value=60, value=2, step=1)

        submitted = st.form_submit_button("Predict")

smoking_map = {
    "Never": 0,
    "Ex-smoker": 1,
    "Current smoker": 2
}

if submitted:
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "smoking_status": smoking_map[smoking_status_label],
        "heart_rate": heart_rate,
        "respiratory_rate": respiratory_rate,
        "temperature": temperature,
        "spo2": spo2,
        "cough_severity": cough_severity,
        "dyspnea_severity": dyspnea_severity,
        "fev1_fvc_ratio": fev1_fvc_ratio,
        "symptom_duration_days": symptom_duration_days
    }])

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    class_labels = model.named_steps["model"].classes_

    prob_df = pd.DataFrame({
        "Class": class_labels,
        "Probability": probabilities
    }).sort_values("Probability", ascending=False)

    with right_col:
        st.subheader("Prediction result")

        if prediction == "Healthy":
            st.success(f"Predicted class: {prediction}")
        elif prediction == "Asthma":
            st.warning(f"Predicted class: {prediction}")
        else:
            st.error(f"Predicted class: {prediction}")

        st.markdown("### Class probabilities")
        st.dataframe(prob_df, use_container_width=True, hide_index=True)
        st.bar_chart(prob_df.set_index("Class"))

        top_class = prob_df.iloc[0]["Class"]
        top_prob = prob_df.iloc[0]["Probability"] * 100

        st.markdown(
            f"""
            **Interpretation:** The model found the strongest similarity with **{top_class}**
            and assigned it a probability of **{top_prob:.2f}%**.
            """
        )

with st.expander("Feature guide"):
    st.markdown("""
    - **Age**: Age in years.
    - **Sex**: Biological sex used in the dataset.
    - **Smoking status**: Never, Ex-smoker, or Current smoker.
    - **Heart rate**: Beats per minute.
    - **Respiratory rate**: Breaths per minute.
    - **Temperature**: Body temperature in degrees Celsius.
    - **SpO2**: Peripheral oxygen saturation.
    - **Cough severity**: 0=None, 1=Mild, 2=Moderate, 3=Severe.
    - **Dyspnoea severity**: 0=None, 1=Mild, 2=Moderate, 3=Severe.
    - **FEV1/FVC ratio**: Spirometric ratio used in respiratory assessment.
    - **Symptom duration**: Duration of symptoms in days.
    """)

with st.expander("About the model"):
    st.markdown("""
    This app uses a trained Random Forest pipeline built on the final V5 respiratory dataset.
    The model predicts one of three classes:
    - Healthy
    - Asthma
    - Pneumonia
    """)

with st.expander("Disclaimer"):
    st.markdown("""
    This tool is an academic dissertation prototype based on synthetic data.
    It is not a medical device, not a clinical decision support system, and must not be used
    to diagnose, treat, or manage any real patient.
    """)
