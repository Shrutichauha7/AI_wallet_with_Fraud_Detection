import streamlit as st
import requests
import pandas as pd
import random

st.set_page_config(page_title="AI Wallet Fraud Detector", layout="wide")

st.title("💳 AI Wallet Fraud Detection Dashboard")

API_URL = "http://127.0.0.1:8000/predict"

# ==============================
# Sidebar
# ==============================
st.sidebar.header("⚙ Controls")

mode = st.sidebar.radio(
    "Choose Mode",
    ["Single Transaction", "Random Transaction", "Bulk CSV Upload"]
)

# ==============================
# FUNCTION TO DISPLAY RESULT
# ==============================
def display_result(result):
    prob = result["fraud_probability"]

    st.subheader("📊 Prediction Result")

    st.progress(prob)
    st.metric("Fraud Probability", f"{prob*100:.2f}%")

    if result["risk_level"] == "HIGH RISK":
        st.error("🚨 HIGH RISK TRANSACTION")
    elif result["risk_level"] == "MEDIUM RISK":
        st.warning("⚠ MEDIUM RISK TRANSACTION")
    else:
        st.success("✅ LOW RISK TRANSACTION")

# ==============================
# MODE 1 — Single Transaction
# ==============================
if mode == "Single Transaction":

    st.subheader("Enter 30 Feature Values")

    features = []

    cols = st.columns(5)  # 5 columns layout
    for i in range(30):
        with cols[i % 5]:
            value = st.number_input(f"F{i+1}", value=0.0)
            features.append(value)

    if st.button("Predict Fraud"):
        payload = {"features": features}
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            display_result(response.json())
        else:
            st.error(response.text)

# ==============================
# MODE 2 — Random Transaction
# ==============================
elif mode == "Random Transaction":

    st.subheader("🎲 Generate Random Transaction")

    if st.button("Generate & Predict"):
        features = [random.uniform(-3, 3) for _ in range(30)]

        payload = {"features": features}
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            st.write("Generated Features:")
            st.write(features)
            display_result(response.json())
        else:
            st.error(response.text)

# ==============================
# MODE 3 — Bulk CSV Upload
# ==============================
elif mode == "Bulk CSV Upload":

    st.subheader("📂 Upload CSV File (30 Columns Required)")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.write("Preview Data:")
        st.dataframe(df.head())

        if df.shape[1] != 30:
            st.error("CSV must contain exactly 30 columns!")
        else:
            if st.button("Predict All Transactions"):
                probabilities = []
                risk_levels = []

                for _, row in df.iterrows():
                    payload = {"features": row.tolist()}
                    response = requests.post(API_URL, json=payload)

                    if response.status_code == 200:
                        result = response.json()
                        probabilities.append(result["fraud_probability"])
                        risk_levels.append(result["risk_level"])
                    else:
                        probabilities.append(None)
                        risk_levels.append("Error")

                df["Fraud Probability"] = probabilities
                df["Risk Level"] = risk_levels

                st.success("Bulk Prediction Completed!")
                st.dataframe(df)