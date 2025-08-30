import streamlit as st
import pandas as pd
from joblib import load
import json

# Load model, scaler, and metadata
model = load("models/model.joblib")
scaler = load("models/preprocess.joblib")
with open("models/metadata.json", "r") as f:
    metadata = json.load(f)
threshold = metadata["threshold"]

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("💳 Credit Card Fraud Detection (RandomForest)")

# Expected column order
expected_cols = ['Time','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
                 'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
                 'V21','V22','V23','V24','V25','V26','V27','V28','Amount']

# Load dataset for pre-filling Single Prediction
df_data = pd.read_csv("data/creditcard.csv")

# Tabs
tab1, tab2 = st.tabs(["📂 Batch Prediction", "🔍 Single Prediction"])

# --- Batch Prediction ---
with tab1:
    st.header("Batch Prediction")
    uploaded = st.file_uploader("Upload a CSV file (same format as dataset)", type="csv")

    if uploaded:
        df = pd.read_csv(uploaded)

        # Drop 'Class' if exists
        if 'Class' in df.columns:
            df = df.drop('Class', axis=1)

        # Ensure correct column order
        df = df[expected_cols]

        # Scale Amount & Time
        df[['Amount', 'Time']] = scaler.transform(df[['Amount', 'Time']])

        # Predict
        scores = model.predict_proba(df)[:, 1]
        df['fraud_score'] = scores
        df['fraud_pred'] = (scores >= threshold).astype(int)

        # Display metrics
        st.metric("Total Transactions", len(df))
        st.metric("Legitimate Transactions", (df['fraud_pred'] == 0).sum())
        st.metric("Fraudulent Transactions", (df['fraud_pred'] == 1).sum())

        # Display separate tables
        st.subheader("Legitimate Transactions")
        st.dataframe(df[df['fraud_pred'] == 0])

        st.subheader("Fraudulent Transactions")
        st.dataframe(df[df['fraud_pred'] == 1])

# --- Single Prediction ---
with tab2:
    st.header("Single Transaction Prediction")

    # Pre-fill with a random real transaction
    sample_row = df_data.sample(1).iloc[0]

    input_data = {}
    for col in expected_cols:
        input_data[col] = st.number_input(col, value=float(sample_row[col]), format="%.4f")

    if st.button("Predict Transaction"):
        row = pd.DataFrame([input_data])
        row[['Amount','Time']] = scaler.transform(row[['Amount','Time']])
        score = model.predict_proba(row)[:, 1][0]
        st.metric("Fraud Score", round(score, 4))
        st.metric("Prediction", "⚠️ FRAUD" if score >= threshold else "✅ LEGIT")
