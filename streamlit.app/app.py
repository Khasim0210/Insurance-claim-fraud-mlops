import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Insurance Fraud Detection", layout="wide")

st.title("🚨 Insurance Claim Fraud Detection (MLOps)")
st.write("This app predicts whether an insurance claim is fraudulent using a trained ML pipeline.")

# Load model bundle
bundle = joblib.load("models/final_model.joblib")
model = bundle["model"]
threshold = bundle["threshold"]

st.sidebar.header("Options")
mode = st.sidebar.radio("Choose Input Mode", ["Upload CSV", "Manual Input (1 claim)"])
st.sidebar.write(f"**Decision Threshold:** {threshold}")

def predict_df(df: pd.DataFrame):
    probs = model.predict_proba(df)[:, 1]
    preds = ["Y" if p >= threshold else "N" for p in probs]
    out = df.copy()
    out["fraud_probability"] = probs
    out["fraud_prediction"] = preds
    return out

if mode == "Upload CSV":
    st.subheader("📤 Upload a CSV file")
    uploaded = st.file_uploader("Upload CSV (must match training features)", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        if st.button("Predict Fraud"):
            try:
                results = predict_df(df)
                st.success("✅ Prediction completed!")
                st.dataframe(results.head(20))

                st.download_button(
                    label="Download predictions as CSV",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="fraud_predictions.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Prediction failed: {e}")

else:
    st.subheader("📝 Manual Input (Single Claim)")
    st.info("Enter values for one claim. Categorical options are taken from X_test.")

    example_df = pd.read_csv("data/processed/X_test.csv")
    cols = example_df.columns.tolist()

    user_input = {}
    col1, col2, col3 = st.columns(3)

    for i, col in enumerate(cols):
        box = [col1, col2, col3][i % 3]
        if example_df[col].dtype in ["int64", "float64"]:
            default_val = float(example_df[col].median())
            user_input[col] = box.number_input(col, value=default_val)
        else:
            options = sorted(example_df[col].dropna().astype(str).unique().tolist())
            default_val = str(example_df[col].mode()[0]) if len(example_df[col].mode()) > 0 else options[0]
            idx = options.index(default_val) if default_val in options else 0
            user_input[col] = box.selectbox(col, options=options, index=idx)

    if st.button("Predict Fraud (Single)"):
        input_df = pd.DataFrame([user_input])
        results = predict_df(input_df)

        prob = results["fraud_probability"].iloc[0]
        pred = results["fraud_prediction"].iloc[0]

        st.metric("Fraud Probability", f"{prob:.3f}")
        if pred == "Y":
            st.error("🚨 Prediction: FRAUD (Y)")
        else:
            st.success("✅ Prediction: NOT FRAUD (N)")

