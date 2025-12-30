import os
from pathlib import Path
from typing import Optional, Dict, List, Any

import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Insurance Fraud Detection", layout="wide")

MODEL_PATH = Path("models/final_model.joblib")
X_TEST_PATH = Path("data/processed/X_test.csv")  # optional; app works without it

st.title("🚨 Insurance Claim Fraud Detection (MLOps)")
st.write("This app predicts whether an insurance claim is fraudulent using a trained ML pipeline.")

# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource
def load_bundle(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}. Make sure it's committed to GitHub.")
    return joblib.load(path)

@st.cache_data
def load_example_df(path: Path) -> Optional[pd.DataFrame]:
    """Optional example dataset for dropdown choices and defaults.
    Returns None if file doesn't exist.
    """
    if path.exists():
        return pd.read_csv(path)
    return None

def safe_predict_proba(model, df: pd.DataFrame) -> pd.Series:
    """Return fraud probability for each row."""
    if hasattr(model, "predict_proba"):
        return pd.Series(model.predict_proba(df)[:, 1], index=df.index)
    # fallback if model has decision_function
    if hasattr(model, "decision_function"):
        scores = model.decision_function(df)
        # convert to (0,1) via sigmoid approximation
        import numpy as np
        probs = 1 / (1 + np.exp(-scores))
        return pd.Series(probs, index=df.index)
    raise AttributeError("Model has neither predict_proba nor decision_function.")

def predict_df(model, threshold: float, df: pd.DataFrame) -> pd.DataFrame:
    probs = safe_predict_proba(model, df)
    preds = (probs >= threshold).map({True: "Y", False: "N"})
    out = df.copy()
    out["fraud_probability"] = probs.values
    out["fraud_prediction"] = preds.values
    return out

def build_categorical_values(example_df: Optional[pd.DataFrame]) -> Dict[str, List[str]]:
    """Build dropdown options for categorical columns."""
    if example_df is not None:
        cat_cols = list(example_df.select_dtypes(include=["object"]).columns)
        return {
            col: sorted(example_df[col].dropna().astype(str).unique().tolist())
            for col in cat_cols
        }

    # Fallback options (edit these to match your dataset)
    return {
        "policy_state": ["IL", "IN", "OH"],
        "insured_sex": ["MALE", "FEMALE"],
        "incident_state": ["IL", "IN", "OH"],
        "insured_education_level": ["High School", "College", "JD", "MD", "PhD"],
        "insured_occupation": ["craft-repair", "exec-managerial", "handlers-cleaners", "other-service"],
        "insured_relationship": ["husband", "wife", "own-child", "other-relative", "unmarried"],
        "incident_type": ["Collision", "Multi-vehicle Collision", "Parked Car", "Single Vehicle Collision", "Vehicle Theft"],
        "collision_type": ["Front Collision", "Rear Collision", "Side Collision", "?", "Unknown"],
        "authorities_contacted": ["Police", "Fire", "Ambulance", "Other", "None"],
        "property_damage": ["YES", "NO", "?"],
        "police_report_available": ["YES", "NO", "?"],
    }

def is_numeric_dtype(dtype) -> bool:
    return str(dtype).startswith("int") or str(dtype).startswith("float")

# -----------------------------
# Load model + threshold
# -----------------------------
try:
    bundle = load_bundle(MODEL_PATH)
    model = bundle.get("model", bundle)  # if you saved model directly, still works
    threshold = float(bundle.get("threshold", 0.5))
except Exception as e:
    st.error(f"❌ Failed to load model bundle: {e}")
    st.stop()

# Optional example data for UI defaults
example_df = load_example_df(X_TEST_PATH)
categorical_values = build_categorical_values(example_df)

# Try to infer feature list from example_df or model
if example_df is not None:
    feature_cols = example_df.columns.tolist()
else:
    # If model is a pipeline with feature_names_in_
    if hasattr(model, "feature_names_in_"):
        feature_cols = list(model.feature_names_in_)
    else:
        feature_cols = []  # we'll handle manual input without strict schema

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Options")
mode = st.sidebar.radio("Choose Input Mode", ["Upload CSV", "Manual Input (1 claim)"])
st.sidebar.write(f"**Decision Threshold:** {threshold}")

# -----------------------------
# Upload CSV mode
# -----------------------------
if mode == "Upload CSV":
    st.subheader("📤 Upload a CSV file")
    uploaded = st.file_uploader("Upload CSV (must match training features)", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        if st.button("Predict Fraud"):
            try:
                results = predict_df(model, threshold, df)
                st.success("✅ Prediction completed!")
                st.dataframe(results.head(50))

                st.download_button(
                    label="Download predictions as CSV",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="fraud_predictions.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# -----------------------------
# Manual input mode
# -----------------------------
else:
    st.subheader("📝 Manual Input (Single Claim)")

    if example_df is None and not feature_cols:
        st.warning(
            "No `data/processed/X_test.csv` found and model doesn't expose feature names. "
            "Manual input needs a known feature list.\n\n"
            "✅ Fix options:\n"
            "1) Commit `data/processed/X_test.csv` (or a small schema CSV) to the repo, OR\n"
            "2) Ensure your saved pipeline has `feature_names_in_`, OR\n"
            "3) Hardcode `feature_cols` below."
        )

    st.info("Enter values for one claim. If X_test.csv is missing, fallback dropdown options are used.")

    # If we still don't have feature cols, ask user to upload a schema CSV
    if not feature_cols:
        schema_file = st.file_uploader("Upload a schema CSV (one row is enough) to build the manual form", type=["csv"])
        if schema_file is None:
            st.stop()
        tmp_df = pd.read_csv(schema_file)
        feature_cols = tmp_df.columns.tolist()
        example_df = tmp_df

    user_input: Dict[str, Any] = {}
    col1, col2, col3 = st.columns(3)

    # Build per-column defaults from example_df if available
    for i, col in enumerate(feature_cols):
        box = [col1, col2, col3][i % 3]

        # Prefer dtype from example_df if present, else treat as categorical if in categorical_values
        if example_df is not None and col in example_df.columns:
            dtype = example_df[col].dtype
        else:
            dtype = "object" if col in categorical_values else "float"

        if is_numeric_dtype(dtype):
            default_val = 0.0
            if example_df is not None and col in example_df.columns:
                try:
                    default_val = float(example_df[col].median())
                except Exception:
                    default_val = 0.0
            user_input[col] = box.number_input(col, value=float(default_val))
        else:
            # categorical
            if example_df is not None and col in example_df.columns:
                options = sorted(example_df[col].dropna().astype(str).unique().tolist())
                if not options:
                    options = categorical_values.get(col, ["UNKNOWN"])
            else:
                options = categorical_values.get(col, ["UNKNOWN"])

            default_val = options[0]
            user_input[col] = box.selectbox(col, options=options, index=0)

    if st.button("Predict Fraud (Single)"):
        try:
            input_df = pd.DataFrame([user_input])
            results = predict_df(model, threshold, input_df)

            prob = float(results["fraud_probability"].iloc[0])
            pred = str(results["fraud_prediction"].iloc[0])

            st.metric("Fraud Probability", f"{prob:.3f}")
            if pred == "Y":
                st.error("🚨 Prediction: FRAUD (Y)")
            else:
                st.success("✅ Prediction: NOT FRAUD (N)")

            with st.expander("Show input row"):
                st.dataframe(input_df)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
