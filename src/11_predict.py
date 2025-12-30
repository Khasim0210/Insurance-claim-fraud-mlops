import joblib
import pandas as pd
from pathlib import Path

bundle = joblib.load("models/final_model.joblib")
model = bundle["model"]
threshold = bundle["threshold"]

# Example: load first row from X_test
X_test = pd.read_csv("data/processed/X_test.csv")
sample = X_test.iloc[[0]]

proba = model.predict_proba(sample)[:, 1][0]
pred = "Y" if proba >= threshold else "N"

print("✅ Predicted probability (fraud):", proba)
print("✅ Threshold:", threshold)
print("✅ Final prediction:", pred)
