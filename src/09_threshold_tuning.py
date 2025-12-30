import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    f1_score,
    confusion_matrix,
)

import mlflow
import mlflow.sklearn

# -----------------------
# Load data
# -----------------------
data_dir = Path("data/processed")
X_train = pd.read_csv(data_dir / "X_train.csv")
X_test = pd.read_csv(data_dir / "X_test.csv")
y_train = pd.read_csv(data_dir / "y_train.csv").squeeze("columns")
y_test = pd.read_csv(data_dir / "y_test.csv").squeeze("columns")

y_test_bin = (y_test == "Y").astype(int)

# -----------------------
# Preprocessing
# -----------------------
numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = [c for c in X_train.columns if c not in numeric_features]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

# -----------------------
# Best model (from Exp02)
# -----------------------
model = GradientBoostingClassifier(
    learning_rate=0.05,
    max_depth=2,
    n_estimators=200,
    random_state=42,
)

pipe = Pipeline([
    ("preprocess", preprocess),
    ("model", model),
])

pipe.fit(X_train, y_train)

# -----------------------
# Probabilities
# -----------------------
y_proba = pipe.predict_proba(X_test)[:, 1]

# -----------------------
# ROC Curve
# -----------------------
fpr, tpr, roc_thresholds = roc_curve(y_test_bin, y_proba)
roc_auc = auc(fpr, tpr)

# -----------------------
# Precision-Recall Curve
# -----------------------
precision, recall, pr_thresholds = precision_recall_curve(y_test_bin, y_proba)

# -----------------------
# Threshold tuning (maximize F1)
# -----------------------
thresholds = np.linspace(0.1, 0.9, 81)
f1_scores = []

for t in thresholds:
    y_pred_t = (y_proba >= t).astype(int)
    f1_scores.append(f1_score(y_test_bin, y_pred_t))

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

y_pred_best = (y_proba >= best_threshold).astype(int)
cm = confusion_matrix(y_test_bin, y_pred_best)

# -----------------------
# Plot ROC
# -----------------------
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()

roc_path = Path("artifacts/exp03_roc_curve.png")
roc_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(roc_path)
plt.close()

# -----------------------
# Plot Precision-Recall
# -----------------------
plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.tight_layout()

pr_path = Path("artifacts/exp03_pr_curve.png")
plt.savefig(pr_path)
plt.close()

# -----------------------
# MLflow logging
# -----------------------
mlflow.set_experiment("insurance-claim-fraud-mlops")

with mlflow.start_run(run_name="Exp03_Threshold_Tuning"):
    mlflow.log_metric("roc_auc", float(roc_auc))
    mlflow.log_metric("best_f1_threshold", float(best_threshold))
    mlflow.log_metric("best_f1_score", float(best_f1))

    mlflow.log_artifact(str(roc_path))
    mlflow.log_artifact(str(pr_path))

    print("✅ ROC AUC:", roc_auc)
    print("✅ Best Threshold:", best_threshold)
    print("✅ Best F1:", best_f1)
    print("✅ Confusion Matrix @ Best Threshold:\n", cm)
