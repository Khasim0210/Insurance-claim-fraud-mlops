import joblib
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier

import mlflow
import mlflow.sklearn

from src.12_feature_engineering import add_engineered_features




BEST_THRESHOLD = 0.10

# Load full train data
data_dir = Path("data/processed")
X_train = pd.read_csv(data_dir / "X_train.csv")
y_train = pd.read_csv(data_dir / "y_train.csv").squeeze("columns")

# ✅ ADD FEATURE ENGINEERING HERE
X_train = add_engineered_features(X_train)

# Feature types
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

# Save locally
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

model_path = models_dir / "final_model.joblib"
joblib.dump({"model": pipe, "threshold": BEST_THRESHOLD}, model_path)

print("✅ Saved model bundle to:", model_path)

# Log to MLflow too
mlflow.set_experiment("insurance-claim-fraud-mlops")
with mlflow.start_run(run_name="Final_Model_Bundle"):
    mlflow.log_param("threshold", BEST_THRESHOLD)
    mlflow.sklearn.log_model(pipe, artifact_path="final_model")
    mlflow.log_artifact(str(model_path))
    print("✅ Logged final model + threshold to MLflow")
