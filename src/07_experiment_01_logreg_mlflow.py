import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix

import mlflow
import mlflow.sklearn

from src.feature_engineering import add_engineered_features



# -----------------------
# Load train/test splits
# -----------------------
data_dir = Path("data/processed")
X_train = pd.read_csv(data_dir / "X_train.csv")
X_test  = pd.read_csv(data_dir / "X_test.csv")
y_train = pd.read_csv(data_dir / "y_train.csv").squeeze("columns")
y_test  = pd.read_csv(data_dir / "y_test.csv").squeeze("columns")

# ✅ ADD FEATURE ENGINEERING HERE (before feature typing + preprocessing)
X_train = add_engineered_features(X_train)
X_test  = add_engineered_features(X_test)

# -----------------------
# Define feature types
# -----------------------
numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = [c for c in X_train.columns if c not in numeric_features]

# -----------------------
# Preprocessing pipelines
# -----------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# -----------------------
# Model
# -----------------------
clf = LogisticRegression(max_iter=500, class_weight=None)

pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", clf)
])

# -----------------------
# CV + Hyperparameter tuning (GridSearch)
# -----------------------
param_grid = {
    "model__C": [0.1, 1.0, 10.0],
    "model__penalty": ["l2"],
    "model__solver": ["lbfgs"]
}

cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv3,
    n_jobs=-1
)

# -----------------------
# MLflow logging
# -----------------------
mlflow.set_experiment("insurance-claim-fraud-mlops")

with mlflow.start_run(run_name="logreg_gridsearch_cv3"):
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_params = grid.best_params_
    best_cv = grid.best_score_

    y_pred = best_model.predict(X_test)

    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_pos = f1_score((y_test == "Y").astype(int), (y_pred == "Y").astype(int))

    mlflow.log_params(best_params)
    mlflow.log_metric("cv_f1_macro_mean", float(best_cv))
    mlflow.log_metric("test_f1_macro", float(f1_macro))
    mlflow.log_metric("test_f1_positive_class", float(f1_pos))

    cm = confusion_matrix(y_test, y_pred, labels=["N", "Y"])
    report = classification_report(y_test, y_pred)

    out_dir = Path("artifacts/exp01")
    out_dir.mkdir(parents=True, exist_ok=True)

    cm_path = out_dir / "confusion_matrix.txt"
    rep_path = out_dir / "classification_report.txt"

    cm_path.write_text(str(cm))
    rep_path.write_text(report)

    mlflow.log_artifact(str(cm_path))
    mlflow.log_artifact(str(rep_path))

    mlflow.sklearn.log_model(best_model, artifact_path="model")

    print("✅ Best Params:", best_params)
    print("✅ CV F1 Macro:", best_cv)
    print("✅ Test F1 Macro:", f1_macro)
    print("✅ Confusion Matrix:\n", cm)
