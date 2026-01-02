import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report

import mlflow
import mlflow.sklearn

from src.12_feature_engineering import add_engineered_features



# -----------------------
# Load splits
# -----------------------
data_dir = Path("data/processed")
X_train = pd.read_csv(data_dir / "X_train.csv")
X_test  = pd.read_csv(data_dir / "X_test.csv")
y_train = pd.read_csv(data_dir / "y_train.csv").squeeze("columns")
y_test  = pd.read_csv(data_dir / "y_test.csv").squeeze("columns")

# ✅ ADD FEATURE ENGINEERING HERE
X_train = add_engineered_features(X_train)
X_test  = add_engineered_features(X_test)

# -----------------------
# Preprocess (trees don't need scaling)
# -----------------------
numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = [c for c in X_train.columns if c not in numeric_features]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

mlflow.set_experiment("insurance-claim-fraud-mlops")

def run_model(model_name, model, param_grid):
    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", model)
    ])

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv3,
        n_jobs=-1,
    )

    with mlflow.start_run(run_name=model_name):
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

        out_dir = Path(f"artifacts/exp02/{model_name}")
        out_dir.mkdir(parents=True, exist_ok=True)

        (out_dir / "confusion_matrix.txt").write_text(str(cm))
        (out_dir / "classification_report.txt").write_text(report)

        mlflow.log_artifact(str(out_dir / "confusion_matrix.txt"))
        mlflow.log_artifact(str(out_dir / "classification_report.txt"))

        mlflow.sklearn.log_model(best_model, artifact_path="model")

        print(f"\n✅ {model_name}")
        print("Best Params:", best_params)
        print("CV F1 Macro:", best_cv)
        print("Test F1 Macro:", f1_macro)
        print("Confusion Matrix:\n", cm)

# Random Forest
run_model(
    "Exp02_RandomForest",
    RandomForestClassifier(random_state=42),
    {
        "model__n_estimators": [200, 500],
        "model__max_depth": [None, 10, 20],
        "model__class_weight": [None, "balanced"]
    }
)

# Gradient Boosting
run_model(
    "Exp02_GradientBoosting",
    GradientBoostingClassifier(random_state=42),
    {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [2, 3],
    }
)

# HistGradientBoosting
run_model(
    "Exp02_HistGradientBoosting",
    HistGradientBoostingClassifier(random_state=42),
    {
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [3, 5],
        "model__max_iter": [200, 500],
    }
)
