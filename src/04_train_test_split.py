import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split

# DB connection
DB_PATH = Path("db/insurance.db")
engine = create_engine(f"sqlite:///{DB_PATH}")

# SQL JOIN (same as Step 6)
query = """
SELECT
    c.claim_id,
    p.policy_number,
    p.policy_bind_date,
    p.policy_state,
    p.policy_csl,
    p.policy_deductable,
    p.policy_annual_premium,
    p.umbrella_limit,

    ph.insured_age,
    ph.insured_sex,
    ph.insured_education_level,
    ph.insured_occupation,
    ph.insured_relationship,
    ph.insured_hobbies,

    v.auto_make,
    v.auto_model,
    v.auto_year,

    c.incident_date,
    c.incident_type,
    c.collision_type,
    c.incident_severity,
    c.incident_state,
    c.incident_city,
    c.incident_location,
    c.number_of_vehicles_involved,
    c.property_damage,
    c.bodily_injuries,
    c.witnesses,
    c.police_report_available,
    c.fraud_reported,

    cp.total_claim_amount,
    cp.injury_claim,
    cp.property_claim,
    cp.vehicle_claim
FROM claims c
JOIN policies p ON c.policy_id = p.policy_id
JOIN policy_holders ph ON p.holder_id = ph.holder_id
JOIN vehicles v ON p.vehicle_id = v.vehicle_id
JOIN claim_payments cp ON c.claim_id = cp.claim_id
;
"""

df = pd.read_sql_query(query, engine)

# -------------------------
# 1) Check target imbalance
# -------------------------
y = df["fraud_reported"]
print("Target distribution:")
print(y.value_counts())
print("\nTarget %:")
print((y.value_counts(normalize=True) * 100).round(2))

# -------------------------
# 2) Define X and y
# -------------------------
X = df.drop(columns=["fraud_reported"])

# Optional: claim_id is an identifier, not a real feature
X = X.drop(columns=["claim_id"])

# -------------------------
# 3) Stratified train/test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain size:", X_train.shape, " Test size:", X_test.shape)
print("\nTrain target distribution:")
print(y_train.value_counts())
print("\nTest target distribution:")
print(y_test.value_counts())

# -------------------------
# 4) Save outputs for next steps
# -------------------------
out_dir = Path("data/processed")
out_dir.mkdir(parents=True, exist_ok=True)

X_train.to_csv(out_dir / "X_train.csv", index=False)
X_test.to_csv(out_dir / "X_test.csv", index=False)
y_train.to_csv(out_dir / "y_train.csv", index=False)
y_test.to_csv(out_dir / "y_test.csv", index=False)

print("\n✅ Saved train/test splits to data/processed/")
