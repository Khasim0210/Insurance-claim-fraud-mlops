import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text

CSV_PATH = Path("data/raw/insurance_claims.csv")
DB_PATH = Path("db/insurance.db")
engine = create_engine(f"sqlite:///{DB_PATH}")

df = pd.read_csv(CSV_PATH).rename(columns={"age": "insured_age"})

# -----------------------------
# 0) Clear tables (child first)
# -----------------------------
with engine.begin() as conn:
    conn.execute(text("DELETE FROM claim_payments"))
    conn.execute(text("DELETE FROM claims"))
    conn.execute(text("DELETE FROM policies"))
    conn.execute(text("DELETE FROM vehicles"))
    conn.execute(text("DELETE FROM policy_holders"))

    # reset AUTOINCREMENT counters so IDs restart at 1
    conn.execute(text("""
        DELETE FROM sqlite_sequence
        WHERE name IN ('policy_holders','vehicles','policies','claims','claim_payments');
    """))


# -----------------------
# 1) policy_holders
# -----------------------
holder_key = [
    "insured_sex",
    "insured_education_level",
    "insured_occupation",
    "insured_relationship",
    "insured_hobbies",
    "insured_age",
]
holders = df[holder_key].drop_duplicates().reset_index(drop=True)
holders.to_sql("policy_holders", engine, if_exists="append", index=False)

# Pull back TRUE IDs from DB and merge
holders_db = pd.read_sql_query(
    "SELECT holder_id, insured_sex, insured_education_level, insured_occupation, "
    "insured_relationship, insured_hobbies, insured_age FROM policy_holders",
    engine,
)
df = df.merge(holders_db, on=holder_key, how="left")

# -----------------------
# 2) vehicles
# -----------------------
vehicle_key = ["auto_make", "auto_model", "auto_year"]
vehicles = df[vehicle_key].drop_duplicates().reset_index(drop=True)
vehicles.to_sql("vehicles", engine, if_exists="append", index=False)

vehicles_db = pd.read_sql_query(
    "SELECT vehicle_id, auto_make, auto_model, auto_year FROM vehicles",
    engine,
)
df = df.merge(vehicles_db, on=vehicle_key, how="left")

# -----------------------
# 3) policies
# -----------------------
policy_key = [
    "policy_number",
    "policy_bind_date",
    "policy_state",
    "policy_csl",
    "policy_deductable",
    "policy_annual_premium",
    "umbrella_limit",
    "holder_id",
    "vehicle_id",
]
policies = df[policy_key].drop_duplicates().reset_index(drop=True)
policies.to_sql("policies", engine, if_exists="append", index=False)

policies_db = pd.read_sql_query(
    "SELECT policy_id, policy_number, policy_bind_date, policy_state, policy_csl, "
    "policy_deductable, policy_annual_premium, umbrella_limit, holder_id, vehicle_id "
    "FROM policies",
    engine,
)
df = df.merge(policies_db, on=policy_key, how="left")

# -----------------------
# 4) claims (keep source_row_id for mapping)
# -----------------------
df["source_row_id"] = range(1, len(df) + 1)

claims_cols = [
    "source_row_id",
    "policy_id",
    "incident_date",
    "incident_type",
    "collision_type",
    "incident_severity",
    "incident_state",
    "incident_city",
    "incident_location",
    "number_of_vehicles_involved",
    "property_damage",
    "bodily_injuries",
    "witnesses",
    "police_report_available",
    "fraud_reported",
]
claims = df[claims_cols].copy()
claims.to_sql("claims", engine, if_exists="append", index=False)

# Pull claim_id by source_row_id (stable!)
claims_db = pd.read_sql_query(
    "SELECT claim_id, source_row_id FROM claims",
    engine,
)
df = df.merge(claims_db, on="source_row_id", how="left")

# -----------------------
# 5) claim_payments
# -----------------------
payments_cols = [
    "claim_id",
    "total_claim_amount",
    "injury_claim",
    "property_claim",
    "vehicle_claim",
]
payments = df[payments_cols].copy()
payments.to_sql("claim_payments", engine, if_exists="append", index=False)

# -----------------------
# Validation
# -----------------------
with engine.connect() as conn:
    tables = ["policy_holders", "vehicles", "policies", "claims", "claim_payments"]
    for t in tables:
        c = conn.execute(text(f"SELECT COUNT(*) FROM {t}")).fetchone()[0]
        print(f"{t}: {c}")

    join_count = conn.execute(text("""
        SELECT COUNT(*)
        FROM claims c
        JOIN policies p ON c.policy_id=p.policy_id
        JOIN policy_holders ph ON p.holder_id=ph.holder_id
        JOIN vehicles v ON p.vehicle_id=v.vehicle_id
        JOIN claim_payments cp ON c.claim_id=cp.claim_id
    """)).fetchone()[0]
    print("JOIN COUNT (should be 1000):", join_count)

print("✅ Reload complete with correct keys.")
