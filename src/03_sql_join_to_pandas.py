import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine

DB_PATH = Path("db/insurance.db")
engine = create_engine(f"sqlite:///{DB_PATH}")

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

df_joined = pd.read_sql_query(query, engine)

print("Joined DF shape:", df_joined.shape)
print(df_joined.head())
print("\nFraud label distribution:")
print(df_joined["fraud_reported"].value_counts())
