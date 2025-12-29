import os
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine
from ydata_profiling import ProfileReport

print("✅ Script started")
print("CWD:", os.getcwd())

DB_PATH = Path("db/insurance.db")
engine = create_engine(f"sqlite:///{DB_PATH}")

query = """
SELECT
    p.policy_annual_premium,
    p.policy_deductable,
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

    c.number_of_vehicles_involved,
    c.bodily_injuries,
    c.witnesses,
    c.property_damage,
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
print("✅ Data loaded:", df.shape)

out_dir = Path("reports")
out_dir.mkdir(parents=True, exist_ok=True)

report_path = out_dir / "data_profile_report.html"

report = ProfileReport(df, title="Insurance Claim Fraud – Data Profiling", explorative=True)
report.to_file(report_path)

print("✅ Report saved to:", report_path.resolve())
