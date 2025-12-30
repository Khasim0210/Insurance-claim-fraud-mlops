import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine

CSV_PATH = Path("data/raw/insurance_claims.csv")
DB_PATH = Path("db/insurance.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)
print("Dataset shape:", df.shape)
print("Target column exists:", "fraud_reported" in df.columns)

engine = create_engine(f"sqlite:///{DB_PATH}")

create_sql = """
DROP TABLE IF EXISTS claim_payments;
DROP TABLE IF EXISTS claims;
DROP TABLE IF EXISTS policies;
DROP TABLE IF EXISTS vehicles;
DROP TABLE IF EXISTS policy_holders;

CREATE TABLE policy_holders (
    holder_id INTEGER PRIMARY KEY AUTOINCREMENT,
    insured_sex TEXT,
    insured_education_level TEXT,
    insured_occupation TEXT,
    insured_relationship TEXT,
    insured_hobbies TEXT,
    insured_age INTEGER
);

CREATE TABLE vehicles (
    vehicle_id INTEGER PRIMARY KEY AUTOINCREMENT,
    auto_make TEXT,
    auto_model TEXT,
    auto_year INTEGER
);

CREATE TABLE policies (
    policy_id INTEGER PRIMARY KEY AUTOINCREMENT,
    policy_number INTEGER,
    policy_bind_date TEXT,
    policy_state TEXT,
    policy_csl TEXT,
    policy_deductable INTEGER,
    policy_annual_premium REAL,
    umbrella_limit INTEGER,
    holder_id INTEGER,
    vehicle_id INTEGER,
    FOREIGN KEY(holder_id) REFERENCES policy_holders(holder_id),
    FOREIGN KEY(vehicle_id) REFERENCES vehicles(vehicle_id)
);

CREATE TABLE claims (
    claim_id INTEGER PRIMARY KEY AUTOINCREMENT,
    policy_id INTEGER,
    incident_date TEXT,
    incident_type TEXT,
    collision_type TEXT,
    incident_severity TEXT,
    incident_state TEXT,
    incident_city TEXT,
    incident_location TEXT,
    number_of_vehicles_involved INTEGER,
    property_damage TEXT,
    bodily_injuries INTEGER,
    witnesses INTEGER,
    police_report_available TEXT,
    fraud_reported TEXT,
    source_row_id INTEGER,
    FOREIGN KEY(policy_id) REFERENCES policies(policy_id)
);

CREATE TABLE claim_payments (
    payment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id INTEGER,
    total_claim_amount REAL,
    injury_claim REAL,
    property_claim REAL,
    vehicle_claim REAL,
    FOREIGN KEY(claim_id) REFERENCES claims(claim_id)
);
"""

with engine.begin() as conn:
    for stmt in create_sql.split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.exec_driver_sql(stmt)


print("✅ SQLite DB and tables created:", DB_PATH)
