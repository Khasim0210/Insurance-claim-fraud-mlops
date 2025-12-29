import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

DB_PATH = Path("db/insurance.db")
engine = create_engine(f"sqlite:///{DB_PATH}")

query = """
SELECT
    p.policy_annual_premium,
    p.policy_deductable,
    p.umbrella_limit,
    ph.insured_age,
    v.auto_year,
    c.number_of_vehicles_involved,
    c.bodily_injuries,
    c.witnesses,
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
corr = df.corr(numeric_only=True)

out_dir = Path("reports")
out_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(10, 8))
plt.imshow(corr.values)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.colorbar()
plt.title("Correlation Matrix (Numeric Features)")
plt.tight_layout()

out_path = out_dir / "correlation_matrix.png"
plt.savefig(out_path)
plt.close()

print("✅ Saved:", out_path)
