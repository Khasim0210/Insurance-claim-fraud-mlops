# Data Cleanup & Preprocessing Tasks

- **Target imbalance:** fraud_reported is imbalanced (~75% N vs ~25% Y), so use stratified split and evaluate with F1-score.
- **Drop identifiers:** claim_id and policy_number are identifiers; remove from modeling features.
- **Categorical encoding:** OneHotEncode nominal categorical features (policy_state, incident_type, collision_type, auto_make, etc.).
- **Scaling:** Scale numeric features (premium, deductible, umbrella_limit, claim amounts) using StandardScaler or MinMaxScaler.
- **Skew handling:** Claim amount features are right-skewed; try log transform for total_claim_amount and related amounts.
- **Correlated features:** total_claim_amount is correlated with injury/property/vehicle_claim; handle via feature selection and/or PCA.
- **Missing/special values:** Treat "?" / Unknown-like values as explicit categories or impute with most frequent.
