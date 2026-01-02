from __future__ import annotations

import re
from typing import List, Optional

import numpy as np
import pandas as pd


# -----------------------------
# Helpers
# -----------------------------
def _to_datetime_safe(s: pd.Series) -> pd.Series:
    """Parse a pandas Series to datetime safely."""
    return pd.to_datetime(s, errors="coerce", infer_datetime_format=True)


def _safe_div(a: pd.Series, b: pd.Series, eps: float = 1e-9) -> pd.Series:
    """Elementwise safe division."""
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return a / (b.replace(0, np.nan) + eps)


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    """In-place numeric coercion if columns exist."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _clean_categorical_text(s: pd.Series) -> pd.Series:
    """Normalize categorical strings: lower, strip, collapse spaces."""
    out = s.astype("string")
    out = out.str.strip().str.lower()
    out = out.str.replace(r"\s+", " ", regex=True)
    return out


# -----------------------------
# Main feature engineering
# -----------------------------
def add_engineered_features(
    X: pd.DataFrame,
    *,
    drop_source_date_cols: bool = False,
    clip_negative_amounts_to_zero: bool = True,
) -> pd.DataFrame:
    """
    Create engineered features for the insurance claim fraud dataset.

    Parameters
    ----------
    X : pd.DataFrame
        Feature dataframe (no target).
    drop_source_date_cols : bool
        If True, drops raw date columns after creating features.
    clip_negative_amounts_to_zero : bool
        If True, any negative monetary amounts are clipped to 0.

    Returns
    -------
    pd.DataFrame
        Copy of X with additional engineered columns.
    """
    df = X.copy()

    # ---- Common monetary columns (seen in many insurance fraud datasets) ----
    money_cols = [
        "policy_annual_premium",
        "policy_deductable",
        "umbrella_limit",
        "injury_claim",
        "property_claim",
        "vehicle_claim",
        "total_claim_amount",
    ]
    _coerce_numeric(df, money_cols)

    if clip_negative_amounts_to_zero:
        for c in money_cols:
            if c in df.columns:
                df[c] = df[c].clip(lower=0)

    # If total_claim_amount missing but components exist, create it
    if "total_claim_amount" not in df.columns:
        parts = [c for c in ["injury_claim", "property_claim", "vehicle_claim"] if c in df.columns]
        if len(parts) >= 2:
            df["total_claim_amount"] = df[parts].sum(axis=1, min_count=1)

    # Ratios of claim parts to total
    if "total_claim_amount" in df.columns:
        if "injury_claim" in df.columns:
            df["injury_to_total_ratio"] = _safe_div(df["injury_claim"], df["total_claim_amount"])
        if "property_claim" in df.columns:
            df["property_to_total_ratio"] = _safe_div(df["property_claim"], df["total_claim_amount"])
        if "vehicle_claim" in df.columns:
            df["vehicle_to_total_ratio"] = _safe_div(df["vehicle_claim"], df["total_claim_amount"])

        # Log transform (helps tree + linear models)
        df["log_total_claim_amount"] = np.log1p(df["total_claim_amount"].fillna(0))

    # Premium vs claim ratio
    if "policy_annual_premium" in df.columns and "total_claim_amount" in df.columns:
        df["claim_to_premium_ratio"] = _safe_div(df["total_claim_amount"], df["policy_annual_premium"])
        df["premium_minus_claim"] = (df["policy_annual_premium"] - df["total_claim_amount"]).fillna(0)

    # Deductible share
    if "policy_deductable" in df.columns and "total_claim_amount" in df.columns:
        df["deductible_to_total_ratio"] = _safe_div(df["policy_deductable"], df["total_claim_amount"])

    # ---- Dates and time gaps ----
    # Typical columns: policy_bind_date, incident_date (sometimes claim_date)
    date_cols = [c for c in ["policy_bind_date", "incident_date", "claim_date"] if c in df.columns]
    for c in date_cols:
        df[c] = _to_datetime_safe(df[c])

    if "policy_bind_date" in df.columns:
        df["policy_bind_year"] = df["policy_bind_date"].dt.year
        df["policy_bind_month"] = df["policy_bind_date"].dt.month
        df["policy_bind_dayofweek"] = df["policy_bind_date"].dt.dayofweek

    if "incident_date" in df.columns:
        df["incident_year"] = df["incident_date"].dt.year
        df["incident_month"] = df["incident_date"].dt.month
        df["incident_dayofweek"] = df["incident_date"].dt.dayofweek

    if "policy_bind_date" in df.columns and "incident_date" in df.columns:
        df["days_since_policy_bind"] = (df["incident_date"] - df["policy_bind_date"]).dt.days
        # Short policy age can be a fraud signal
        df["is_policy_new_30d"] = (df["days_since_policy_bind"] <= 30).astype("Int64")
        df["is_policy_new_90d"] = (df["days_since_policy_bind"] <= 90).astype("Int64")

    if "claim_date" in df.columns and "incident_date" in df.columns:
        df["days_incident_to_claim"] = (df["claim_date"] - df["incident_date"]).dt.days

    if drop_source_date_cols and date_cols:
        df = df.drop(columns=date_cols, errors="ignore")

    # ---- Vehicles & counts ----
    # Common columns: auto_year, number_of_vehicles_involved, bodily_injuries, witnesses
    count_cols = ["auto_year", "number_of_vehicles_involved", "bodily_injuries", "witnesses"]
    _coerce_numeric(df, count_cols)

    if "auto_year" in df.columns and "incident_date" in df.columns:
        df["vehicle_age_at_incident"] = df["incident_date"].dt.year - df["auto_year"]
    elif "auto_year" in df.columns:
        # If incident date not available, approximate using current year is risky; skip.
        pass

    if "number_of_vehicles_involved" in df.columns:
        df["multi_vehicle_flag"] = (df["number_of_vehicles_involved"] >= 2).astype("Int64")

    if "bodily_injuries" in df.columns:
        df["any_bodily_injury"] = (df["bodily_injuries"] > 0).astype("Int64")

    if "witnesses" in df.columns:
        df["has_witness"] = (df["witnesses"] > 0).astype("Int64")

    # ---- Normalize text categoricals (optional but helps reduce sparse categories) ----
    # Only apply to object-like columns that are likely categorical
    for c in df.select_dtypes(include=["object", "string"]).columns:
        # skip high-cardinality IDs if present
        if re.search(r"(id|number)$", c, flags=re.IGNORECASE):
            continue
        df[c] = _clean_categorical_text(df[c])

    # ---- Simple anomaly flags (optional) ----
    # Total claim equals sum of parts check
    if all(c in df.columns for c in ["injury_claim", "property_claim", "vehicle_claim", "total_claim_amount"]):
        parts_sum = df[["injury_claim", "property_claim", "vehicle_claim"]].sum(axis=1, min_count=1)
        df["claim_parts_mismatch"] = (np.abs(parts_sum - df["total_claim_amount"]) > 1e-6).astype("Int64")

    return df


# -----------------------------
# Quick local test (optional)
# -----------------------------
if __name__ == "__main__":
    # Minimal smoke test (won’t crash if columns missing)
    sample = pd.DataFrame(
        {
            "policy_annual_premium": [1200, 900],
            "policy_deductable": [500, 1000],
            "injury_claim": [2000, 0],
            "property_claim": [1500, 1000],
            "vehicle_claim": [5000, 2000],
            "policy_bind_date": ["2015-01-01", "2019-05-10"],
            "incident_date": ["2015-01-20", "2019-06-12"],
            "number_of_vehicles_involved": [2, 1],
            "witnesses": [1, 0],
            "insured_sex": ["MALE", "FEMALE"],
            "incident_type": ["Collision", "Theft"],
        }
    )

    out = add_engineered_features(sample)
    print("Input cols:", sample.columns.tolist())
    print("Output cols:", out.columns.tolist())
    print(out.head())
