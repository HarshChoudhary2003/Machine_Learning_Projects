

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

# ---------- Utility ----------
def parse_numeric(x):
    """Convert strings with commas/₹ to float safely"""
    if pd.isna(x): return np.nan
    s = str(x).replace(",", "").replace("₹", "").strip()
    try:
        return float(s)
    except:
        return np.nan

# ---------- Load + Clean ----------
def load_and_clean(path):
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded dataset → {df.shape[0]:,} rows, {df.shape[1]} cols")

    # Convert numeric-like object columns
    num_cols_guess = [
        "age","monthly_salary","monthly_rent","school_fees","college_fees",
        "travel_expenses","groceries_utilities","other_monthly_expenses",
        "current_emi_amount","credit_score","bank_balance","emergency_fund",
        "requested_amount","requested_tenure","max_monthly_emi"
    ]
    for c in num_cols_guess:
        if c in df.columns:
            df[c] = df[c].apply(parse_numeric)

    # Fix credit score range
    if "credit_score" in df.columns:
        df["credit_score"] = df["credit_score"].clip(lower=300, upper=850)

    # Fill missing
    for c in df.select_dtypes(include=[np.number]).columns:
        df[c].fillna(df[c].median(), inplace=True)
    for c in df.select_dtypes(include=["object"]).columns:
        df[c].fillna("Unknown", inplace=True)

    return df

# ---------- Feature Engineering ----------
def add_features(df):
    df["total_monthly_expenses"] = (
        df[["monthly_rent","school_fees","college_fees",
            "travel_expenses","groceries_utilities",
            "other_monthly_expenses","current_emi_amount"]]
        .sum(axis=1)
    )

    df["dti_ratio"] = df["total_monthly_expenses"] / (df["monthly_salary"].replace(0,np.nan))
    df["affordability_ratio"] = (df["bank_balance"] + df["emergency_fund"]) / (df["requested_amount"] + 1)

    df["log_bank_balance"] = np.log1p(df["bank_balance"].clip(lower=0))
    df["log_emergency_fund"] = np.log1p(df["emergency_fund"].clip(lower=0))

    if "emi_scenario" in df.columns:
        df["emi_scenario_code"] = df["emi_scenario"].astype("category").cat.codes

    return df

# ---------- Split ----------
def split_and_save(df, out_dir="data/processed", test_size=0.15, val_size=0.10, seed=42):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    strat_col = df["emi_eligibility"] if "emi_eligibility" in df.columns else None

    train_val, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=strat_col)
    train, val = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=seed, stratify=train_val["emi_eligibility"] if strat_col is not None else None)

    train.to_parquet(f"{out_dir}/train.parquet", index=False)
    val.to_parquet(f"{out_dir}/val.parquet", index=False)
    test.to_parquet(f"{out_dir}/test.parquet", index=False)
    print(f"✅ Saved train/val/test splits to {out_dir}/")

# ---------- Run ----------
if __name__ == "__main__":
    raw_path = "H:\EMI\data\emi_prediction_dataset.csv"
    df = load_and_clean(raw_path)
    df = add_features(df)
    split_and_save(df)
