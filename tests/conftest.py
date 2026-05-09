"""Shared test fixtures — synthetic data matching the loan schema."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def seed():
    return 42


@pytest.fixture
def synthetic_train(seed):
    """200-row synthetic train set matching the competition schema."""
    rng = np.random.RandomState(seed)
    n = 200

    dates_approved = pd.date_range("2025-01-01", periods=n, freq="D")
    dates_disbursed = dates_approved + pd.to_timedelta(
        rng.randint(1, 15, n), unit="D"
    )
    first_pay = dates_disbursed + pd.to_timedelta(
        rng.randint(20, 40, n), unit="D"
    )
    maturity = dates_disbursed + pd.to_timedelta(
        rng.randint(180, 730, n), unit="D"
    )
    dobs = pd.date_range("1970-01-01", periods=n, freq="30D")

    product_codes = rng.choice([0, 1, 2, 3, 4, 5], n)
    provinces = rng.choice(
        ["Harare", "Bulawayo", "Manicaland", "Mashonaland_West",
         "Midlands"],
        n,
    )
    sectors = rng.choice(
        ["Agriculture", "Mining", "Retail_Trade", "Government",
         "Education", "Informal_Sector"],
        n,
    )
    purposes = rng.choice(
        ["Working_Capital", "School_Fees", "Medical",
         "Equipment", "Personal"],
        n,
    )
    channels = rng.choice(
        ["EcoCash", "Bank_Transfer", "Cash", "InnBucks"], n
    )
    genders = rng.choice(["Male", "Female"], n)
    marital = rng.choice(
        ["Married", "Single", "Divorced", "Widowed"], n
    )
    collateral = rng.choice(
        ["None", "Vehicle", "Property", "Livestock", "Savings",
         "Guarantor"],
        n,
    )
    frequencies = rng.choice(["Monthly", "Bi-Weekly", "Weekly"], n)

    amount = rng.uniform(50, 50000, n).round(2)
    rate = rng.uniform(8, 200, n).round(2)
    term = rng.choice([3, 6, 12, 18, 24, 36, 48], n)
    income = rng.uniform(31, 2800, n).round(2)
    dependents = rng.randint(0, 9, n).astype(float)
    months_emp = rng.randint(1, 240, n).astype(float)
    obligations = rng.randint(0, 9, n)
    target = rng.choice([0, 1], n, p=[0.75, 0.25])

    # Inject ~5% NaN in some columns
    for arr in [rate, dependents, months_emp, income]:
        mask = rng.random(n) < 0.05
        arr[mask] = np.nan

    df = pd.DataFrame({
        "ID": [f"TS{i:05d}" for i in range(n)],
        "product_code": pd.Categorical(product_codes),
        "date_approved": dates_approved,
        "date_disbursed": dates_disbursed,
        "first_payment_due": first_pay,
        "maturity_date": maturity,
        "amount_usd": amount,
        "annual_rate_pct": rate,
        "term_months": term,
        "payment_frequency": pd.Categorical(frequencies),
        "loan_purpose": pd.Categorical(purposes),
        "client_gender": pd.Categorical(genders),
        "client_dob": dobs,
        "marital_status": pd.Categorical(marital),
        "num_dependents": dependents,
        "employment_sector": pd.Categorical(sectors),
        "months_at_employer": months_emp,
        "monthly_income_usd": income,
        "existing_obligations": obligations,
        "collateral_type": pd.Categorical(collateral),
        "disbursement_channel": pd.Categorical(channels),
        "province": pd.Categorical(provinces),
        "Target": target,
    })

    # Add a couple NaN in categoricals
    df.loc[5, "loan_purpose"] = np.nan
    df.loc[10, "marital_status"] = np.nan
    df.loc[15, "collateral_type"] = np.nan

    return df


@pytest.fixture
def synthetic_test(synthetic_train):
    """50-row synthetic test set (no Target column)."""
    rng = np.random.RandomState(99)
    test = synthetic_train.head(50).copy()
    test = test.drop(columns=["Target"])
    test["ID"] = [f"TT{i:05d}" for i in range(50)]
    # Extra NaN in test
    mask = rng.random(50) < 0.08
    test.loc[mask, "monthly_income_usd"] = np.nan
    return test


@pytest.fixture
def sample_submission(synthetic_test):
    """Matching sample submission for synthetic test."""
    return pd.DataFrame({
        "ID": synthetic_test["ID"].values,
        "Target": np.zeros(len(synthetic_test)),
    })
