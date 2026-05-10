"""Base feature engineering — shared across all encoding variants.

All transformations are fit on train, applied to train+test.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.loader import (
    CATEGORICAL_COLS,
    DATE_COLS,
    ID_COL,
    NUMERIC_COLS,
    TARGET_COL,
)

logger = logging.getLogger(__name__)

COLLATERAL_RANK = {
    "None": 0,
    "Guarantor": 1,
    "Savings": 1,
    "Livestock": 2,
    "Vehicle": 3,
    "Property": 4,
}


class BaseFeatureEngineer:
    """Fit-transform pattern for shared numeric/date features.

    Attributes:
        numeric_medians_: Medians fitted on train for imputation.
        cat_modes_: Modes fitted on train for categorical imputation.
    """

    def __init__(self) -> None:
        self.numeric_medians_: dict[str, float] = {}
        self.cat_modes_: dict[str, str] = {}
        self._fitted = False

    def fit(self, train: pd.DataFrame) -> BaseFeatureEngineer:
        """Compute imputation values from training data only."""
        for col in NUMERIC_COLS:
            if col in train.columns:
                self.numeric_medians_[col] = train[col].median()

        for col in CATEGORICAL_COLS:
            if col in train.columns:
                mode_vals = train[col].mode()
                self.cat_modes_[col] = (
                    mode_vals.iloc[0] if len(mode_vals) > 0 else "Unknown"
                )

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering. Does not modify the input."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        df = df.copy()
        df = self._missing_indicators(df)
        df = self._impute(df)
        df = self._date_features(df)
        df = self._numeric_ratios(df)
        df = self._interaction_features(df)
        df = self._drop_raw_dates(df)
        return df

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        """Convenience: fit on train, then transform it."""
        return self.fit(train).transform(train)

    def _missing_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary flags for columns with informative missingness."""
        indicator_cols = [
            "collateral_type", "monthly_income_usd",
            "num_dependents", "months_at_employer",
            "employment_sector", "loan_purpose",
        ]
        for col in indicator_cols:
            if col in df.columns:
                df[f"{col}_was_missing"] = (
                    df[col].isna().astype(np.float32)
                )
        return df

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, med in self.numeric_medians_.items():
            if col in df.columns:
                df[col] = df[col].fillna(med)

        for col, mode in self.cat_modes_.items():
            if col in df.columns:
                df[col] = df[col].cat.add_categories(
                    [c for c in ["Unknown", mode]
                     if c not in df[col].cat.categories]
                )
                df[col] = df[col].fillna(mode)

        return df

    def _date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "date_disbursed" in df.columns and "date_approved" in df.columns:
            df["approval_to_disburse_days"] = (
                df["date_disbursed"] - df["date_approved"]
            ).dt.days

        if (
            "first_payment_due" in df.columns
            and "date_disbursed" in df.columns
        ):
            df["disburse_to_first_pay_days"] = (
                df["first_payment_due"] - df["date_disbursed"]
            ).dt.days

        if "maturity_date" in df.columns and "date_disbursed" in df.columns:
            df["loan_age_days"] = (
                df["maturity_date"] - df["date_disbursed"]
            ).dt.days

        if "date_approved" in df.columns and "client_dob" in df.columns:
            df["client_age_at_approval"] = (
                df["date_approved"] - df["client_dob"]
            ).dt.days / 365.25

        if "date_approved" in df.columns:
            month = df["date_approved"].dt.month
            df["approval_month_sin"] = np.sin(2 * np.pi * month / 12)
            df["approval_month_cos"] = np.cos(2 * np.pi * month / 12)
            df["approval_quarter"] = df["date_approved"].dt.quarter

        return df

    def _numeric_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        income = df.get("monthly_income_usd")
        amount = df.get("amount_usd")
        rate = df.get("annual_rate_pct")
        term = df.get("term_months")
        obligations = df.get("existing_obligations")
        dependents = df.get("num_dependents")
        months_emp = df.get("months_at_employer")
        age = df.get("client_age_at_approval")

        if income is not None and amount is not None:
            safe_income = income.clip(lower=1)
            df["debt_to_income"] = amount / safe_income

        if (
            amount is not None
            and rate is not None
            and term is not None
            and income is not None
        ):
            monthly_rate = rate / 1200.0
            safe_term = term.clip(lower=1)
            numerator = amount * monthly_rate
            denominator = 1 - (1 + monthly_rate) ** (-safe_term)
            denominator = denominator.replace(0, np.nan)
            df["monthly_payment_est"] = numerator / denominator
            df["payment_to_income"] = (
                df["monthly_payment_est"] / income.clip(lower=1)
            )

        if obligations is not None and income is not None:
            df["obligation_burden"] = (
                obligations / income.clip(lower=1)
            )

        if income is not None and dependents is not None:
            df["income_per_dependent"] = (
                income / (dependents + 1)
            )

        if months_emp is not None and age is not None:
            safe_age_months = (age * 12).clip(lower=1)
            df["employer_stability"] = months_emp / safe_age_months

        if amount is not None:
            df["log_amount_usd"] = np.log1p(amount)

        if income is not None:
            df["log_monthly_income_usd"] = np.log1p(income)

        # --- additional ratio/flag features ---

        # Bank vs MFI flag (annual_rate_pct is bimodal around 40%)
        if rate is not None:
            df["is_mfi_loan"] = (rate > 40.0).astype(np.float32)
            df["rate_bucket"] = pd.cut(
                rate,
                bins=[0, 15, 40, 100, 210],
                labels=False,
            ).astype(np.float32)

        # Amount per month — simple repayment proxy
        if amount is not None and term is not None:
            safe_term = term.clip(lower=1)
            df["amount_per_month"] = amount / safe_term

        # Total cost of the loan
        if amount is not None and rate is not None and term is not None:
            df["total_loan_cost"] = amount * (1 + rate / 100 * term / 12)

        # High-obligation borrower flag
        if obligations is not None:
            df["has_many_obligations"] = (
                obligations >= 3
            ).astype(np.float32)

        # Short-tenure employee flag
        if months_emp is not None:
            df["is_new_employee"] = (
                months_emp < 12
            ).astype(np.float32)

        return df

    def _interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "province" in df.columns and "employment_sector" in df.columns:
            df["province_sector"] = (
                df["province"].astype(str)
                + "_"
                + df["employment_sector"].astype(str)
            ).astype("category")

        if "product_code" in df.columns and "loan_purpose" in df.columns:
            df["product_purpose"] = (
                df["product_code"].astype(str)
                + "_"
                + df["loan_purpose"].astype(str)
            ).astype("category")

        if "collateral_type" in df.columns:
            df["collateral_score"] = (
                df["collateral_type"]
                .astype(str)
                .map(COLLATERAL_RANK)
                .fillna(0)
                .astype(int)
            )

        if (
            "disbursement_channel" in df.columns
            and "province" in df.columns
        ):
            df["channel_province"] = (
                df["disbursement_channel"].astype(str)
                + "_"
                + df["province"].astype(str)
            ).astype("category")

        # Amount rank within product code
        if "amount_usd" in df.columns and "product_code" in df.columns:
            df["amount_rank_in_product"] = df.groupby("product_code")[
                "amount_usd"
            ].rank(pct=True)

        # Income rank within province
        if "monthly_income_usd" in df.columns and "province" in df.columns:
            df["income_rank_in_province"] = df.groupby("province")[
                "monthly_income_usd"
            ].rank(pct=True)

        # Has any collateral at all
        if "collateral_type" in df.columns:
            df["has_collateral"] = (
                df["collateral_type"].astype(str) != "None"
            ).astype(np.float32)

        return df

    def _drop_raw_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        drop = [c for c in DATE_COLS if c in df.columns]
        return df.drop(columns=drop)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return all columns usable as features (exclude ID and target)."""
    exclude = {ID_COL, TARGET_COL}
    return [c for c in df.columns if c not in exclude]
