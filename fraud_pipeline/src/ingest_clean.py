"""
Data ingestion and cleaning module.
Loads raw transaction data, handles date issues, engineers features, and exports cleaned dataset.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from . import config
from .utils import LOGGER, save_csv, check_missing_values


def load_raw_data(filepath: Path) -> pd.DataFrame:
    """
    Load raw CSV data.
    """
    LOGGER.info(f"Loading raw data from {filepath}")
    df = pd.read_csv(filepath)
    LOGGER.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    normalized_columns = pd.Index(df.columns).str.lower().str.replace(" ", "_")

    # Generate synthetic transactionid only if no transaction id column exists after normalization
    if "transactionid" not in normalized_columns:
        LOGGER.info("  Generating synthetic transactionid values...")
        df["transactionid"] = pd.Series([f"TXN_{i:08d}" for i in range(len(df))], index=df.index)
    else:
        source_column = df.columns[normalized_columns.get_loc("transactionid")]
        if df[source_column].isnull().all():
            LOGGER.info("  Generating synthetic transactionid values...")
            df[source_column] = pd.Series([f"TXN_{i:08d}" for i in range(len(df))], index=df.index)
    
    return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names: lowercase, replace spaces with underscores.
    """
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    if df.columns.duplicated().any():
        LOGGER.warning(
            "Duplicate column names found after normalization; keeping first occurrence for "
            f"{df.columns[df.columns.duplicated()].tolist()}"
        )
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse date columns safely.
    - transactiondate: datetime of transaction
    - previoustransactiondate: datetime of previous transaction for account
    """
    LOGGER.info("Parsing date columns...")

    # Parse main transaction date
    df["transactiondate"] = pd.to_datetime(df["transactiondate"], errors="coerce")

    # Parse previous transaction date (may have issues)
    df["previoustransactiondate"] = pd.to_datetime(df["previoustransactiondate"], errors="coerce")

    if df["transactiondate"].isnull().any():
        LOGGER.warning(f"  {df['transactiondate'].isnull().sum()} invalid transactiondate values")

    if df["previoustransactiondate"].isnull().any():
        LOGGER.warning(f"  {df['previoustransactiondate'].isnull().sum()} invalid previoustransactiondate values")

    return df


def fix_previous_transaction_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle unrealistic PreviousTransactionDate values.

    Issue: Raw data has PreviousTransactionDate in 2024 (future) relative to
    TransactionDate in 2023. This is invalid for real transaction histories.

    Solution: If enabled in config, regenerate synthetic previous dates using:
    - Random interval between MIN_DAYS_SINCE_PREVIOUS and MAX_DAYS_SINCE_PREVIOUS
    - Account for multiple transactions per account by grouping
    - Add a flag column to track which rows have regenerated dates

    Returns:
        DataFrame with corrected dates and new 'previous_date_regenerated' flag
    """
    LOGGER.info("Inspecting PreviousTransactionDate values...")

    df["previous_date_regenerated"] = False

    # Identify rows with unrealistic dates
    unrealistic_mask = df["previoustransactiondate"] > df["transactiondate"]
    unrealistic_count = unrealistic_mask.sum()
    LOGGER.info(f"  Found {unrealistic_count} rows with future PreviousTransactionDate")

    if not config.REGENERATE_PREVIOUS_DATES:
        LOGGER.info("  REGENERATE_PREVIOUS_DATES=False; keeping original dates")
        return df

    # For unrealistic dates, regenerate synthetic ones
    if unrealistic_count > 0:
        LOGGER.info(f"  Regenerating {unrealistic_count} synthetic previous dates...")

        # Sort by account and date to maintain consistency
        df_sorted = df.sort_values(["accountid", "transactiondate"]).reset_index(drop=True)

        for idx in range(len(df_sorted)):
            row = df_sorted.iloc[idx]

            # Check if date is unrealistic
            if row["previoustransactiondate"] > row["transactiondate"]:
                # Generate synthetic interval
                days_back = np.random.randint(
                    config.MIN_DAYS_SINCE_PREVIOUS,
                    config.MAX_DAYS_SINCE_PREVIOUS + 1,
                )
                new_prev_date = row["transactiondate"] - timedelta(days=days_back)

                df_sorted.at[idx, "previoustransactiondate"] = new_prev_date
                df_sorted.at[idx, "previous_date_regenerated"] = True

        df = df_sorted.reset_index(drop=True)
        LOGGER.info(f"  Regenerated {(df['previous_date_regenerated']).sum()} synthetic previous dates")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer derived features for anomaly detection.

    Features:
    - time_since_previous_transaction: Days between current and previous transaction
    - transaction_amount_to_balance_ratio: Transaction amount / post-transaction balance
    - login_attempt_risk: Normalized login attempts (0-1 scale)
    - device_change_flag: First appearance of device for this account
    - ip_change_flag: First appearance of IP for this account
    - account_transaction_count: Total transactions per account
    - merchant_transaction_count: Total transactions per merchant
    - location_transaction_count: Total transactions per location
    """
    LOGGER.info("Engineering features...")

    # 1. Time since previous transaction
    df["time_since_previous_transaction"] = (
        df["transactiondate"] - df["previoustransactiondate"]
    ).dt.days.fillna(0)
    df["time_since_previous_transaction"] = df["time_since_previous_transaction"].clip(lower=0)

    # 2. Transaction amount to balance ratio
    df["transaction_amount_to_balance_ratio"] = (
        df["transactionamount"] / df["accountbalance"].replace(0, 1)
    ).clip(0, 1000)  # Cap at 1000x to handle edge cases

    # 3. Login attempt risk (normalize to 0-1)
    max_login_attempts = df["loginattempts"].max()
    df["login_attempt_risk"] = (df["loginattempts"] / max(max_login_attempts, 1)).clip(0, 1)

    # 4. Device change flag: First device for account?
    df["device_change_flag"] = df.groupby("accountid")["deviceid"].transform(
        lambda x: x != x.iloc[0]
    ).astype(int)

    # 5. IP change flag: First IP for account?
    df["ip_change_flag"] = df.groupby("accountid")["ip_address"].transform(
        lambda x: x != x.iloc[0]
    ).astype(int)

    # 6. Account transaction count (total transactions per account)
    df["account_transaction_count"] = df.groupby("accountid")["transactionid"].transform("size")

    # 7. Merchant transaction count (total transactions per merchant)
    df["merchant_transaction_count"] = df.groupby("merchantid")["transactionid"].transform("size")

    # 8. Location transaction count (total transactions per location)
    df["location_transaction_count"] = df.groupby("location")["transactionid"].transform("size")

    LOGGER.info(f"  Engineered 8 features")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate rows. Log if any found.
    """
    before_len = len(df)
    df = df.drop_duplicates(subset=["transactionid"]).reset_index(drop=True)
    after_len = len(df)

    if before_len > after_len:
        LOGGER.warning(f"Removed {before_len - after_len} duplicate transactions")

    return df


def validate_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure correct data types.
    """
    LOGGER.info("Validating and converting data types...")

    dtype_map = {
        "transactionid": "string",
        "accountid": "string",
        "transactionamount": "float64",
        "transactiondate": "datetime64[ns]",
        "transactiontype": "category",
        "location": "string",
        "deviceid": "string",
        "ip_address": "string",
        "merchantid": "string",
        "channel": "category",
        "customerage": "int64",
        "customeroccupation": "category",
        "transactionduration": "int64",
        "loginattempts": "int64",
        "accountbalance": "float64",
        "previoustransactiondate": "datetime64[ns]",
    }

    for col, dtype in dtype_map.items():
        if col in df.columns:
            try:
                if dtype.startswith("datetime"):
                    # Already parsed
                    pass
                elif dtype == "string":
                    df[col] = df[col].astype("string")
                elif dtype == "category":
                    df[col] = df[col].astype(dtype)
                else:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)
            except Exception as e:
                LOGGER.warning(f"  Could not convert {col} to {dtype}: {e}")

    return df


def generate_data_quality_report(df: pd.DataFrame) -> None:
    """
    Print summary of data quality after cleaning.
    """
    LOGGER.info("\n=== DATA QUALITY REPORT ===")
    LOGGER.info(f"Rows: {len(df)}")
    LOGGER.info(f"Columns: {len(df.columns)}")

    missing = check_missing_values(df)
    if len(missing) > 0:
        LOGGER.info("Missing values:")
        for _, row in missing.iterrows():
            LOGGER.info(f"  {row['column']}: {row['missing_count']} ({row['missing_pct']:.2f}%)")
    else:
        LOGGER.info("  No missing values")

    # Check for duplicates
    dup_count = df.duplicated(subset=["transactionid"]).sum()
    LOGGER.info(f"Duplicate TransactionIDs: {dup_count}")

    # Check date range
    if "transactiondate" in df.columns:
        date_min = df["transactiondate"].min()
        date_max = df["transactiondate"].max()
        LOGGER.info(f"Transaction Date Range: {date_min} to {date_max}")

    # Numeric statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    LOGGER.info(f"Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols)}")

    LOGGER.info("=== END QUALITY REPORT ===\n")


def clean_transactions_dataframe(
    df: pd.DataFrame,
    output_file: Path = None,
    save_output: bool = True,
) -> pd.DataFrame:
    """Clean and enrich a raw transaction dataframe already loaded in memory."""
    # Clean column names
    df = clean_column_names(df)

    # Parse dates
    df = parse_dates(df)

    # Fix date issues
    df = fix_previous_transaction_dates(df)

    # Remove duplicates
    df = remove_duplicates(df)

    # Validate data types
    df = validate_data_types(df)

    # Engineer features
    df = engineer_features(df)

    # Quality report
    generate_data_quality_report(df)

    # Save cleaned data
    if save_output and output_file is not None:
        save_csv(df, output_file)

    return df


def load_and_clean(
    input_file: Path = None,
    output_file: Path = None,
    save_output: bool = True,
) -> pd.DataFrame:
    """
    End-to-end data loading and cleaning pipeline.

    Args:
        input_file: Path to raw CSV (default: config.RAW_DATA_FILE)
        output_file: Path to save cleaned CSV (default: config.CLEANED_DATA_FILE)
        save_output: Persist cleaned CSV to disk when True

    Returns:
        Cleaned DataFrame
    """
    if input_file is None:
        input_file = config.RAW_DATA_FILE
    if output_file is None:
        output_file = config.CLEANED_DATA_FILE

    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("STAGE 1: DATA INGESTION & CLEANING")
    LOGGER.info("=" * 60)

    # Load
    df = load_raw_data(input_file)

    df = clean_transactions_dataframe(df, output_file=output_file, save_output=save_output)

    LOGGER.info(f"\nStage 1 complete. Cleaned data saved to {output_file}\n")

    return df


if __name__ == "__main__":
    # Run standalone for testing
    df = load_and_clean()
    print(f"\nCleaned DataFrame shape: {df.shape}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nColumns: {list(df.columns)}")
