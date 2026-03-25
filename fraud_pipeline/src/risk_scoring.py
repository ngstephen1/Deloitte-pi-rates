"""
Risk scoring and ranking module.
Combines anomaly detection, graph features, and transaction metadata into a
transparent composite risk score.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .utils import LOGGER, save_csv, normalize_to_01


def compute_amount_outlier_risk(df: pd.DataFrame) -> np.ndarray:
    """
    Flag unusually high or low transaction amounts as risky.

    Returns:
        Risk scores [0, 1]
    """
    amounts = df["transactionamount"].values

    # Z-score based outlier detection
    mean = np.nanmean(amounts)
    std = np.nanstd(amounts)

    if std == 0:
        z_scores = np.zeros_like(amounts)
    else:
        z_scores = np.abs((amounts - mean) / std)

    # Normalize to [0, 1]
    risk = normalize_to_01(z_scores)

    return risk


def combine_risk_signals(
    df: pd.DataFrame,
    anomaly_scores: pd.DataFrame,
    graph_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine all risk signals into a single composite score using
    transparent, configurable weights from config.RISK_WEIGHTS.

    Args:
        df: Original cleaned transaction DataFrame
        anomaly_scores: DataFrame with anomaly detection scores
        graph_features: DataFrame with graph-derived features

    Returns:
        DataFrame with composite risk score and component breakdown
    """
    LOGGER.info("Combining risk signals...")

    # Start with transaction IDs
    risk_df = df[["transactionid", "accountid", "merchantid", "deviceid", "ip_address", "location"]].copy()

    # Merge anomaly scores
    risk_df = risk_df.merge(anomaly_scores, on="transactionid", how="left")

    # Merge graph features if available
    if graph_features is not None:
        risk_df = risk_df.merge(graph_features, on="transactionid", how="left")
    else:
        LOGGER.info("Graph features not provided. Skipping graph feature merge.")

    # Compute amount outlier risk
    amount_outlier_risk = compute_amount_outlier_risk(df)

    # Ensure amount_outlier_risk aligns with risk_df index
    if len(amount_outlier_risk) != len(risk_df):
        LOGGER.error("Length mismatch: amount_outlier_risk does not align with risk_df index.")
        raise ValueError("Length of amount_outlier_risk does not match risk_df index.")

    risk_df["amount_outlier_risk"] = amount_outlier_risk

    # Handle missing previous_date_regenerated column
    if "previous_date_regenerated" in df.columns:
        risk_df["previous_date_regenerated"] = df["previous_date_regenerated"].astype(float).values
    else:
        risk_df["previous_date_regenerated"] = 0.0

    # Fill any NaN with 0 for scoring
    risk_df = risk_df.fillna(0)

    # Compute composite score using configured weights
    composite_score = np.zeros(len(risk_df))

    for component, weight in config.RISK_WEIGHTS.items():
        if component in risk_df.columns:
            component_values = risk_df[component].values
            # Ensure values are in [0, 1]
            component_values = normalize_to_01(component_values)
            composite_score += weight * component_values
        else:
            LOGGER.warning(f"  Warning: risk component '{component}' not found in data")

    # Normalize composite score to [0, 1]
    risk_df["composite_risk_score"] = normalize_to_01(composite_score)

    # Categorize risk level
    def risk_level(score):
        if score < config.RISK_LEVEL_LOW:
            return "Low"
        elif score < config.RISK_LEVEL_MEDIUM:
            return "Medium"
        else:
            return "High"

    risk_df["risk_level"] = risk_df["composite_risk_score"].apply(risk_level)

    LOGGER.info(f"  Composite score range: [{risk_df['composite_risk_score'].min():.3f}, {risk_df['composite_risk_score'].max():.3f}]")
    LOGGER.info(f"  Risk level distribution:")
    for level in ["Low", "Medium", "High"]:
        count = (risk_df["risk_level"] == level).sum()
        pct = 100 * count / len(risk_df)
        LOGGER.info(f"    {level}: {count} ({pct:.1f}%)")

    return risk_df


def rank_transactions(risk_df: pd.DataFrame, top_n: int = None) -> pd.DataFrame:
    """
    Rank transactions by composite risk score (descending).

    Returns:
        Ranked DataFrame with risk_rank column
    """
    ranked = risk_df.sort_values("composite_risk_score", ascending=False).reset_index(drop=True)
    ranked["risk_rank"] = range(1, len(ranked) + 1)

    return ranked


def summarize_account_risk(risk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate risk by account.

    Returns:
        DataFrame with account-level risk metrics
    """
    LOGGER.info("Aggregating account-level risk...")

    account_risk = risk_df.groupby("accountid").agg({
        "composite_risk_score": ["mean", "max", "std"],
        "transactionid": "count",
        "risk_level": lambda x: (x == "High").sum(),
    }).reset_index()

    account_risk.columns = [
        "accountid",
        "avg_risk_score",
        "max_risk_score",
        "risk_score_std",
        "transaction_count",
        "high_risk_transaction_count",
    ]

    # Compute account-level composite risk
    account_risk["account_risk_score"] = normalize_to_01(account_risk["max_risk_score"].values)
    account_risk["high_risk_transaction_pct"] = (
        100 * account_risk["high_risk_transaction_count"] / account_risk["transaction_count"]
    )

    # Sort by account risk score
    account_risk = account_risk.sort_values("account_risk_score", ascending=False).reset_index(drop=True)
    account_risk["account_risk_rank"] = range(1, len(account_risk) + 1)

    return account_risk


def summarize_merchant_risk(risk_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate risk by merchant."""
    merchant_risk = risk_df.groupby("merchantid").agg({
        "composite_risk_score": ["mean", "max"],
        "transactionid": "count",
        "risk_level": lambda x: (x == "High").sum(),
    }).reset_index()

    merchant_risk.columns = [
        "merchantid",
        "avg_risk_score",
        "max_risk_score",
        "transaction_count",
        "high_risk_count",
    ]

    merchant_risk["high_risk_pct"] = 100 * merchant_risk["high_risk_count"] / merchant_risk["transaction_count"]
    merchant_risk = merchant_risk.sort_values("avg_risk_score", ascending=False).reset_index(drop=True)

    return merchant_risk


def summarize_device_risk(risk_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate risk by device."""
    device_risk = risk_df.groupby("deviceid").agg({
        "composite_risk_score": ["mean", "max"],
        "transactionid": "count",
        "risk_level": lambda x: (x == "High").sum(),
    }).reset_index()

    device_risk.columns = [
        "deviceid",
        "avg_risk_score",
        "max_risk_score",
        "transaction_count",
        "high_risk_count",
    ]

    device_risk["high_risk_pct"] = 100 * device_risk["high_risk_count"] / device_risk["transaction_count"]
    device_risk = device_risk.sort_values("avg_risk_score", ascending=False).reset_index(drop=True)

    return device_risk


def summarize_ip_risk(risk_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate risk by IP address."""
    ip_risk = risk_df.groupby("ip_address").agg({
        "composite_risk_score": ["mean", "max"],
        "transactionid": "count",
        "risk_level": lambda x: (x == "High").sum(),
    }).reset_index()

    ip_risk.columns = [
        "ip_address",
        "avg_risk_score",
        "max_risk_score",
        "transaction_count",
        "high_risk_count",
    ]

    ip_risk["high_risk_pct"] = 100 * ip_risk["high_risk_count"] / ip_risk["transaction_count"]
    ip_risk = ip_risk.sort_values("avg_risk_score", ascending=False).reset_index(drop=True)

    return ip_risk


def risk_scoring(
    df: pd.DataFrame,
    anomaly_scores: pd.DataFrame,
    graph_features: pd.DataFrame,
) -> dict:
    """
    Run full risk scoring and ranking pipeline.

    Returns:
        Dictionary with all ranking outputs
    """
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("STAGE 5: RISK SCORING & RANKING")
    LOGGER.info("=" * 60)

    # Combine signals
    risk_df = combine_risk_signals(df, anomaly_scores, graph_features)

    # Rank transactions
    risk_ranked = rank_transactions(risk_df)

    # Aggregate by entity
    account_risk = summarize_account_risk(risk_ranked)
    merchant_risk = summarize_merchant_risk(risk_ranked)
    device_risk = summarize_device_risk(risk_ranked)
    ip_risk = summarize_ip_risk(risk_ranked)

    # Save outputs
    LOGGER.info("Saving ranking outputs...")
    save_csv(risk_ranked, config.RISK_TRANSACTIONS_FILE)
    save_csv(account_risk, config.RISK_ACCOUNTS_FILE)
    save_csv(merchant_risk, config.REPORTS_DIR / "risk_ranked_merchants.csv")
    save_csv(device_risk, config.REPORTS_DIR / "risk_ranked_devices.csv")
    save_csv(ip_risk, config.REPORTS_DIR / "risk_ranked_ips.csv")

    LOGGER.info(f"\nStage 5 complete. Ranking outputs saved to {config.REPORTS_DIR}\n")

    return {
        "transactions_ranked": risk_ranked,
        "accounts_ranked": account_risk,
        "merchants_ranked": merchant_risk,
        "devices_ranked": device_risk,
        "ips_ranked": ip_risk,
    }


if __name__ == "__main__":
    # For testing
    from .ingest_clean import load_and_clean
    from .anomaly_detection import run_anomaly_detection
    from .graph_analysis import graph_analysis

    df = load_and_clean()
    anomaly_scores = run_anomaly_detection(df)
    graph_features, _ = graph_analysis(df)
    results = risk_scoring(df, anomaly_scores, graph_features)
    print(results["transactions_ranked"].head())
