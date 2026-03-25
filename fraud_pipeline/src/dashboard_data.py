"""
Helpers for Streamlit upload workflows and active dashboard data bundles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd

from . import config
from .anomaly_detection import run_anomaly_detection
from .graph_analysis import graph_analysis
from .ingest_clean import clean_transactions_dataframe
from .risk_scoring import risk_scoring


CSV_UPLOAD_OPTIONS = [
    "Raw transaction dataset",
    "Processed / scored transaction dataset",
    "Analyst review log",
]


CSV_TYPE_EXPECTATIONS: Dict[str, List[str]] = {
    "Raw transaction dataset": [
        "transactionid",
        "accountid",
        "transactionamount",
        "transactiondate",
        "location",
        "deviceid",
        "ip_address",
        "merchantid",
        "channel",
        "loginattempts",
        "accountbalance",
        "previoustransactiondate",
    ],
    "Processed / scored transaction dataset": [
        "transactionid",
        "composite_risk_score",
        "risk_level",
        "accountid",
    ],
    "Analyst review log": [
        "transactionid",
        "decision",
        "updated_at",
    ],
}


def normalize_uploaded_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = normalized.columns.str.strip().str.lower().str.replace(" ", "_")
    return normalized


def validate_uploaded_csv(df: pd.DataFrame, csv_type: str) -> Dict[str, Any]:
    normalized = normalize_uploaded_columns(df)
    expected = CSV_TYPE_EXPECTATIONS.get(csv_type, [])
    missing = [column for column in expected if column not in normalized.columns]
    return {
        "normalized_df": normalized,
        "expected_columns": expected,
        "missing_columns": missing,
        "is_valid": len(missing) == 0,
    }


def build_entity_summary(transactions: pd.DataFrame, entity_col: str) -> pd.DataFrame:
    if transactions.empty or entity_col not in transactions.columns:
        return pd.DataFrame()
    valid = transactions.dropna(subset=[entity_col]).copy()
    if valid.empty:
        return pd.DataFrame()
    summary = valid.groupby(entity_col).agg(
        avg_risk_score=("composite_risk_score", "mean"),
        max_risk_score=("composite_risk_score", "max"),
        transaction_count=("transactionid", "count"),
        high_risk_count=("risk_level", lambda values: (values == "High").sum()),
    ).reset_index()
    summary["high_risk_pct"] = 100 * summary["high_risk_count"] / summary["transaction_count"]
    return summary.sort_values(["max_risk_score", "avg_risk_score"], ascending=False).reset_index(drop=True)


def build_account_summary(transactions: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty or "accountid" not in transactions.columns:
        return pd.DataFrame()
    accounts = (
        transactions.groupby("accountid")
        .agg(
            avg_risk_score=("composite_risk_score", "mean"),
            max_risk_score=("composite_risk_score", "max"),
            risk_score_std=("composite_risk_score", "std"),
            transaction_count=("transactionid", "count"),
            high_risk_transaction_count=("risk_level", lambda values: (values == "High").sum()),
        )
        .reset_index()
    )
    accounts["account_risk_score"] = accounts["max_risk_score"]
    accounts["high_risk_transaction_pct"] = 100 * accounts["high_risk_transaction_count"] / accounts["transaction_count"]
    accounts["account_risk_rank"] = range(1, len(accounts) + 1)
    return accounts.sort_values("account_risk_score", ascending=False).reset_index(drop=True)


def build_summary_snapshot(
    transactions: pd.DataFrame,
    accounts: pd.DataFrame,
    merchants: pd.DataFrame,
    devices: pd.DataFrame,
    locations: pd.DataFrame,
) -> Dict[str, Any]:
    if transactions.empty:
        return {}
    return {
        "total_transactions": int(len(transactions)),
        "total_accounts": int(transactions["accountid"].nunique(dropna=True)) if "accountid" in transactions else 0,
        "total_merchants": int(transactions["merchantid"].nunique(dropna=True)) if "merchantid" in transactions else 0,
        "total_locations": int(transactions["location"].nunique(dropna=True)) if "location" in transactions else 0,
        "high_risk_count": int((transactions["risk_level"] == "High").sum()) if "risk_level" in transactions else 0,
        "high_risk_pct": float(100 * (transactions["risk_level"] == "High").mean()) if "risk_level" in transactions else 0.0,
        "medium_risk_count": int((transactions["risk_level"] == "Medium").sum()) if "risk_level" in transactions else 0,
        "medium_risk_pct": float(100 * (transactions["risk_level"] == "Medium").mean()) if "risk_level" in transactions else 0.0,
        "avg_composite_score": float(transactions["composite_risk_score"].mean()) if "composite_risk_score" in transactions else 0.0,
        "max_composite_score": float(transactions["composite_risk_score"].max()) if "composite_risk_score" in transactions else 0.0,
        "median_composite_score": float(transactions["composite_risk_score"].median()) if "composite_risk_score" in transactions else 0.0,
        "total_transaction_volume": float(transactions["transactionamount"].sum()) if "transactionamount" in transactions else 0.0,
        "high_risk_transaction_volume": float(
            transactions.loc[transactions["risk_level"] == "High", "transactionamount"].sum()
        )
        if {"risk_level", "transactionamount"}.issubset(transactions.columns)
        else 0.0,
        "flagged_transactions": int(transactions["risk_level"].isin(["High", "Medium"]).sum()) if "risk_level" in transactions else 0,
        "high_risk_accounts": int((accounts["account_risk_score"] >= config.RISK_LEVEL_MEDIUM).sum()) if "account_risk_score" in accounts else 0,
        "high_risk_merchants": int((merchants["max_risk_score"] >= config.RISK_LEVEL_MEDIUM).sum()) if "max_risk_score" in merchants else 0,
        "high_risk_devices": int((devices["max_risk_score"] >= config.RISK_LEVEL_MEDIUM).sum()) if "max_risk_score" in devices else 0,
        "high_risk_locations": int((locations["max_risk_score"] >= config.RISK_LEVEL_MEDIUM).sum()) if "max_risk_score" in locations else 0,
    }


def bundle_from_transactions(transactions: pd.DataFrame, source_label: str) -> Dict[str, Any]:
    ranked_transactions = transactions.copy()
    if "composite_risk_score" in ranked_transactions.columns:
        ranked_transactions = ranked_transactions.sort_values("composite_risk_score", ascending=False).reset_index(drop=True)
    accounts = build_account_summary(ranked_transactions)
    merchants = build_entity_summary(ranked_transactions, "merchantid")
    devices = build_entity_summary(ranked_transactions, "deviceid")
    ips = build_entity_summary(ranked_transactions, "ip_address")
    locations = build_entity_summary(ranked_transactions, "location")
    summary = build_summary_snapshot(ranked_transactions, accounts, merchants, devices, locations)
    return {
        "transactions": ranked_transactions,
        "accounts": accounts,
        "merchants": merchants,
        "devices": devices,
        "ips": ips,
        "locations": locations,
        "summary": summary,
        "explanations": {},
        "source_label": source_label,
        "uploaded_type": "transactions",
    }


def analyze_uploaded_raw_transactions(raw_df: pd.DataFrame, source_label: str) -> Dict[str, Any]:
    cleaned = clean_transactions_dataframe(raw_df.copy(), save_output=False)
    anomaly_scores = run_anomaly_detection(cleaned, save_output=False)
    graph_features, _ = graph_analysis(cleaned, save_output=False)
    risk_results = risk_scoring(cleaned, anomaly_scores, graph_features, save_output=False)
    bundle = bundle_from_transactions(risk_results["transactions_ranked"], source_label)
    bundle["accounts"] = risk_results["accounts_ranked"]
    bundle["merchants"] = risk_results["merchants_ranked"]
    bundle["devices"] = risk_results["devices_ranked"]
    bundle["ips"] = risk_results["ips_ranked"]
    bundle["summary"] = build_summary_snapshot(
        bundle["transactions"],
        bundle["accounts"],
        bundle["merchants"],
        bundle["devices"],
        bundle["locations"],
    )
    bundle["cleaned_transactions"] = cleaned
    bundle["anomaly_scores"] = anomaly_scores
    bundle["graph_features"] = graph_features
    bundle["source_label"] = source_label
    bundle["uploaded_type"] = "raw_transaction_dataset"
    return bundle


def bundle_from_uploaded_csv(df: pd.DataFrame, csv_type: str, source_label: str) -> Dict[str, Any]:
    normalized = normalize_uploaded_columns(df)

    if csv_type == "Raw transaction dataset":
        return analyze_uploaded_raw_transactions(normalized, source_label)

    if csv_type == "Processed / scored transaction dataset":
        return bundle_from_transactions(normalized, source_label)

    if csv_type == "Analyst review log":
        review_log = normalized.copy()
        return {
            "transactions": pd.DataFrame(),
            "accounts": pd.DataFrame(),
            "merchants": pd.DataFrame(),
            "devices": pd.DataFrame(),
            "ips": pd.DataFrame(),
            "locations": pd.DataFrame(),
            "summary": {},
            "review_log": review_log,
            "explanations": {},
            "source_label": source_label,
            "uploaded_type": "review_log",
        }

    raise ValueError(f"Unsupported CSV type: {csv_type}")
