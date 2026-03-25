"""
Compact data loading and publish helpers for ChatOps / OpenClaw consumers.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .. import config
from ..dashboard_data import bundle_from_transactions
from ..review_store import ReviewStore
from ..utils import LOGGER


ACTIVE_BUNDLE_FILES = {
    "transactions": "transactions.csv",
    "accounts": "accounts.csv",
    "merchants": "merchants.csv",
    "devices": "devices.csv",
    "ips": "ips.csv",
    "locations": "locations.csv",
    "review_log": "review_log.csv",
    "anomaly_scores": "anomaly_scores.csv",
    "graph_features": "graph_features.csv",
    "tda_features": "tda_features.csv",
    "ai_review_recommendations": "ai_review_recommendations.csv",
}


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        LOGGER.warning(f"Could not read CSV {path}: {exc}")
        return pd.DataFrame()


def _safe_read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        LOGGER.warning(f"Could not read JSON {path}: {exc}")
        return {}


def build_review_summary(transactions: pd.DataFrame, review_log: pd.DataFrame) -> Dict[str, Any]:
    review_log = review_log.copy() if isinstance(review_log, pd.DataFrame) else pd.DataFrame()
    if not review_log.empty and "transactionid" in review_log.columns:
        review_log["transactionid"] = review_log["transactionid"].astype(str)

    active_ids = set()
    if isinstance(transactions, pd.DataFrame) and not transactions.empty and "transactionid" in transactions.columns:
        active_ids = set(transactions["transactionid"].astype(str))

    reviewed_ids = set(review_log["transactionid"].astype(str)) if not review_log.empty and "transactionid" in review_log else set()
    decision_counts = review_log["decision"].value_counts().to_dict() if not review_log.empty and "decision" in review_log else {}

    pending_review_ids = set()
    if not review_log.empty and {"transactionid", "decision"}.issubset(review_log.columns):
        needs_review = review_log.loc[review_log["decision"] == "Needs Review", "transactionid"].astype(str)
        pending_review_ids.update(needs_review.tolist())
    pending_review_ids.update(active_ids - reviewed_ids)

    pending_flagged = pd.DataFrame()
    pending_high_risk = pd.DataFrame()
    if isinstance(transactions, pd.DataFrame) and not transactions.empty and {"transactionid", "risk_level"}.issubset(transactions.columns):
        pending_flagged = transactions.loc[
            transactions["transactionid"].astype(str).isin(pending_review_ids)
            & transactions["risk_level"].isin(["High", "Medium"])
        ]
        pending_high_risk = transactions.loc[
            transactions["transactionid"].astype(str).isin(pending_review_ids)
            & (transactions["risk_level"] == "High")
        ]

    return {
        "reviewed_total": int(len(review_log)),
        "approved_count": int(decision_counts.get("Approve Flag", 0)),
        "dismissed_count": int(decision_counts.get("Dismiss", 0)),
        "needs_review_count": int(decision_counts.get("Needs Review", 0)),
        "unreviewed_count": int(len(active_ids - reviewed_ids)),
        "pending_review_total": int(len(pending_review_ids)),
        "pending_flagged_count": int(len(pending_flagged)),
        "pending_high_risk_count": int(len(pending_high_risk)),
    }


def load_report_bundle() -> Dict[str, Any]:
    transactions = _safe_read_csv(config.RISK_TRANSACTIONS_FILE)
    accounts = _safe_read_csv(config.RISK_ACCOUNTS_FILE)
    merchants = _safe_read_csv(config.RISK_MERCHANTS_FILE)
    devices = _safe_read_csv(config.RISK_DEVICES_FILE)
    ips = _safe_read_csv(config.RISK_IPS_FILE)
    anomaly_scores = _safe_read_csv(config.ANOMALY_SCORES_FILE)
    graph_features = _safe_read_csv(config.GRAPH_FEATURES_FILE)
    tda_features = _safe_read_csv(config.TDA_FEATURES_FILE)
    ai_review_recommendations = _safe_read_csv(config.AI_REVIEW_RECOMMENDATIONS_FILE)
    review_log = ReviewStore().get_all_decisions()

    if not transactions.empty:
        bundle = bundle_from_transactions(transactions, "Pipeline outputs")
    else:
        bundle = {
            "transactions": pd.DataFrame(),
            "accounts": pd.DataFrame(),
            "merchants": pd.DataFrame(),
            "devices": pd.DataFrame(),
            "ips": pd.DataFrame(),
            "locations": pd.DataFrame(),
            "summary": {},
            "explanations": {},
            "source_label": "Pipeline outputs",
            "uploaded_type": "pipeline_outputs",
        }

    if not accounts.empty:
        bundle["accounts"] = accounts
    if not merchants.empty:
        bundle["merchants"] = merchants
    if not devices.empty:
        bundle["devices"] = devices
    if not ips.empty:
        bundle["ips"] = ips

    file_summary = _safe_read_json(config.EXECUTIVE_SUMMARY_FILE)
    review_summary = build_review_summary(bundle.get("transactions", pd.DataFrame()), review_log)
    bundle["summary"] = {**bundle.get("summary", {}), **file_summary, **review_summary}
    bundle["review_log"] = review_log
    bundle["anomaly_scores"] = anomaly_scores
    bundle["graph_features"] = graph_features
    bundle["tda_features"] = tda_features
    bundle["ai_review_recommendations"] = ai_review_recommendations
    bundle["source_label"] = "Pipeline outputs"
    bundle["uploaded_type"] = "pipeline_outputs"
    return bundle


def _bundle_from_active_dir(active_dir: Path) -> Dict[str, Any]:
    manifest = _safe_read_json(active_dir / "manifest.json")
    data_frames = {key: _safe_read_csv(active_dir / filename) for key, filename in ACTIVE_BUNDLE_FILES.items()}
    transactions = data_frames["transactions"]

    if not transactions.empty:
        bundle = bundle_from_transactions(transactions, manifest.get("source_label", "Published dashboard context"))
    else:
        bundle = {
            "transactions": pd.DataFrame(),
            "accounts": pd.DataFrame(),
            "merchants": pd.DataFrame(),
            "devices": pd.DataFrame(),
            "ips": pd.DataFrame(),
            "locations": pd.DataFrame(),
            "summary": {},
            "explanations": {},
            "source_label": manifest.get("source_label", "Published dashboard context"),
            "uploaded_type": manifest.get("uploaded_type", "published_context"),
        }

    for key in [
        "accounts",
        "merchants",
        "devices",
        "ips",
        "locations",
        "review_log",
        "anomaly_scores",
        "graph_features",
        "tda_features",
        "ai_review_recommendations",
    ]:
        if not data_frames[key].empty:
            bundle[key] = data_frames[key]

    bundle["summary"] = {
        **bundle.get("summary", {}),
        **(manifest.get("summary") or {}),
        **build_review_summary(bundle.get("transactions", pd.DataFrame()), bundle.get("review_log", pd.DataFrame())),
    }
    bundle["source_label"] = manifest.get("source_label", bundle.get("source_label", "Published dashboard context"))
    bundle["uploaded_type"] = manifest.get("uploaded_type", bundle.get("uploaded_type", "published_context"))
    bundle["chatops_manifest"] = manifest
    return bundle


def load_active_bundle(prefer_published: bool = True) -> Dict[str, Any]:
    if prefer_published and config.CHATOPS_MANIFEST_FILE.exists():
        bundle = _bundle_from_active_dir(config.CHATOPS_ACTIVE_DIR)
        if any(isinstance(bundle.get(key), pd.DataFrame) and not bundle.get(key).empty for key in ["transactions", "review_log"]):
            return bundle
    return load_report_bundle()


def publish_bundle_context(
    bundle: Dict[str, Any],
    *,
    source_label: str | None = None,
    publish_reason: str = "streamlit",
) -> Dict[str, Any]:
    config.CHATOPS_ACTIVE_DIR.mkdir(parents=True, exist_ok=True)

    for key, filename in ACTIVE_BUNDLE_FILES.items():
        frame = bundle.get(key)
        path = config.CHATOPS_ACTIVE_DIR / filename
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            frame.to_csv(path, index=False)
        elif path.exists():
            path.unlink()

    review_log = bundle.get("review_log")
    if not isinstance(review_log, pd.DataFrame):
        review_log = ReviewStore().get_all_decisions()
        if not review_log.empty:
            review_log.to_csv(config.CHATOPS_ACTIVE_DIR / ACTIVE_BUNDLE_FILES["review_log"], index=False)

    manifest = {
        "source_label": source_label or bundle.get("source_label", "Published dashboard context"),
        "uploaded_type": bundle.get("uploaded_type", "published_context"),
        "publish_reason": publish_reason,
        "published_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            **(bundle.get("summary", {}) or {}),
            **build_review_summary(bundle.get("transactions", pd.DataFrame()), review_log),
        },
    }
    config.CHATOPS_MANIFEST_FILE.write_text(json.dumps(manifest, indent=2))
    LOGGER.info(
        "Published ChatOps context to %s from %s",
        config.CHATOPS_ACTIVE_DIR,
        manifest["source_label"],
    )
    return manifest
