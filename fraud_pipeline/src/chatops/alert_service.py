"""
Transparent fraud alert rules, deduplication, and ChatOps delivery helpers.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from .. import config
from ..utils import LOGGER
from .context_loader import build_review_summary, load_active_bundle, publish_bundle_context
from .message_formatter import (
    build_alert_message,
    build_case_reminder_message,
    build_decision_update_message,
    build_qna_message,
    build_reminder_message,
    build_report_message,
)
from .openclaw_bridge import deliver_message


def _load_alert_state() -> Dict[str, Any]:
    if not config.CHATOPS_ALERT_STATE_FILE.exists():
        return {}
    try:
        return json.loads(config.CHATOPS_ALERT_STATE_FILE.read_text())
    except Exception:
        return {}


def _save_alert_state(state: Dict[str, Any]) -> None:
    config.CHATOPS_ALERT_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.CHATOPS_ALERT_STATE_FILE.write_text(json.dumps(state, indent=2))


def _recently_sent(state: Dict[str, Any], alert_id: str) -> bool:
    sent_at_raw = (state.get(alert_id) or {}).get("sent_at")
    if not sent_at_raw:
        return False
    try:
        sent_at = datetime.fromisoformat(sent_at_raw)
    except Exception:
        return False
    return datetime.now(timezone.utc) - sent_at < timedelta(hours=config.OPENCLAW_ALERT_DEDUPE_HOURS)


def _mark_sent(state: Dict[str, Any], alert_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    state[alert_id] = {
        "sent_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
    }


def _location_risk_table(transactions: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty or "location" not in transactions.columns:
        return pd.DataFrame()
    flagged = transactions.loc[transactions["risk_level"].isin(["High", "Medium"])].copy() if "risk_level" in transactions else transactions.copy()
    if flagged.empty:
        return pd.DataFrame()
    return (
        flagged.groupby("location", dropna=False)
        .agg(
            flagged_count=("transactionid", "count"),
            avg_risk_score=("composite_risk_score", "mean"),
            max_risk_score=("composite_risk_score", "max"),
        )
        .reset_index()
        .sort_values(["flagged_count", "avg_risk_score"], ascending=False)
    )


def _pending_review_transactions(bundle: Dict[str, Any]) -> pd.DataFrame:
    transactions = bundle.get("transactions", pd.DataFrame())
    review_log = bundle.get("review_log", pd.DataFrame())
    if transactions.empty or "transactionid" not in transactions.columns:
        return pd.DataFrame()

    reviewed_ids = set()
    needs_review_ids = set()
    if isinstance(review_log, pd.DataFrame) and not review_log.empty and "transactionid" in review_log.columns:
        reviewed_ids = set(review_log["transactionid"].astype(str))
        if "decision" in review_log.columns:
            needs_review_ids = set(
                review_log.loc[review_log["decision"] == "Needs Review", "transactionid"].astype(str).tolist()
            )

    pending_ids = (set(transactions["transactionid"].astype(str)) - reviewed_ids) | needs_review_ids
    pending = transactions.loc[transactions["transactionid"].astype(str).isin(pending_ids)].copy()
    if "composite_risk_score" in pending.columns:
        pending = pending.sort_values("composite_risk_score", ascending=False)
    return pending


def generate_fraud_alerts(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    transactions = bundle.get("transactions", pd.DataFrame())
    accounts = bundle.get("accounts", pd.DataFrame())
    merchants = bundle.get("merchants", pd.DataFrame())
    devices = bundle.get("devices", pd.DataFrame())
    summary = bundle.get("summary", {}) or {}
    alerts: List[Dict[str, Any]] = []

    if isinstance(transactions, pd.DataFrame) and not transactions.empty:
        critical_transactions = transactions.loc[
            transactions["composite_risk_score"] >= config.OPENCLAW_ALERT_TRANSACTION_CRITICAL_THRESHOLD
        ].head(min(3, config.OPENCLAW_ALERT_MAX_ITEMS))
        for row in critical_transactions.itertuples():
            alerts.append(
                {
                    "alert_id": f"transaction::{row.transactionid}",
                    "alert_type": "fraud.alert.transaction",
                    "severity": "critical",
                    "title": f"Critical Transaction Risk Alert • {row.transactionid}",
                    "entity_type": "transaction",
                    "entity_id": str(row.transactionid),
                    "risk_score": float(getattr(row, "composite_risk_score", 0) or 0),
                    "reason": (
                        f"Transaction {row.transactionid} exceeded the critical composite-risk threshold "
                        f"at {float(getattr(row, 'composite_risk_score', 0) or 0):.3f}."
                    ),
                    "evidence": [
                        f"Account {getattr(row, 'accountid', 'N/A')}",
                        f"Merchant {getattr(row, 'merchantid', 'N/A')}",
                        f"Channel {getattr(row, 'channel', 'N/A')}",
                        f"Amount ${float(getattr(row, 'transactionamount', 0) or 0):,.2f}",
                    ],
                    "next_action": "Validate account ownership, merchant pattern, device/IP reuse, and any repeated login attempts before escalation.",
                }
            )

    if isinstance(accounts, pd.DataFrame) and not accounts.empty:
        risky_accounts = accounts.loc[
            accounts["account_risk_score"] >= config.OPENCLAW_ALERT_ACCOUNT_THRESHOLD
        ].head(2)
        for row in risky_accounts.itertuples():
            alerts.append(
                {
                    "alert_id": f"account::{row.accountid}",
                    "alert_type": "fraud.alert.account",
                    "severity": "warning",
                    "title": f"High-Risk Account Alert • {row.accountid}",
                    "entity_type": "account",
                    "entity_id": str(row.accountid),
                    "risk_score": float(getattr(row, "account_risk_score", 0) or 0),
                    "reason": (
                        f"Account {row.accountid} is in the highest account-risk tier at "
                        f"{float(getattr(row, 'account_risk_score', 0) or 0):.3f}."
                    ),
                    "evidence": [
                        f"Transactions {int(getattr(row, 'transaction_count', 0) or 0)}",
                        f"High-risk transactions {int(getattr(row, 'high_risk_transaction_count', 0) or 0)}",
                        f"High-risk share {float(getattr(row, 'high_risk_transaction_pct', 0) or 0):.1f}%",
                    ],
                    "next_action": "Prioritize transaction review for this account and confirm whether the merchant/device patterns are consistent with known behavior.",
                }
            )

    if isinstance(merchants, pd.DataFrame) and not merchants.empty:
        suspicious_merchants = merchants.loc[
            merchants["high_risk_count"] >= config.OPENCLAW_ALERT_MERCHANT_HIGH_RISK_COUNT
        ].head(2)
        for row in suspicious_merchants.itertuples():
            alerts.append(
                {
                    "alert_id": f"merchant::{row.merchantid}",
                    "alert_type": "fraud.alert.merchant",
                    "severity": "warning",
                    "title": f"Repeated Suspicious Merchant Activity • {row.merchantid}",
                    "entity_type": "merchant",
                    "entity_id": str(row.merchantid),
                    "risk_score": float(getattr(row, "max_risk_score", 0) or 0),
                    "reason": f"Merchant {row.merchantid} is repeatedly appearing in higher-risk activity.",
                    "evidence": [
                        f"High-risk count {int(getattr(row, 'high_risk_count', 0) or 0)}",
                        f"Transactions {int(getattr(row, 'transaction_count', 0) or 0)}",
                        f"Average risk {float(getattr(row, 'avg_risk_score', 0) or 0):.3f}",
                    ],
                    "next_action": "Review recent transactions tied to this merchant and determine whether tighter approval controls or merchant escalation is warranted.",
                }
            )

    if isinstance(devices, pd.DataFrame) and not devices.empty:
        risky_devices = devices.loc[
            (devices["high_risk_count"] >= config.OPENCLAW_ALERT_DEVICE_HIGH_RISK_COUNT)
            | (devices["max_risk_score"] >= config.OPENCLAW_ALERT_ACCOUNT_THRESHOLD)
        ].head(2)
        for row in risky_devices.itertuples():
            alerts.append(
                {
                    "alert_id": f"device::{row.deviceid}",
                    "alert_type": "fraud.alert.device",
                    "severity": "warning",
                    "title": f"Elevated Device Anomaly Alert • {row.deviceid}",
                    "entity_type": "device",
                    "entity_id": str(row.deviceid),
                    "risk_score": float(getattr(row, "max_risk_score", 0) or 0),
                    "reason": f"Device {row.deviceid} is linked to elevated fraud risk and repeat exposure.",
                    "evidence": [
                        f"High-risk count {int(getattr(row, 'high_risk_count', 0) or 0)}",
                        f"Transactions {int(getattr(row, 'transaction_count', 0) or 0)}",
                        f"Max risk {float(getattr(row, 'max_risk_score', 0) or 0):.3f}",
                    ],
                    "next_action": "Inspect shared-account behavior, device-change patterns, and any linked merchant or IP concentration.",
                }
            )

    location_table = _location_risk_table(transactions) if isinstance(transactions, pd.DataFrame) else pd.DataFrame()
    if not location_table.empty:
        risky_locations = location_table.loc[
            location_table["flagged_count"] >= config.OPENCLAW_ALERT_LOCATION_FLAGGED_COUNT
        ].head(2)
        for row in risky_locations.itertuples():
            alerts.append(
                {
                    "alert_id": f"location::{row.location}",
                    "alert_type": "fraud.alert.location",
                    "severity": "warning",
                    "title": f"Elevated Location Risk Concentration • {row.location}",
                    "entity_type": "location",
                    "entity_id": str(row.location),
                    "risk_score": float(getattr(row, "max_risk_score", 0) or 0),
                    "reason": f"{row.location} shows concentrated flagged activity above the configured threshold.",
                    "evidence": [
                        f"Flagged transactions {int(getattr(row, 'flagged_count', 0) or 0)}",
                        f"Average flagged risk {float(getattr(row, 'avg_risk_score', 0) or 0):.3f}",
                    ],
                    "next_action": "Validate whether the location spike aligns with expected business volume or needs targeted monitoring escalation.",
                }
            )

    review_summary = build_review_summary(transactions, bundle.get("review_log", pd.DataFrame()))
    if review_summary.get("pending_flagged_count", 0) >= config.OPENCLAW_ALERT_PENDING_REVIEW_THRESHOLD:
        pending = _pending_review_transactions(bundle).head(5)
        pending_ids = ", ".join(pending["transactionid"].astype(str).tolist()) if not pending.empty else "N/A"
        alerts.append(
            {
                "alert_id": "review-queue::pending-flagged",
                "alert_type": "fraud.alert.review_queue",
                "severity": "warning",
                "title": "Pending Flagged Review Queue Alert",
                "entity_type": "review_queue",
                "entity_id": "pending-flagged",
                "risk_score": None,
                "reason": (
                    f"There are {review_summary.get('pending_flagged_count', 0)} flagged items still awaiting analyst closure."
                ),
                "evidence": [
                    f"Pending review total {review_summary['pending_review_total']}",
                    f"Pending high-risk {review_summary['pending_high_risk_count']}",
                    f"Pending flagged {review_summary.get('pending_flagged_count', 0)}",
                    f"Top pending IDs {pending_ids}",
                ],
                "next_action": "Clear the highest-ranked pending cases first so unresolved exposure does not accumulate in the analyst queue.",
            }
        )

    if not alerts and summary.get("high_risk_count", 0):
        LOGGER.info("No threshold-based ChatOps alerts generated beyond the standard report summary.")
    return alerts[: config.OPENCLAW_ALERT_MAX_ITEMS]


def send_report_message(
    bundle: Optional[Dict[str, Any]] = None,
    *,
    headline: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    active_bundle = bundle or load_active_bundle()
    message = build_report_message(active_bundle, headline=headline)
    result = deliver_message(message, dry_run=dry_run)
    return {
        "message": message,
        "delivery": result,
    }


def publish_and_send_report(
    bundle: Dict[str, Any],
    *,
    headline: Optional[str] = None,
    publish_reason: str = "streamlit",
    dry_run: bool = False,
) -> Dict[str, Any]:
    manifest = publish_bundle_context(bundle, publish_reason=publish_reason)
    result = send_report_message(bundle, headline=headline, dry_run=dry_run)
    return {
        "manifest": manifest,
        **result,
    }


def send_alert_notifications(
    bundle: Optional[Dict[str, Any]] = None,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> Dict[str, Any]:
    active_bundle = bundle or load_active_bundle()
    state = _load_alert_state()
    alerts = generate_fraud_alerts(active_bundle)
    source_label = active_bundle.get("source_label", "Fraud dashboard")
    deliveries: List[Dict[str, Any]] = []

    for alert in alerts:
        suppressed = (not force) and _recently_sent(state, alert["alert_id"])
        if suppressed and not dry_run:
            deliveries.append({"alert": alert, "skipped": True, "reason": "deduped"})
            continue

        message = build_alert_message(alert, source_label=source_label)
        delivery = deliver_message(message, dry_run=dry_run)
        deliveries.append({"alert": alert, "delivery": delivery, "skipped": False})
        if delivery.delivered and not dry_run:
            _mark_sent(state, alert["alert_id"], {"severity": alert.get("severity"), "entity_id": alert.get("entity_id")})

    if not dry_run:
        _save_alert_state(state)

    return {
        "alerts": alerts,
        "deliveries": deliveries,
    }


def send_monitoring_reminder(
    bundle: Optional[Dict[str, Any]] = None,
    *,
    reminder_index: int = 0,
    dry_run: bool = False,
) -> Dict[str, Any]:
    active_bundle = bundle or load_active_bundle()
    if reminder_index > 0:
        message = build_case_reminder_message(active_bundle, reminder_index=reminder_index)
    else:
        message = build_case_reminder_message(active_bundle, reminder_index=0)
    result = deliver_message(message, dry_run=dry_run)
    return {
        "message": message,
        "delivery": result,
    }


def send_decision_update(
    *,
    case_summary: Dict[str, Any],
    decision: str,
    notes: str,
    source_label: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    message = build_decision_update_message(
        case_summary=case_summary,
        decision=decision,
        notes=notes,
        source_label=source_label,
    )
    result = deliver_message(message, dry_run=dry_run)
    return {
        "message": message,
        "delivery": result,
    }


def send_qna_update(
    *,
    question: str,
    answer: str,
    source_label: str,
    used_ai: bool,
    dry_run: bool = False,
) -> Dict[str, Any]:
    message = build_qna_message(
        question=question,
        answer=answer,
        source_label=source_label,
        used_ai=used_ai,
    )
    result = deliver_message(message, dry_run=dry_run)
    return {
        "message": message,
        "delivery": result,
    }
