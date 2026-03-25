"""
Business-friendly message formatting for Discord and OpenClaw delivery.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

from .. import config
from ..ai_assistant import rule_based_recommendations, rule_based_reminders
from .contracts import ChatOpsMessage


SEVERITY_LABELS = {
    "critical": "Critical",
    "warning": "Warning",
    "info": "Info",
}

SEVERITY_COLORS = {
    "critical": 0xC23B2A,
    "warning": 0xD18A1B,
    "info": 0x3A8E3A,
}


def _truncate(text: str, max_length: int) -> str:
    return text if len(text) <= max_length else f"{text[: max_length - 3]}..."


def _clean_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.2f}"
        return f"{value:.3f}"
    return str(value)


def format_table_block(df: pd.DataFrame, columns: list[str], max_rows: int = 5) -> Optional[str]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    available_columns = [column for column in columns if column in df.columns]
    if not available_columns:
        return None
    preview = df.head(max_rows)[available_columns].copy()
    for column in preview.columns:
        preview[column] = preview[column].map(_clean_value)
    table_text = preview.to_string(index=False)
    return _truncate(table_text, 900)


def _recommendation_highlights(bundle: Dict[str, Any], limit: int = 3) -> list[str]:
    highlights = [item for item in rule_based_recommendations(bundle) if item]
    return highlights[:limit]


def _build_case_candidates(bundle: Dict[str, Any]) -> list[Dict[str, Any]]:
    transactions = bundle.get("transactions", pd.DataFrame())
    accounts = bundle.get("accounts", pd.DataFrame())
    merchants = bundle.get("merchants", pd.DataFrame())
    review_log = bundle.get("review_log", pd.DataFrame())
    candidates: list[Dict[str, Any]] = []

    if isinstance(transactions, pd.DataFrame) and not transactions.empty and "transactionid" in transactions.columns:
        pending_ids = set()
        reviewed_ids = set()
        if isinstance(review_log, pd.DataFrame) and not review_log.empty and "transactionid" in review_log.columns:
            reviewed_ids = set(review_log["transactionid"].astype(str))
            if "decision" in review_log.columns:
                pending_ids.update(
                    review_log.loc[review_log["decision"] == "Needs Review", "transactionid"].astype(str).tolist()
                )
        pending_ids.update(set(transactions["transactionid"].astype(str)) - reviewed_ids)
        pending_transactions = transactions.loc[transactions["transactionid"].astype(str).isin(pending_ids)].copy()
        if "composite_risk_score" in pending_transactions.columns:
            pending_transactions = pending_transactions.sort_values("composite_risk_score", ascending=False)
        for row in pending_transactions.head(6).to_dict(orient="records"):
            candidates.append({"case_type": "transaction", "case": row})

    if isinstance(accounts, pd.DataFrame) and not accounts.empty:
        for row in accounts.head(4).to_dict(orient="records"):
            candidates.append({"case_type": "account", "case": row})

    if isinstance(merchants, pd.DataFrame) and not merchants.empty:
        for row in merchants.head(4).to_dict(orient="records"):
            candidates.append({"case_type": "merchant", "case": row})

    return candidates


def build_report_message(
    bundle: Dict[str, Any],
    *,
    headline: Optional[str] = None,
    recommendations: Optional[list[str]] = None,
) -> ChatOpsMessage:
    summary = bundle.get("summary", {}) or {}
    transactions = bundle.get("transactions", pd.DataFrame())
    accounts = bundle.get("accounts", pd.DataFrame())
    merchants = bundle.get("merchants", pd.DataFrame())
    anomaly_scores = bundle.get("anomaly_scores", pd.DataFrame())
    source_label = bundle.get("source_label", "Fraud dashboard")

    top_transactions = format_table_block(
        transactions,
        ["transactionid", "accountid", "merchantid", "composite_risk_score", "risk_level"],
        max_rows=config.OPENCLAW_REPORT_TOP_N,
    )
    anomaly_table = format_table_block(
        anomaly_scores,
        ["transactionid", "ensemble_anomaly_score", "is_anomalous"],
        max_rows=config.OPENCLAW_REPORT_TOP_N,
    )

    facts = [
        f"Transactions: {summary.get('total_transactions', len(transactions)):,}",
        f"Flagged: {summary.get('flagged_transactions', 0):,}",
        f"High risk: {summary.get('high_risk_count', 0):,}",
        f"Higher-risk accounts: {summary.get('high_risk_accounts', 0):,}",
        f"Higher-risk merchants: {summary.get('high_risk_merchants', 0):,}",
        f"Pending review: {summary.get('pending_review_total', 0):,}",
    ]

    top_account = accounts.iloc[0]["accountid"] if isinstance(accounts, pd.DataFrame) and not accounts.empty and "accountid" in accounts else "N/A"
    top_merchant = merchants.iloc[0]["merchantid"] if isinstance(merchants, pd.DataFrame) and not merchants.empty and "merchantid" in merchants else "N/A"

    text = (
        f"{headline or 'Fraud analysis report ready.'} "
        f"Current focus should start with account {top_account}, merchant {top_merchant}, and the highest-ranked flagged transactions."
    )
    if anomaly_table:
        text += " Top anomaly scores are included below for fast triage."

    table_text = top_transactions or anomaly_table
    table_title = "Top suspicious transactions" if top_transactions else "Top anomaly scores"
    next_action = "Review the top flagged transactions first, then work through the exposed accounts, merchants, and pending analyst queue."
    recommendation_list = recommendations or _recommendation_highlights(bundle)

    return ChatOpsMessage(
        message_type="fraud.report",
        title=f"Fraud Analysis Report • {source_label}",
        text=text,
        severity="info" if summary.get("high_risk_count", 0) < 1 else "warning",
        facts=facts,
        highlights=recommendation_list,
        table_title=table_title,
        table_text=table_text,
        next_action=next_action,
        source_label=source_label,
        metadata={"source_label": source_label},
    )


def build_alert_message(alert: Dict[str, Any], *, source_label: str) -> ChatOpsMessage:
    severity = str(alert.get("severity", "warning")).lower()
    facts = [str(item) for item in alert.get("evidence", []) if str(item).strip()]
    if alert.get("entity_id"):
        facts.insert(0, f"{alert.get('entity_type', 'Entity').title()} ID: {alert['entity_id']}")
    if alert.get("risk_score") is not None:
        facts.append(f"Risk score: {float(alert['risk_score']):.3f}")

    return ChatOpsMessage(
        message_type=alert.get("alert_type", "fraud.alert"),
        title=alert.get("title", "Fraud alert"),
        text=alert.get("reason", ""),
        severity=severity,
        facts=facts,
        next_action=alert.get("next_action"),
        source_label=source_label,
        metadata={
            "alert_id": alert.get("alert_id"),
            "entity_type": alert.get("entity_type"),
            "entity_id": alert.get("entity_id"),
        },
    )


def build_reminder_message(bundle: Dict[str, Any]) -> ChatOpsMessage:
    summary = bundle.get("summary", {}) or {}
    reminders = rule_based_reminders(bundle)
    reminder_facts = list(reminders[:4]) or [
        f"Flagged transactions currently in view: {summary.get('flagged_transactions', 0):,}",
        f"Pending review cases: {summary.get('pending_review_total', 0):,}",
    ]
    return ChatOpsMessage(
        message_type="fraud.reminder",
        title="Fraud Monitoring Reminder",
        text="OOF monitoring reminder generated from the latest published fraud context.",
        severity="warning" if summary.get("pending_review_total", 0) else "info",
        facts=reminder_facts,
        next_action="Clear the oldest pending high-risk reviews first and confirm controls on the highest-risk channel and merchant clusters.",
        source_label=bundle.get("source_label", "Published fraud context"),
        metadata={"generated_at": datetime.now(timezone.utc).isoformat()},
    )


def build_case_reminder_message(bundle: Dict[str, Any], *, reminder_index: int = 0) -> ChatOpsMessage:
    candidates = _build_case_candidates(bundle)
    if not candidates:
        return build_reminder_message(bundle)

    selected = candidates[reminder_index % len(candidates)]
    case_type = selected["case_type"]
    case = selected["case"]
    source_label = bundle.get("source_label", "Published fraud context")

    if case_type == "transaction":
        transaction_id = str(case.get("transactionid", "N/A"))
        return ChatOpsMessage(
            message_type="fraud.reminder.case_transaction",
            title=f"Case Reminder • {transaction_id}",
            text=(
                f"Follow up on transaction {transaction_id}. It remains one of the highest-priority items in the active fraud queue."
            ),
            severity="critical" if str(case.get("risk_level", "")).lower() == "high" else "warning",
            facts=[
                f"Account {case.get('accountid', 'N/A')}",
                f"Merchant {case.get('merchantid', 'N/A')}",
                f"Location {case.get('location', 'N/A')}",
                f"Channel {case.get('channel', 'N/A')}",
                f"Amount ${float(case.get('transactionamount', 0) or 0):,.2f}",
                f"Composite risk {float(case.get('composite_risk_score', 0) or 0):.3f}",
            ],
            highlights=[
                "Validate whether the alert has already been reviewed or still needs analyst action.",
                "Check linked device/IP reuse and merchant clustering before closing the case.",
            ],
            next_action="Open the transaction case first, confirm the review decision, and document the rationale in the analyst log.",
            source_label=source_label,
            metadata={"case_type": case_type, "case_id": transaction_id},
        )

    if case_type == "account":
        account_id = str(case.get("accountid", "N/A"))
        return ChatOpsMessage(
            message_type="fraud.reminder.case_account",
            title=f"Account Reminder • {account_id}",
            text=f"Account {account_id} is still a top fraud-monitoring priority in the active portfolio.",
            severity="warning",
            facts=[
                f"Account risk {float(case.get('account_risk_score', 0) or 0):.3f}",
                f"Transactions {int(case.get('transaction_count', 0) or 0)}",
                f"High-risk transactions {int(case.get('high_risk_transaction_count', 0) or 0)}",
                f"High-risk share {float(case.get('high_risk_transaction_pct', 0) or 0):.1f}%",
            ],
            highlights=[
                "Review the highest-scoring transactions tied to this account before moving to lower-ranked cases.",
                "Check whether the merchant and channel pattern fits expected account behavior.",
            ],
            next_action="Review the account exposure trail and confirm whether additional transaction-level escalation is required.",
            source_label=source_label,
            metadata={"case_type": case_type, "case_id": account_id},
        )

    merchant_id = str(case.get("merchantid", "N/A"))
    return ChatOpsMessage(
        message_type="fraud.reminder.case_merchant",
        title=f"Merchant Reminder • {merchant_id}",
        text=f"Merchant {merchant_id} continues to show elevated suspicious activity in the current fraud view.",
        severity="warning",
        facts=[
            f"Average risk {float(case.get('avg_risk_score', 0) or 0):.3f}",
            f"Max risk {float(case.get('max_risk_score', 0) or 0):.3f}",
            f"Transactions {int(case.get('transaction_count', 0) or 0)}",
            f"High-risk count {int(case.get('high_risk_count', 0) or 0)}",
        ],
        highlights=[
            "Check whether recent flagged transactions cluster around one account segment or channel.",
            "Confirm whether merchant-level controls need to be tightened before the next review cycle.",
        ],
        next_action="Inspect the latest flagged transactions tied to this merchant and decide whether merchant monitoring needs escalation.",
        source_label=source_label,
        metadata={"case_type": case_type, "case_id": merchant_id},
    )


def build_decision_update_message(
    *,
    case_summary: Dict[str, Any],
    decision: str,
    notes: str,
    source_label: str,
) -> ChatOpsMessage:
    transaction_id = str(case_summary.get("transactionid", "N/A"))
    return ChatOpsMessage(
        message_type="fraud.review.update",
        title=f"Analyst Review Update • {transaction_id}",
        text=f"An analyst recorded `{decision}` for transaction {transaction_id} from the Streamlit dashboard.",
        severity="info",
        facts=[
            f"Account {case_summary.get('accountid', 'N/A')}",
            f"Merchant {case_summary.get('merchantid', 'N/A')}",
            f"Location {case_summary.get('location', 'N/A')}",
            f"Channel {case_summary.get('channel', 'N/A')}",
            f"Composite risk {float(case_summary.get('composite_risk_score', 0) or 0):.3f}",
        ],
        highlights=[notes] if notes.strip() else [],
        next_action="Use the updated review status when triaging the remaining queue and when answering Discord analyst questions.",
        source_label=source_label,
        metadata={"decision": decision, "case_id": transaction_id},
    )


def build_qna_message(question: str, answer: str, *, source_label: str, used_ai: bool) -> ChatOpsMessage:
    return ChatOpsMessage(
        message_type="fraud.answer",
        title="Fraud Analyst Answer",
        text=_truncate(answer, 1500),
        severity="info",
        facts=[
            f"Question: {question}",
            f"Source: {source_label}",
            f"AI used: {'yes' if used_ai else 'fallback'}",
        ],
        source_label=source_label,
    )


def build_discord_embed(message: ChatOpsMessage) -> Dict[str, Any]:
    facts = [f"• {fact}" for fact in message.facts if fact][:6]
    highlights = [f"• {item}" for item in message.highlights if item][:4]
    embed: Dict[str, Any] = {
        "title": _truncate(message.title, 256),
        "description": _truncate(message.text, 1800),
        "color": SEVERITY_COLORS.get(message.severity, SEVERITY_COLORS["info"]),
        "fields": [],
        "footer": {"text": _truncate(message.source_label or "Fraud ChatOps", 2048)},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if facts:
        embed["fields"].append({
            "name": "Evidence",
            "value": _truncate("\n".join(facts), 1024),
            "inline": False,
        })
    if highlights:
        embed["fields"].append({
            "name": "Recommendations",
            "value": _truncate("\n".join(highlights), 1024),
            "inline": False,
        })
    if message.table_text:
        embed["fields"].append({
            "name": _truncate(message.table_title or "Top items", 256),
            "value": f"```{_truncate(message.table_text, 980)}```",
            "inline": False,
        })
    if message.next_action:
        embed["fields"].append({
            "name": "Recommended Next Action",
            "value": _truncate(message.next_action, 1024),
            "inline": False,
        })
    return embed


def build_openclaw_payload(message: ChatOpsMessage) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "type": message.message_type,
        "title": message.title,
        "text": message.text,
        "severity": SEVERITY_LABELS.get(message.severity, message.severity.title()),
        "summary": {
            "sourceLabel": message.source_label,
            "facts": message.facts,
            "highlights": message.highlights,
            "nextAction": message.next_action,
        },
        "metadata": message.metadata or {},
    }
    if message.table_text:
        payload["table"] = {
            "title": message.table_title,
            "text": message.table_text,
        }
    return payload
