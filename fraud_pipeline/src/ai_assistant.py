"""
Shared AI assistant helpers for the fraud dashboard.

These helpers keep OpenAI usage optional and grounded in the current
data bundle loaded in Streamlit or produced by the backend pipeline.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pandas as pd

from . import config
from .utils import LOGGER


def is_ai_enabled() -> bool:
    """Return True when AI features are enabled in config and an API key exists."""
    return bool(config.ENABLE_AI_FEATURES and has_openai_api_key())


def has_openai_api_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def ai_availability_message() -> str:
    if not config.ENABLE_AI_FEATURES:
        return "AI features are disabled in config.py."
    if not has_openai_api_key():
        return "Set OPENAI_API_KEY to enable live AI recommendations, Q&A, and case explanations."
    return "AI features are available."


def _get_client():
    from openai import OpenAI

    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def request_ai_response(
    instructions: str,
    prompt: str,
    *,
    max_output_tokens: int = config.AI_MAX_OUTPUT_TOKENS,
) -> Optional[str]:
    """Call the OpenAI Responses API and return output text when available."""
    if not is_ai_enabled():
        return None

    try:
        client = _get_client()
        response = client.responses.create(
            model=config.OPENAI_MODEL,
            instructions=instructions,
            input=prompt,
            max_output_tokens=max_output_tokens,
            store=False,
        )
        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text.strip()
    except Exception as exc:
        LOGGER.warning(f"AI request failed: {exc}")
    return None


def bundle_context_summary(bundle: Dict[str, Any], max_rows: int = config.AI_MAX_CONTEXT_ROWS) -> str:
    """
    Build a compact, grounded context block from the active data bundle.
    """
    summary = bundle.get("summary", {}) or {}
    transactions = bundle.get("transactions", pd.DataFrame())
    accounts = bundle.get("accounts", pd.DataFrame())
    merchants = bundle.get("merchants", pd.DataFrame())
    devices = bundle.get("devices", pd.DataFrame())
    locations = bundle.get("locations", pd.DataFrame())

    context_blocks = [
        f"Source: {bundle.get('source_label', 'Pipeline outputs')}",
        (
            "Summary: "
            f"transactions={summary.get('total_transactions', len(transactions))}, "
            f"flagged={summary.get('flagged_transactions', 0)}, "
            f"high_risk={summary.get('high_risk_count', 0)}, "
            f"high_risk_accounts={summary.get('high_risk_accounts', 0)}, "
            f"high_risk_merchants={summary.get('high_risk_merchants', 0)}, "
            f"high_risk_devices={summary.get('high_risk_devices', 0)}, "
            f"high_risk_locations={summary.get('high_risk_locations', 0)}, "
            f"volume={summary.get('total_transaction_volume', 0):,.2f}"
        ),
    ]

    if not transactions.empty:
        risk_mix = transactions["risk_level"].value_counts().to_dict() if "risk_level" in transactions else {}
        context_blocks.append(f"Risk mix: {risk_mix}")

        top_transactions = transactions.head(max_rows)[
            [
                column
                for column in [
                    "transactionid",
                    "accountid",
                    "merchantid",
                    "location",
                    "channel",
                    "transactionamount",
                    "composite_risk_score",
                    "risk_level",
                ]
                if column in transactions.columns
            ]
        ]
        if not top_transactions.empty:
            context_blocks.append(f"Top transactions:\n{top_transactions.to_csv(index=False)}")

        channel_table = summarize_channel_risk(transactions)
        if not channel_table.empty:
            context_blocks.append(f"Channel risk:\n{channel_table.head(max_rows).to_csv(index=False)}")

        feature_signals = compute_feature_signal_summary(transactions)
        if feature_signals:
            context_blocks.append(f"Signal summary: {feature_signals}")

    for label, df, columns in [
        ("Top accounts", accounts, ["accountid", "account_risk_score", "transaction_count", "high_risk_transaction_count"]),
        ("Top merchants", merchants, ["merchantid", "avg_risk_score", "max_risk_score", "transaction_count", "high_risk_count"]),
        ("Top devices", devices, ["deviceid", "avg_risk_score", "max_risk_score", "transaction_count", "high_risk_count"]),
        ("Top locations", locations, ["location", "avg_risk_score", "max_risk_score", "transaction_count", "high_risk_count"]),
    ]:
        if isinstance(df, pd.DataFrame) and not df.empty:
            view = df.head(max_rows)[[column for column in columns if column in df.columns]]
            if not view.empty:
                context_blocks.append(f"{label}:\n{view.to_csv(index=False)}")

    return "\n\n".join(context_blocks)


def summarize_channel_risk(transactions: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty or "channel" not in transactions.columns:
        return pd.DataFrame()
    return (
        transactions.groupby("channel", dropna=False, observed=False)
        .agg(
            avg_risk_score=("composite_risk_score", "mean"),
            max_risk_score=("composite_risk_score", "max"),
            transaction_count=("transactionid", "count"),
            high_risk_count=("risk_level", lambda values: (values == "High").sum()),
        )
        .reset_index()
        .sort_values(["avg_risk_score", "high_risk_count"], ascending=False)
    )


def compute_feature_signal_summary(transactions: pd.DataFrame) -> Dict[str, float]:
    if transactions.empty or "composite_risk_score" not in transactions.columns:
        return {}

    summary: Dict[str, float] = {}
    flagged = transactions[transactions["risk_level"].isin(["High", "Medium"])] if "risk_level" in transactions else transactions
    if flagged.empty:
        flagged = transactions

    for column in ["login_attempt_risk", "device_change_flag", "ip_change_flag", "time_since_previous_transaction"]:
        if column in transactions.columns and pd.api.types.is_numeric_dtype(transactions[column]):
            summary[f"{column}_overall_mean"] = round(float(transactions[column].mean()), 3)
            summary[f"{column}_flagged_mean"] = round(float(flagged[column].mean()), 3)
    return summary


def rule_based_recommendations(bundle: Dict[str, Any]) -> List[str]:
    """
    Deterministic recommendations grounded in current results.
    """
    recommendations: List[str] = []
    transactions = bundle.get("transactions", pd.DataFrame())
    accounts = bundle.get("accounts", pd.DataFrame())
    merchants = bundle.get("merchants", pd.DataFrame())
    locations = bundle.get("locations", pd.DataFrame())

    if isinstance(accounts, pd.DataFrame) and not accounts.empty:
        top_accounts = accounts.head(10)["accountid"].astype(str).tolist()
        recommendations.append(
            f"Review these accounts first: {', '.join(top_accounts[:5])}"
            + ("..." if len(top_accounts) > 5 else "")
        )

    if isinstance(merchants, pd.DataFrame) and not merchants.empty:
        top_merchant = merchants.iloc[0]
        merchant_id = top_merchant.get("merchantid", "N/A")
        recommendations.append(
            f"Prioritize merchant {merchant_id}; it has the highest merchant-level risk exposure in the current dataset."
        )

    if isinstance(locations, pd.DataFrame) and not locations.empty:
        top_location = locations.iloc[0]
        location_name = top_location.get("location", "N/A")
        recommendations.append(
            f"Increase monitoring on {location_name}; it is the most risk-concentrated location in the active view."
        )

    if isinstance(transactions, pd.DataFrame) and not transactions.empty:
        channel_table = summarize_channel_risk(transactions)
        if not channel_table.empty:
            top_channel = channel_table.iloc[0]
            recommendations.append(
                f"Apply tighter controls to the {top_channel['channel']} channel; it shows the highest average transaction risk."
            )

        signal_summary = compute_feature_signal_summary(transactions)
        if signal_summary:
            if signal_summary.get("login_attempt_risk_flagged_mean", 0) > signal_summary.get("login_attempt_risk_overall_mean", 0):
                recommendations.append("Repeated login attempts are elevated in flagged transactions and should remain a frontline review signal.")
            if signal_summary.get("device_change_flagged_mean", 0) > signal_summary.get("device_change_flag_overall_mean", 0):
                recommendations.append("Device-change anomalies are materially higher in flagged activity; review shared-device clusters early.")
            if signal_summary.get("ip_change_flagged_mean", 0) > signal_summary.get("ip_change_flag_overall_mean", 0):
                recommendations.append("IP-change behavior is elevated in flagged activity; inspect location and access pattern shifts.")

    return recommendations[:5]


def rule_based_reminders(bundle: Dict[str, Any]) -> List[str]:
    reminders: List[str] = []
    summary = bundle.get("summary", {}) or {}
    transactions = bundle.get("transactions", pd.DataFrame())

    flagged_count = summary.get("flagged_transactions", 0)
    if flagged_count:
        reminders.append(f"Review the highest-ranked {min(10, flagged_count)} flagged transactions first.")

    if isinstance(transactions, pd.DataFrame) and not transactions.empty:
        channel_table = summarize_channel_risk(transactions)
        if not channel_table.empty:
            top_channel = channel_table.iloc[0]["channel"]
            reminders.append(f"{top_channel} transactions currently show the highest average risk score.")

        feature_signals = compute_feature_signal_summary(transactions)
        if feature_signals.get("device_change_flagged_mean", 0) > feature_signals.get("device_change_flag_overall_mean", 0):
            reminders.append("Device-change anomalies are elevated in the flagged cohort.")
        if feature_signals.get("login_attempt_risk_flagged_mean", 0) > feature_signals.get("login_attempt_risk_overall_mean", 0):
            reminders.append("Repeated login attempts remain a leading risk signal in the active portfolio.")

    return reminders[:4]


def generate_ai_recommendations(bundle: Dict[str, Any]) -> Dict[str, Any]:
    baseline = rule_based_recommendations(bundle)
    reminders = rule_based_reminders(bundle)
    response_text = request_ai_response(
        instructions=(
            "You are a fraud analytics advisor preparing concise executive recommendations "
            "for the Office of Oversight and Finance. Stay grounded in the supplied metrics. "
            "Return 3 to 5 short bullet points only."
        ),
        prompt=(
            "Use the following grounded dataset context to produce executive monitoring recommendations.\n\n"
            f"{bundle_context_summary(bundle)}\n\n"
            f"Existing deterministic observations: {baseline + reminders}"
        ),
        max_output_tokens=220,
    )
    ai_bullets = [line.strip("- ").strip() for line in (response_text or "").splitlines() if line.strip()]
    return {
        "ai_available": is_ai_enabled(),
        "availability_message": ai_availability_message(),
        "baseline_recommendations": baseline,
        "reminders": reminders,
        "ai_recommendations": ai_bullets,
    }


def explain_case(
    *,
    entity_type: str,
    case_summary: Dict[str, Any],
    bundle: Dict[str, Any],
) -> Dict[str, Any]:
    baseline = _baseline_case_explanation(entity_type, case_summary)
    prompt = (
        f"Entity type: {entity_type}\n"
        f"Case summary: {case_summary}\n\n"
        f"Dataset context:\n{bundle_context_summary(bundle)}\n\n"
        "Provide a short business-facing explanation covering: what was flagged, why it looks suspicious, and what to review next."
    )
    ai_text = request_ai_response(
        instructions=(
            "You are a fraud investigator. Answer in 3 short sentences maximum. "
            "Be specific, factual, and grounded in the data provided."
        ),
        prompt=prompt,
        max_output_tokens=180,
    )
    return {
        "baseline": baseline,
        "ai_text": ai_text,
        "ai_available": is_ai_enabled(),
    }


def answer_data_question(question: str, bundle: Dict[str, Any]) -> Dict[str, Any]:
    heuristic_answer = heuristic_question_answer(question, bundle)
    ai_text = request_ai_response(
        instructions=(
            "You are an executive fraud analytics assistant. Answer using only the supplied dataset context. "
            "Be concise, business-facing, and avoid unsupported claims."
        ),
        prompt=(
            f"Question: {question}\n\n"
            f"Dataset context:\n{bundle_context_summary(bundle)}\n\n"
            f"Heuristic answer draft: {heuristic_answer}"
        ),
        max_output_tokens=260,
    )
    return {
        "heuristic_answer": heuristic_answer,
        "ai_answer": ai_text,
        "ai_available": is_ai_enabled(),
        "availability_message": ai_availability_message(),
    }


def heuristic_question_answer(question: str, bundle: Dict[str, Any]) -> str:
    question_lower = question.lower()
    transactions = bundle.get("transactions", pd.DataFrame())
    accounts = bundle.get("accounts", pd.DataFrame())
    merchants = bundle.get("merchants", pd.DataFrame())
    locations = bundle.get("locations", pd.DataFrame())

    if "top 5 suspicious account" in question_lower or "riskiest account" in question_lower:
        if isinstance(accounts, pd.DataFrame) and not accounts.empty:
            top_accounts = accounts.head(5)[["accountid", "account_risk_score"]]
            return "Top accounts by current risk score: " + ", ".join(
                f"{row.accountid} ({row.account_risk_score:.3f})" for row in top_accounts.itertuples()
            )

    if "merchant" in question_lower and isinstance(merchants, pd.DataFrame) and not merchants.empty:
        top_merchants = merchants.head(5)[["merchantid", "avg_risk_score"]]
        return "Top merchants by average risk: " + ", ".join(
            f"{row.merchantid} ({row.avg_risk_score:.3f})" for row in top_merchants.itertuples()
        )

    if "location" in question_lower and isinstance(locations, pd.DataFrame) and not locations.empty:
        top_location = locations.iloc[0]
        return (
            f"{top_location['location']} currently has the highest location-level risk, "
            f"with avg risk {top_location['avg_risk_score']:.3f}."
        )

    if "device changes" in question_lower or "login attempts" in question_lower:
        signals = compute_feature_signal_summary(transactions)
        if signals:
            login_gap = signals.get("login_attempt_risk_flagged_mean", 0) - signals.get("login_attempt_risk_overall_mean", 0)
            device_gap = signals.get("device_change_flagged_mean", 0) - signals.get("device_change_flag_overall_mean", 0)
            if login_gap >= device_gap:
                return "Repeated login attempts show the stronger association with flagged risk in the active dataset."
            return "Device-change behavior shows the stronger association with flagged risk in the active dataset."

    if "summarize" in question_lower or "pattern" in question_lower:
        summary = bundle.get("summary", {}) or {}
        return (
            f"The active dataset contains {summary.get('flagged_transactions', 0)} flagged transactions, "
            f"{summary.get('high_risk_accounts', 0)} higher-risk accounts, and concentration in the highest-risk channel and entity clusters."
        )

    return "Use the ranked transactions, accounts, merchants, and locations in the active dataset as the primary investigation starting points."


def _baseline_case_explanation(entity_type: str, case_summary: Dict[str, Any]) -> str:
    if entity_type == "transaction":
        return (
            f"Transaction {case_summary.get('transactionid', 'N/A')} was flagged because it combines a "
            f"risk score of {case_summary.get('composite_risk_score', 0):.3f} with elevated anomaly or graph signals. "
            "Review account history, merchant behavior, and channel context next."
        )
    if entity_type == "account":
        return (
            f"Account {case_summary.get('accountid', 'N/A')} shows concentrated suspicious activity across "
            f"{case_summary.get('transaction_count', 0)} transactions. Review its highest-risk transactions first."
        )
    if entity_type == "merchant":
        return (
            f"Merchant {case_summary.get('merchantid', 'N/A')} is elevated because multiple transactions cluster at a high risk level. "
            "Review linked accounts, channels, and reused devices next."
        )
    if entity_type == "location":
        return (
            f"Location {case_summary.get('location', 'N/A')} stands out for concentrated high-risk activity. "
            "Review channel mix, connected merchants, and access-pattern changes next."
        )
    return "This case is elevated relative to the rest of the portfolio and should be reviewed against its linked entities and strongest risk signals."
