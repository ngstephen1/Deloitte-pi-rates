"""
Shared AI assistant helpers for the fraud dashboard.

These helpers keep OpenAI usage optional and grounded in the current
data bundle loaded in Streamlit or produced by the backend pipeline.
"""

from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

import pandas as pd

from . import config
from .utils import LOGGER


def has_openai_api_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def is_ai_enabled() -> bool:
    """Return True when AI features are enabled in config and an API key exists."""
    return bool(config.ENABLE_AI_FEATURES and has_openai_api_key())


def ai_availability_message() -> str:
    if not config.ENABLE_AI_FEATURES:
        return "AI features are disabled in config.py."
    if not has_openai_api_key():
        return "Set OPENAI_API_KEY to enable live AI recommendations, Q&A, and case explanations."
    return f"AI features are available via {config.OPENAI_MODEL}."


def _build_response_payload(instructions: str, prompt: str, max_output_tokens: int) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": config.OPENAI_MODEL,
        "instructions": instructions,
        "input": prompt,
        "max_output_tokens": max_output_tokens,
        "store": False,
        "text": {"verbosity": "low"},
    }
    reasoning_effort = str(config.OPENAI_REASONING_EFFORT or "").strip().lower()
    if reasoning_effort and reasoning_effort not in {"none", "default"}:
        payload["reasoning"] = {"effort": config.OPENAI_REASONING_EFFORT}
    return payload


def _extract_output_text(response_payload: Dict[str, Any]) -> Optional[str]:
    if not isinstance(response_payload, dict):
        return None

    output_text = response_payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    fragments: List[str] = []
    for item in response_payload.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            text = content.get("text") or content.get("output_text")
            if isinstance(text, str) and text.strip():
                fragments.append(text.strip())

    return "\n".join(fragments).strip() or None


def _perform_http_request(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ssl_context = None
    try:
        import certifi

        ssl_context = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ssl_context = None

    request = urllib.request.Request(
        url=f"{config.OPENAI_API_BASE_URL.rstrip('/')}/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=config.OPENAI_REQUEST_TIMEOUT_SECONDS,
            context=ssl_context,
        ) as response:
            body = response.read().decode("utf-8")
        return json.loads(body)
    except urllib.error.HTTPError as exc:
        try:
            error_body = exc.read().decode("utf-8")
        except Exception:
            error_body = ""
        LOGGER.warning(f"AI HTTP request failed: status={exc.code}, body={error_body[:300]}")
    except Exception as exc:
        LOGGER.warning(f"AI HTTP transport failed: {exc}")
    return None


def _request_ai_response_http(instructions: str, prompt: str, max_output_tokens: int) -> Optional[str]:
    payload = _build_response_payload(instructions, prompt, max_output_tokens)
    response_payload = _perform_http_request(payload)
    if not response_payload:
        return None

    output_text = _extract_output_text(response_payload)
    if output_text:
        return output_text

    incomplete_reason = (
        response_payload.get("incomplete_details", {}) or {}
    ).get("reason")
    if response_payload.get("status") == "incomplete" and incomplete_reason == "max_output_tokens":
        retry_tokens = min(max(max_output_tokens * 2, 300), 1200)
        retry_payload = _build_response_payload(instructions, prompt, retry_tokens)
        retry_response_payload = _perform_http_request(retry_payload)
        if retry_response_payload:
            retry_output = _extract_output_text(retry_response_payload)
            if retry_output:
                return retry_output
            LOGGER.debug(
                "AI HTTP request remained incomplete after retry: "
                f"reason={(retry_response_payload.get('incomplete_details', {}) or {}).get('reason')}"
            )
        return None

    if response_payload.get("status") == "completed":
        LOGGER.debug("AI HTTP request completed without extractable text output.")
    return None


def _request_ai_response_sdk(instructions: str, prompt: str, max_output_tokens: int) -> Optional[str]:
    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.responses.create(
            model=config.OPENAI_MODEL,
            instructions=instructions,
            input=prompt,
            max_output_tokens=max_output_tokens,
            store=False,
            reasoning={"effort": config.OPENAI_REASONING_EFFORT} if config.OPENAI_REASONING_EFFORT else None,
        )
        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text.strip()
        return _extract_output_text(response.model_dump())
    except Exception as exc:
        LOGGER.warning(f"AI SDK transport failed: {exc}")
    return None


def request_ai_response(
    instructions: str,
    prompt: str,
    *,
    max_output_tokens: int = config.AI_MAX_OUTPUT_TOKENS,
) -> Optional[str]:
    """Call the OpenAI Responses API and return output text when available."""
    if not is_ai_enabled():
        return None

    transport_preference = str(config.OPENAI_TRANSPORT_PREFERENCE).lower()
    if transport_preference == "sdk":
        transports = ["sdk", "http"]
    elif transport_preference in {"auto", "both"}:
        transports = ["http", "sdk"]
    else:
        transports = ["http"]

    for transport in transports:
        if transport == "http":
            text = _request_ai_response_http(instructions, prompt, max_output_tokens)
        else:
            text = _request_ai_response_sdk(instructions, prompt, max_output_tokens)
        if text:
            return text
    return None


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


def _top_entity_rows(df: pd.DataFrame, columns: List[str], max_rows: int) -> Optional[str]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    view = df.head(max_rows)[[column for column in columns if column in df.columns]]
    if view.empty:
        return None
    return view.to_csv(index=False)


def _flagged_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty or "risk_level" not in transactions.columns:
        return transactions
    flagged = transactions[transactions["risk_level"].isin(["High", "Medium"])].copy()
    return flagged if not flagged.empty else transactions.copy()


def bundle_context_summary(
    bundle: Dict[str, Any],
    max_rows: int = config.AI_MAX_CONTEXT_ROWS,
    detail: str = "compact",
) -> str:
    """
    Build a compact, grounded context block from the active data bundle.
    """
    summary = bundle.get("summary", {}) or {}
    transactions = bundle.get("transactions", pd.DataFrame())
    accounts = bundle.get("accounts", pd.DataFrame())
    merchants = bundle.get("merchants", pd.DataFrame())
    devices = bundle.get("devices", pd.DataFrame())
    locations = bundle.get("locations", pd.DataFrame())
    flagged = _flagged_transactions(transactions)

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

    include_full_tables = detail == "full"
    include_flagged_table = detail in {"compact", "full"}

    if not transactions.empty:
        risk_mix = transactions["risk_level"].value_counts().to_dict() if "risk_level" in transactions else {}
        context_blocks.append(f"Risk mix: {risk_mix}")

        transaction_columns = [
            column for column in [
                "transactionid",
                "accountid",
                "merchantid",
                "location",
                "channel",
                "transactionamount",
                "composite_risk_score",
                "risk_level",
            ] if column in transactions.columns
        ]
        top_flagged = flagged.head(max_rows)[transaction_columns]
        if include_flagged_table and not top_flagged.empty:
            context_blocks.append(f"Flagged transactions:\n{top_flagged.to_csv(index=False)}")

        channel_table = summarize_channel_risk(transactions)
        if not channel_table.empty:
            limit = 1 if detail == "minimal" else min(max_rows, 3)
            context_blocks.append(f"Channel risk:\n{channel_table.head(limit).to_csv(index=False)}")

        feature_signals = compute_feature_signal_summary(transactions)
        if feature_signals:
            context_blocks.append(f"Signal summary: {feature_signals}")

        if include_full_tables:
            top_transactions = transactions.head(max_rows)[transaction_columns]
            if not top_transactions.empty:
                context_blocks.append(f"Top transactions:\n{top_transactions.to_csv(index=False)}")

    for label, df, columns in [
        ("Top accounts", accounts, ["accountid", "account_risk_score", "transaction_count", "high_risk_transaction_count"]),
        ("Top merchants", merchants, ["merchantid", "avg_risk_score", "max_risk_score", "transaction_count", "high_risk_count"]),
        ("Top devices", devices, ["deviceid", "avg_risk_score", "max_risk_score", "transaction_count", "high_risk_count"]),
        ("Top locations", locations, ["location", "avg_risk_score", "max_risk_score", "transaction_count", "high_risk_count"]),
    ]:
        entity_limit = 1 if detail == "minimal" else min(max_rows, 3)
        table = _top_entity_rows(df, columns, entity_limit)
        if table:
            context_blocks.append(f"{label}:\n{table}")

    return "\n\n".join(context_blocks)


def rule_based_recommendations(bundle: Dict[str, Any]) -> List[str]:
    """Deterministic recommendations grounded in current results."""
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
        recommendations.append(
            f"Prioritize merchant {top_merchant.get('merchantid', 'N/A')}; it leads the merchant risk ranking in the current portfolio."
        )

    if isinstance(locations, pd.DataFrame) and not locations.empty:
        top_location = locations.iloc[0]
        recommendations.append(
            f"Increase monitoring on {top_location.get('location', 'N/A')}; it is the most risk-concentrated location in the active view."
        )

    if isinstance(transactions, pd.DataFrame) and not transactions.empty:
        channel_table = summarize_channel_risk(transactions)
        if not channel_table.empty:
            top_channel = channel_table.iloc[0]
            recommendations.append(
                f"Apply tighter controls to the {top_channel['channel']} channel; it has the highest average transaction risk."
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
            reminders.append(f"{channel_table.iloc[0]['channel']} transactions currently show the highest average risk score.")

        feature_signals = compute_feature_signal_summary(transactions)
        if feature_signals.get("device_change_flagged_mean", 0) > feature_signals.get("device_change_flag_overall_mean", 0):
            reminders.append("Device-change anomalies are elevated in the flagged cohort.")
        if feature_signals.get("login_attempt_risk_flagged_mean", 0) > feature_signals.get("login_attempt_risk_overall_mean", 0):
            reminders.append("Repeated login attempts remain a leading risk signal in the active portfolio.")

    return reminders[:4]


def _strongest_risk_components(case_row: pd.Series) -> str:
    component_map = {
        "Isolation Forest": float(case_row.get("isolation_forest_score", 0) or 0),
        "Local Outlier Factor": float(case_row.get("lof_score", 0) or 0),
        "K-Means": float(case_row.get("kmeans_anomaly_score", 0) or 0),
        "Graph Risk": float(case_row.get("graph_risk_score", 0) or 0),
        "Amount Outlier": float(case_row.get("amount_outlier_risk", 0) or 0),
        "Login Attempts": float(case_row.get("login_attempt_risk", 0) or 0),
    }
    ranked = sorted(component_map.items(), key=lambda item: item[1], reverse=True)
    strongest = [f"{label} ({score:.3f})" for label, score in ranked if score > 0][:3]
    return ", ".join(strongest) if strongest else "composite portfolio risk aggregation"


def _find_matching_value(question: str, df: pd.DataFrame, column: str) -> Optional[str]:
    if df.empty or column not in df.columns:
        return None
    question_lower = question.lower()
    for value in df[column].dropna().astype(str).unique().tolist()[:1000]:
        if value.lower() in question_lower:
            return value
    return None


def generate_ai_recommendations(bundle: Dict[str, Any]) -> Dict[str, Any]:
    baseline = rule_based_recommendations(bundle)
    reminders = rule_based_reminders(bundle)
    response_text = request_ai_response(
        instructions=(
            "You are a senior enterprise fraud strategy advisor supporting the Office of Oversight and Finance. "
            "Use only the supplied evidence. Return 3 to 5 short bullets. Each bullet must include a priority, "
            "the observed pattern, and the control action to take next."
        ),
        prompt=(
            "Produce executive monitoring recommendations from this grounded dataset context.\n\n"
            f"{bundle_context_summary(bundle, detail='minimal')}\n\n"
            f"Deterministic observations already found: {baseline + reminders}"
        ),
        max_output_tokens=260,
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
        f"Dataset context:\n{bundle_context_summary(bundle, detail='minimal')}\n\n"
        "Explain what was flagged, which evidence supports the concern, and what the analyst should review next."
    )
    ai_text = request_ai_response(
        instructions=(
            "You are a senior fraud investigator. Use only the supplied evidence. "
            "Answer in three compact parts: finding, evidence, next review step. "
            "If evidence is limited, say so explicitly."
        ),
        prompt=prompt,
        max_output_tokens=220,
    )
    return {
        "baseline": baseline,
        "ai_text": ai_text,
        "ai_available": is_ai_enabled(),
    }


def _is_help_intent(question: str) -> bool:
    question_clean = " ".join(question.lower().strip().split())
    help_phrases = {
        "hi",
        "hello",
        "hey",
        "help",
        "what can you help me with",
        "what can you do",
        "how can you help",
        "who are you",
    }
    if question_clean in help_phrases:
        return True
    return any(
        phrase in question_clean
        for phrase in [
            "what can you help me with",
            "what can you do",
            "how can you help",
            "how do i use this",
            "what should i ask",
        ]
    )


def _help_intent_answer(bundle: Dict[str, Any]) -> str:
    summary = bundle.get("summary", {}) or {}
    return (
        "I can help you investigate the currently loaded fraud dataset. "
        f"Right now the active view contains {summary.get('flagged_transactions', 0)} flagged transactions and "
        f"{summary.get('high_risk_accounts', 0)} higher-risk accounts.\n\n"
        "You can ask things like:\n"
        "- What are the riskiest merchants?\n"
        "- Which location has the most high-risk activity?\n"
        "- Why was transaction TX000275 flagged?\n"
        "- Are login attempts or device changes more associated with risk?\n"
        "- Give me 3 recommendations for OOF based on this dataset."
    )


def answer_data_question(question: str, bundle: Dict[str, Any]) -> Dict[str, Any]:
    if _is_help_intent(question):
        help_answer = _help_intent_answer(bundle)
        return {
            "heuristic_answer": help_answer,
            "ai_answer": None,
            "ai_available": is_ai_enabled(),
            "availability_message": ai_availability_message(),
        }

    heuristic_answer = heuristic_question_answer(question, bundle)
    ai_text = request_ai_response(
        instructions=(
            "You are an enterprise fraud analytics assistant. Answer using only the supplied dataset context. "
            "Be precise, business-facing, and robust on edge cases. "
            "Structure the answer as: direct answer, supporting evidence, recommended next step. "
            "If the data does not support a conclusion, say so plainly."
        ),
        prompt=(
            f"Question: {question}\n\n"
            f"Dataset context:\n{bundle_context_summary(bundle, detail='compact')}\n\n"
            f"Deterministic grounded answer draft: {heuristic_answer}"
        ),
        max_output_tokens=320,
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
    devices = bundle.get("devices", pd.DataFrame())
    summary = bundle.get("summary", {}) or {}
    flagged = _flagged_transactions(transactions)

    transaction_id = _find_matching_value(question, transactions, "transactionid")
    if transaction_id:
        tx = transactions.loc[transactions["transactionid"].astype(str) == str(transaction_id)].iloc[0]
        return (
            f"Transaction {transaction_id} is elevated at {float(tx.get('composite_risk_score', 0) or 0):.3f}. "
            f"The strongest drivers are { _strongest_risk_components(tx) }. "
            f"Review account {tx.get('accountid', 'N/A')}, merchant {tx.get('merchantid', 'N/A')}, and channel {tx.get('channel', 'N/A')} next."
        )

    if ("top" in question_lower or "riskiest" in question_lower) and "transaction" in question_lower and not transactions.empty:
        top_transactions = transactions.head(5)[[column for column in ["transactionid", "accountid", "merchantid", "composite_risk_score"] if column in transactions.columns]]
        return "Top suspicious transactions: " + ", ".join(
            f"{row.transactionid} ({row.composite_risk_score:.3f})"
            for row in top_transactions.itertuples()
        )

    if ("top 5 suspicious account" in question_lower or "riskiest account" in question_lower or "top account" in question_lower) and isinstance(accounts, pd.DataFrame) and not accounts.empty:
        top_accounts = accounts.head(5)[["accountid", "account_risk_score"]]
        return "Top accounts by current risk score: " + ", ".join(
            f"{row.accountid} ({row.account_risk_score:.3f})" for row in top_accounts.itertuples()
        )

    if "merchant" in question_lower and isinstance(merchants, pd.DataFrame) and not merchants.empty:
        if "most often" in question_lower or "flagged" in question_lower:
            merchant_counts = (
                flagged.groupby("merchantid")
                .agg(flagged_count=("transactionid", "count"), avg_risk_score=("composite_risk_score", "mean"))
                .reset_index()
                .sort_values(["flagged_count", "avg_risk_score"], ascending=False)
            )
            if not merchant_counts.empty:
                top_merchants = merchant_counts.head(5)
                return "Merchants appearing most often in flagged transactions: " + ", ".join(
                    f"{row.merchantid} ({int(row.flagged_count)} flagged, avg risk {row.avg_risk_score:.3f})"
                    for row in top_merchants.itertuples()
                )
        top_merchants = merchants.head(5)[["merchantid", "avg_risk_score"]]
        return "Top merchants by average risk: " + ", ".join(
            f"{row.merchantid} ({row.avg_risk_score:.3f})" for row in top_merchants.itertuples()
        )

    if "location" in question_lower and isinstance(locations, pd.DataFrame) and not locations.empty:
        if not flagged.empty and "location" in flagged.columns:
            location_counts = (
                flagged.groupby("location")
                .agg(flagged_count=("transactionid", "count"), avg_risk_score=("composite_risk_score", "mean"))
                .reset_index()
                .sort_values(["flagged_count", "avg_risk_score"], ascending=False)
            )
            if not location_counts.empty:
                top_location = location_counts.iloc[0]
                return (
                    f"{top_location['location']} has the most flagged activity with {int(top_location['flagged_count'])} flagged transactions "
                    f"and average flagged risk {float(top_location['avg_risk_score']):.3f}."
                )

    if "device" in question_lower and isinstance(devices, pd.DataFrame) and not devices.empty:
        top_device = devices.iloc[0]
        return (
            f"Device {top_device.get('deviceid', 'N/A')} currently leads the device-risk ranking with "
            f"max risk {float(top_device.get('max_risk_score', 0) or 0):.3f}."
        )

    if "channel" in question_lower and not transactions.empty:
        channel_table = summarize_channel_risk(transactions)
        if not channel_table.empty:
            top_channel = channel_table.iloc[0]
            return (
                f"{top_channel['channel']} is the highest-risk channel right now with average risk "
                f"{float(top_channel['avg_risk_score']):.3f} across {int(top_channel['transaction_count'])} transactions."
            )

    if "device changes" in question_lower or "device change" in question_lower or "login attempts" in question_lower:
        signals = compute_feature_signal_summary(transactions)
        if signals:
            login_gap = signals.get("login_attempt_risk_flagged_mean", 0) - signals.get("login_attempt_risk_overall_mean", 0)
            device_gap = signals.get("device_change_flagged_mean", 0) - signals.get("device_change_flag_overall_mean", 0)
            if login_gap >= device_gap:
                return (
                    "Repeated login attempts show the stronger association with flagged risk in the active dataset. "
                    f"Flagged mean login-attempt risk exceeds the portfolio baseline by {login_gap:.3f}."
                )
            return (
                "Device-change behavior shows the stronger association with flagged risk in the active dataset. "
                f"Flagged mean device-change signal exceeds the portfolio baseline by {device_gap:.3f}."
            )

    if "recommendation" in question_lower or "oof" in question_lower or "control" in question_lower:
        recommendations = rule_based_recommendations(bundle)
        if recommendations:
            return "Top recommendations: " + " | ".join(recommendations[:3])

    if "summarize" in question_lower or "pattern" in question_lower or "overview" in question_lower:
        channel_table = summarize_channel_risk(transactions)
        top_channel = channel_table.iloc[0]["channel"] if not channel_table.empty else "the leading channel"
        return (
            f"The active dataset contains {summary.get('flagged_transactions', 0)} flagged transactions, "
            f"{summary.get('high_risk_accounts', 0)} higher-risk accounts, and concentrated exposure in {top_channel}. "
            "The strongest recurring warning signs are the highest-risk entities and elevated access-pattern anomalies."
        )

    if len(question_lower.strip()) <= 3:
        return _help_intent_answer(bundle)

    return (
        f"The strongest investigation starting points are {summary.get('flagged_transactions', 0)} flagged transactions, "
        f"{summary.get('high_risk_accounts', 0)} higher-risk accounts, and the top-ranked merchants, locations, and channels."
    )


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
