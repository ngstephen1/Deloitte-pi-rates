"""
Grounded analyst Q&A and workflow helpers for Discord/OpenClaw consumers.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .. import config
from ..ai_assistant import (
    answer_data_question,
    build_deep_case_explanation,
    generate_multi_agent_oof_brief,
)
from ..review_judge import judge_case_disposition
from .context_loader import build_review_summary, load_active_bundle


def _pending_review_answer(bundle: Dict[str, Any]) -> str:
    transactions = bundle.get("transactions", pd.DataFrame())
    review_log = bundle.get("review_log", pd.DataFrame())
    if transactions.empty or "transactionid" not in transactions.columns:
        return "No active transaction-level dataset is currently available for review-queue analysis."

    review_summary = build_review_summary(transactions, review_log)
    reviewed_ids = set(review_log["transactionid"].astype(str)) if not review_log.empty and "transactionid" in review_log.columns else set()
    needs_review_ids = set()
    if not review_log.empty and {"transactionid", "decision"}.issubset(review_log.columns):
        needs_review_ids = set(review_log.loc[review_log["decision"] == "Needs Review", "transactionid"].astype(str))

    pending_ids = (set(transactions["transactionid"].astype(str)) - reviewed_ids) | needs_review_ids
    pending = transactions.loc[transactions["transactionid"].astype(str).isin(pending_ids)].copy()
    if "composite_risk_score" in pending.columns:
        pending = pending.sort_values("composite_risk_score", ascending=False)
    top_pending = pending.loc[pending["risk_level"] == "High"].head(5) if "risk_level" in pending.columns else pending.head(5)
    if top_pending.empty:
        top_pending = pending.head(5)
    if top_pending.empty:
        return "There are currently no pending high-priority reviews in the active fraud dataset."

    top_items = ", ".join(
        f"{row.transactionid} ({float(row.composite_risk_score):.3f}, {row.risk_level})"
        for row in top_pending.itertuples()
    )
    return (
        f"There are {review_summary['pending_review_total']} pending reviews, including "
        f"{review_summary['pending_high_risk_count']} high-risk items and "
        f"{review_summary.get('pending_flagged_count', 0)} flagged items overall. "
        f"Start with: {top_items}."
    )


def _daily_summary_answer(bundle: Dict[str, Any]) -> str:
    summary = bundle.get("summary", {}) or {}
    transactions = bundle.get("transactions", pd.DataFrame())
    accounts = bundle.get("accounts", pd.DataFrame())
    merchants = bundle.get("merchants", pd.DataFrame())

    top_transaction = transactions.iloc[0]["transactionid"] if isinstance(transactions, pd.DataFrame) and not transactions.empty and "transactionid" in transactions else "N/A"
    top_account = accounts.iloc[0]["accountid"] if isinstance(accounts, pd.DataFrame) and not accounts.empty and "accountid" in accounts else "N/A"
    top_merchant = merchants.iloc[0]["merchantid"] if isinstance(merchants, pd.DataFrame) and not merchants.empty and "merchantid" in merchants else "N/A"

    return (
        f"Current fraud indicators show {summary.get('flagged_transactions', 0)} flagged transactions and "
        f"{summary.get('high_risk_count', 0)} high-risk transactions. "
        f"Priority items include transaction {top_transaction}, account {top_account}, and merchant {top_merchant}. "
        f"There are {summary.get('pending_review_total', 0)} items still pending analyst review."
    )


def _extra_operational_context(bundle: Dict[str, Any], conversation_context: Optional[str]) -> str:
    summary = bundle.get("summary", {}) or {}
    review_context = (
        f"Review status: approved={summary.get('approved_count', 0)}, "
        f"dismissed={summary.get('dismissed_count', 0)}, "
        f"needs_review={summary.get('needs_review_count', 0)}, "
        f"pending_total={summary.get('pending_review_total', 0)}, "
        f"pending_flagged={summary.get('pending_flagged_count', 0)}, "
        f"pending_high_risk={summary.get('pending_high_risk_count', 0)}."
    )
    if conversation_context:
        return f"{review_context}\n\nRecent Discord conversation:\n{conversation_context}"
    return review_context


def _find_row(df: pd.DataFrame, column: str, value: str) -> Optional[Dict[str, Any]]:
    if df.empty or column not in df.columns:
        return None
    matches = df.loc[df[column].astype(str).str.lower() == value.lower()]
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()


def _append_judge_summary(answer: str, judgment: Dict[str, Any]) -> str:
    return (
        f"{answer}\n\n"
        f"Suggested disposition: {judgment['decision']} ({judgment['confidence']} confidence).\n"
        f"Judge rationale: {judgment['rationale']}\n"
        + "\n".join(f"- {item}" for item in judgment.get("checks", [])[:3])
    )


def explain_transaction_case(transaction_id: str, bundle: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    active_bundle = bundle or load_active_bundle()
    transactions = active_bundle.get("transactions", pd.DataFrame())
    case_summary = _find_row(transactions, "transactionid", transaction_id)
    if not case_summary:
        return {
            "found": False,
            "answer": f"Transaction {transaction_id} is not present in the active fraud context.",
            "case_type": "transaction",
            "case_id": transaction_id,
            "bundle": active_bundle,
        }

    explanation = build_deep_case_explanation(
        "transaction",
        case_summary,
        active_bundle,
        model=config.OPENCLAW_OPENAI_MODEL,
        reasoning_effort=config.OPENCLAW_OPENAI_REASONING_EFFORT,
    )
    judgment = judge_case_disposition(
        "transaction",
        case_summary,
        active_bundle,
        model=config.OPENCLAW_OPENAI_MODEL,
        reasoning_effort=config.OPENCLAW_OPENAI_REASONING_EFFORT,
    )
    fallback = (
        f"{explanation['summary']}\n\n"
        "Supporting evidence:\n"
        + "\n".join(f"- {line}" for line in explanation["evidence_lines"])
        + "\n\nRecommended next steps:\n"
        + "\n".join(f"- {line}" for line in explanation["next_steps"])
    )
    return {
        "found": True,
        "answer": _append_judge_summary(explanation["ai_text"] or fallback, judgment),
        "used_ai": bool(explanation["ai_text"] or judgment.get("used_ai")),
        "case_type": "transaction",
        "case_id": transaction_id,
        "case_summary": case_summary,
        "judge": judgment,
        "bundle": active_bundle,
    }


def explain_account_case(account_id: str, bundle: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    active_bundle = bundle or load_active_bundle()
    accounts = active_bundle.get("accounts", pd.DataFrame())
    case_summary = _find_row(accounts, "accountid", account_id)
    if not case_summary:
        return {
            "found": False,
            "answer": f"Account {account_id} is not present in the active fraud context.",
            "case_type": "account",
            "case_id": account_id,
            "bundle": active_bundle,
        }

    explanation = build_deep_case_explanation(
        "account",
        case_summary,
        active_bundle,
        model=config.OPENCLAW_OPENAI_MODEL,
        reasoning_effort=config.OPENCLAW_OPENAI_REASONING_EFFORT,
    )
    judgment = judge_case_disposition(
        "account",
        case_summary,
        active_bundle,
        model=config.OPENCLAW_OPENAI_MODEL,
        reasoning_effort=config.OPENCLAW_OPENAI_REASONING_EFFORT,
    )
    fallback = (
        f"{explanation['summary']}\n\n"
        "Supporting evidence:\n"
        + "\n".join(f"- {line}" for line in explanation["evidence_lines"])
        + "\n\nRecommended next steps:\n"
        + "\n".join(f"- {line}" for line in explanation["next_steps"])
    )
    return {
        "found": True,
        "answer": _append_judge_summary(explanation["ai_text"] or fallback, judgment),
        "used_ai": bool(explanation["ai_text"] or judgment.get("used_ai")),
        "case_type": "account",
        "case_id": account_id,
        "case_summary": case_summary,
        "judge": judgment,
        "bundle": active_bundle,
    }


def merchant_summary(merchant_id: str, bundle: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    active_bundle = bundle or load_active_bundle()
    merchants = active_bundle.get("merchants", pd.DataFrame())
    transactions = active_bundle.get("transactions", pd.DataFrame())
    case_summary = _find_row(merchants, "merchantid", merchant_id)
    if not case_summary:
        return {
            "found": False,
            "answer": f"Merchant {merchant_id} is not present in the active fraud context.",
            "case_type": "merchant",
            "case_id": merchant_id,
            "bundle": active_bundle,
        }

    explanation = build_deep_case_explanation(
        "merchant",
        case_summary,
        active_bundle,
        model=config.OPENCLAW_OPENAI_MODEL,
        reasoning_effort=config.OPENCLAW_OPENAI_REASONING_EFFORT,
    )
    judgment = judge_case_disposition(
        "merchant",
        case_summary,
        active_bundle,
        model=config.OPENCLAW_OPENAI_MODEL,
        reasoning_effort=config.OPENCLAW_OPENAI_REASONING_EFFORT,
    )
    linked = pd.DataFrame()
    if not transactions.empty and "merchantid" in transactions.columns:
        linked = transactions.loc[transactions["merchantid"].astype(str).str.lower() == merchant_id.lower()].copy()
        if "composite_risk_score" in linked.columns:
            linked = linked.sort_values("composite_risk_score", ascending=False)
    top_transactions = ", ".join(linked.head(3)["transactionid"].astype(str).tolist()) if not linked.empty and "transactionid" in linked.columns else "N/A"
    fallback = (
        f"{explanation['summary']}\n\n"
        "Supporting evidence:\n"
        + "\n".join(f"- {line}" for line in explanation["evidence_lines"])
        + f"\n- Top linked transactions: {top_transactions}"
    )
    return {
        "found": True,
        "answer": _append_judge_summary(explanation["ai_text"] or fallback, judgment),
        "used_ai": bool(explanation["ai_text"] or judgment.get("used_ai")),
        "case_type": "merchant",
        "case_id": merchant_id,
        "case_summary": case_summary,
        "judge": judgment,
        "bundle": active_bundle,
    }


def top_accounts_summary(bundle: Optional[Dict[str, Any]] = None, limit: int = 5) -> Dict[str, Any]:
    active_bundle = bundle or load_active_bundle()
    accounts = active_bundle.get("accounts", pd.DataFrame())
    if accounts.empty:
        return {"answer": "No ranked account risk data is currently available.", "bundle": active_bundle}
    top = accounts.head(limit)
    answer = "Top risky accounts: " + ", ".join(
        f"{row.accountid} ({float(row.account_risk_score):.3f})" for row in top.itertuples()
    )
    return {"answer": answer, "bundle": active_bundle}


def pending_review_summary(bundle: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    active_bundle = bundle or load_active_bundle()
    return {"answer": _pending_review_answer(active_bundle), "bundle": active_bundle}


def create_oof_brief(
    *,
    bundle: Optional[Dict[str, Any]] = None,
    focus: str = "",
    export_path: Optional[Path] = None,
) -> Dict[str, Any]:
    active_bundle = bundle or load_active_bundle()
    brief = generate_multi_agent_oof_brief(
        active_bundle,
        focus=focus,
        model=config.OPENCLAW_OPENAI_MODEL,
        reasoning_effort=config.OPENCLAW_OPENAI_REASONING_EFFORT,
    )
    output_path = export_path
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(brief["brief_markdown"])
    return {
        "answer": brief["brief_markdown"],
        "bundle": active_bundle,
        "used_ai": brief["used_ai"],
        "role_outputs": brief["role_outputs"],
        "export_path": output_path,
    }


def parse_command(command_text: str) -> Optional[Dict[str, str]]:
    normalized = command_text.strip()
    if not normalized.startswith("/"):
        return None
    parts = normalized.split(maxsplit=1)
    command = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""
    return {"command": command, "arg": arg}


def run_command_workflow(command_text: str, *, bundle: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    parsed = parse_command(command_text)
    active_bundle = bundle or load_active_bundle()
    if not parsed:
        return {"handled": False, "bundle": active_bundle}

    command = parsed["command"]
    arg = parsed["arg"]

    if command == "/triage":
        if not arg:
            return {"handled": True, "answer": "Usage: `/triage TX000275` or `/triage AC00454`.", "bundle": active_bundle}
        token = arg.split()[0]
        if token.upper().startswith("TX"):
            result = explain_transaction_case(token, active_bundle)
        elif token.upper().startswith("AC"):
            result = explain_account_case(token, active_bundle)
        else:
            result = {"answer": "Use a transaction ID like `TX000275` or an account ID like `AC00454` with `/triage`.", "bundle": active_bundle}
        return {"handled": True, **result}

    if command == "/top-accounts":
        return {"handled": True, **top_accounts_summary(active_bundle)}

    if command == "/pending-review":
        return {"handled": True, **pending_review_summary(active_bundle)}

    if command == "/merchant":
        if not arg:
            return {"handled": True, "answer": "Usage: `/merchant M026`.", "bundle": active_bundle}
        token = arg.split()[0]
        return {"handled": True, **merchant_summary(token, active_bundle)}

    if command == "/send-oof-brief":
        return {"handled": True, **create_oof_brief(bundle=active_bundle, focus=arg)}

    if command == "/why-flagged":
        if not arg:
            return {"handled": True, "answer": "Usage: `/why-flagged TX000275`.", "bundle": active_bundle}
        token = arg.split()[0]
        return {"handled": True, **explain_transaction_case(token, active_bundle)}

    return {"handled": False, "bundle": active_bundle}


def answer_analyst_question(
    question: str,
    *,
    bundle: Optional[Dict[str, Any]] = None,
    conversation_context: Optional[str] = None,
) -> Dict[str, Any]:
    active_bundle = bundle or load_active_bundle()
    command_result = run_command_workflow(question, bundle=active_bundle)
    if command_result.get("handled"):
        return {
            "question": question,
            "answer": command_result["answer"],
            "used_ai": bool(command_result.get("used_ai")),
            "source_label": active_bundle.get("source_label", "Fraud dashboard"),
            "bundle": active_bundle,
            "case_type": command_result.get("case_type"),
            "case_id": command_result.get("case_id"),
            "case_summary": command_result.get("case_summary"),
            "export_path": command_result.get("export_path"),
            "role_outputs": command_result.get("role_outputs"),
        }

    question_lower = question.lower()
    if "pending analyst review" in question_lower or "pending review" in question_lower or "unreviewed" in question_lower:
        heuristic_answer = _pending_review_answer(active_bundle)
    elif "today" in question_lower or "summary" in question_lower or "key fraud indicators" in question_lower:
        heuristic_answer = _daily_summary_answer(active_bundle)
    else:
        heuristic_answer = None

    response = answer_data_question(
        question,
        active_bundle,
        model=config.OPENCLAW_OPENAI_MODEL,
        reasoning_effort=config.OPENCLAW_OPENAI_REASONING_EFFORT,
        extra_context=_extra_operational_context(active_bundle, conversation_context),
        max_output_tokens=config.OPENCLAW_OPENAI_MAX_OUTPUT_TOKENS,
    )
    answer_text = response["ai_answer"] or heuristic_answer or response["heuristic_answer"]

    transaction_match = re.search(r"\b(TX\d+)\b", question, re.IGNORECASE)
    account_match = re.search(r"\b(AC\d+)\b", question, re.IGNORECASE)
    case_type = None
    case_id = None
    if transaction_match:
        case_type, case_id = "transaction", transaction_match.group(1).upper()
    elif account_match:
        case_type, case_id = "account", account_match.group(1).upper()

    return {
        "question": question,
        "answer": answer_text,
        "used_ai": bool(response["ai_answer"]),
        "source_label": active_bundle.get("source_label", "Fraud dashboard"),
        "bundle": active_bundle,
        "case_type": case_type,
        "case_id": case_id,
    }
