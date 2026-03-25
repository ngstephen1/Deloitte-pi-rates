"""
Grounded analyst Q&A helpers for Discord/OpenClaw consumers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from .. import config
from ..ai_assistant import answer_data_question
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


def answer_analyst_question(
    question: str,
    *,
    bundle: Optional[Dict[str, Any]] = None,
    conversation_context: Optional[str] = None,
) -> Dict[str, Any]:
    active_bundle = bundle or load_active_bundle()
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
    return {
        "question": question,
        "answer": answer_text,
        "used_ai": bool(response["ai_answer"]),
        "source_label": active_bundle.get("source_label", "Fraud dashboard"),
        "bundle": active_bundle,
    }
