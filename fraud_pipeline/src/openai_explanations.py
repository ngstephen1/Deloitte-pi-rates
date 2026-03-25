"""
OpenAI-backed explanation helpers.

These wrappers preserve the original module entry points while routing
requests through the shared AI assistant utilities and the current SDK.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

from . import config
from .ai_assistant import has_openai_api_key, request_ai_response


def get_api_key() -> Optional[str]:
    """Get OpenAI API key from environment, with safety checks."""
    return os.environ.get("OPENAI_API_KEY")


def explain_transaction(
    tx_dict: Dict,
    risk_scores: Dict,
    max_tokens: int = 150,
) -> Optional[str]:
    """Generate a concise explanation for a flagged transaction."""
    if not config.USE_OPENAI_EXPLANATIONS or not has_openai_api_key():
        return None

    top_risks = sorted(risk_scores.items(), key=lambda item: item[1], reverse=True)[:3]
    risk_summary = ", ".join([f"{name}={score:.2f}" for name, score in top_risks])
    return request_ai_response(
        instructions=(
            "You are a fraud analyst. Explain flagged transactions in one or two concise, "
            "business-facing sentences using only the supplied facts."
        ),
        prompt=(
            f"Transaction ID: {tx_dict.get('transactionid', 'N/A')}\n"
            f"Amount: {tx_dict.get('transactionamount', 0):.2f}\n"
            f"Account: {tx_dict.get('accountid', 'N/A')}\n"
            f"Merchant: {tx_dict.get('merchantid', 'N/A')}\n"
            f"Channel: {tx_dict.get('channel', 'N/A')}\n"
            f"Location: {tx_dict.get('location', 'N/A')}\n"
            f"Top risk signals: {risk_summary}\n\n"
            "Explain why this case was flagged and what an analyst should review next."
        ),
        max_output_tokens=max_tokens,
    )


def explain_account_risk(
    account_id: str,
    account_stats: Dict,
    max_tokens: int = 100,
) -> Optional[str]:
    """Generate explanation for an account's risk profile."""
    if not config.USE_OPENAI_EXPLANATIONS or not has_openai_api_key():
        return None

    return request_ai_response(
        instructions=(
            "You are a fraud risk analyst. Provide a short, factual explanation of the account risk profile."
        ),
        prompt=(
            f"Account: {account_id}\n"
            f"Risk score: {account_stats.get('account_risk_score', 0):.3f}\n"
            f"Transaction count: {account_stats.get('transaction_count', 0)}\n"
            f"High-risk transaction count: {account_stats.get('high_risk_transaction_count', 0)}\n"
            f"High-risk transaction percentage: {account_stats.get('high_risk_transaction_pct', 0):.1f}%\n\n"
            "Summarize why this account is elevated and what should be reviewed next."
        ),
        max_output_tokens=max_tokens,
    )
