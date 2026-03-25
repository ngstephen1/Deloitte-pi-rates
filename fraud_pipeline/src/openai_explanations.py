"""
OpenAI-based natural language explanations for fraud detection.

Provides utilities for generating human-readable explanations using OpenAI API.
Safely handles missing API keys and failures.

Usage:
    from src.openai_explanations import explain_transaction
    explanation = explain_transaction(transaction_row, risk_scores)
"""

import os
from typing import Optional, Dict

from .utils import LOGGER
from . import config


def get_api_key() -> Optional[str]:
    """Get OpenAI API key from environment, with safety checks."""
    return os.environ.get("OPENAI_API_KEY")


def explain_transaction(
    tx_dict: Dict,
    risk_scores: Dict,
    max_tokens: int = 150,
) -> Optional[str]:
    """
    Generate a concise explanation for a flagged transaction.
    
    Args:
        tx_dict: Transaction data (amount, merchant, account, channel, etc.)
        risk_scores: Risk component breakdown (anomaly scores, graph score, etc.)
        max_tokens: Max length of response
        
    Returns:
        Explanation string or None if API unavailable
    """
    if not config.USE_OPENAI_EXPLANATIONS:
        return None
    
    api_key = get_api_key()
    if not api_key:
        return None
    
    try:
        import openai
        openai.api_key = api_key
        
        # Build concise prompt
        top_risks = sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        risk_summary = ", ".join([f"{k}={v:.2f}" for k, v in top_risks])
        
        prompt = f"""Analyze this high-risk transaction (keep explanation to 1 sentence):
Amount: ${tx_dict.get('transactionamount', 'N/A'):.2f}
Merchant: {tx_dict.get('merchantid', 'N/A')}
Channel: {tx_dict.get('channel', 'N/A')}
Top Risk Signals: {risk_summary}

Why is this high-risk?"""
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a fraud analyst. Provide concise, factual risk explanations."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        
        explanation = response["choices"][0]["message"]["content"].strip()
        return explanation
        
    except ImportError:
        LOGGER.debug("OpenAI package not installed; skipping explanation")
        return None
    except Exception as e:
        LOGGER.debug(f"OpenAI API error: {e}")
        return None


def explain_account_risk(
    account_id: str,
    account_stats: Dict,
    max_tokens: int = 100,
) -> Optional[str]:
    """
    Generate explanation for an account's risk profile.
    
    Args:
        account_id: Account identifier
        account_stats: Dict with risk_score, transaction_count, high_risk_count, etc.
        max_tokens: Max response length
        
    Returns:
        Explanation or None
    """
    if not config.USE_OPENAI_EXPLANATIONS:
        return None
    
    api_key = get_api_key()
    if not api_key:
        return None
    
    try:
        import openai
        openai.api_key = api_key
        
        prompt = f"""Account {account_id} has this risk profile (1 sentence):
Risk Score: {account_stats.get('account_risk_score', 0):.3f}
Total Transactions: {account_stats.get('transaction_count', 0)}
High-Risk Transactions: {account_stats.get('high_risk_transaction_count', 0)}

Summarize the risk in one short sentence."""
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a fraud risk analyst. Be concise and factual."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        
        explanation = response["choices"][0]["message"]["content"].strip()
        return explanation
        
    except (ImportError, Exception) as e:
        LOGGER.debug(f"OpenAI account explanation error: {e}")
        return None
