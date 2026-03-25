"""
Link extracted image findings back to the active fraud context when possible.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

import pandas as pd


def _flatten_candidates(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (int, float)):
        return [str(value)]
    if isinstance(value, dict):
        values: list[str] = []
        for nested in value.values():
            values.extend(_flatten_candidates(nested))
        return values
    if isinstance(value, Iterable):
        values = []
        for item in value:
            values.extend(_flatten_candidates(item))
        return values
    return []


def _normalized_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _row_summary(row: pd.Series, columns: list[str]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for column in columns:
        if column in row.index and pd.notna(row[column]):
            summary[column] = row[column]
    return summary


def _match_rows(df: pd.DataFrame, column: str, candidates: list[str], columns: list[str], *, max_rows: int = 3) -> list[Dict[str, Any]]:
    if df.empty or column not in df.columns:
        return []
    normalized_candidates = {_normalized_text(candidate) for candidate in candidates if _normalized_text(candidate)}
    if not normalized_candidates:
        return []
    matches = df.loc[df[column].astype(str).str.lower().isin(normalized_candidates)]
    if matches.empty and column == "location":
        matches = df.loc[
            df[column].astype(str).str.lower().apply(
                lambda value: any(value in candidate or candidate in value for candidate in normalized_candidates if candidate)
            )
        ]
    return [_row_summary(row, columns) for _, row in matches.head(max_rows).iterrows()]


def _extract_candidate_ids(raw_text: str) -> Dict[str, list[str]]:
    text = raw_text or ""
    return {
        "transaction_ids": sorted(set(re.findall(r"\bTXN?\d{5,}\b", text, flags=re.IGNORECASE))),
        "account_ids": sorted(set(re.findall(r"\bAC\d{4,}\b", text, flags=re.IGNORECASE))),
        "merchant_ids": sorted(set(re.findall(r"\bM\d{2,5}\b", text, flags=re.IGNORECASE))),
        "device_ids": sorted(set(re.findall(r"\b(?:DV-[A-Z0-9-]+|D\d{4,})\b", text, flags=re.IGNORECASE))),
        "ip_addresses": sorted(set(re.findall(r"\b\d{1,3}(?:\.\d{1,3}){2,3}(?:\.xx)?\b", text, flags=re.IGNORECASE))),
    }


def link_image_findings(
    bundle: Dict[str, Any],
    *,
    extracted_entities: Dict[str, Any],
    raw_text_excerpt: str = "",
    file_name: str = "",
) -> Dict[str, Any]:
    transactions = bundle.get("transactions", pd.DataFrame())
    accounts = bundle.get("accounts", pd.DataFrame())
    merchants = bundle.get("merchants", pd.DataFrame())
    devices = bundle.get("devices", pd.DataFrame())
    locations = bundle.get("locations", pd.DataFrame())
    review_log = bundle.get("review_log", pd.DataFrame())

    parsed_ids = _extract_candidate_ids(f"{raw_text_excerpt}\n{file_name}")
    candidate_transactions = _flatten_candidates(extracted_entities.get("transaction_ids")) + parsed_ids["transaction_ids"]
    candidate_accounts = _flatten_candidates(extracted_entities.get("account_ids")) + parsed_ids["account_ids"]
    candidate_merchants = _flatten_candidates(extracted_entities.get("merchant_ids")) + parsed_ids["merchant_ids"]
    candidate_devices = _flatten_candidates(extracted_entities.get("device_ids")) + parsed_ids["device_ids"]
    candidate_locations = _flatten_candidates(extracted_entities.get("locations"))

    linked_transactions = _match_rows(
        transactions,
        "transactionid",
        candidate_transactions,
        ["transactionid", "accountid", "merchantid", "location", "channel", "composite_risk_score", "risk_level"],
    )
    linked_accounts = _match_rows(
        accounts,
        "accountid",
        candidate_accounts,
        ["accountid", "account_risk_score", "transaction_count", "high_risk_transaction_count", "high_risk_transaction_pct"],
    )
    linked_merchants = _match_rows(
        merchants,
        "merchantid",
        candidate_merchants,
        ["merchantid", "avg_risk_score", "max_risk_score", "transaction_count", "high_risk_count"],
    )
    linked_devices = _match_rows(
        devices,
        "deviceid",
        candidate_devices,
        ["deviceid", "avg_risk_score", "max_risk_score", "transaction_count", "high_risk_count"],
    )
    linked_locations = _match_rows(
        locations,
        "location",
        candidate_locations,
        ["location", "avg_risk_score", "max_risk_score", "transaction_count", "high_risk_count"],
    )

    review_matches = pd.DataFrame()
    if not review_log.empty:
        transaction_candidates = {_normalized_text(item) for item in candidate_transactions}
        if "transactionid" in review_log.columns and transaction_candidates:
            review_matches = review_log.loc[review_log["transactionid"].astype(str).str.lower().isin(transaction_candidates)]
        elif "case_id" in review_log.columns and transaction_candidates:
            review_matches = review_log.loc[review_log["case_id"].astype(str).str.lower().isin(transaction_candidates)]

    linked_review_rows = [
        _row_summary(row, ["case_id", "transactionid", "decision", "analyst_notes", "updated_at"])
        for _, row in review_matches.head(3).iterrows()
    ]

    highlight_lines: list[str] = []
    if linked_transactions:
        first = linked_transactions[0]
        highlight_lines.append(
            f"Matched transaction {first.get('transactionid')} in the active fraud context at risk {float(first.get('composite_risk_score', 0) or 0):.3f} ({first.get('risk_level', 'N/A')})."
        )
    if linked_accounts:
        first = linked_accounts[0]
        highlight_lines.append(
            f"Matched account {first.get('accountid')} with account risk {float(first.get('account_risk_score', 0) or 0):.3f}."
        )
    if linked_merchants:
        first = linked_merchants[0]
        highlight_lines.append(
            f"Matched merchant {first.get('merchantid')} with max observed risk {float(first.get('max_risk_score', 0) or 0):.3f}."
        )
    if linked_devices:
        first = linked_devices[0]
        highlight_lines.append(
            f"Matched device {first.get('deviceid')} with max observed risk {float(first.get('max_risk_score', 0) or 0):.3f}."
        )
    if linked_locations:
        first = linked_locations[0]
        highlight_lines.append(
            f"Matched location {first.get('location')} with concentrated flagged activity in the active context."
        )
    if linked_review_rows:
        first = linked_review_rows[0]
        highlight_lines.append(
            f"Review log already contains {first.get('decision', 'a decision')} for case {first.get('case_id') or first.get('transactionid')}."
        )

    primary_case_type = None
    primary_case_id = None
    if linked_transactions:
        primary_case_type = "transaction"
        primary_case_id = str(linked_transactions[0].get("transactionid", ""))
    elif linked_accounts:
        primary_case_type = "account"
        primary_case_id = str(linked_accounts[0].get("accountid", ""))

    return {
        "matches": {
            "transactions": linked_transactions,
            "accounts": linked_accounts,
            "merchants": linked_merchants,
            "devices": linked_devices,
            "locations": linked_locations,
            "review_log": linked_review_rows,
        },
        "highlight_lines": highlight_lines,
        "primary_case_type": primary_case_type,
        "primary_case_id": primary_case_id,
        "has_match": any([linked_transactions, linked_accounts, linked_merchants, linked_devices, linked_locations, linked_review_rows]),
    }
