"""
Grounded image analysis for Discord/OpenClaw fraud workflows.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .. import config
from .context_linker import link_image_findings
from .context_loader import load_active_bundle
from .image_extractors import extract_structured_image_review
from .image_response_builder import build_image_review_text, export_image_review_artifacts
from .image_router import (
    IMAGE_TYPE_BANK_SCREENSHOT,
    IMAGE_TYPE_CARD_STATEMENT,
    IMAGE_TYPE_DEVICE_VERIFICATION,
    IMAGE_TYPE_INVOICE_RECEIPT,
    IMAGE_TYPE_RISK_DASHBOARD,
    IMAGE_TYPE_SUSPICIOUS_EMAIL,
    detect_image_type,
)


def _to_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = re.sub(r"[^0-9.\-]", "", str(value))
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except Exception:
        return None


def _normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (int, float)):
        return [str(value)]
    if isinstance(value, list):
        items: list[str] = []
        for item in value:
            items.extend(_normalize_list(item))
        return items
    return []


def _dedupe(items: List[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        cleaned = str(item).strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(cleaned)
    return ordered


def _contains_any(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def _derive_indicator_signals(
    *,
    image_type: str,
    extracted_entities: Dict[str, Any],
    raw_text_excerpt: str,
    linked_entities: Dict[str, Any],
    bundle: Dict[str, Any],
    user_prompt: str,
) -> Dict[str, list[str]]:
    indicators: list[str] = []
    recommendations: list[str] = []
    confidence_notes: list[str] = []

    amounts = [_to_float(value) for value in _normalize_list(extracted_entities.get("amounts"))]
    amounts = [value for value in amounts if value is not None]
    merchant_names = _normalize_list(extracted_entities.get("merchant_names"))
    merchant_ids = _normalize_list(extracted_entities.get("merchant_ids"))
    channels = [item.lower() for item in _normalize_list(extracted_entities.get("channels"))]
    locations = _normalize_list(extracted_entities.get("locations"))
    device_ids = _normalize_list(extracted_entities.get("device_ids"))
    cta_text = _normalize_list(extracted_entities.get("cta_text"))
    status_labels = _normalize_list(extracted_entities.get("status_labels"))
    login_attempt_count = extracted_entities.get("login_attempt_count")
    login_attempt_count = int(login_attempt_count) if str(login_attempt_count).isdigit() else None
    sender_email = str(extracted_entities.get("sender_email") or "").strip()
    sender_domain = str(extracted_entities.get("sender_domain") or "").strip()
    subject = str(extracted_entities.get("subject") or "").strip()
    raw_text = str(raw_text_excerpt or "")

    max_amount = max(amounts) if amounts else None
    if max_amount is not None and max_amount >= 1000:
        indicators.append(f"Large visible amount outlier appears in the image at approximately ${max_amount:,.2f}.")

    repeated_merchants = [name for name in set(merchant_names + merchant_ids) if (merchant_names + merchant_ids).count(name) >= 2]
    if repeated_merchants:
        indicators.append(
            f"Repeated merchant activity is visible around {', '.join(repeated_merchants[:3])}, which can indicate clustering or split-charge behavior."
        )

    if channels.count("online") >= 2 or (channels and len(set(channels)) == 1 and channels[0] == "online"):
        indicators.append("The visible activity is concentrated in the online channel, which is consistent with elevated access and payment-fraud exposure.")

    if len({location.lower() for location in locations if location}) >= 2:
        indicators.append("The visible locations suggest a geographic inconsistency that requires verification.")

    if login_attempt_count is not None and login_attempt_count >= 3:
        indicators.append(f"Elevated failed-login activity is visible ({login_attempt_count} attempts).")

    if len(device_ids) >= 2 or _contains_any(raw_text, ["new device", "trusted device"]):
        indicators.append("The image suggests a device-change anomaly between a trusted device and a newly seen device.")

    if _contains_any(raw_text, ["urgent", "immediate action required", "critical", "verify immediately", "prevent further unauthorized access"]):
        indicators.append("Urgency or pressure language is visible, which is a common fraud, phishing, or rushed-approval signal.")

    if image_type == IMAGE_TYPE_SUSPICIOUS_EMAIL:
        if "@" in sender_email and sender_domain and not any(token in sender_domain.lower() for token in ["northriver", "summit", "federal", "bank"]):
            indicators.append("The sender domain does not clearly align with a trusted institution name and should be verified independently.")
        if _contains_any(subject + "\n" + raw_text, ["unusual payment activity", "verify your identity", "account alert", "review transaction"]):
            indicators.append("The email requests payment or account verification based on suspicious-activity language.")
        if cta_text:
            indicators.append(f"A call-to-action is visible ({', '.join(cta_text[:2])}), which increases phishing risk if the link target is unverified.")
        recommendations.extend(
            [
                "Do not click links or buttons directly from the email; verify the alert through a trusted channel first.",
                "Cross-check the referenced account, transaction, and device details against the fraud dashboard before taking action.",
            ]
        )

    if image_type == IMAGE_TYPE_INVOICE_RECEIPT:
        if _contains_any(raw_text, ["merchant profile recently updated", "exceptional operational need", "pending secondary verification"]):
            indicators.append("The invoice includes operational-exception or profile-change language that can signal rushed or weak approval controls.")
        recommendations.extend(
            [
                "Verify the vendor independently and confirm the invoice reference, merchant profile change, and approval trail.",
                "Review supporting documentation before approving payment or closing the case.",
            ]
        )

    if image_type in {IMAGE_TYPE_BANK_SCREENSHOT, IMAGE_TYPE_CARD_STATEMENT}:
        if _contains_any(raw_text, ["11:30 pm", "11:56 pm", "late", "overnight"]):
            indicators.append("The visible timing suggests late-night activity, which may warrant extra review.")
        recommendations.extend(
            [
                "Review the highest visible amount and any repeated merchant rows first.",
                "Check whether the visible location and channel pattern fits the account's recent behavior.",
            ]
        )

    if image_type == IMAGE_TYPE_DEVICE_VERIFICATION:
        recommendations.extend(
            [
                "Verify the user identity, require MFA or step-up authentication, and review recent transactions immediately.",
                "If the device and location mismatch cannot be validated, freeze or restrict the account until manual review is complete.",
            ]
        )

    if image_type == IMAGE_TYPE_RISK_DASHBOARD:
        recommendations.extend(
            [
                "Prioritize the top visible high-risk accounts, merchants, and devices before lower-ranked entities.",
                "Reduce queue risk first by clearing unresolved high-risk or needs-review items from the analyst backlog.",
            ]
        )

    if linked_entities.get("has_match"):
        recommendations.append("Open the matched case thread or dashboard record first so the image evidence is tied to the existing fraud case.")
    else:
        confidence_notes.append("No direct entity match was found in the current active fraud outputs, so this image should be treated as new or external evidence until linked.")

    if "compare" in user_prompt.lower():
        summary = bundle.get("summary", {}) or {}
        recommendations.append(
            f"Compare this image against the current portfolio baseline of {summary.get('flagged_transactions', 0)} flagged transactions and {summary.get('high_risk_count', 0)} high-risk items before escalating."
        )

    if "oof" in user_prompt.lower() or "oversight" in user_prompt.lower() or "finance" in user_prompt.lower():
        recommendations.append("Document the evidence in an OOF-ready brief with the suspected entity, visible indicators, verification gaps, and next-control action.")

    if not indicators:
        confidence_notes.append("Only partial or limited fraud indicators were visible in the image, so the assessment should be treated as preliminary.")

    return {
        "suspicious_indicators": _dedupe(indicators),
        "recommendations": _dedupe(recommendations),
        "confidence_notes": _dedupe(confidence_notes),
    }


def analyze_uploaded_image(
    image_path: Path,
    *,
    user_prompt: str = "",
    bundle: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    active_bundle = bundle or load_active_bundle()
    extraction = extract_structured_image_review(
        image_path,
        user_prompt=user_prompt,
        model=config.OPENCLAW_OPENAI_MODEL,
        reasoning_effort=config.OPENCLAW_OPENAI_REASONING_EFFORT,
    )
    structured = extraction.get("structured", {}) or {}
    routing = detect_image_type(
        file_name=image_path.name,
        user_text=user_prompt,
        raw_text_excerpt=str(structured.get("raw_text_excerpt", "")),
        model_guess=str(structured.get("image_type", "")),
    )
    structured["image_type"] = routing["image_type"]
    linked_entities = link_image_findings(
        active_bundle,
        extracted_entities=structured.get("extracted_entities", {}) or {},
        raw_text_excerpt=str(structured.get("raw_text_excerpt", "")),
        file_name=image_path.name,
    )
    derived = _derive_indicator_signals(
        image_type=routing["image_type"],
        extracted_entities=structured.get("extracted_entities", {}) or {},
        raw_text_excerpt=str(structured.get("raw_text_excerpt", "")),
        linked_entities=linked_entities,
        bundle=active_bundle,
        user_prompt=user_prompt,
    )
    structured["suspicious_indicators"] = _dedupe(
        _normalize_list(structured.get("suspicious_indicators")) + derived["suspicious_indicators"]
    )
    structured["confidence_notes"] = _dedupe(
        _normalize_list(structured.get("confidence_notes")) + [routing["reason"]] + derived["confidence_notes"]
    )
    structured["recommendations"] = _dedupe(
        _normalize_list(structured.get("recommendations")) + derived["recommendations"]
    )

    if not structured.get("summary"):
        structured["summary"] = (
            f"The image appears to be a {routing['label'].lower()}. "
            "The visible evidence has been reviewed for fraud, anomaly, and control-risk signals."
        )

    if not linked_entities.get("has_match"):
        transactions = active_bundle.get("transactions", pd.DataFrame())
        accounts = active_bundle.get("accounts", pd.DataFrame())
        top_transaction = (
            str(transactions.iloc[0]["transactionid"])
            if isinstance(transactions, pd.DataFrame) and not transactions.empty and "transactionid" in transactions.columns
            else "N/A"
        )
        top_account = (
            str(accounts.iloc[0]["accountid"])
            if isinstance(accounts, pd.DataFrame) and not accounts.empty and "accountid" in accounts.columns
            else "N/A"
        )
        linked_entities["highlight_lines"].append(
            f"The current fraud portfolio is led by transaction {top_transaction} and account {top_account}; the image identifiers do not directly match them."
        )

    analysis = {
        "file_name": image_path.name,
        "image_path": str(image_path),
        "user_prompt": user_prompt.strip(),
        "used_ai": bool(extraction.get("used_ai")),
        "structured_output": structured,
        "linked_entities_if_found": linked_entities,
        "classification_confidence": routing["confidence"],
        "primary_case_type": linked_entities.get("primary_case_type"),
        "primary_case_id": linked_entities.get("primary_case_id"),
        "source_label": active_bundle.get("source_label", "Published fraud context"),
    }
    export_paths = export_image_review_artifacts(analysis, stem=Path(image_path.name).stem)
    analysis["export_paths"] = export_paths
    analysis["reply_text"] = build_image_review_text(analysis)
    return analysis
