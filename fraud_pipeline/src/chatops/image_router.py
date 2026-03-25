"""
Rule-based image routing for Discord/OpenClaw fraud image analysis.
"""

from __future__ import annotations

import re
from typing import Dict


IMAGE_TYPE_BANK_SCREENSHOT = "bank_transaction_screenshot"
IMAGE_TYPE_CARD_STATEMENT = "card_statement_page"
IMAGE_TYPE_INVOICE_RECEIPT = "invoice_or_payment_receipt"
IMAGE_TYPE_SUSPICIOUS_EMAIL = "suspicious_email_screenshot"
IMAGE_TYPE_DEVICE_VERIFICATION = "identity_device_verification_screen"
IMAGE_TYPE_RISK_DASHBOARD = "dashboard_screenshot_with_risky_transactions"


IMAGE_TYPE_METADATA: dict[str, dict[str, str]] = {
    IMAGE_TYPE_BANK_SCREENSHOT: {
        "label": "Bank transaction screenshot",
        "description": "Banking activity log, recent transactions, balances, channels, and flagged rows.",
    },
    IMAGE_TYPE_CARD_STATEMENT: {
        "label": "Credit/debit card statement page",
        "description": "Statement-period transaction table with balances, credits, debits, and merchant/location detail.",
    },
    IMAGE_TYPE_INVOICE_RECEIPT: {
        "label": "Invoice or payment receipt",
        "description": "Invoice/receipt with vendor, payment metadata, totals, line items, or approval text.",
    },
    IMAGE_TYPE_SUSPICIOUS_EMAIL: {
        "label": "Suspicious payment or login email screenshot",
        "description": "Email screenshot with sender, subject, body, CTA, and phishing or alert wording.",
    },
    IMAGE_TYPE_DEVICE_VERIFICATION: {
        "label": "Identity or device verification screen",
        "description": "Device verification or account-security prompt with login attempts, device IDs, IPs, and warnings.",
    },
    IMAGE_TYPE_RISK_DASHBOARD: {
        "label": "Dashboard screenshot with risky transactions",
        "description": "Fraud-monitoring dashboard with KPIs, risky entities, charts, and review queue detail.",
    },
}


_ALIASES = {
    IMAGE_TYPE_BANK_SCREENSHOT: {
        "bank transaction screenshot",
        "transaction screenshot",
        "recent account activities",
        "recent activity log",
        IMAGE_TYPE_BANK_SCREENSHOT,
    },
    IMAGE_TYPE_CARD_STATEMENT: {
        "credit debit card statement page",
        "card statement page",
        "statement page",
        "statement screenshot",
        IMAGE_TYPE_CARD_STATEMENT,
    },
    IMAGE_TYPE_INVOICE_RECEIPT: {
        "invoice or payment receipt",
        "payment receipt",
        "invoice",
        "receipt",
        IMAGE_TYPE_INVOICE_RECEIPT,
    },
    IMAGE_TYPE_SUSPICIOUS_EMAIL: {
        "suspicious email screenshot",
        "payment alert email",
        "phishing email",
        "email screenshot",
        IMAGE_TYPE_SUSPICIOUS_EMAIL,
    },
    IMAGE_TYPE_DEVICE_VERIFICATION: {
        "identity device verification screen",
        "device verification screen",
        "identity verification screen",
        "account verification screen",
        IMAGE_TYPE_DEVICE_VERIFICATION,
    },
    IMAGE_TYPE_RISK_DASHBOARD: {
        "dashboard screenshot with risky transactions",
        "risk dashboard",
        "fraud dashboard",
        "monitoring dashboard",
        IMAGE_TYPE_RISK_DASHBOARD,
    },
}


_KEYWORDS_BY_TYPE = {
    IMAGE_TYPE_BANK_SCREENSHOT: [
        "recent activity",
        "recent account activity",
        "account overview",
        "current balance",
        "transaction type",
        "merchant",
        "debit",
        "credit",
        "channel",
    ],
    IMAGE_TYPE_CARD_STATEMENT: [
        "statement period",
        "statement summary",
        "opening balance",
        "closing balance",
        "account holder",
        "transaction table",
        "total credits",
        "total debits",
    ],
    IMAGE_TYPE_INVOICE_RECEIPT: [
        "invoice",
        "payment receipt",
        "billed to",
        "payment date",
        "invoice #",
        "unit price",
        "line item",
        "total",
        "paid",
    ],
    IMAGE_TYPE_SUSPICIOUS_EMAIL: [
        "subject",
        "to me",
        "from",
        "review transaction",
        "this message may be a phishing attempt",
        "urgent",
        "security team",
        "verify",
        "account alert",
    ],
    IMAGE_TYPE_DEVICE_VERIFICATION: [
        "verify recent account activity",
        "failed login attempts",
        "new device",
        "trusted device",
        "report unauthorized activity",
        "confirm it was me",
        "ip address change",
        "usual account region",
    ],
    IMAGE_TYPE_RISK_DASHBOARD: [
        "fraud monitoring dashboard",
        "flagged transactions",
        "risky merchants",
        "risky devices",
        "risky locations",
        "analyst review queue",
        "strategic recommendation panel",
        "high-risk accounts",
    ],
}


def image_type_label(image_type: str) -> str:
    return IMAGE_TYPE_METADATA.get(image_type, {}).get("label", "Image artifact")


def normalize_image_type(value: str | None) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(value or "").strip().lower()).strip()
    if not normalized:
        return IMAGE_TYPE_BANK_SCREENSHOT
    for image_type, aliases in _ALIASES.items():
        if normalized == image_type.replace("_", " "):
            return image_type
        if normalized in aliases:
            return image_type
        if any(alias in normalized for alias in aliases):
            return image_type
    return IMAGE_TYPE_BANK_SCREENSHOT


def detect_image_type(
    *,
    file_name: str = "",
    user_text: str = "",
    raw_text_excerpt: str = "",
    model_guess: str = "",
) -> Dict[str, str]:
    candidate_guess = normalize_image_type(model_guess) if model_guess else ""
    if candidate_guess and candidate_guess != IMAGE_TYPE_BANK_SCREENSHOT:
        return {
            "image_type": candidate_guess,
            "label": image_type_label(candidate_guess),
            "confidence": "high",
            "reason": "Vision-model classification matched a known fraud image type.",
        }

    haystack = " ".join(
        part for part in [file_name.lower(), user_text.lower(), raw_text_excerpt.lower()] if part
    )
    scores = {image_type: 0 for image_type in IMAGE_TYPE_METADATA}

    for image_type, keywords in _KEYWORDS_BY_TYPE.items():
        for keyword in keywords:
            if keyword in haystack:
                scores[image_type] += 2

    file_lower = file_name.lower()
    if "dashboard" in file_lower:
        scores[IMAGE_TYPE_RISK_DASHBOARD] += 4
    if "phishing" in file_lower or "email" in file_lower:
        scores[IMAGE_TYPE_SUSPICIOUS_EMAIL] += 4
    if "receipt" in file_lower or "invoice" in file_lower:
        scores[IMAGE_TYPE_INVOICE_RECEIPT] += 4
    if "statement" in file_lower or "transaction-table" in file_lower:
        scores[IMAGE_TYPE_CARD_STATEMENT] += 4
    if "verify" in file_lower or "account_activities" in file_lower:
        scores[IMAGE_TYPE_DEVICE_VERIFICATION] += 4
    if "activity" in file_lower or "recent-activity" in file_lower:
        scores[IMAGE_TYPE_BANK_SCREENSHOT] += 2

    selected = max(scores, key=scores.get)
    score = scores[selected]
    confidence = "high" if score >= 6 else "medium" if score >= 3 else "low"
    reason = (
        "Filename and visible text strongly match the routed image type."
        if confidence == "high"
        else "Routing is based on visible text or filename hints and may need verification."
    )
    return {
        "image_type": selected,
        "label": image_type_label(selected),
        "confidence": confidence,
        "reason": reason,
    }
