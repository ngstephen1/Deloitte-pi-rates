"""
Formatting and artifact export helpers for fraud image analysis.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .. import config
from .image_router import image_type_label


def _safe_slug(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", text.strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned or "image-review"


def _bullets(items: list[str], *, limit: int = 5) -> str:
    filtered = [str(item).strip() for item in items if str(item).strip()]
    if not filtered:
        return "- None identified from the visible evidence."
    return "\n".join(f"- {item}" for item in filtered[:limit])


def _entity_lines(extracted_entities: Dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if extracted_entities.get("account_ids"):
        lines.append(f"Accounts: {', '.join(map(str, extracted_entities['account_ids'][:4]))}")
    if extracted_entities.get("transaction_ids"):
        lines.append(f"Transactions: {', '.join(map(str, extracted_entities['transaction_ids'][:4]))}")
    if extracted_entities.get("merchant_ids"):
        lines.append(f"Merchant IDs: {', '.join(map(str, extracted_entities['merchant_ids'][:4]))}")
    if extracted_entities.get("merchant_names"):
        lines.append(f"Merchants: {', '.join(map(str, extracted_entities['merchant_names'][:4]))}")
    if extracted_entities.get("amounts"):
        lines.append(f"Amounts: {', '.join(map(str, extracted_entities['amounts'][:5]))}")
    if extracted_entities.get("locations"):
        lines.append(f"Locations: {', '.join(map(str, extracted_entities['locations'][:4]))}")
    if extracted_entities.get("channels"):
        lines.append(f"Channels: {', '.join(map(str, extracted_entities['channels'][:4]))}")
    if extracted_entities.get("device_ids"):
        lines.append(f"Devices: {', '.join(map(str, extracted_entities['device_ids'][:4]))}")
    if extracted_entities.get("ip_addresses"):
        lines.append(f"IP addresses: {', '.join(map(str, extracted_entities['ip_addresses'][:4]))}")
    if extracted_entities.get("invoice_numbers"):
        lines.append(f"Invoice references: {', '.join(map(str, extracted_entities['invoice_numbers'][:3]))}")
    if extracted_entities.get("sender_email"):
        lines.append(f"Sender email: {extracted_entities['sender_email']}")
    if extracted_entities.get("subject"):
        lines.append(f"Subject: {extracted_entities['subject']}")
    if extracted_entities.get("login_attempt_count") not in {None, ""}:
        lines.append(f"Login attempts visible: {extracted_entities['login_attempt_count']}")
    return lines


def build_image_review_text(analysis: Dict[str, Any]) -> str:
    structured = analysis.get("structured_output", {}) or {}
    extracted_entities = structured.get("extracted_entities", {}) or {}
    linked = analysis.get("linked_entities_if_found", {}) or {}
    highlight_lines = linked.get("highlight_lines", []) or []
    no_match_line = (
        "No reliable direct entity match was found in the current fraud outputs."
        if not linked.get("has_match")
        else None
    )

    sections = [
        f"**Detected image type:** {image_type_label(structured.get('image_type', ''))} ({analysis.get('classification_confidence', 'low')} confidence)",
        "",
        f"**Assessment:** {structured.get('summary') or 'The image was reviewed for fraud or anomaly indicators.'}",
        "",
        "**Key details extracted**",
        _bullets(_entity_lines(extracted_entities), limit=8),
        "",
        "**Suspicious indicators**",
        _bullets(structured.get("suspicious_indicators", []), limit=6),
        "",
        "**Cross-check against current fraud context**",
        _bullets(highlight_lines or ([no_match_line] if no_match_line else []), limit=5),
        "",
        "**Confidence / caution notes**",
        _bullets(structured.get("confidence_notes", []), limit=5),
        "",
        "**Recommended next actions**",
        _bullets(structured.get("recommendations", []), limit=5),
    ]
    return "\n".join(section for section in sections if section is not None).strip()


def build_image_review_markdown(analysis: Dict[str, Any]) -> str:
    structured = analysis.get("structured_output", {}) or {}
    linked = analysis.get("linked_entities_if_found", {}) or {}
    extracted_entities = structured.get("extracted_entities", {}) or {}
    raw_excerpt = structured.get("raw_text_excerpt", "")
    return (
        "# Fraud Image Review\n\n"
        f"- Source image: `{analysis.get('file_name', 'unknown')}`\n"
        f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`\n"
        f"- Detected image type: `{structured.get('image_type', '')}`\n"
        f"- Detected label: {image_type_label(structured.get('image_type', ''))}\n"
        f"- Classification confidence: `{analysis.get('classification_confidence', 'low')}`\n"
        f"- User goal: {analysis.get('user_prompt') or 'General image fraud review'}\n\n"
        "## Assessment\n\n"
        f"{structured.get('summary') or 'No summary available.'}\n\n"
        "## Key Details Extracted\n\n"
        f"{_bullets(_entity_lines(extracted_entities), limit=10)}\n\n"
        "## Suspicious Indicators\n\n"
        f"{_bullets(structured.get('suspicious_indicators', []), limit=10)}\n\n"
        "## Cross-links To Current Fraud Context\n\n"
        f"{_bullets(linked.get('highlight_lines', []) or ['No reliable direct entity match was found in the current fraud outputs.'], limit=8)}\n\n"
        "## Confidence Notes\n\n"
        f"{_bullets(structured.get('confidence_notes', []), limit=8)}\n\n"
        "## Recommended Next Actions\n\n"
        f"{_bullets(structured.get('recommendations', []), limit=8)}\n\n"
        "## Raw Text Excerpt\n\n"
        f"```\n{raw_excerpt[:1600]}\n```\n"
    )


def export_image_review_artifacts(analysis: Dict[str, Any], *, stem: str | None = None) -> Dict[str, Path]:
    export_dir = config.CHATOPS_IMAGE_EXPORTS_DIR
    export_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    file_stem = _safe_slug(stem or Path(analysis.get("file_name", "image-review")).stem)
    md_path = export_dir / f"{timestamp}-{file_stem}-image-review.md"
    json_path = export_dir / f"{timestamp}-{file_stem}-image-review.json"

    md_path.write_text(build_image_review_markdown(analysis))
    json_path.write_text(json.dumps(analysis.get("structured_output", {}), indent=2, default=str))
    return {"markdown": md_path, "json": json_path}
