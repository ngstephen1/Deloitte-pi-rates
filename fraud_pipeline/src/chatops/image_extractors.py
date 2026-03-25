"""
Image upload storage and multimodal extraction helpers for Discord/OpenClaw.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .. import config
from ..ai_assistant import ai_availability_message, is_ai_enabled, request_ai_content_response


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
SUPPORTED_IMAGE_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}


def is_supported_image_attachment(file_name: str = "", content_type: str = "") -> bool:
    suffix = Path(file_name or "").suffix.lower()
    if suffix in SUPPORTED_IMAGE_EXTENSIONS:
        return True
    return content_type.lower() in SUPPORTED_IMAGE_CONTENT_TYPES


def save_uploaded_image(file_bytes: bytes, *, file_name: str, channel_id: str) -> Path:
    safe_name = Path(file_name or "upload.png").name
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target_dir = config.CHATOPS_IMAGE_UPLOADS_DIR / str(channel_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{timestamp}-{safe_name}"
    target_path.write_bytes(file_bytes)
    return target_path


def _guess_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    return mime_type or "image/png"


def _as_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{_guess_mime_type(path)};base64,{encoded}"


def _strip_json_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _parse_json_response(text: str) -> Dict[str, Any] | None:
    cleaned = _strip_json_fences(text)
    try:
        return json.loads(cleaned)
    except Exception:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
    return None


def _fallback_image_extraction(path: Path, user_prompt: str) -> Dict[str, Any]:
    return {
        "image_type": "",
        "image_type_reason": "AI image extraction is unavailable, so only filename-level routing can be used.",
        "summary": "The image could not be OCR-processed locally without OpenAI multimodal analysis.",
        "raw_text_excerpt": "",
        "extracted_entities": {
            "transaction_ids": [],
            "account_ids": [],
            "merchant_ids": [],
            "merchant_names": [],
            "invoice_numbers": [],
            "amounts": [],
            "dates": [],
            "times": [],
            "locations": [],
            "channels": [],
            "device_ids": [],
            "ip_addresses": [],
            "cta_text": [],
            "status_labels": [],
            "sender_name": "",
            "sender_email": "",
            "sender_domain": "",
            "subject": "",
            "login_attempt_count": None,
        },
        "suspicious_indicators": [],
        "confidence_notes": [ai_availability_message()],
        "recommendations": [
            "Enable OPENAI_API_KEY for multimodal extraction so the bot can read visible text and compare it to fraud context."
        ],
        "user_prompt": user_prompt.strip(),
        "file_name": path.name,
    }


def extract_structured_image_review(
    image_path: Path,
    *,
    user_prompt: str = "",
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> Dict[str, Any]:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not is_ai_enabled():
        return {
            "used_ai": False,
            "structured": _fallback_image_extraction(image_path, user_prompt),
            "raw_response_text": None,
        }

    schema_hint = {
        "image_type": "one of: bank_transaction_screenshot, card_statement_page, invoice_or_payment_receipt, suspicious_email_screenshot, identity_device_verification_screen, dashboard_screenshot_with_risky_transactions",
        "image_type_reason": "short reason",
        "summary": "2 sentence cautious summary",
        "raw_text_excerpt": "up to 500 visible characters from the image",
        "extracted_entities": {
            "transaction_ids": [],
            "account_ids": [],
            "merchant_ids": [],
            "merchant_names": [],
            "invoice_numbers": [],
            "amounts": [],
            "dates": [],
            "times": [],
            "locations": [],
            "channels": [],
            "device_ids": [],
            "ip_addresses": [],
            "cta_text": [],
            "status_labels": [],
            "sender_name": "",
            "sender_email": "",
            "sender_domain": "",
            "subject": "",
            "login_attempt_count": None,
        },
        "suspicious_indicators": ["up to 4 short bullets"],
        "confidence_notes": ["up to 3 short notes"],
        "recommendations": ["up to 4 short actions"],
    }

    prompt_text = (
        "Analyze this fraud-relevant image carefully.\n"
        f"User request: {user_prompt or 'Analyze this image for fraud or anomaly indicators.'}\n\n"
        "Return exactly one JSON object and nothing else. Do not wrap it in markdown.\n"
        "Do not invent text that is not visible. Use null, empty strings, or empty arrays when evidence is missing.\n"
        "Prefer caution wording such as possible, appears, suggests, and requires verification.\n"
        "Keep the response compact: no prose outside the required fields, no extra commentary, and keep arrays short.\n"
        f"Schema:\n{json.dumps(schema_hint, indent=2)}"
    )
    input_content = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt_text},
                {"type": "input_image", "image_url": _as_data_url(image_path)},
            ],
        }
    ]
    response_text = request_ai_content_response(
        instructions=(
            "You are a cautious fraud-operations vision analyst. "
            "Classify the image, extract only visible evidence, list suspicious indicators, and recommend next steps. "
            "Return valid JSON only."
        ),
        input_content=input_content,
        max_output_tokens=config.OPENCLAW_IMAGE_MAX_OUTPUT_TOKENS,
        model=model or config.OPENCLAW_OPENAI_MODEL,
        reasoning_effort=reasoning_effort or config.OPENCLAW_OPENAI_REASONING_EFFORT,
    )

    if not response_text:
        return {
            "used_ai": False,
            "structured": _fallback_image_extraction(image_path, user_prompt),
            "raw_response_text": None,
        }

    parsed = _parse_json_response(response_text)
    if not isinstance(parsed, dict):
        fallback = _fallback_image_extraction(image_path, user_prompt)
        fallback["confidence_notes"].append("The multimodal model response could not be parsed as structured JSON.")
        fallback["summary"] = "The image was processed, but the structured extraction did not parse cleanly."
        fallback["raw_text_excerpt"] = response_text[:800]
        return {
            "used_ai": True,
            "structured": fallback,
            "raw_response_text": response_text,
        }

    parsed.setdefault("image_type", "")
    parsed.setdefault("image_type_reason", "")
    parsed.setdefault("summary", "")
    parsed.setdefault("raw_text_excerpt", "")
    parsed.setdefault("extracted_entities", {})
    parsed.setdefault("suspicious_indicators", [])
    parsed.setdefault("confidence_notes", [])
    parsed.setdefault("recommendations", [])
    parsed["file_name"] = image_path.name
    parsed["user_prompt"] = user_prompt.strip()
    return {
        "used_ai": True,
        "structured": parsed,
        "raw_response_text": response_text,
    }
