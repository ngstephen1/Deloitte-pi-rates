"""
Delivery bridge for Discord/OpenClaw-style webhook notifications.
"""

from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.request
from typing import Optional

from .. import config
from ..utils import LOGGER
from .contracts import ChatOpsMessage, DeliveryResult, NotificationTarget, resolve_notification_target
from .message_formatter import build_discord_embed, build_openclaw_payload


def _build_ssl_context() -> Optional[ssl.SSLContext]:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return None


def _post_json(webhook_url: str, payload: dict) -> DeliveryResult:
    request = urllib.request.Request(
        url=webhook_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "OpenClawHire-Fraud-ChatOps/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10, context=_build_ssl_context()) as response:
            status = getattr(response, "status", 200)
        if status >= 400:
            return DeliveryResult(delivered=False, delivery_error=f"Webhook returned status {status}")
        return DeliveryResult(delivered=True)
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8")
        except Exception:
            detail = ""
        return DeliveryResult(delivered=False, delivery_error=f"Webhook returned {exc.code}: {detail[:200]}")
    except Exception as exc:
        return DeliveryResult(delivered=False, delivery_error=str(exc))


def build_webhook_payload(message: ChatOpsMessage, target: NotificationTarget) -> dict:
    webhook_format = (target.webhook_format or config.OPENCLAW_WEBHOOK_FORMAT or "discord").lower()
    if webhook_format == "discord":
        return {
            "content": message.title,
            "embeds": [build_discord_embed(message)],
        }

    payload = build_openclaw_payload(message)
    if target.channel_id:
        payload["channelId"] = target.channel_id
    if target.conversation_id:
        payload["conversationId"] = target.conversation_id
    if target.actor_id:
        payload["actorId"] = target.actor_id
    return payload


def deliver_message(
    message: ChatOpsMessage,
    *,
    target: Optional[NotificationTarget] = None,
    dry_run: bool = False,
) -> DeliveryResult:
    resolved_target = resolve_notification_target(target)
    if not config.OPENCLAW_ENABLED:
        return DeliveryResult(delivered=False, delivery_error="OpenClaw integration is disabled in config.")
    if resolved_target is None or not resolved_target.webhook_url:
        return DeliveryResult(delivered=False, delivery_error="No ChatOps webhook target is configured.")

    payload = build_webhook_payload(message, resolved_target)
    if dry_run:
        return DeliveryResult(
            delivered=False,
            webhook_format=resolved_target.webhook_format,
            delivery_error="dry_run",
            payload_preview=payload,
        )

    result = _post_json(resolved_target.webhook_url, payload)
    result.webhook_format = resolved_target.webhook_format
    if result.delivered:
        LOGGER.info("Delivered ChatOps message '%s' via %s", message.message_type, resolved_target.webhook_format or "webhook")
    else:
        LOGGER.warning("ChatOps delivery failed for '%s': %s", message.message_type, result.delivery_error)
    return result
