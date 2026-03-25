"""
Delivery bridge for Discord/OpenClaw-style webhook notifications.
"""

from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

from .. import config
from ..utils import LOGGER
from .contracts import ChatOpsMessage, DeliveryResult, NotificationTarget, resolve_notification_target
from .discord_state import get_case_thread, read_discord_state, touch_case_thread, upsert_case_thread, write_discord_state
from .message_formatter import build_discord_embed, build_openclaw_payload


def _build_ssl_context() -> Optional[ssl.SSLContext]:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return None


def _build_webhook_url(webhook_url: str, *, thread_id: str | None = None, wait: bool = False) -> str:
    parsed = urllib.parse.urlparse(webhook_url)
    query = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
    if thread_id:
        query["thread_id"] = thread_id
    if wait:
        query["wait"] = "true"
    return urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(query)))


def _post_json(webhook_url: str, payload: dict, *, thread_id: str | None = None, wait: bool = False) -> DeliveryResult:
    final_url = _build_webhook_url(webhook_url, thread_id=thread_id, wait=wait)
    request = urllib.request.Request(
        url=final_url,
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
            body = response.read().decode("utf-8") if wait else ""
        if status >= 400:
            return DeliveryResult(delivered=False, delivery_error=f"Webhook returned status {status}")
        response_data = None
        if body:
            try:
                response_data = json.loads(body)
            except Exception:
                response_data = None
        return DeliveryResult(delivered=True, response_data=response_data)
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8")
        except Exception:
            detail = ""
        return DeliveryResult(delivered=False, delivery_error=f"Webhook returned {exc.code}: {detail[:200]}")
    except Exception as exc:
        return DeliveryResult(delivered=False, delivery_error=str(exc))


def _case_identity_from_message(message: ChatOpsMessage) -> tuple[str | None, str | None]:
    metadata = message.metadata or {}
    case_type = metadata.get("case_type") or metadata.get("entity_type")
    case_id = metadata.get("case_id") or metadata.get("entity_id")
    if not case_type or not case_id:
        return None, None
    return str(case_type), str(case_id)


def _should_create_case_thread(message: ChatOpsMessage, case_type: str | None) -> bool:
    if case_type not in {"transaction", "account"}:
        return False
    return message.message_type.startswith("fraud.alert") or message.message_type.startswith("fraud.review") or message.message_type.startswith("fraud.reminder.case")


def _resolve_case_thread_target(message: ChatOpsMessage, target: NotificationTarget) -> NotificationTarget:
    case_type, case_id = _case_identity_from_message(message)
    if not case_type or not case_id:
        return target

    state = read_discord_state()
    existing = get_case_thread(state, case_type, case_id)
    if not existing:
        return target

    touch_case_thread(state, case_type=case_type, case_id=case_id)
    write_discord_state(state)
    return NotificationTarget(
        webhook_url=target.webhook_url,
        webhook_format=target.webhook_format,
        channel_id=existing.get("channel_id") or target.channel_id,
        conversation_id=target.conversation_id,
        thread_id=existing.get("thread_id"),
        actor_id=target.actor_id,
        metadata=target.metadata,
    )


def _discord_api_request(url: str, payload: dict) -> Optional[Dict[str, object]]:
    token = os.environ.get("DISCORD_BOT_TOKEN")
    if not token:
        return None
    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bot {token}",
            "User-Agent": "OpenClawHire-Fraud-ChatOps/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10, context=_build_ssl_context()) as response:
            body = response.read().decode("utf-8")
        return json.loads(body) if body else {}
    except Exception as exc:
        LOGGER.warning("Discord thread API request failed: %s", exc)
        return None


def _thread_name_for_case(message: ChatOpsMessage, case_type: str, case_id: str) -> str:
    prefix = "Case"
    if case_type == "transaction":
        prefix = "Transaction"
    elif case_type == "account":
        prefix = "Account"
    return f"{prefix} {case_id}"


def _ensure_case_thread_for_delivery(message: ChatOpsMessage, target: NotificationTarget, delivery: DeliveryResult) -> None:
    case_type, case_id = _case_identity_from_message(message)
    if not _should_create_case_thread(message, case_type):
        return
    if not delivery.delivered or not delivery.response_data:
        return

    state = read_discord_state()
    if get_case_thread(state, case_type, case_id):
        touch_case_thread(state, case_type=case_type, case_id=case_id)
        write_discord_state(state)
        return

    channel_id = str(delivery.response_data.get("channel_id") or target.channel_id or "")
    message_id = str(delivery.response_data.get("id") or "")
    if not channel_id or not message_id:
        return

    thread_payload = {
        "name": _thread_name_for_case(message, case_type, case_id),
        "auto_archive_duration": config.OPENCLAW_DISCORD_THREAD_AUTO_ARCHIVE_MINUTES,
    }
    created = _discord_api_request(
        f"https://discord.com/api/v10/channels/{channel_id}/messages/{message_id}/threads",
        thread_payload,
    )
    if not created or "id" not in created:
        return

    upsert_case_thread(
        state,
        case_type=case_type,
        case_id=case_id,
        thread_id=str(created["id"]),
        channel_id=channel_id,
        thread_name=str(created.get("name") or thread_payload["name"]),
        source_message_id=message_id,
    )
    write_discord_state(state)


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

    resolved_target = _resolve_case_thread_target(message, resolved_target)
    payload = build_webhook_payload(message, resolved_target)
    if dry_run:
        return DeliveryResult(
            delivered=False,
            webhook_format=resolved_target.webhook_format,
            delivery_error="dry_run",
            payload_preview=payload,
        )

    wait_for_response = (
        (resolved_target.webhook_format or config.OPENCLAW_WEBHOOK_FORMAT or "discord").lower() == "discord"
        and bool(_should_create_case_thread(message, _case_identity_from_message(message)[0]))
        and not bool(resolved_target.thread_id)
    )
    result = _post_json(
        resolved_target.webhook_url,
        payload,
        thread_id=resolved_target.thread_id,
        wait=wait_for_response,
    )
    result.webhook_format = resolved_target.webhook_format
    if result.delivered:
        _ensure_case_thread_for_delivery(message, resolved_target, result)
        LOGGER.info("Delivered ChatOps message '%s' via %s", message.message_type, resolved_target.webhook_format or "webhook")
    else:
        LOGGER.warning("ChatOps delivery failed for '%s': %s", message.message_type, result.delivery_error)
    return result
