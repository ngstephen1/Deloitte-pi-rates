"""
OpenClaw-style ChatOps contracts used by the fraud monitoring integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .. import config

WEBHOOK_FORMATS = {"openclaw", "discord"}


def _first_non_empty(*values: Optional[str]) -> Optional[str]:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _normalize_webhook_format(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = value.strip().lower()
    return normalized if normalized in WEBHOOK_FORMATS else None


@dataclass
class NotificationTarget:
    webhook_url: Optional[str] = None
    webhook_format: Optional[str] = None
    channel_id: Optional[str] = None
    conversation_id: Optional[str] = None
    actor_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeliveryResult:
    delivered: bool
    webhook_format: Optional[str] = None
    delivery_error: Optional[str] = None
    payload_preview: Optional[Dict[str, Any]] = None


@dataclass
class ChatOpsMessage:
    message_type: str
    title: str
    text: str
    severity: str = "info"
    facts: list[str] = field(default_factory=list)
    table_title: Optional[str] = None
    table_text: Optional[str] = None
    next_action: Optional[str] = None
    source_label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def get_default_notification_target() -> Optional[NotificationTarget]:
    webhook_url = _first_non_empty(
        config.OPENCLAW_WEBHOOK_URL,
        config.OPENCLAW_DEFAULT_WEBHOOK_URL,
        config.OPENCLAW_DISCORD_WEBHOOK_URL,
    )
    webhook_format = _normalize_webhook_format(
        _first_non_empty(
            config.OPENCLAW_WEBHOOK_FORMAT,
            "discord" if webhook_url == config.OPENCLAW_DISCORD_WEBHOOK_URL and webhook_url else None,
        )
    )
    channel_id = _first_non_empty(config.OPENCLAW_DEFAULT_CHANNEL_ID)
    conversation_id = _first_non_empty(config.OPENCLAW_DEFAULT_CONVERSATION_ID)
    actor_id = _first_non_empty(config.OPENCLAW_DEFAULT_ACTOR_ID)

    if not webhook_url and not channel_id and not conversation_id and not actor_id:
        return None

    return NotificationTarget(
        webhook_url=webhook_url,
        webhook_format=webhook_format,
        channel_id=channel_id,
        conversation_id=conversation_id,
        actor_id=actor_id,
    )


def merge_notification_targets(*targets: Optional[NotificationTarget]) -> Optional[NotificationTarget]:
    merged = NotificationTarget()
    for target in targets:
        if target is None:
            continue
        merged.webhook_url = target.webhook_url or merged.webhook_url
        merged.webhook_format = target.webhook_format or merged.webhook_format
        merged.channel_id = target.channel_id or merged.channel_id
        merged.conversation_id = target.conversation_id or merged.conversation_id
        merged.actor_id = target.actor_id or merged.actor_id
        merged.metadata.update(target.metadata or {})

    if not merged.webhook_url and not merged.channel_id and not merged.conversation_id and not merged.actor_id:
        return None
    return merged


def resolve_notification_target(*targets: Optional[NotificationTarget]) -> Optional[NotificationTarget]:
    return merge_notification_targets(get_default_notification_target(), *targets)
