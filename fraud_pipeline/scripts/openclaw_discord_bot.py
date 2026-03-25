#!/usr/bin/env python3
"""
Discord companion bot for grounded fraud ChatOps questions and reminders.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except Exception:
    pass

try:
    import discord
    from discord.ext import tasks
except Exception as exc:  # pragma: no cover - import guard for optional dependency
    raise SystemExit(
        "discord.py is required for the Discord companion bot. Install requirements first."
    ) from exc

from src import config
from src.chatops import load_active_bundle
from src.chatops.alert_service import generate_fraud_alerts
from src.chatops.message_formatter import build_reminder_message, build_report_message
from src.chatops.query_service import answer_analyst_question


def _parse_id_set(raw_value: str) -> set[str]:
    return {item.strip() for item in (raw_value or "").split(",") if item.strip()}


def _read_state() -> Dict[str, Dict[str, str]]:
    if not config.CHATOPS_DISCORD_STATE_FILE.exists():
        return {"channels": {}}
    try:
        return json.loads(config.CHATOPS_DISCORD_STATE_FILE.read_text())
    except Exception:
        return {"channels": {}}


def _write_state(state: Dict[str, Dict[str, str]]) -> None:
    config.CHATOPS_DISCORD_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.CHATOPS_DISCORD_STATE_FILE.write_text(json.dumps(state, indent=2))


def _get_channel_state(state: Dict[str, Dict[str, str]], channel_id: str) -> Dict[str, str]:
    channels = state.setdefault("channels", {})
    return channels.setdefault(
        channel_id,
        {
            "last_human_at": "",
            "last_proactive_at": "",
            "last_daily_reset": "",
            "proactive_count_today": 0,
        },
    )


def _reset_daily_counter_if_needed(channel_state: Dict[str, str], now: datetime) -> None:
    current_day = now.date().isoformat()
    if channel_state.get("last_daily_reset") != current_day:
        channel_state["last_daily_reset"] = current_day
        channel_state["proactive_count_today"] = 0


def _split_for_discord(text: str, max_chars: int = 1800) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break
        slice_text = remaining[:max_chars]
        break_index = max(slice_text.rfind("\n\n"), slice_text.rfind("\n"), slice_text.rfind(". "), slice_text.rfind(" "))
        break_index = break_index if break_index > 250 else max_chars
        chunks.append(remaining[:break_index].strip())
        remaining = remaining[break_index:].strip()
    return chunks


def _message_allowed(message: discord.Message) -> bool:
    if message.author.bot:
        return False

    if isinstance(message.channel, discord.DMChannel):
        return config.DISCORD_ALLOW_DMS

    allowed_channels = _parse_id_set(os.environ.get("DISCORD_ALLOWED_CHANNEL_IDS", ""))
    allowed_guilds = _parse_id_set(os.environ.get("DISCORD_ALLOWED_GUILD_IDS", ""))

    if allowed_channels and str(message.channel.id) not in allowed_channels:
        return False
    if allowed_guilds and message.guild and str(message.guild.id) not in allowed_guilds:
        return False
    return True


def _strip_mention(text: str, bot_user_id: int) -> str:
    return text.replace(f"<@{bot_user_id}>", "").replace(f"<@!{bot_user_id}>", "").strip()


def _format_report_text() -> str:
    bundle = load_active_bundle()
    message = build_report_message(bundle)
    lines = [f"**{message.title}**", message.text]
    lines.extend(f"- {fact}" for fact in message.facts)
    if message.table_text:
        lines.append(f"```{message.table_text}```")
    if message.next_action:
        lines.append(f"Next action: {message.next_action}")
    return "\n".join(lines)


def _format_alert_digest() -> str:
    bundle = load_active_bundle()
    alerts = generate_fraud_alerts(bundle)
    if not alerts:
        return "No threshold-based fraud alerts are currently active in the published context."
    lines = ["**Active fraud alerts**"]
    for alert in alerts[:5]:
        lines.append(
            f"- [{alert['severity'].upper()}] {alert['title']}: {alert['reason']}"
        )
    return "\n".join(lines)


def _format_reminder_text() -> str:
    bundle = load_active_bundle()
    message = build_reminder_message(bundle)
    lines = [f"**{message.title}**", message.text]
    lines.extend(f"- {fact}" for fact in message.facts)
    if message.next_action:
        lines.append(f"Next action: {message.next_action}")
    return "\n".join(lines)


async def _build_transcript(channel: discord.abc.Messageable, limit: int) -> str:
    if isinstance(channel, discord.DMChannel):
        history = []
        async for item in channel.history(limit=limit):
            if not item.content:
                continue
            history.append(f"{item.author.display_name}: {item.content}")
        return "\n".join(reversed(history))

    history = []
    async for item in channel.history(limit=limit):
        if not item.content:
            continue
        author_name = item.author.display_name if hasattr(item.author, "display_name") else item.author.name
        history.append(f"{author_name}: {item.content}")
    return "\n".join(reversed(history))


token = os.environ.get("DISCORD_BOT_TOKEN")
if not token:
    raise SystemExit("DISCORD_BOT_TOKEN is required to run the Discord companion bot.")

proactive_channel_ids = _parse_id_set(os.environ.get("OPENCLAW_DISCORD_PROACTIVE_CHANNEL_IDS", ""))

intents = discord.Intents.default()
intents.message_content = True
intents.messages = True
intents.guilds = True

client = discord.Client(intents=intents)


@client.event
async def on_ready() -> None:
    print(f"OpenClaw fraud bot connected as {client.user}")
    if config.OPENCLAW_DISCORD_PROACTIVE_ENABLED and proactive_channel_ids and not proactive_loop.is_running():
        proactive_loop.start()


@client.event
async def on_message(message: discord.Message) -> None:
    if not _message_allowed(message):
        return

    state = _read_state()
    channel_state = _get_channel_state(state, str(message.channel.id))
    channel_state["last_human_at"] = datetime.now(timezone.utc).isoformat()
    _write_state(state)

    content = message.content.strip()
    if not content:
        return

    if config.DISCORD_REPLY_ONLY_ON_MENTION:
        if client.user not in message.mentions:
            return
        content = _strip_mention(content, client.user.id)

    lowered = content.lower().strip()
    if lowered in {"/fraud-report", "fraud report", "show fraud report"}:
        for chunk in _split_for_discord(_format_report_text()):
            await message.channel.send(chunk)
        return

    if lowered in {"/fraud-alerts", "fraud alerts", "show fraud alerts"}:
        for chunk in _split_for_discord(_format_alert_digest()):
            await message.channel.send(chunk)
        return

    transcript = await _build_transcript(message.channel, config.OPENCLAW_DISCORD_MAX_CONTEXT_MESSAGES)
    response = answer_analyst_question(content, conversation_context=transcript)
    answer = response["answer"]
    for chunk in _split_for_discord(answer):
        await message.channel.send(chunk)


@tasks.loop(minutes=config.OPENCLAW_DISCORD_PROACTIVE_POLL_MINUTES)
async def proactive_loop() -> None:
    if not config.OPENCLAW_DISCORD_PROACTIVE_ENABLED:
        return

    state = _read_state()
    now = datetime.now(timezone.utc)
    for channel_id in proactive_channel_ids:
        channel = client.get_channel(int(channel_id))
        if channel is None:
            continue

        channel_state = _get_channel_state(state, channel_id)
        _reset_daily_counter_if_needed(channel_state, now)

        if int(channel_state.get("proactive_count_today", 0) or 0) >= config.OPENCLAW_DISCORD_PROACTIVE_MAX_PER_DAY:
            continue

        last_human_at = channel_state.get("last_human_at")
        if last_human_at:
            try:
                last_human_dt = datetime.fromisoformat(last_human_at)
            except Exception:
                last_human_dt = now - timedelta(days=1)
        else:
            last_human_dt = now - timedelta(days=1)

        if now - last_human_dt < timedelta(minutes=config.OPENCLAW_DISCORD_PROACTIVE_IDLE_MINUTES):
            continue

        last_proactive_at = channel_state.get("last_proactive_at")
        if last_proactive_at:
            try:
                last_proactive_dt = datetime.fromisoformat(last_proactive_at)
            except Exception:
                last_proactive_dt = now - timedelta(days=1)
        else:
            last_proactive_dt = now - timedelta(days=1)

        if now - last_proactive_dt < timedelta(minutes=config.OPENCLAW_DISCORD_PROACTIVE_MIN_INTERVAL_MINUTES):
            continue

        for chunk in _split_for_discord(_format_reminder_text()):
            await channel.send(chunk)

        channel_state["last_proactive_at"] = now.isoformat()
        channel_state["proactive_count_today"] = int(channel_state.get("proactive_count_today", 0) or 0) + 1
        _write_state(state)


if __name__ == "__main__":
    client.run(token)
