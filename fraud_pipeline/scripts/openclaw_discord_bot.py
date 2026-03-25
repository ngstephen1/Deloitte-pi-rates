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
from typing import Any, Dict, List

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
from src.chatops.discord_upload_service import inspect_discord_csv_upload, process_saved_csv_upload, save_uploaded_csv
from src.chatops.message_formatter import build_case_reminder_message, build_report_message
from src.chatops.query_service import answer_analyst_question


def _parse_id_set(raw_value: str) -> set[str]:
    return {item.strip() for item in (raw_value or "").split(",") if item.strip()}


def _read_state() -> Dict[str, Any]:
    if not config.CHATOPS_DISCORD_STATE_FILE.exists():
        return {"channels": {}}
    try:
        return json.loads(config.CHATOPS_DISCORD_STATE_FILE.read_text())
    except Exception:
        return {"channels": {}}


def _write_state(state: Dict[str, Any]) -> None:
    config.CHATOPS_DISCORD_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.CHATOPS_DISCORD_STATE_FILE.write_text(json.dumps(state, indent=2))


def _get_channel_state(state: Dict[str, Any], channel_id: str) -> Dict[str, Any]:
    channels = state.setdefault("channels", {})
    channel_state = channels.setdefault(
        channel_id,
        {
            "last_human_at": "",
            "last_proactive_at": "",
            "last_daily_reset": "",
            "proactive_count_today": 0,
            "reminder_cursor": 0,
            "pending_upload": None,
        },
    )
    channel_state.setdefault("pending_upload", None)
    return channel_state


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


def _summarize_columns(columns: list[str], max_items: int = 8) -> str:
    if not columns:
        return "(none)"
    preview = columns[:max_items]
    suffix = "..." if len(columns) > max_items else ""
    return ", ".join(preview) + suffix


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


def _find_csv_attachment(message: discord.Message) -> discord.Attachment | None:
    for attachment in message.attachments:
        name = (attachment.filename or attachment.url or "").lower()
        content_type = (attachment.content_type or "").lower()
        if name.endswith(".csv") or content_type in {"text/csv", "application/csv", "application/vnd.ms-excel"}:
            return attachment
    return None


def _looks_like_upload_instruction(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in lowered
        for token in [
            "report",
            "summary",
            "summarize",
            "analysis",
            "analyze",
            "annotated",
            "annotate",
            "csv",
            "cleaned",
            "clean",
            "both",
            "raw",
            "processed",
            "scored",
            "review log",
            "recommend",
            "executive",
            "triage",
        ]
    )


def _render_chatops_message(message) -> str:
    lines = [f"**{message.title}**", message.text]
    if getattr(message, "highlights", None):
        lines.extend(f"- {item}" for item in message.highlights)
    if getattr(message, "facts", None):
        lines.extend(f"- {fact}" for fact in message.facts)
    if getattr(message, "table_text", None):
        lines.append(f"```{message.table_text}```")
    if getattr(message, "next_action", None):
        lines.append(f"Next action: {message.next_action}")
    return "\n".join(lines)


def _format_report_text() -> str:
    bundle = load_active_bundle()
    return _render_chatops_message(build_report_message(bundle))


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


def _format_reminder_text(reminder_index: int) -> str:
    bundle = load_active_bundle()
    message = build_case_reminder_message(bundle, reminder_index=reminder_index)
    return _render_chatops_message(message)


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


async def _send_reply(message: discord.Message, text: str) -> None:
    for chunk in _split_for_discord(text):
        await message.channel.send(chunk)


async def _send_reply_with_files(message: discord.Message, text: str, files: list[Path]) -> None:
    chunks = _split_for_discord(text)
    discord_files = [discord.File(str(path), filename=path.name) for path in files]
    if chunks:
        await message.channel.send(chunks[0], files=discord_files)
        for chunk in chunks[1:]:
            await message.channel.send(chunk)
        return
    await message.channel.send(files=discord_files)


async def _process_pending_upload(
    *,
    message: discord.Message,
    state: Dict[str, Any],
    channel_state: Dict[str, Any],
    instruction_text: str,
) -> bool:
    pending = channel_state.get("pending_upload")
    if not pending:
        return False

    if not _looks_like_upload_instruction(instruction_text):
        return False

    csv_type = pending.get("selected_type")
    if not csv_type:
        await _send_reply(
            message,
            "I still need the CSV type before I can process that file. Tell me whether it is a raw transaction dataset, a processed / scored transaction dataset, or an analyst review log.",
        )
        return True

    actions = inspect_discord_csv_upload(Path(pending["file_path"]).read_bytes(), pending["file_name"], instruction_text)
    if not actions["has_goal"]:
        await _send_reply(
            message,
            "Tell me the goal for this upload too, for example `executive summary for OOF`, `merchant triage`, or `location review`, then I’ll generate the artifacts.",
        )
        return True

    requested_actions = sorted(set(pending.get("requested_actions", []) + actions["requested_actions"]))
    result = process_saved_csv_upload(
        file_path=Path(pending["file_path"]),
        file_name=pending["file_name"],
        csv_type=csv_type,
        requested_actions=requested_actions,
        goal_text=instruction_text.strip(),
    )
    channel_state["pending_upload"] = None
    _write_state(state)
    await _send_reply_with_files(message, result["reply_text"], result["files"])
    return True


async def _handle_csv_upload(
    *,
    message: discord.Message,
    state: Dict[str, Any],
    channel_state: Dict[str, Any],
    content: str,
) -> bool:
    attachment = _find_csv_attachment(message)
    if not attachment:
        return False

    file_bytes = await attachment.read()
    inspection = inspect_discord_csv_upload(file_bytes, attachment.filename, content)
    stored_path = save_uploaded_csv(file_bytes, file_name=attachment.filename, channel_id=str(message.channel.id))

    channel_state["pending_upload"] = {
        "file_path": str(stored_path),
        "file_name": attachment.filename,
        "selected_type": inspection["selected_type"],
        "requested_actions": inspection["requested_actions"],
        "row_count": inspection["row_count"],
        "columns": inspection["columns"],
    }
    _write_state(state)

    if inspection["needs_clarification"]:
        inferred_type = inspection["selected_type"] or "unknown"
        preview_text = (
            f"I loaded `{attachment.filename}` with {inspection['row_count']:,} rows and "
            f"{len(inspection['columns']):,} columns. "
            f"Closest preset: `{inferred_type}`. "
            f"Columns: {_summarize_columns(inspection['columns'])}\n\n"
            f"{inspection['clarification_message']}"
        )
        await _send_reply(message, preview_text)
        return True

    result = process_saved_csv_upload(
        file_path=stored_path,
        file_name=attachment.filename,
        csv_type=inspection["selected_type"],
        requested_actions=inspection["requested_actions"],
        goal_text=inspection["goal_text"],
    )
    channel_state["pending_upload"] = None
    _write_state(state)
    await _send_reply_with_files(message, result["reply_text"], result["files"])
    return True


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

    if config.DISCORD_REPLY_ONLY_ON_MENTION:
        if client.user not in message.mentions:
            return
        content = _strip_mention(content, client.user.id)

    try:
        if await _handle_csv_upload(message=message, state=state, channel_state=channel_state, content=content):
            return

        if channel_state.get("pending_upload") and content and await _process_pending_upload(
            message=message,
            state=state,
            channel_state=channel_state,
            instruction_text=content,
        ):
            return
    except Exception as exc:
        await _send_reply(
            message,
            f"I couldn't process that CSV cleanly: {exc}",
        )
        return

    if not content:
        return

    lowered = content.lower().strip()
    if lowered in {"!help", "/help", "help"}:
        await _send_reply(
            message,
            "I can answer grounded fraud questions, send the current fraud report, show active alerts, and process CSV uploads. "
            "If you upload a CSV, I will ask what you want back if the goal is not clear. "
            "You can ask for a `report`, `annotated csv`, `cleaned csv`, or `both`.",
        )
        return

    if lowered in {"/fraud-report", "fraud report", "show fraud report"}:
        for chunk in _split_for_discord(_format_report_text()):
            await message.channel.send(chunk)
        return

    if lowered in {"!clearupload", "/clear-upload", "clear upload"}:
        channel_state["pending_upload"] = None
        _write_state(state)
        await _send_reply(message, "Cleared the pending CSV upload for this channel.")
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

        reminder_cursor = int(channel_state.get("reminder_cursor", 0) or 0)
        for chunk in _split_for_discord(_format_reminder_text(reminder_cursor)):
            await channel.send(chunk)

        channel_state["last_proactive_at"] = now.isoformat()
        channel_state["proactive_count_today"] = int(channel_state.get("proactive_count_today", 0) or 0) + 1
        channel_state["reminder_cursor"] = reminder_cursor + 1
        _write_state(state)


if __name__ == "__main__":
    client.run(token)
