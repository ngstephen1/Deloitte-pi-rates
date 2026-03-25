#!/usr/bin/env python3
"""
Discord companion bot for grounded fraud ChatOps questions and reminders.
"""

from __future__ import annotations

import asyncio
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
from src.chatops.openclaw_agent import polish_reply_with_openclaw
from src.chatops.discord_state import (
    build_case_key,
    get_case_thread,
    get_channel_state,
    mark_case_thread_human_touch,
    read_discord_state,
    update_channel_workspace,
    upsert_case_thread,
    write_discord_state,
)
from src.chatops.discord_upload_service import inspect_discord_csv_upload, process_saved_csv_upload, save_uploaded_csv
from src.chatops.image_analysis import analyze_uploaded_image
from src.chatops.image_extractors import is_supported_image_attachment, save_uploaded_image
from src.chatops.message_formatter import build_case_reminder_message, build_case_thread_title, build_oof_brief_message, build_report_message
from src.chatops.query_service import answer_analyst_question, create_oof_brief, run_command_workflow


def _parse_id_set(raw_value: str) -> set[str]:
    return {item.strip() for item in (raw_value or "").split(",") if item.strip()}


def _read_state() -> Dict[str, Any]:
    return read_discord_state()


def _write_state(state: Dict[str, Any]) -> None:
    write_discord_state(state)


def _get_channel_state(state: Dict[str, Any], channel_id: str) -> Dict[str, Any]:
    return get_channel_state(state, channel_id)


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


async def _run_blocking(func, /, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)


async def _maybe_polish_reply(
    *,
    user_message: str,
    grounded_answer: str,
    bundle: Dict[str, Any] | None = None,
    transcript: str = "",
    case_type: str | None = None,
    case_id: str | None = None,
) -> str:
    polished = await _run_blocking(
        polish_reply_with_openclaw,
        user_message=user_message,
        grounded_answer=grounded_answer,
        bundle=bundle,
        transcript=transcript,
        case_type=case_type,
        case_id=case_id,
    )
    return polished or grounded_answer


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

    if allowed_channels:
        channel_ids_to_check = {str(message.channel.id)}
        parent_id = getattr(message.channel, "parent_id", None)
        if parent_id is not None:
            channel_ids_to_check.add(str(parent_id))
        if not channel_ids_to_check.intersection(allowed_channels):
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


def _find_image_attachment(message: discord.Message) -> discord.Attachment | None:
    for attachment in message.attachments:
        if is_supported_image_attachment(attachment.filename or "", attachment.content_type or ""):
            return attachment
    return None


def _normalize_image_goal(text: str) -> str:
    normalized = text.strip()
    for prefix in ["/analyze-image", "/fraud-image-review"]:
        if normalized.lower().startswith(prefix):
            return normalized[len(prefix):].strip()
    return normalized


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


def _build_proactive_case_message(reminder_index: int, *, escalated: bool = False):
    bundle = load_active_bundle()
    return build_case_reminder_message(bundle, reminder_index=reminder_index, escalated=escalated)


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


def _is_case_channel(message: discord.Message) -> tuple[str | None, str | None]:
    state = _read_state()
    if not isinstance(message.channel, discord.Thread):
        return None, None
    thread_id = str(message.channel.id)
    for entry in state.get("case_threads", {}).values():
        if str(entry.get("thread_id")) == thread_id:
            return str(entry.get("case_type") or ""), str(entry.get("case_id") or "")
    return None, None


async def _get_or_create_case_thread(
    message: discord.Message,
    *,
    case_type: str,
    case_id: str,
) -> discord.abc.Messageable:
    state = _read_state()
    existing = get_case_thread(state, case_type, case_id)
    if existing and existing.get("thread_id"):
        thread = client.get_channel(int(existing["thread_id"]))
        if thread is None:
            try:
                thread = await client.fetch_channel(int(existing["thread_id"]))
            except Exception:
                thread = None
        if thread is not None:
            return thread

    if isinstance(message.channel, discord.Thread):
        upsert_case_thread(
            state,
            case_type=case_type,
            case_id=case_id,
            thread_id=str(message.channel.id),
            channel_id=str(message.channel.parent_id or message.channel.id),
            thread_name=message.channel.name,
            source_message_id=str(message.id),
        )
        mark_case_thread_human_touch(state, case_type=case_type, case_id=case_id)
        _write_state(state)
        return message.channel

    if not hasattr(message, "create_thread") or isinstance(message.channel, discord.DMChannel):
        return message.channel

    thread = await message.create_thread(
        name=build_case_thread_title(case_type, case_id),
        auto_archive_duration=config.OPENCLAW_DISCORD_THREAD_AUTO_ARCHIVE_MINUTES,
    )
    upsert_case_thread(
        state,
        case_type=case_type,
        case_id=case_id,
        thread_id=str(thread.id),
        channel_id=str(message.channel.id),
        thread_name=thread.name,
        source_message_id=str(message.id),
    )
    mark_case_thread_human_touch(state, case_type=case_type, case_id=case_id)
    _write_state(state)
    return thread


async def _send_text(target: discord.abc.Messageable, text: str, *, files: list[discord.File] | None = None) -> None:
    chunks = _split_for_discord(text)
    if chunks:
        await target.send(chunks[0], files=files or [])
        for chunk in chunks[1:]:
            await target.send(chunk)
    else:
        await target.send(files=files or [])


def _export_brief_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return config.CHATOPS_EXPORTS_DIR / f"{timestamp}-discord-oof-brief.md"


def _update_case_memory(
    *,
    state: Dict[str, Any],
    channel_id: str,
    case_type: str | None = None,
    case_id: str | None = None,
    last_command: str | None = None,
    analyst_intent: str | None = None,
    goal: str | None = None,
) -> None:
    last_discussed_case = None
    if case_type and case_id:
        thread = get_case_thread(state, case_type, case_id)
        last_discussed_case = {
            "case_type": case_type,
            "case_id": case_id,
            "thread_id": thread.get("thread_id") if thread else "",
        }
        mark_case_thread_human_touch(state, case_type=case_type, case_id=case_id)

    update_channel_workspace(
        state,
        channel_id=channel_id,
        last_command=last_command,
        last_goal=goal,
        analyst_intent=analyst_intent,
        last_discussed_case=last_discussed_case,
    )
    _write_state(state)


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


async def _send_image_analysis_result(message: discord.Message, analysis: Dict[str, Any]) -> None:
    files = [
        discord.File(str(analysis["export_paths"]["markdown"]), filename=Path(analysis["export_paths"]["markdown"]).name),
        discord.File(str(analysis["export_paths"]["json"]), filename=Path(analysis["export_paths"]["json"]).name),
    ]
    case_type = analysis.get("primary_case_type")
    case_id = analysis.get("primary_case_id")
    reply_text = await _maybe_polish_reply(
        user_message=str(analysis.get("user_prompt") or "Analyze this image"),
        grounded_answer=analysis["reply_text"],
        case_type=case_type,
        case_id=case_id,
    )
    if case_type and case_id:
        target = await _get_or_create_case_thread(message, case_type=case_type, case_id=case_id)
        await _send_text(target, reply_text, files=files)
        return
    await _send_text(message.channel, reply_text, files=files)


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
    async with message.channel.typing():
        result = await _run_blocking(
            process_saved_csv_upload,
            file_path=Path(pending["file_path"]),
            file_name=pending["file_name"],
            csv_type=csv_type,
            requested_actions=requested_actions,
            goal_text=instruction_text.strip(),
        )
    channel_state["pending_upload"] = None
    update_channel_workspace(
        state,
        channel_id=str(message.channel.id),
        last_goal=instruction_text.strip(),
        analyst_intent="csv_followup_processing",
        uploaded_csv_entry={
            "file_name": pending["file_name"],
            "csv_type": csv_type,
            "goal_text": instruction_text.strip(),
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "export_dir": str(result["export_dir"]),
        },
    )
    _write_state(state)
    transcript = await _build_transcript(message.channel, config.OPENCLAW_DISCORD_MAX_CONTEXT_MESSAGES)
    reply_text = await _maybe_polish_reply(
        user_message=instruction_text.strip(),
        grounded_answer=result["reply_text"],
        bundle=result.get("bundle"),
        transcript=transcript,
    )
    await _send_reply_with_files(message, reply_text, result["files"])
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
    update_channel_workspace(
        state,
        channel_id=str(message.channel.id),
        last_goal=inspection["goal_text"],
        analyst_intent="csv_upload",
        last_command="csv_upload",
    )
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

    async with message.channel.typing():
        result = await _run_blocking(
            process_saved_csv_upload,
            file_path=stored_path,
            file_name=attachment.filename,
            csv_type=inspection["selected_type"],
            requested_actions=inspection["requested_actions"],
            goal_text=inspection["goal_text"],
        )
    channel_state["pending_upload"] = None
    update_channel_workspace(
        state,
        channel_id=str(message.channel.id),
        last_goal=inspection["goal_text"],
        analyst_intent="csv_upload_processing",
        uploaded_csv_entry={
            "file_name": attachment.filename,
            "csv_type": inspection["selected_type"],
            "goal_text": inspection["goal_text"],
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "export_dir": str(result["export_dir"]),
        },
    )
    _write_state(state)
    transcript = await _build_transcript(message.channel, config.OPENCLAW_DISCORD_MAX_CONTEXT_MESSAGES)
    reply_text = await _maybe_polish_reply(
        user_message=content.strip() or attachment.filename,
        grounded_answer=result["reply_text"],
        bundle=result.get("bundle"),
        transcript=transcript,
    )
    await _send_reply_with_files(message, reply_text, result["files"])
    return True


async def _process_pending_image(
    *,
    message: discord.Message,
    state: Dict[str, Any],
    channel_state: Dict[str, Any],
    instruction_text: str,
) -> bool:
    pending = channel_state.get("pending_image")
    if not pending:
        return False

    image_path = Path(str(pending.get("file_path", "")))
    if not image_path.exists():
        channel_state["pending_image"] = None
        _write_state(state)
        await _send_reply(message, "The pending image could not be found anymore. Upload it again and I will review it.")
        return True

    goal_text = _normalize_image_goal(instruction_text) or "Analyze this image for fraud or anomaly indicators."
    async with message.channel.typing():
        analysis = await _run_blocking(analyze_uploaded_image, image_path, user_prompt=goal_text)
    channel_state["pending_image"] = None
    update_channel_workspace(
        state,
        channel_id=str(message.channel.id),
        last_goal=goal_text,
        last_command=instruction_text.strip(),
        analyst_intent="image_followup_processing",
        uploaded_image_entry={
            "file_name": pending.get("file_name", image_path.name),
            "file_path": str(image_path),
            "goal_text": goal_text,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "image_type": analysis["structured_output"].get("image_type", ""),
            "export_dir": str(config.CHATOPS_IMAGE_EXPORTS_DIR),
        },
        last_image_analysis={
            "file_name": image_path.name,
            "image_path": str(image_path),
            "image_type": analysis["structured_output"].get("image_type", ""),
            "primary_case_type": analysis.get("primary_case_type"),
            "primary_case_id": analysis.get("primary_case_id"),
        },
    )
    _write_state(state)
    await _send_image_analysis_result(message, analysis)
    return True


async def _handle_image_upload(
    *,
    message: discord.Message,
    state: Dict[str, Any],
    channel_state: Dict[str, Any],
    content: str,
) -> bool:
    attachment = _find_image_attachment(message)
    if not attachment:
        return False

    file_bytes = await attachment.read()
    stored_path = save_uploaded_image(file_bytes, file_name=attachment.filename, channel_id=str(message.channel.id))
    goal_text = _normalize_image_goal(content)

    channel_state["pending_image"] = {
        "file_path": str(stored_path),
        "file_name": attachment.filename,
    }
    update_channel_workspace(
        state,
        channel_id=str(message.channel.id),
        last_goal=goal_text,
        analyst_intent="image_upload",
        last_command=content.strip() or "image_upload",
    )
    _write_state(state)

    if not goal_text:
        await _send_reply(
            message,
            (
                f"I saved `{attachment.filename}` for review. Tell me what you want me to do with it, for example:\n"
                "- `analyze this`\n"
                "- `is this suspicious?`\n"
                "- `compare this to current flagged patterns`\n"
                "- `what should OOF do next based on this image?`"
            ),
        )
        return True

    async with message.channel.typing():
        analysis = await _run_blocking(analyze_uploaded_image, stored_path, user_prompt=goal_text)
    channel_state["pending_image"] = None
    update_channel_workspace(
        state,
        channel_id=str(message.channel.id),
        last_goal=goal_text,
        last_command=content.strip() or "image_upload",
        analyst_intent="image_upload_processing",
        uploaded_image_entry={
            "file_name": attachment.filename,
            "file_path": str(stored_path),
            "goal_text": goal_text,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "image_type": analysis["structured_output"].get("image_type", ""),
            "export_dir": str(config.CHATOPS_IMAGE_EXPORTS_DIR),
        },
        last_image_analysis={
            "file_name": attachment.filename,
            "image_path": str(stored_path),
            "image_type": analysis["structured_output"].get("image_type", ""),
            "primary_case_type": analysis.get("primary_case_type"),
            "primary_case_id": analysis.get("primary_case_id"),
        },
    )
    _write_state(state)
    await _send_image_analysis_result(message, analysis)
    return True


async def _handle_image_command(
    *,
    message: discord.Message,
    state: Dict[str, Any],
    channel_state: Dict[str, Any],
    content: str,
) -> bool:
    lowered = content.strip().lower()
    if not lowered.startswith("/analyze-image") and not lowered.startswith("/fraud-image-review"):
        return False

    workspace = channel_state.get("workspace", {}) or {}
    image_history = workspace.get("uploaded_image_history", []) or []
    last_image = (image_history[-1] if image_history else None) or (workspace.get("last_image_analysis") or {})
    image_path = Path(str(last_image.get("file_path") or last_image.get("image_path") or ""))
    if not image_path.exists():
        await _send_reply(
            message,
            "Upload an image first, or send `/analyze-image` together with a PNG, JPG, JPEG, or WEBP attachment.",
        )
        return True

    goal_text = _normalize_image_goal(content) or "Analyze this image for fraud or anomaly indicators."
    async with message.channel.typing():
        analysis = await _run_blocking(analyze_uploaded_image, image_path, user_prompt=goal_text)
    update_channel_workspace(
        state,
        channel_id=str(message.channel.id),
        last_goal=goal_text,
        last_command=content.strip(),
        analyst_intent="image_command",
        uploaded_image_entry={
            "file_name": last_image.get("file_name", image_path.name),
            "file_path": str(image_path),
            "goal_text": goal_text,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "image_type": analysis["structured_output"].get("image_type", ""),
            "export_dir": str(config.CHATOPS_IMAGE_EXPORTS_DIR),
        },
        last_image_analysis={
            "file_name": image_path.name,
            "image_path": str(image_path),
            "image_type": analysis["structured_output"].get("image_type", ""),
            "primary_case_type": analysis.get("primary_case_type"),
            "primary_case_id": analysis.get("primary_case_id"),
        },
    )
    _write_state(state)
    await _send_image_analysis_result(message, analysis)
    return True


async def _handle_command_workflow(message: discord.Message, state: Dict[str, Any], content: str) -> bool:
    if not content.strip().startswith("/"):
        return False

    if content.strip().lower() in {"/fraud-report", "/fraud-alerts"}:
        return False

    async with message.channel.typing():
        bundle = await _run_blocking(load_active_bundle)
        command_result = await _run_blocking(run_command_workflow, content, bundle=bundle)
    if not command_result.get("handled"):
        return False

    case_type = command_result.get("case_type")
    case_id = command_result.get("case_id")
    answer = command_result.get("answer", "No response available.")
    used_ai = bool(command_result.get("used_ai"))
    export_files: list[discord.File] = []
    export_path = command_result.get("export_path")
    if export_path:
        export_files.append(discord.File(str(export_path), filename=Path(export_path).name))

    if content.strip().lower().startswith("/send-oof-brief") and not export_path:
        brief_path = _export_brief_path()
        async with message.channel.typing():
            brief = await _run_blocking(create_oof_brief, bundle=bundle, focus=content.strip(), export_path=brief_path)
        answer = build_oof_brief_message(
            brief["answer"],
            source_label=bundle.get("source_label", "Fraud dashboard"),
            focus=content.strip(),
        ).text
        export_files = [discord.File(str(brief_path), filename=brief_path.name)]
        used_ai = bool(brief.get("used_ai"))

    _update_case_memory(
        state=state,
        channel_id=str(message.channel.id),
        case_type=case_type,
        case_id=case_id,
        last_command=content.strip(),
        analyst_intent="command_workflow",
        goal=content.strip(),
    )
    answer = await _maybe_polish_reply(
        user_message=content.strip(),
        grounded_answer=answer,
        bundle=bundle,
        case_type=case_type,
        case_id=case_id,
    )

    if case_type and case_id:
        target = await _get_or_create_case_thread(message, case_type=case_type, case_id=case_id)
        await _send_text(target, answer, files=export_files)
    else:
        await _send_text(message.channel, answer, files=export_files)
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

    thread_case_type, thread_case_id = _is_case_channel(message)
    if thread_case_type and thread_case_id:
        _update_case_memory(
            state=state,
            channel_id=str(message.channel.id),
            case_type=thread_case_type,
            case_id=thread_case_id,
            analyst_intent="case_thread_discussion",
        )

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

        if await _handle_image_upload(message=message, state=state, channel_state=channel_state, content=content):
            return

        if channel_state.get("pending_image") and content and await _process_pending_image(
            message=message,
            state=state,
            channel_state=channel_state,
            instruction_text=content,
        ):
            return

        if content and await _handle_image_command(
            message=message,
            state=state,
            channel_state=channel_state,
            content=content,
        ):
            return

        if content and await _handle_command_workflow(message, state, content):
            return
    except Exception as exc:
        await _send_reply(
            message,
            f"I couldn't process that upload cleanly: {exc}",
        )
        return

    if not content:
        return

    lowered = content.lower().strip()
    if lowered in {"!help", "/help", "help"}:
        await _send_reply(
            message,
            "I can answer grounded fraud questions, send the current fraud report, show active alerts, process CSV uploads, and run workflow commands. "
            "Try `/triage TX000275`, `/top-accounts`, `/pending-review`, `/merchant M026`, `/send-oof-brief`, or `/analyze-image`. "
            "If you upload a CSV, I will ask what you want back if the goal is not clear. You can ask for a `report`, `annotated csv`, `cleaned csv`, or `both`. "
            "If you upload a PNG, JPG, JPEG, or WEBP image, ask things like `analyze this`, `is this suspicious?`, `compare this to current flagged patterns`, or `what should OOF do next based on this image?`.",
        )
        return

    if lowered in {"/fraud-report", "fraud report", "show fraud report"}:
        for chunk in _split_for_discord(_format_report_text()):
            await message.channel.send(chunk)
        return

    if lowered in {"!clearupload", "/clear-upload", "clear upload"}:
        channel_state["pending_upload"] = None
        channel_state["pending_image"] = None
        _write_state(state)
        await _send_reply(message, "Cleared the pending upload state for this channel.")
        return

    if lowered in {"/fraud-alerts", "fraud alerts", "show fraud alerts"}:
        for chunk in _split_for_discord(_format_alert_digest()):
            await message.channel.send(chunk)
        return

    transcript = await _build_transcript(message.channel, config.OPENCLAW_DISCORD_MAX_CONTEXT_MESSAGES)
    async with message.channel.typing():
        response = await _run_blocking(answer_analyst_question, content, conversation_context=transcript)
        answer = await _maybe_polish_reply(
            user_message=content,
            grounded_answer=response["answer"],
            bundle=response.get("bundle"),
            transcript=transcript,
            case_type=response.get("case_type"),
            case_id=response.get("case_id"),
        )
    _update_case_memory(
        state=state,
        channel_id=str(message.channel.id),
        case_type=response.get("case_type"),
        case_id=response.get("case_id"),
        last_command=content,
        analyst_intent="natural_language_qna",
        goal=content,
    )
    if response.get("case_type") and response.get("case_id"):
        target = await _get_or_create_case_thread(message, case_type=response["case_type"], case_id=response["case_id"])
        await _send_text(target, answer)
        return
    await _send_text(message.channel, answer)


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
        reminder_message = _build_proactive_case_message(reminder_cursor, escalated=False)
        case_type = (reminder_message.metadata or {}).get("case_type")
        case_id = (reminder_message.metadata or {}).get("case_id")
        escalated = False
        if case_type and case_id:
            case_entry = get_case_thread(state, str(case_type), str(case_id))
            last_touch_raw = (case_entry or {}).get("last_human_touch_at") or (case_entry or {}).get("last_activity_at")
            if last_touch_raw:
                try:
                    last_touch_dt = datetime.fromisoformat(str(last_touch_raw))
                except Exception:
                    last_touch_dt = now - timedelta(hours=config.OPENCLAW_DISCORD_CASE_ESCALATION_HOURS + 1)
                escalated = now - last_touch_dt >= timedelta(hours=config.OPENCLAW_DISCORD_CASE_ESCALATION_HOURS)
            if escalated:
                reminder_message = _build_proactive_case_message(reminder_cursor, escalated=True)

        target_channel = channel
        if case_type and case_id:
            case_entry = get_case_thread(state, str(case_type), str(case_id))
            thread_id = (case_entry or {}).get("thread_id")
            if thread_id:
                target_channel = client.get_channel(int(thread_id)) or await client.fetch_channel(int(thread_id))

        for chunk in _split_for_discord(_render_chatops_message(reminder_message)):
            await target_channel.send(chunk)

        channel_state["last_proactive_at"] = now.isoformat()
        channel_state["proactive_count_today"] = int(channel_state.get("proactive_count_today", 0) or 0) + 1
        channel_state["reminder_cursor"] = reminder_cursor + 1
        update_channel_workspace(
            state,
            channel_id=channel_id,
            analyst_intent="proactive_case_reminder",
            last_discussed_case={"case_type": case_type, "case_id": case_id, "thread_id": str(getattr(target_channel, 'id', ''))} if case_type and case_id else None,
        )
        _write_state(state)


if __name__ == "__main__":
    client.run(token)
