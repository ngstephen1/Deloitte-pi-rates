"""
Shared Discord state helpers for case threads and channel workspaces.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .. import config


def read_discord_state() -> Dict[str, Any]:
    if not config.CHATOPS_DISCORD_STATE_FILE.exists():
        return {"channels": {}, "case_threads": {}}
    try:
        state = json.loads(config.CHATOPS_DISCORD_STATE_FILE.read_text())
    except Exception:
        return {"channels": {}, "case_threads": {}}

    state.setdefault("channels", {})
    state.setdefault("case_threads", {})
    return state


def write_discord_state(state: Dict[str, Any]) -> None:
    config.CHATOPS_DISCORD_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state.setdefault("channels", {})
    state.setdefault("case_threads", {})
    config.CHATOPS_DISCORD_STATE_FILE.write_text(json.dumps(state, indent=2))


def get_channel_state(state: Dict[str, Any], channel_id: str) -> Dict[str, Any]:
    channels = state.setdefault("channels", {})
    channel_state = channels.setdefault(
        str(channel_id),
        {
            "last_human_at": "",
            "last_proactive_at": "",
            "last_daily_reset": "",
            "proactive_count_today": 0,
            "reminder_cursor": 0,
            "pending_upload": None,
            "pending_image": None,
            "workspace": {
                "uploaded_csv_history": [],
                "uploaded_image_history": [],
                "last_goal": "",
                "last_discussed_case": {},
                "last_command": "",
                "analyst_intent": "",
                "last_image_analysis": {},
            },
        },
    )
    channel_state.setdefault("pending_upload", None)
    channel_state.setdefault("pending_image", None)
    workspace = channel_state.setdefault("workspace", {})
    workspace.setdefault("uploaded_csv_history", [])
    workspace.setdefault("uploaded_image_history", [])
    workspace.setdefault("last_goal", "")
    workspace.setdefault("last_discussed_case", {})
    workspace.setdefault("last_command", "")
    workspace.setdefault("analyst_intent", "")
    workspace.setdefault("last_image_analysis", {})
    return channel_state


def update_channel_workspace(
    state: Dict[str, Any],
    *,
    channel_id: str,
    last_command: Optional[str] = None,
    last_goal: Optional[str] = None,
    analyst_intent: Optional[str] = None,
    last_discussed_case: Optional[Dict[str, Any]] = None,
    uploaded_csv_entry: Optional[Dict[str, Any]] = None,
    uploaded_image_entry: Optional[Dict[str, Any]] = None,
    last_image_analysis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    channel_state = get_channel_state(state, channel_id)
    workspace = channel_state.setdefault("workspace", {})

    if last_command is not None:
        workspace["last_command"] = last_command
    if last_goal is not None:
        workspace["last_goal"] = last_goal
    if analyst_intent is not None:
        workspace["analyst_intent"] = analyst_intent
    if last_discussed_case is not None:
        workspace["last_discussed_case"] = last_discussed_case
    if uploaded_csv_entry is not None:
        history = workspace.setdefault("uploaded_csv_history", [])
        history.append(uploaded_csv_entry)
        workspace["uploaded_csv_history"] = history[-6:]
    if uploaded_image_entry is not None:
        history = workspace.setdefault("uploaded_image_history", [])
        history.append(uploaded_image_entry)
        workspace["uploaded_image_history"] = history[-6:]
    if last_image_analysis is not None:
        workspace["last_image_analysis"] = last_image_analysis

    return channel_state


def build_case_key(case_type: str, case_id: str) -> str:
    return f"{case_type}::{case_id}"


def get_case_thread(state: Dict[str, Any], case_type: str, case_id: str) -> Optional[Dict[str, Any]]:
    return (state.setdefault("case_threads", {})).get(build_case_key(case_type, case_id))


def upsert_case_thread(
    state: Dict[str, Any],
    *,
    case_type: str,
    case_id: str,
    thread_id: str,
    channel_id: str,
    thread_name: str,
    source_message_id: Optional[str] = None,
) -> Dict[str, Any]:
    case_threads = state.setdefault("case_threads", {})
    current = case_threads.get(build_case_key(case_type, case_id), {})
    current.update(
        {
            "case_type": case_type,
            "case_id": case_id,
            "thread_id": str(thread_id),
            "channel_id": str(channel_id),
            "thread_name": thread_name,
            "source_message_id": str(source_message_id) if source_message_id else current.get("source_message_id"),
            "last_activity_at": datetime.now(timezone.utc).isoformat(),
            "last_human_touch_at": current.get("last_human_touch_at", ""),
            "created_at": current.get("created_at") or datetime.now(timezone.utc).isoformat(),
        }
    )
    case_threads[build_case_key(case_type, case_id)] = current
    return current


def touch_case_thread(state: Dict[str, Any], *, case_type: str, case_id: str) -> Optional[Dict[str, Any]]:
    current = get_case_thread(state, case_type, case_id)
    if not current:
        return None
    current["last_activity_at"] = datetime.now(timezone.utc).isoformat()
    return current


def mark_case_thread_human_touch(state: Dict[str, Any], *, case_type: str, case_id: str) -> Optional[Dict[str, Any]]:
    current = get_case_thread(state, case_type, case_id)
    if not current:
        return None
    now = datetime.now(timezone.utc).isoformat()
    current["last_activity_at"] = now
    current["last_human_touch_at"] = now
    return current
