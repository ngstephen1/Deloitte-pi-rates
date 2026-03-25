"""
Optional bridge to the real OpenClaw CLI agent runtime.

This lets the Discord companion bot hand final reply synthesis to OpenClaw
while still grounding the answer in the local fraud outputs first.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any, Dict, Optional

from .. import config
from ..ai_assistant import bundle_context_summary
from ..utils import LOGGER


def openclaw_agent_enabled() -> bool:
    return bool(config.OPENCLAW_USE_AGENT_RUNTIME)


def openclaw_agent_available() -> bool:
    return shutil.which(config.OPENCLAW_CLI_LAUNCHER) is not None


def _openclaw_command(prompt: str) -> list[str]:
    if config.OPENCLAW_CLI_LAUNCHER == "npx":
        return [
            config.OPENCLAW_CLI_LAUNCHER,
            config.OPENCLAW_CLI_PACKAGE,
            "agent",
            "--agent",
            config.OPENCLAW_AGENT,
            "--message",
            prompt,
            "--thinking",
            config.OPENCLAW_DISCORD_THINKING,
            "--json",
        ]
    return [
        config.OPENCLAW_CLI_LAUNCHER,
        "agent",
        "--agent",
        config.OPENCLAW_AGENT,
        "--message",
        prompt,
        "--thinking",
        config.OPENCLAW_DISCORD_THINKING,
        "--json",
    ]


def _extract_agent_text(stdout: str) -> str:
    cleaned = stdout.strip()
    if not cleaned:
        return ""
    try:
        parsed = json.loads(cleaned)
    except Exception:
        return cleaned

    payloads = (
        parsed.get("result", {}).get("payloads")
        or parsed.get("payloads")
        or parsed.get("result", {}).get("outputs")
        or []
    )
    fragments: list[str] = []
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        text = payload.get("text") or payload.get("content") or payload.get("message")
        if isinstance(text, str) and text.strip():
            fragments.append(text.strip())
    if fragments:
        return "\n\n".join(fragments)

    fallback = parsed.get("result", {}).get("text") or parsed.get("text")
    return str(fallback).strip() if fallback else cleaned


def run_openclaw_agent(prompt: str) -> str:
    command = _openclaw_command(prompt)
    result = subprocess.run(
        command,
        cwd=str(config.PROJECT_ROOT.parent),
        env=None,
        capture_output=True,
        text=True,
        timeout=max(10, int(config.OPENCLAW_DISCORD_AGENT_TIMEOUT_MS / 1000)),
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(stderr or f"OpenClaw agent exited with code {result.returncode}")
    text = _extract_agent_text(result.stdout or "")
    if not text:
        raise RuntimeError("OpenClaw agent returned no text payload.")
    return text


def polish_reply_with_openclaw(
    *,
    user_message: str,
    grounded_answer: str,
    bundle: Optional[Dict[str, Any]] = None,
    transcript: str = "",
    case_type: str | None = None,
    case_id: str | None = None,
) -> Optional[str]:
    if not openclaw_agent_enabled():
        return None
    if not openclaw_agent_available():
        LOGGER.warning(
            "OpenClaw runtime is enabled but `%s` is not on PATH. Falling back to the local fraud assistant.",
            config.OPENCLAW_CLI_LAUNCHER,
        )
        return None

    case_context = f"Case context: {case_type or 'general'} {case_id or ''}".strip()
    bundle_context = bundle_context_summary(bundle or {}, detail="minimal") if bundle else "[no active fraud bundle]"
    prompt = "\n".join(
        [
            "You are answering through the real OpenClaw agent runtime for a fraud-monitoring system.",
            "Use the grounded answer below as the factual source of truth.",
            "Do not contradict the grounded answer or invent new facts.",
            "Improve clarity, structure, and analyst usefulness only.",
            "Keep the reply concise, business-facing, and suitable for one Discord assistant message.",
            "",
            case_context,
            "",
            "Fraud data context:",
            bundle_context,
            "",
            "Recent Discord transcript:",
            transcript or "[no recent transcript]",
            "",
            f"Newest user message:\n{user_message}",
            "",
            f"Grounded answer:\n{grounded_answer}",
            "",
            "Return one final Discord reply only.",
        ]
    )
    try:
        return run_openclaw_agent(prompt)
    except Exception as exc:
        LOGGER.warning("OpenClaw agent reply synthesis failed: %s", exc)
        return None
