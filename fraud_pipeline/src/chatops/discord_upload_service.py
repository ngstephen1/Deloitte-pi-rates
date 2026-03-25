"""
Discord CSV upload helpers for fraud-analysis workflows.
"""

from __future__ import annotations

import io
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .. import config
from ..ai_assistant import (
    answer_data_question,
    generate_ai_recommendations,
    generate_multi_agent_oof_brief,
    rule_based_recommendations,
)
from ..dashboard_data import CSV_TYPE_EXPECTATIONS, bundle_from_uploaded_csv, validate_uploaded_csv
from ..utils import LOGGER
from .context_loader import build_review_summary, publish_bundle_context
from .message_formatter import build_report_message


ACTION_REPORT = "report"
ACTION_ANNOTATED = "annotated_csv"
ACTION_CLEANED = "cleaned_csv"


def _safe_slug(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", text.strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned or "upload"


def _load_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def _infer_csv_type(df: pd.DataFrame) -> Dict[str, Any]:
    normalized = df.copy()
    normalized.columns = normalized.columns.str.strip().str.lower().str.replace(" ", "_")

    scored: list[Dict[str, Any]] = []
    for csv_type, expected_columns in CSV_TYPE_EXPECTATIONS.items():
        missing = [column for column in expected_columns if column not in normalized.columns]
        scored.append(
            {
                "csv_type": csv_type,
                "missing_columns": missing,
                "matched_columns": len(expected_columns) - len(missing),
            }
        )

    scored.sort(key=lambda item: (len(item["missing_columns"]), -item["matched_columns"], item["csv_type"]))
    best = scored[0] if scored else None
    return {
        "best_type": best["csv_type"] if best else None,
        "best_missing_columns": best["missing_columns"] if best else [],
        "candidates": scored,
    }


def _extract_requested_type(text: str) -> Optional[str]:
    lowered = text.lower()
    if "analyst review" in lowered or "review log" in lowered:
        return "Analyst review log"
    if "processed" in lowered or "scored" in lowered or "ranked" in lowered:
        return "Processed / scored transaction dataset"
    if "raw" in lowered or "transaction dataset" in lowered or "transaction csv" in lowered:
        return "Raw transaction dataset"
    return None


def _parse_requested_actions(text: str) -> Dict[str, Any]:
    lowered = text.lower()
    wants_report = any(token in lowered for token in ["report", "summary", "summarize", "analysis", "analyze", "recommend", "recommendation"])
    wants_annotated = any(token in lowered for token in ["annotated", "annotate", "annotated csv", "tag each row", "row recommendation"])
    wants_cleaned = any(token in lowered for token in ["cleaned", "clean", "normalized"])
    wants_both = "both" in lowered or "all" in lowered or ("report" in lowered and "csv" in lowered)

    requested_actions: list[str] = []
    if wants_report or wants_both or not lowered.strip():
        requested_actions.append(ACTION_REPORT)
    if wants_annotated or wants_both:
        requested_actions.append(ACTION_ANNOTATED)
    if wants_cleaned:
        requested_actions.append(ACTION_CLEANED)

    if not requested_actions:
        requested_actions = [ACTION_REPORT]

    generic_prompts = {
        "",
        "analyze",
        "analyse",
        "review",
        "check this",
        "look at this",
        "help",
        "please analyze",
        "please review",
    }
    normalized_prompt = " ".join(lowered.split())
    has_goal = normalized_prompt not in generic_prompts and len(normalized_prompt.split()) >= 4

    return {
        "requested_actions": requested_actions,
        "has_goal": has_goal,
        "goal_text": text.strip(),
    }


def _markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 8) -> str:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return "_No rows available._"
    available = [column for column in columns if column in df.columns]
    if not available:
        return "_No matching columns available._"

    preview = df.head(max_rows)[available].copy()
    for column in preview.columns:
        preview[column] = preview[column].map(
            lambda value: f"{float(value):.3f}" if isinstance(value, float) else str(value)
        )

    header = "| " + " | ".join(available) + " |"
    separator = "| " + " | ".join(["---"] * len(available)) + " |"
    rows = ["| " + " | ".join(str(row[column]) for column in available) + " |" for _, row in preview.iterrows()]
    return "\n".join([header, separator, *rows])


def _row_recommendation(row: pd.Series) -> tuple[str, str]:
    risk_level = str(row.get("risk_level", "")).strip().lower()
    score = float(row.get("composite_risk_score", 0) or 0)
    if risk_level == "high" or score >= 0.8:
        return "Immediate Review", "High composite risk or high-risk tier."
    if risk_level == "medium" or score >= config.RISK_LEVEL_LOW:
        return "Needs Review", "Flagged by the composite risk threshold."
    return "Monitor", "Lower relative risk within the current portfolio."


def _review_focus(row: pd.Series) -> str:
    focus_parts = []
    for label, column in [
        ("merchant", "merchantid"),
        ("location", "location"),
        ("device", "deviceid"),
        ("channel", "channel"),
    ]:
        value = row.get(column)
        if pd.notna(value) and str(value).strip():
            focus_parts.append(f"{label}:{value}")
    return ", ".join(focus_parts[:3]) or "transaction context"


def build_annotated_transactions_csv(bundle: Dict[str, Any]) -> pd.DataFrame:
    transactions = bundle.get("transactions", pd.DataFrame())
    if not isinstance(transactions, pd.DataFrame) or transactions.empty:
        return pd.DataFrame()

    annotated = transactions.copy()
    if "composite_risk_score" in annotated.columns:
        annotated = annotated.sort_values("composite_risk_score", ascending=False).reset_index(drop=True)
    annotated["case_id"] = annotated.get("transactionid", pd.Series(range(1, len(annotated) + 1))).astype(str)
    annotated["review_priority"] = range(1, len(annotated) + 1)
    decisions = annotated.apply(_row_recommendation, axis=1, result_type="expand")
    annotated["recommended_action"] = decisions[0]
    annotated["recommendation_reason"] = decisions[1]
    annotated["review_focus"] = annotated.apply(_review_focus, axis=1)
    return annotated


def build_annotated_review_log_csv(review_log: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(review_log, pd.DataFrame) or review_log.empty:
        return pd.DataFrame()
    annotated = review_log.copy()
    annotated["case_id"] = annotated.get("case_id", annotated.get("transactionid", pd.Series(dtype=str))).astype(str)
    annotated["follow_up_action"] = annotated["decision"].map(
        {
            "Approve Flag": "Escalate flagged case to the next control layer.",
            "Dismiss": "Retain the case history and no further action is required unless new signals emerge.",
            "Needs Review": "Keep the case open and prioritize a documented analyst follow-up.",
        }
    ).fillna("Review the case history and confirm whether additional analyst action is needed.")
    return annotated


def _report_summary_lines(bundle: Dict[str, Any], recommendations: Dict[str, Any]) -> list[str]:
    summary = bundle.get("summary", {}) or {}
    lines = [
        f"Transactions: {summary.get('total_transactions', 0):,}",
        f"Flagged transactions: {summary.get('flagged_transactions', 0):,}",
        f"High-risk transactions: {summary.get('high_risk_count', 0):,}",
        f"High-risk accounts: {summary.get('high_risk_accounts', 0):,}",
        f"Pending review: {summary.get('pending_review_total', 0):,}",
    ]
    lines.extend(recommendations.get("baseline_recommendations", [])[:3])
    lines.extend(recommendations.get("ai_recommendations", [])[:2])
    return lines[:7]


def build_markdown_report(
    *,
    file_name: str,
    csv_type: str,
    bundle: Dict[str, Any],
    validation: Dict[str, Any],
    recommendations: Dict[str, Any],
    requested_goal: str,
) -> str:
    source_label = bundle.get("source_label", file_name)
    summary = bundle.get("summary", {}) or {}
    transactions = bundle.get("transactions", pd.DataFrame())
    accounts = bundle.get("accounts", pd.DataFrame())
    merchants = bundle.get("merchants", pd.DataFrame())
    review_log = bundle.get("review_log", pd.DataFrame())

    sections = [
        "# Fraud Analysis Report",
        "",
        f"- Source: `{source_label}`",
        f"- CSV type: `{csv_type}`",
        f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Requested goal: {requested_goal or 'General fraud analysis and recommendations'}",
        "",
        "## Validation",
        "",
        f"- Status: {'Valid' if validation.get('is_valid') else 'Invalid'}",
        f"- Missing required columns: {', '.join(validation.get('missing_columns', [])) or 'None'}",
        f"- Detected columns: {len(validation.get('normalized_df', pd.DataFrame()).columns):,}",
        f"- Row count: {len(validation.get('normalized_df', pd.DataFrame())):,}",
        "",
    ]

    if csv_type == "Analyst review log":
        review_summary = build_review_summary(pd.DataFrame(), review_log)
        sections.extend(
            [
                "## Review Summary",
                "",
                f"- Approved: {review_summary.get('approved_count', 0):,}",
                f"- Dismissed: {review_summary.get('dismissed_count', 0):,}",
                f"- Needs review: {review_summary.get('needs_review_count', 0):,}",
                f"- Total review rows: {len(review_log):,}",
                "",
                "## Recent Review History",
                "",
                _markdown_table(
                    review_log,
                    ["case_id", "transactionid", "accountid", "decision", "analyst_notes", "updated_at"],
                    max_rows=10,
                ),
            ]
        )
        return "\n".join(sections).strip() + "\n"

    report_message = build_report_message(bundle)
    sections.extend(
        [
            "## Executive Summary",
            "",
            f"- Transactions: {summary.get('total_transactions', len(transactions)):,}",
            f"- Flagged transactions: {summary.get('flagged_transactions', 0):,}",
            f"- High-risk transactions: {summary.get('high_risk_count', 0):,}",
            f"- High-risk accounts: {summary.get('high_risk_accounts', 0):,}",
            f"- High-risk merchants: {summary.get('high_risk_merchants', 0):,}",
            f"- High-risk devices: {summary.get('high_risk_devices', 0):,}",
            f"- High-risk locations: {summary.get('high_risk_locations', 0):,}",
            "",
            "## Analyst Take",
            "",
            report_message.text,
            "",
            "## Recommendations",
            "",
        ]
    )

    recommendation_lines = recommendations.get("ai_recommendations") or recommendations.get("baseline_recommendations") or rule_based_recommendations(bundle)
    sections.extend([f"- {item}" for item in recommendation_lines[:5]] or ["- No recommendation output available."])
    sections.extend(
        [
            "",
            "## Top Suspicious Transactions",
            "",
            _markdown_table(
                transactions,
                ["transactionid", "accountid", "merchantid", "location", "channel", "composite_risk_score", "risk_level"],
                max_rows=10,
            ),
            "",
            "## Top Risky Accounts",
            "",
            _markdown_table(
                accounts,
                ["accountid", "account_risk_score", "transaction_count", "high_risk_transaction_count"],
                max_rows=8,
            ),
            "",
            "## Top Risky Merchants",
            "",
            _markdown_table(
                merchants,
                ["merchantid", "avg_risk_score", "max_risk_score", "transaction_count", "high_risk_count"],
                max_rows=8,
            ),
            "",
            "## Recommended Next Step",
            "",
            report_message.next_action or "Start with the highest-ranked flagged cases and update the review log as decisions are made.",
            "",
        ]
    )
    return "\n".join(sections).strip() + "\n"


def inspect_discord_csv_upload(file_bytes: bytes, file_name: str, user_text: str = "") -> Dict[str, Any]:
    df = _load_csv_bytes(file_bytes)
    parsed_request = _parse_requested_actions(user_text)
    inferred = _infer_csv_type(df)
    requested_type = _extract_requested_type(user_text)
    selected_type = requested_type or inferred["best_type"]
    validation = validate_uploaded_csv(df, selected_type) if selected_type else {
        "normalized_df": df.copy(),
        "expected_columns": [],
        "missing_columns": [],
        "is_valid": False,
    }
    needs_clarification = (
        not parsed_request["has_goal"]
        or not selected_type
        or not validation["is_valid"]
    )

    if not selected_type:
        clarification_message = (
            f"I loaded `{file_name}` with {len(df):,} rows, but I could not confidently match it to an approved CSV preset. "
            "Tell me whether it is a raw transaction dataset, a processed / scored transaction dataset, or an analyst review log. "
            "Also tell me what you want back: a fraud report, an annotated CSV, a cleaned CSV, or both."
        )
    elif not validation["is_valid"]:
        clarification_message = (
            f"I loaded `{file_name}` and it looks closest to `{selected_type}`, but it is missing required columns: "
            f"{', '.join(validation['missing_columns'])}. "
            "Confirm the intended CSV type or upload a corrected file. Also tell me the goal, such as executive summary, analyst triage, or location review."
        )
    else:
        clarification_message = (
            f"I loaded `{file_name}` with {len(df):,} rows and {len(df.columns):,} columns. "
            f"It validates as `{selected_type}`. Before I process it, tell me the goal and desired output: "
            "`report`, `annotated csv`, `cleaned csv`, or `both`."
        )

    return {
        "dataframe": df,
        "row_count": int(len(df)),
        "columns": df.columns.astype(str).tolist(),
        "selected_type": selected_type,
        "requested_type": requested_type,
        "requested_actions": parsed_request["requested_actions"],
        "goal_text": parsed_request["goal_text"],
        "has_goal": parsed_request["has_goal"],
        "validation": validation,
        "needs_clarification": needs_clarification,
        "clarification_message": clarification_message,
        "inference": inferred,
    }


def save_uploaded_csv(file_bytes: bytes, *, file_name: str, channel_id: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = _safe_slug(Path(file_name).stem)
    path = config.CHATOPS_UPLOADS_DIR / f"{channel_id}-{timestamp}-{safe_name}.csv"
    path.write_bytes(file_bytes)
    return path


def _load_saved_csv(file_path: Path) -> pd.DataFrame:
    return pd.read_csv(file_path)


def _report_file_stem(file_name: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{_safe_slug(Path(file_name).stem)}"


def process_saved_csv_upload(
    *,
    file_path: Path,
    file_name: str,
    csv_type: str,
    requested_actions: list[str],
    goal_text: str,
) -> Dict[str, Any]:
    df = _load_saved_csv(file_path)
    validation = validate_uploaded_csv(df, csv_type)
    if not validation["is_valid"]:
        raise ValueError(
            f"Uploaded file does not validate as `{csv_type}`. Missing columns: {', '.join(validation['missing_columns'])}"
        )

    bundle = bundle_from_uploaded_csv(df, csv_type, f"Discord upload: {file_name}")
    bundle["chatops_upload_goal"] = goal_text
    manifest = publish_bundle_context(bundle, publish_reason="discord_csv_upload")

    recommendations = (
        generate_ai_recommendations(bundle)
        if csv_type != "Analyst review log"
        else {
            "ai_recommendations": [],
            "baseline_recommendations": [],
            "reminders": [],
            "availability_message": "",
            "ai_available": False,
        }
    )

    goal_response = None
    if goal_text and csv_type != "Analyst review log":
        try:
            response = answer_data_question(
                goal_text,
                bundle,
                model=config.OPENCLAW_OPENAI_MODEL,
                reasoning_effort=config.OPENCLAW_OPENAI_REASONING_EFFORT,
                max_output_tokens=config.OPENCLAW_OPENAI_MAX_OUTPUT_TOKENS,
            )
            goal_response = response["ai_answer"] or response["heuristic_answer"]
        except Exception as exc:
            LOGGER.warning("Could not build upload goal response: %s", exc)

    stem = _report_file_stem(file_name)
    export_dir = config.CHATOPS_EXPORTS_DIR / stem
    export_dir.mkdir(parents=True, exist_ok=True)

    markdown_report = build_markdown_report(
        file_name=file_name,
        csv_type=csv_type,
        bundle=bundle,
        validation=validation,
        recommendations=recommendations,
        requested_goal=goal_text,
    )
    report_path = export_dir / f"{stem}-fraud-report.md"
    report_path.write_text(markdown_report)

    files = [report_path]
    annotated_frame = pd.DataFrame()
    cleaned_path = None

    oof_brief = None
    if any(token in goal_text.lower() for token in ["oof", "executive", "brief", "oversight", "finance"]) and csv_type != "Analyst review log":
        oof_brief = generate_multi_agent_oof_brief(
            bundle,
            focus=goal_text,
            model=config.OPENCLAW_OPENAI_MODEL,
            reasoning_effort=config.OPENCLAW_OPENAI_REASONING_EFFORT,
        )
        oof_brief_path = export_dir / f"{stem}-oof-brief.md"
        oof_brief_path.write_text(oof_brief["brief_markdown"])
        files.append(oof_brief_path)

    if csv_type == "Analyst review log":
        annotated_frame = build_annotated_review_log_csv(bundle.get("review_log", pd.DataFrame()))
    else:
        annotated_frame = build_annotated_transactions_csv(bundle)

    if ACTION_ANNOTATED in requested_actions and not annotated_frame.empty:
        annotated_path = export_dir / f"{stem}-annotated.csv"
        annotated_frame.to_csv(annotated_path, index=False)
        files.append(annotated_path)

    cleaned_frame = pd.DataFrame()
    if ACTION_CLEANED in requested_actions:
        if csv_type == "Raw transaction dataset":
            cleaned_frame = bundle.get("cleaned_transactions", pd.DataFrame())
        elif csv_type == "Analyst review log":
            cleaned_frame = validation["normalized_df"].copy()
        else:
            cleaned_frame = validation["normalized_df"].copy()
        if isinstance(cleaned_frame, pd.DataFrame) and not cleaned_frame.empty:
            cleaned_path = export_dir / f"{stem}-cleaned.csv"
            cleaned_frame.to_csv(cleaned_path, index=False)
            files.append(cleaned_path)

    summary_lines = _report_summary_lines(bundle, recommendations)
    if goal_response:
        summary_lines.insert(0, goal_response)

    reply_lines = [
        f"I processed `{file_name}` as `{csv_type}` and published it as the active fraud context for this Discord thread.",
        *[f"- {line}" for line in summary_lines[:6]],
        "I attached the generated artifacts below.",
    ]

    return {
        "bundle": bundle,
        "manifest": manifest,
        "validation": validation,
        "csv_type": csv_type,
        "requested_actions": requested_actions,
        "goal_text": goal_text,
        "reply_text": "\n".join(reply_lines),
        "goal_response": goal_response,
        "recommendations": recommendations,
        "report_path": report_path,
        "files": files,
        "export_dir": export_dir,
        "cleaned_path": cleaned_path,
        "oof_brief": oof_brief,
    }
