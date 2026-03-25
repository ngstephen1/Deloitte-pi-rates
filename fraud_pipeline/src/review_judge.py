"""
Structured AI-assisted review judgments for flagged fraud cases.

The judge suggests a disposition and rationale but does not write decisions
directly into the analyst log. Human reviewers remain the final control point.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from . import config
from .ai_assistant import (
    bundle_context_summary,
    is_ai_enabled,
    request_ai_response,
    summarize_case_evidence,
)
from .utils import LOGGER, save_csv


REVIEW_DECISIONS = tuple(config.DECISION_OPTIONS)


def _safe_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return 0.0
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return 0.0


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(cleaned[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _normalize_decision(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    mapping = {
        "approve": "Approve Flag",
        "approve flag": "Approve Flag",
        "escalate": "Approve Flag",
        "dismiss": "Dismiss",
        "reject": "Dismiss",
        "needs review": "Needs Review",
        "review": "Needs Review",
        "manual review": "Needs Review",
    }
    return mapping.get(normalized, "Needs Review")


def _normalize_confidence(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"low", "medium", "high"}:
        return normalized
    return "medium"


def heuristic_review_decision(entity_type: str, case_summary: Dict[str, Any], bundle: Dict[str, Any]) -> Dict[str, Any]:
    evidence = summarize_case_evidence(entity_type, case_summary, bundle)
    row = evidence.get("case_row", {}) or {}

    risk_level = str(row.get("risk_level") or case_summary.get("risk_level") or "").strip().lower()
    primary_score = _safe_float(
        row.get("composite_risk_score")
        or row.get("account_risk_score")
        or row.get("max_risk_score")
        or row.get("avg_risk_score")
        or case_summary.get("composite_risk_score")
        or case_summary.get("account_risk_score")
        or case_summary.get("max_risk_score")
        or case_summary.get("avg_risk_score")
        or 0.0
    )
    strong_driver_count = sum(
        1
        for column in [
            "isolation_forest_score",
            "lof_score",
            "kmeans_anomaly_score",
            "autoencoder_score",
            "graph_risk_score",
            "tda_risk_score",
            "login_attempt_risk",
            "amount_outlier_risk",
        ]
        if _safe_float(row.get(column, 0) or 0) >= 0.6
    )
    linked_high_risk_count = int(
        row.get("high_risk_transaction_count")
        or row.get("high_risk_count")
        or case_summary.get("high_risk_transaction_count")
        or case_summary.get("high_risk_count")
        or 0
    )

    if risk_level == "high" or primary_score >= 0.78 or strong_driver_count >= 3 or linked_high_risk_count >= 2:
        decision = "Approve Flag"
        confidence = "high" if primary_score >= 0.85 or strong_driver_count >= 4 else "medium"
        rationale = "The case is materially elevated versus the portfolio and has enough converging evidence to escalate."
    elif risk_level == "low" and primary_score <= 0.22 and strong_driver_count == 0:
        decision = "Dismiss"
        confidence = "medium"
        rationale = "The visible signals are weak relative to the rest of the portfolio and do not justify escalation."
    else:
        decision = "Needs Review"
        confidence = "medium"
        rationale = "The case is elevated but still needs manual validation of the linked entities and access pattern."

    checks = evidence.get("next_steps", [])[:3]
    if not checks:
        checks = ["Validate the visible evidence against linked entities before saving a final analyst decision."]

    return {
        "decision": decision,
        "confidence": confidence,
        "rationale": rationale,
        "checks": checks,
        "used_ai": False,
        "evidence": evidence,
    }


def judge_case_disposition(
    entity_type: str,
    case_summary: Dict[str, Any],
    bundle: Dict[str, Any],
    *,
    use_ai: bool = True,
    model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> Dict[str, Any]:
    baseline = heuristic_review_decision(entity_type, case_summary, bundle)
    if not (config.ENABLE_AI_REVIEW_JUDGE and use_ai and is_ai_enabled()):
        return baseline

    evidence = baseline["evidence"]
    prompt = (
        f"Entity type: {entity_type}\n"
        f"Case summary: {case_summary}\n"
        f"Structured evidence: {json.dumps(evidence, indent=2)}\n"
        f"Baseline heuristic judgment: {json.dumps({k: baseline[k] for k in ['decision', 'confidence', 'rationale', 'checks']}, indent=2)}\n\n"
        f"Dataset context:\n{bundle_context_summary(bundle, detail='minimal')}\n\n"
        "Return strict JSON with keys: decision, confidence, rationale, checks. "
        "Decision must be one of Approve Flag, Dismiss, Needs Review. "
        "Confidence must be one of low, medium, high. "
        "Checks must be a short list of follow-up review actions."
    )
    response_text = request_ai_response(
        instructions=(
            "You are a senior fraud review judge supporting analysts. "
            "Use only the supplied evidence. Recommend a disposition, but do not overclaim certainty. "
            "Prefer Needs Review when evidence is mixed. Output JSON only."
        ),
        prompt=prompt,
        max_output_tokens=config.AI_REVIEW_MAX_OUTPUT_TOKENS,
        model=model or config.OPENCLAW_OPENAI_MODEL,
        reasoning_effort=reasoning_effort or config.OPENCLAW_OPENAI_REASONING_EFFORT,
    )
    parsed = _extract_json_object(response_text or "")
    if not parsed:
        LOGGER.debug("AI review judge returned non-JSON output; falling back to heuristic result.")
        return baseline

    decision = _normalize_decision(parsed.get("decision"))
    confidence = _normalize_confidence(parsed.get("confidence"))
    rationale = str(parsed.get("rationale") or baseline["rationale"]).strip()
    checks = parsed.get("checks") or baseline["checks"]
    if isinstance(checks, str):
        checks = [checks]
    checks = [str(item).strip() for item in checks if str(item).strip()][:4] or baseline["checks"]

    return {
        "decision": decision,
        "confidence": confidence,
        "rationale": rationale,
        "checks": checks,
        "used_ai": True,
        "evidence": evidence,
        "raw_ai_text": response_text,
    }


def generate_review_judgments(
    bundle: Dict[str, Any],
    *,
    top_n: int = config.AI_REVIEW_MAX_CASES,
    ai_case_limit: int = config.AI_REVIEW_MAX_AI_CASES,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    transactions = bundle.get("transactions", pd.DataFrame())
    if not isinstance(transactions, pd.DataFrame) or transactions.empty:
        return pd.DataFrame()

    ranked = transactions.copy()
    if "composite_risk_score" in ranked.columns:
        ranked = ranked.sort_values("composite_risk_score", ascending=False)

    rows: List[Dict[str, Any]] = []
    for index, (_, row) in enumerate(ranked.head(top_n).iterrows()):
        row_dict = row.to_dict()
        judgment = judge_case_disposition(
            "transaction",
            row_dict,
            bundle,
            use_ai=index < ai_case_limit,
        )
        rows.append(
            {
                "transactionid": row_dict.get("transactionid"),
                "accountid": row_dict.get("accountid"),
                "merchantid": row_dict.get("merchantid"),
                "composite_risk_score": row_dict.get("composite_risk_score"),
                "risk_level": row_dict.get("risk_level"),
                "suggested_decision": judgment["decision"],
                "judge_confidence": judgment["confidence"],
                "judge_rationale": judgment["rationale"],
                "judge_checks": " | ".join(judgment["checks"]),
                "used_ai": bool(judgment["used_ai"]),
            }
        )

    judgments = pd.DataFrame(rows)
    target_path = output_path or config.AI_REVIEW_RECOMMENDATIONS_FILE
    if not judgments.empty:
        save_csv(judgments, target_path)
    return judgments
