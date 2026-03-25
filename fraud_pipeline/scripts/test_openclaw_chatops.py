#!/usr/bin/env python3
"""
Local smoke test for grounded OpenClaw-style fraud Q&A.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chatops import load_active_bundle, load_report_bundle, publish_bundle_context
from src.chatops.query_service import answer_analyst_question


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a grounded analyst question against the latest fraud context.")
    parser.add_argument("--question", required=True, help="Question to ask against the active fraud context.")
    parser.add_argument("--transcript", default="", help="Optional recent conversation context.")
    parser.add_argument(
        "--use-pipeline-outputs",
        action="store_true",
        help="Ignore any published Streamlit context and use the latest saved pipeline outputs.",
    )
    parser.add_argument(
        "--publish-pipeline-context",
        action="store_true",
        help="Publish the loaded context into the shared ChatOps active-context directory before answering.",
    )
    args = parser.parse_args()

    bundle = load_report_bundle() if args.use_pipeline_outputs else load_active_bundle()
    if args.publish_pipeline_context:
        publish_bundle_context(bundle, publish_reason="manual_query_test")

    result = answer_analyst_question(args.question, bundle=bundle, conversation_context=args.transcript or None)
    print(f"SOURCE: {result['source_label']}")
    print(f"AI_USED: {result['used_ai']}")
    print("ANSWER:")
    print(result["answer"])


if __name__ == "__main__":
    main()
