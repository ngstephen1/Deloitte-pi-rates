#!/usr/bin/env python3
"""
Local smoke test for Discord/OpenClaw fraud image analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.chatops.context_loader import load_active_bundle, load_report_bundle
from src.chatops.image_analysis import analyze_uploaded_image


SAMPLE_IMAGES = [
    config.PROJECT_ROOT / "discord-img" / "OOF_Fraud_Monitor_Dashboard.png",
    config.PROJECT_ROOT / "discord-img" / "OOF_phishing_email_alert.png",
    config.PROJECT_ROOT / "discord-img" / "Payment_receipt_invoice.png",
    config.PROJECT_ROOT / "discord-img" / "Recent_Account_Activities.png",
    config.PROJECT_ROOT / "discord-img" / "Recent-Activity-Logs.png",
    config.PROJECT_ROOT / "discord-img" / "Transaction-table.png",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local fraud image-analysis smoke tests.")
    parser.add_argument("--image", type=Path, help="Path to one image to analyze.")
    parser.add_argument(
        "--prompt",
        default="Analyze this image for fraud indicators and recommend next steps.",
        help="Prompt or goal to use for the image analysis.",
    )
    parser.add_argument("--all-samples", action="store_true", help="Analyze all bundled sample images.")
    parser.add_argument(
        "--use-pipeline-outputs",
        action="store_true",
        help="Use saved pipeline outputs instead of the last published ChatOps context.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    bundle = load_report_bundle() if args.use_pipeline_outputs else load_active_bundle()

    image_paths: list[Path] = []
    if args.image:
        image_paths = [args.image]
    elif args.all_samples:
        image_paths = SAMPLE_IMAGES
    else:
        raise SystemExit("Provide --image PATH or --all-samples.")

    for image_path in image_paths:
        print("=" * 80)
        print(f"IMAGE: {image_path}")
        analysis = analyze_uploaded_image(image_path, user_prompt=args.prompt, bundle=bundle)
        print(f"USED_AI: {analysis['used_ai']}")
        print(f"IMAGE_TYPE: {analysis['structured_output'].get('image_type')}")
        print(f"PRIMARY_CASE: {analysis.get('primary_case_type')}::{analysis.get('primary_case_id')}")
        print("REPLY:")
        print(analysis["reply_text"])
        print("EXPORTS:")
        for label, path in analysis["export_paths"].items():
            print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
