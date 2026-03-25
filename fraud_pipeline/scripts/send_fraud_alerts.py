#!/usr/bin/env python3
"""
Manual ChatOps trigger for fraud reports and alert delivery.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chatops import load_active_bundle, load_report_bundle, publish_bundle_context, send_alert_notifications, send_report_message


def main() -> None:
    parser = argparse.ArgumentParser(description="Send fraud report and alerts to the configured ChatOps target.")
    parser.add_argument("--dry-run", action="store_true", help="Build the payloads without posting to the webhook.")
    parser.add_argument("--force", action="store_true", help="Ignore alert dedupe state for this run.")
    parser.add_argument("--report-only", action="store_true", help="Send only the report digest.")
    parser.add_argument("--alerts-only", action="store_true", help="Send only threshold-based alerts.")
    parser.add_argument(
        "--use-pipeline-outputs",
        action="store_true",
        help="Ignore any published Streamlit context and load the latest saved pipeline outputs instead.",
    )
    args = parser.parse_args()

    bundle = load_report_bundle() if args.use_pipeline_outputs else load_active_bundle()
    publish_bundle_context(bundle, publish_reason="manual_chatops_script")

    if not args.alerts_only:
        report = send_report_message(bundle, dry_run=args.dry_run)
        delivery = report["delivery"]
        print(f"REPORT delivered={delivery.delivered} format={delivery.webhook_format} error={delivery.delivery_error}")
        if delivery.payload_preview:
            print("REPORT payload preview:")
            print(delivery.payload_preview)

    if not args.report_only:
        alerts = send_alert_notifications(bundle, dry_run=args.dry_run, force=args.force)
        print(f"ALERTS generated={len(alerts['alerts'])}")
        for item in alerts["deliveries"]:
            alert = item["alert"]
            if item.get("skipped"):
                print(f" - SKIPPED {alert['alert_id']} reason={item.get('reason')}")
                continue
            delivery = item["delivery"]
            print(
                f" - {alert['alert_id']} delivered={delivery.delivered} format={delivery.webhook_format} error={delivery.delivery_error}"
            )
            if delivery.payload_preview:
                print(delivery.payload_preview)


if __name__ == "__main__":
    main()
