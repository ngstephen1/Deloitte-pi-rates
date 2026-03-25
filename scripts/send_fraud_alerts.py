#!/usr/bin/env python3
from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).resolve().parent.parent / "fraud_pipeline" / "scripts" / "send_fraud_alerts.py"),
    run_name="__main__",
)
