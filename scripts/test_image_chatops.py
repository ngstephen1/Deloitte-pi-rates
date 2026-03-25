#!/usr/bin/env python3
from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).resolve().parent.parent / "fraud_pipeline" / "scripts" / "test_image_chatops.py"),
    run_name="__main__",
)
