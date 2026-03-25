"""
Main pipeline orchestrator.
Runs all stages end-to-end with optional memory-efficient modes:
1. Data ingestion & cleaning
2. EDA & profiling
3. Anomaly detection (optional)
4. TDA (stub)
5. Graph analysis (optional)
6. Risk scoring
7. Reporting & visualizations

Usage:
    python run_pipeline.py                    # Full pipeline
    python run_pipeline.py --fast             # Skip graph analysis
    python run_pipeline.py --sample 0.10      # Run on 10% of data
    python run_pipeline.py --minimal          # Skip graph + anomaly
    python run_pipeline.py --sample 0.25 --fast  # Combination modes
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

from src import config
from src.utils import LOGGER, setup_logger
from src.ingest_clean import load_and_clean
from src.eda_profile import eda_and_profile
from src.anomaly_detection import run_anomaly_detection
from src.graph_analysis import graph_analysis
from src.risk_scoring import risk_scoring
from src.tda_analysis import tda_analysis_stub
from src.reporting import generate_report


def run_pipeline(skip_streamlit: bool = False, skip_graph: bool = False, 
                skip_anomaly: bool = False, sample_size: float = None):
    """
    Execute fraud detection pipeline with optional memory-efficient modes.

    Args:
        skip_streamlit: If True, don't launch Streamlit after completion
        skip_graph: If True, skip graph analysis (saves ~100MB)
        skip_anomaly: If True, skip anomaly detection (saves ~150MB)
        sample_size: If set, run on fraction of data (e.g., 0.10 for 10%)
    """
    LOGGER.info("\n")
    LOGGER.info("=" * 70)
    LOGGER.info("FRAUD DETECTION PIPELINE - FULL RUN")
    LOGGER.info("=" * 70)
    LOGGER.info(f"Started at: {datetime.now()}")
    LOGGER.info(f"Project root: {config.PROJECT_ROOT}")
    if skip_graph or skip_anomaly or sample_size:
        mode_str = []
        if skip_graph:
            mode_str.append("skip_graph")
        if skip_anomaly:
            mode_str.append("skip_anomaly")
        if sample_size:
            mode_str.append(f"sample={sample_size*100:.0f}%")
        LOGGER.info(f"Memory mode: {', '.join(mode_str)}")
    LOGGER.info("")

    try:
        # ====================================================================
        # STAGE 1: DATA INGESTION & CLEANING
        # ====================================================================
        df = load_and_clean()
        
        # Apply sampling if requested
        if sample_size and sample_size < 1.0:
            orig_rows = len(df)
            df = df.sample(frac=sample_size, random_state=42)
            LOGGER.info(f"Sampled {len(df)} rows from {orig_rows} ({sample_size*100:.1f}%)")

        # ====================================================================
        # STAGE 2: EDA & PROFILING
        # ====================================================================
        eda_results = eda_and_profile(df)

        # ====================================================================
        # STAGE 3: ANOMALY DETECTION
        # ====================================================================
        if not skip_anomaly:
            anomaly_scores = run_anomaly_detection(df)
        else:
            LOGGER.info("⏭️  SKIPPING STAGE 3 (Anomaly Detection) - Memory mode")
            anomaly_scores = None

        # ====================================================================
        # STAGE 3b: TDA (STUB FOR NOW)
        # ====================================================================
        tda_features = tda_analysis_stub(df)

        # ====================================================================
        # STAGE 4: GRAPH ANALYSIS
        # ====================================================================
        if not skip_graph:
            graph_features, graph = graph_analysis(df)
        else:
            LOGGER.info("⏭️  SKIPPING STAGE 4 (Graph Analysis) - Memory mode")
            graph_features = None
            graph = None

        # ====================================================================
        # STAGE 5: RISK SCORING
        # ====================================================================
        risk_results = risk_scoring(df, anomaly_scores, graph_features)

        # ====================================================================
        # STAGE 6: REPORTING & VISUALIZATIONS (Step 7)
        # ====================================================================
        report_artifacts = generate_report(
            risk_results["transactions_ranked"],
            risk_results
        )

        # ====================================================================
        # SUMMARY
        # ====================================================================
        LOGGER.info("=" * 70)
        LOGGER.info("PIPELINE EXECUTION COMPLETE ✓")
        LOGGER.info("=" * 70)
        LOGGER.info("")
        LOGGER.info("📊 Output Files Created:")
        LOGGER.info(f"  Cleaned Data:           {config.CLEANED_DATA_FILE}")
        LOGGER.info(f"  Anomaly Scores:         {config.ANOMALY_SCORES_FILE}")
        LOGGER.info(f"  Graph Features:         {config.GRAPH_FEATURES_FILE}")
        LOGGER.info(f"  Ranked Transactions:    {config.RISK_TRANSACTIONS_FILE}")
        LOGGER.info(f"  Ranked Accounts:        {config.RISK_ACCOUNTS_FILE}")
        LOGGER.info(f"  Visualizations:         {config.FIGURES_DIR}/")
        LOGGER.info(f"  Summary Tables:         {config.REPORTS_DIR}/")
        LOGGER.info("")

        # Summary statistics
        transactions = risk_results["transactions_ranked"]
        accounts = risk_results["accounts_ranked"]

        high_risk_count = (transactions["risk_level"] == "High").sum()
        medium_risk_count = (transactions["risk_level"] == "Medium").sum()
        low_risk_count = (transactions["risk_level"] == "Low").sum()

        LOGGER.info("📈 Risk Distribution:")
        LOGGER.info(f"  High Risk:   {high_risk_count:4d} ({100*high_risk_count/len(transactions):5.1f}%)")
        LOGGER.info(f"  Medium Risk: {medium_risk_count:4d} ({100*medium_risk_count/len(transactions):5.1f}%)")
        LOGGER.info(f"  Low Risk:    {low_risk_count:4d} ({100*low_risk_count/len(transactions):5.1f}%)")
        LOGGER.info("")

        top_5_accounts = accounts.head(5)
        LOGGER.info("🔴 Top 5 Highest Risk Accounts:")
        for idx, row in top_5_accounts.iterrows():
            LOGGER.info(
                f"  {row['accountid']:10s} - Risk: {row['account_risk_score']:.3f} "
                f"({row['high_risk_transaction_count']} high-risk transactions)"
            )
        LOGGER.info("")

        LOGGER.info("✅ Next Steps:")
        LOGGER.info(f"  1. Review outputs in {config.REPORTS_DIR}/")
        LOGGER.info(f"  2. Launch Streamlit app: streamlit run app/streamlit_app.py")
        LOGGER.info(f"  3. Import data into Tableau: Use CSV files from {config.REPORTS_DIR}/")
        LOGGER.info("")
        LOGGER.info(f"Completed at: {datetime.now()}")
        LOGGER.info("=" * 70)

        if not skip_streamlit:
            LOGGER.info("")
            LOGGER.info("💡 To launch the Streamlit review interface, run:")
            LOGGER.info("   streamlit run app/streamlit_app.py")
            LOGGER.info("")

        return True

    except Exception as e:
        LOGGER.error(f"\n❌ PIPELINE FAILED: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run fraud detection pipeline with optional memory-efficient modes"
    )
    parser.add_argument(
        "--skip-streamlit",
        action="store_true",
        help="Skip Streamlit launch message (don't actually launch)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="FAST mode: skip graph analysis (saves ~100MB memory, ~10s runtime)",
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=None,
        metavar="FRACTION",
        help="SAMPLE mode: run on fraction of data (e.g. 0.10 for 10 percent, default: 1.0 for full)",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="MINIMAL mode: skip both graph and anomaly detection (saves ~250MB, ~20s runtime)",
    )
    args = parser.parse_args()

    # Determine which stages to skip
    skip_graph = args.fast or args.minimal
    skip_anomaly = args.minimal
    sample_size = args.sample if args.sample else 1.0
    
    # Validate sample size
    if sample_size is not None and (sample_size <= 0 or sample_size > 1.0):
        LOGGER.error("Sample fraction must be between 0.0 and 1.0")
        sys.exit(1)

    success = run_pipeline(
        skip_streamlit=args.skip_streamlit,
        skip_graph=skip_graph,
        skip_anomaly=skip_anomaly,
        sample_size=sample_size if sample_size < 1.0 else None
    )
    sys.exit(0 if success else 1)
