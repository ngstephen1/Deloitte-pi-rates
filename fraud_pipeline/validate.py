"""
Quick validation script to verify the pipeline structure and imports.
This tests basic functionality without running the full pipeline.
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("PIPELINE VALIDATION")
print("=" * 70)
print()

# Test 1: Config
print("✓ Testing config module...")
try:
    from src import config
    print(f"  Project root: {config.PROJECT_ROOT}")
    print(f"  Raw data file: {config.RAW_DATA_FILE}")
    print(f"  Risk weights sum: {sum(config.RISK_WEIGHTS.values()):.2f}")
    print("  ✓ Config loaded")
except Exception as e:
    print(f"  ✗ Config error: {e}")
    sys.exit(1)

# Test 2: Utils
print("\n✓ Testing utils module...")
try:
    from src.utils import normalize_to_01, setup_logger
    import numpy as np
    test_vals = np.array([0, 50, 100])
    normalized = normalize_to_01(test_vals)
    assert normalized.min() >= 0 and normalized.max() <= 1
    print(f"  Normalization test: {test_vals} -> {normalized}")
    print("  ✓ Utils loaded")
except Exception as e:
    print(f"  ✗ Utils error: {e}")
    sys.exit(1)

# Test 3: Check raw data
print("\n✓ Testing data availability...")
try:
    import pandas as pd
    if not config.RAW_DATA_FILE.exists():
        raise FileNotFoundError(f"Raw data not found at {config.RAW_DATA_FILE}")
    
    df_sample = pd.read_csv(config.RAW_DATA_FILE, nrows=5)
    print(f"  Raw data shape: {df_sample.shape}")
    print(f"  Columns: {list(df_sample.columns)}")
    print("  ✓ Data available")
except Exception as e:
    print(f"  ✗ Data error: {e}")
    sys.exit(1)

# Test 4: Module imports
print("\n✓ Testing module imports...")
try:
    from src.ingest_clean import clean_column_names, parse_dates
    from src.eda_profile import compute_summary_statistics
    from src.anomaly_detection import prepare_features_for_anomaly_detection
    from src.graph_analysis import build_transaction_graph
    from src.risk_scoring import compute_amount_outlier_risk
    from src.benford import benford_expected_distribution
    from src.review_store import ReviewStore
    print("  ✓ All modules imported successfully")
except Exception as e:
    print(f"  ✗ Import error: {e}")
    sys.exit(1)

# Test 5: Data cleaning functions
print("\n✓ Testing data cleaning functions...")
try:
    df = pd.read_csv(config.RAW_DATA_FILE, nrows=10)
    df_clean = clean_column_names(df)
    assert all(col.islower() for col in df_clean.columns)
    print(f"  Cleaned column names: {list(df_clean.columns[:3])}...")
    
    df_dates = parse_dates(df_clean)
    assert df_dates['transactiondate'].dtype == 'datetime64[ns]'
    print("  ✓ Data cleaning functions work")
except Exception as e:
    print(f"  ✗ Cleaning error: {e}")
    sys.exit(1)

# Test 6: Output directories
print("\n✓ Testing output directories...")
try:
    assert config.PROCESSED_DATA_DIR.exists()
    assert config.FIGURES_DIR.exists()
    assert config.REPORTS_DIR.exists()
    print(f"  Processed data dir: {config.PROCESSED_DATA_DIR}")
    print(f"  Figures dir: {config.FIGURES_DIR}")
    print(f"  Reports dir: {config.REPORTS_DIR}")
    print("  ✓ All directories exist")
except Exception as e:
    print(f"  ✗ Directory error: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ ALL VALIDATION CHECKS PASSED")
print("=" * 70)
print("\nNext steps:")
print("1. Install dependencies: pip install -r requirements.txt")
print("2. Run full pipeline: python run_pipeline.py")
print("3. Launch Streamlit: streamlit run app/streamlit_app.py")
print()
