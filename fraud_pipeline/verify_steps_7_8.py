#!/usr/bin/env python3
"""
Quick verification script for Steps 7-8 implementation.
Run before launching pipeline: python verify_steps_7_8.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def check_modules():
    """Verify all required modules exist and import correctly."""
    print("\n✓ Checking modules...")
    
    modules_to_check = [
        "src.config",
        "src.utils",
        "src.reporting",
        "src.openai_explanations",
        "src.review_store",
    ]
    
    for module_name in modules_to_check:
        try:
            __import__(module_name)
            print(f"  ✓ {module_name}")
        except ImportError as e:
            print(f"  ✗ {module_name}: {e}")
            return False
    
    return True


def check_dependencies():
    """Check that key dependencies are installed."""
    print("\n✓ Checking dependencies...")
    
    dependencies = {
        "pandas": "pandas",
        "numpy": "numpy",
        "plotly": "plotly",
        "streamlit": "streamlit",
        "sklearn": "scikit-learn",
        "networkx": "networkx",
    }
    
    for import_name, package_name in dependencies.items():
        try:
            __import__(import_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ✗ {package_name} not installed")
            return False
    
    return True


def check_config():
    """Verify configuration is sensible."""
    print("\n✓ Checking configuration...")
    
    from src import config
    
    # Check weights sum to 1.0
    weight_sum = sum(config.RISK_WEIGHTS.values())
    if 0.99 <= weight_sum <= 1.01:
        print(f"  ✓ Risk weights sum to {weight_sum:.2f}")
    else:
        print(f"  ✗ Risk weights sum to {weight_sum:.2f} (should be ~1.0)")
        return False
    
    # Check directories
    for dir_name in [config.FIGURES_DIR, config.REPORTS_DIR, config.PROCESSED_DATA_DIR]:
        if dir_name.exists():
            print(f"  ✓ {dir_name.name} directory exists")
        else:
            print(f"  ✗ {dir_name.name} directory missing")
            return False
    
    # Check OpenAI flag
    print(f"  ✓ USE_OPENAI_EXPLANATIONS = {config.USE_OPENAI_EXPLANATIONS}")
    
    return True


def check_data():
    """Check if raw data is available."""
    print("\n✓ Checking data...")
    
    from src import config
    
    if config.RAW_DATA_FILE.exists():
        size_mb = config.RAW_DATA_FILE.stat().st_size / (1024 * 1024)
        print(f"  ✓ Raw data file found ({size_mb:.1f} MB)")
        return True
    else:
        print(f"  ✗ Raw data not found: {config.RAW_DATA_FILE}")
        print(f"     Copy it with: cp /path/to/bank_transactions_data.csv {config.RAW_DATA_FILE}")
        return False


def check_reporting():
    """Verify reporting module features."""
    print("\n✓ Checking reporting capabilities...")
    
    from src import reporting
    
    functions_to_check = [
        "generate_report",
        "plot_risk_by_account",
        "plot_risk_by_merchant",
        "create_executive_summary",
        "save_summary_for_tableau",
        "generate_openai_explanation",
    ]
    
    for func_name in functions_to_check:
        if hasattr(reporting, func_name):
            print(f"  ✓ {func_name}")
        else:
            print(f"  ✗ {func_name} not found")
            return False
    
    return True


def check_streamlit():
    """Verify Streamlit app can import."""
    print("\n✓ Checking Streamlit app...")
    
    app_file = Path(__file__).parent / "app" / "streamlit_app.py"
    if app_file.exists():
        print(f"  ✓ Streamlit app found at {app_file}")
        return True
    else:
        print(f"  ✗ Streamlit app not found")
        return False


def main():
    print("=" * 60)
    print("VERIFICATION: Steps 7-8 Implementation")
    print("=" * 60)
    
    checks = [
        ("Modules", check_modules),
        ("Dependencies", check_dependencies),
        ("Configuration", check_config),
        ("Data", check_data),
        ("Reporting", check_reporting),
        ("Streamlit", check_streamlit),
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        try:
            if not check_func():
                all_passed = False
        except Exception as e:
            print(f"\n✗ {check_name} check failed: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL CHECKS PASSED")
        print("\nNext steps:")
        print("  1. Run pipeline:        python run_pipeline.py")
        print("  2. Launch Streamlit:    streamlit run app/streamlit_app.py")
        print("=" * 60)
        return 0
    else:
        print("❌ SOME CHECKS FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
