"""
Topological Data Analysis (TDA) placeholder module.

Current Status: Stub implementation with clear TODOs
Reason: TDA libraries (KeplerMapper, Ripser) have heavy or unstable dependencies.
This module is structured for future enhancement without blocking the pipeline.

Future enhancements could include:
- KeplerMapper for persistent topology of transaction graphs
- Ripser/Persim for persistent homology diagrams
- Vectorization of topological features into the risk score

For now, this returns minimal dummy outputs so the pipeline can proceed.
"""

from pathlib import Path

import pandas as pd

from . import config
from .utils import LOGGER, save_csv


def tda_analysis_stub(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stub TDA analysis: returns a DataFrame with TDA features (currently all zeros).

    Args:
        df: Cleaned transaction DataFrame

    Returns:
        DataFrame with transactionid and placeholder TDA features
    """
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("STAGE 3b: TOPOLOGICAL DATA ANALYSIS (STUB)")
    LOGGER.info("=" * 60)

    LOGGER.info("TDA module currently a stub (no heavy dependencies).")
    LOGGER.info("Returning placeholder features for future enhancement.\n")

    # Create stub output
    tda_df = pd.DataFrame({
        "transactionid": df["transactionid"],
        "mapper_connected_component_id": 0,
        "mapper_distance_to_core": 0.0,
        "persistence_homology_feature_1": 0.0,
        "persistence_homology_feature_2": 0.0,
    })

    # Save stub output
    output_file = config.REPORTS_DIR / "tda_features.csv"
    save_csv(tda_df, output_file)

    LOGGER.info("Saved stub TDA features. To enable real TDA:")
    LOGGER.info("  1. Install KeplerMapper: pip install scikit-tda")
    LOGGER.info("  2. Uncomment TDA logic in tda_analysis.py")
    LOGGER.info("  3. Re-run pipeline\n")

    return tda_df


if __name__ == "__main__":
    # For testing
    import pandas as pd

    dummy_df = pd.DataFrame({"transactionid": ["TX001", "TX002"]})
    result = tda_analysis_stub(dummy_df)
    print(result)
