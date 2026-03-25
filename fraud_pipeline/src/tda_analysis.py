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

import numpy as np
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

    # Create stub output with arrays (one value per transaction)
    n_rows = len(df)
    
    # Extract transactionid safely (handle 2D case)
    txn_id = df["transactionid"].values
    if txn_id.ndim > 1:
        txn_id = txn_id[:, 0]
    
    tda_df = pd.DataFrame({
        "transactionid": txn_id,
        "mapper_connected_component_id": np.zeros(n_rows, dtype=int),
        "mapper_distance_to_core": np.zeros(n_rows, dtype=float),
        "persistence_homology_feature_1": np.zeros(n_rows, dtype=float),
        "persistence_homology_feature_2": np.zeros(n_rows, dtype=float),
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
