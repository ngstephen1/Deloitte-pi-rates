"""
Unsupervised anomaly detection module.
Combines Isolation Forest, Local Outlier Factor, and K-Means clustering.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from . import config
from .utils import LOGGER, normalize_to_01, save_csv, standardize_features


def prepare_features_for_anomaly_detection(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Select and standardize numeric features for anomaly detection.

    Args:
        df: Cleaned transaction DataFrame

    Returns:
        Tuple of (feature_df, standardized_features)
    """
    LOGGER.info("Preparing features for anomaly detection...")

    # Select configured anomaly features
    available_features = [f for f in config.ANOMALY_FEATURES if f in df.columns]
    missing_features = set(config.ANOMALY_FEATURES) - set(available_features)

    if missing_features:
        LOGGER.warning(f"  Missing features: {missing_features}")

    feature_df = df[available_features].copy()

    # Fill NaN with median
    feature_df = feature_df.fillna(feature_df.median())

    # Standardize
    scaler = StandardScaler()
    features_standardized = scaler.fit_transform(feature_df)

    LOGGER.info(f"  Selected {len(available_features)} features")
    LOGGER.info(f"  Standardized to mean=0, std=1")

    return feature_df, features_standardized


def run_isolation_forest(
    features: np.ndarray,
    contamination: float = None,
) -> np.ndarray:
    """
    Run Isolation Forest anomaly detection.

    Returns:
        Anomaly scores normalized to [0, 1] (higher = more anomalous)
    """
    if contamination is None:
        contamination = config.ISOLATION_FOREST_CONTAMINATION

    LOGGER.info(f"Running Isolation Forest (contamination={contamination})...")

    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=config.ISOLATION_FOREST_RANDOM_STATE,
        n_jobs=-1,
    )

    # Fit the model and get scores
    iso_forest.fit(features)
    
    # Raw scores: negative values = anomalous, positive = normal
    raw_scores = iso_forest.score_samples(features)

    # Normalize to [0, 1]: invert so anomalies get higher scores
    scores = 1 / (1 + np.exp(raw_scores))  # Sigmoid transformation
    scores = normalize_to_01(scores)

    anomaly_count = (scores > 0.5).sum()
    LOGGER.info(f"  Identified {anomaly_count} anomalies (score > 0.5)")

    return scores


def run_local_outlier_factor(
    features: np.ndarray,
    n_neighbors: int = None,
    contamination: float = None,
) -> np.ndarray:
    """
    Run Local Outlier Factor anomaly detection.

    Returns:
        Anomaly scores normalized to [0, 1]
    """
    if n_neighbors is None:
        n_neighbors = config.LOF_N_NEIGHBORS
    if contamination is None:
        contamination = config.LOF_CONTAMINATION

    n_samples = len(features)
    if n_samples <= 2:
        LOGGER.warning("Too few samples for LOF; returning zero scores.")
        return np.zeros(n_samples)
    n_neighbors = min(n_neighbors, n_samples - 1)

    LOGGER.info(f"Running Local Outlier Factor (n_neighbors={n_neighbors}, contamination={contamination})...")

    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
    )

    # Raw LOF scores: <1 = inlier, >1 = outlier
    raw_scores = lof.fit_predict(features)
    lof_scores = lof.negative_outlier_factor_

    # Normalize to [0, 1]
    scores = normalize_to_01(-lof_scores)  # Negate so anomalies get higher scores

    anomaly_count = (raw_scores == -1).sum()
    LOGGER.info(f"  Identified {anomaly_count} anomalies (LOF outlier)")

    return scores


def run_kmeans_clustering(
    features: np.ndarray,
    n_clusters: int = None,
    contamination: float = None,
) -> np.ndarray:
    """
    Run K-Means clustering and flag small/isolated clusters as anomalies.

    Returns:
        Anomaly scores normalized to [0, 1]
    """
    if n_clusters is None:
        n_clusters = config.KMEANS_N_CLUSTERS
    if contamination is None:
        contamination = config.KMEANS_CONTAMINATION
    n_samples = len(features)
    if n_samples == 0:
        return np.array([])
    if n_samples <= 2:
        LOGGER.warning("Too few samples for K-Means clustering; returning zero scores.")
        return np.zeros(n_samples)
    n_clusters = min(n_clusters, max(2, n_samples // 2))

    LOGGER.info(f"Running K-Means clustering (n_clusters={n_clusters})...")

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=config.KMEANS_RANDOM_STATE,
        n_init=10,
    )

    labels = kmeans.fit_predict(features)

    # Compute cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique, counts))

    # Compute distance to cluster center for each point
    distances = kmeans.transform(features).min(axis=1)

    # Points in small clusters + far from center = anomalous
    # Identify small clusters (bottom contamination%)
    size_threshold = np.percentile(list(cluster_sizes.values()), 100 * contamination)
    is_small_cluster = np.array([cluster_sizes[label] <= size_threshold for label in labels])

    # Distance-based scoring within clusters
    distance_scores = normalize_to_01(distances)

    # Combined score: more weight on being in small cluster
    anomaly_scores = 0.6 * is_small_cluster.astype(float) + 0.4 * distance_scores

    anomaly_count = (anomaly_scores > 0.5).sum()
    LOGGER.info(f"  Identified {anomaly_count} anomalies (small clusters or far from center)")

    return anomaly_scores


def run_anomaly_detection(df: pd.DataFrame, save_output: bool = True) -> pd.DataFrame:
    """
    Run full unsupervised anomaly detection pipeline.

    Returns:
        DataFrame with anomaly scores and flags
    """
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("STAGE 3: UNSUPERVISED ANOMALY DETECTION")
    LOGGER.info("=" * 60)

    # Prepare features
    feature_df, features_std = prepare_features_for_anomaly_detection(df)

    # Run each method
    isolation_forest_scores = run_isolation_forest(features_std)
    lof_scores = run_local_outlier_factor(features_std)
    kmeans_scores = run_kmeans_clustering(features_std)

    # Ensure all scores are 1D arrays
    isolation_forest_scores = np.asarray(isolation_forest_scores).flatten()
    lof_scores = np.asarray(lof_scores).flatten()
    kmeans_scores = np.asarray(kmeans_scores).flatten()

    # Extract transactionid and accountid safely (handle 2D case)
    txn_id = df["transactionid"].values
    if txn_id.ndim > 1:
        txn_id = txn_id[:, 0]
    
    acct_id = df["accountid"].values
    if acct_id.ndim > 1:
        acct_id = acct_id[:, 0]

    # Validate dimensions
    LOGGER.info(f"  isolation_forest_scores shape: {isolation_forest_scores.shape}")
    LOGGER.info(f"  lof_scores shape: {lof_scores.shape}")
    LOGGER.info(f"  kmeans_scores shape: {kmeans_scores.shape}")
    LOGGER.info(f"  txn_id shape: {txn_id.shape}")
    LOGGER.info(f"  acct_id shape: {acct_id.shape}")

    # Create anomaly output dataframe
    anomaly_df = pd.DataFrame({
        "transactionid": txn_id,
        "accountid": acct_id,
        "isolation_forest_score": isolation_forest_scores,
        "lof_score": lof_scores,
        "kmeans_anomaly_score": kmeans_scores,
    })

    # Ensure transactionid is unique
    if anomaly_df["transactionid"].duplicated().any():
        LOGGER.warning("Duplicate transactionid found in anomaly_df. Removing duplicates.")
        anomaly_df = anomaly_df.drop_duplicates(subset="transactionid")

    # Ensemble score: average of all methods (equally weighted)
    anomaly_df["ensemble_anomaly_score"] = (
        anomaly_df["isolation_forest_score"]
        + anomaly_df["lof_score"]
        + anomaly_df["kmeans_anomaly_score"]
    ) / 3

    # Flag rows with high ensemble score
    ensemble_threshold = 0.6
    anomaly_df["is_anomalous"] = (anomaly_df["ensemble_anomaly_score"] > ensemble_threshold).astype(int)

    anomaly_count = anomaly_df["is_anomalous"].sum()
    LOGGER.info(f"\nEnsemble anomaly detection:")
    LOGGER.info(f"  Total anomalies flagged: {anomaly_count} ({100*anomaly_count/len(anomaly_df):.2f}%)")
    LOGGER.info(f"  Threshold: {ensemble_threshold}")

    # Save anomaly scores
    output_file = config.ANOMALY_SCORES_FILE
    if save_output:
        save_csv(anomaly_df, output_file)
        LOGGER.info(f"\nStage 3 complete. Anomaly scores saved to {output_file}\n")
    else:
        LOGGER.info("\nStage 3 complete. Anomaly scores computed in memory.\n")

    return anomaly_df


if __name__ == "__main__":
    # For testing: load cleaned data and run anomaly detection
    from .ingest_clean import load_and_clean

    df = load_and_clean()
    anomaly_results = run_anomaly_detection(df)
    print(f"\nAnomalies: {anomaly_results['is_anomalous'].sum()}")
    print(anomaly_results.head())
