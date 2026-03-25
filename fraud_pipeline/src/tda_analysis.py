"""
Topological Data Analysis (TDA) features for the fraud pipeline.

This stage now computes real Mapper-style and persistent-homology-derived
features when the optional TDA dependencies are installed. It still falls back
to a lightweight deterministic output if those dependencies are missing so the
rest of the pipeline remains runnable during bootstrap.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import config
from .utils import LOGGER, normalize_to_01, save_csv


def _import_tda_dependencies():
    try:
        import kmapper as km
        from gph import ripser_parallel

        return km, ripser_parallel
    except Exception as exc:  # pragma: no cover - exercised only when deps missing
        return None, exc


def _prepare_tda_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    numeric_candidates = []
    for column in config.TDA_FEATURE_COLUMNS:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            numeric_candidates.append(column)
    if not numeric_candidates:
        raise ValueError("No numeric feature columns available for TDA analysis.")

    feature_df = df[numeric_candidates].copy()
    feature_df = feature_df.fillna(feature_df.median(numeric_only=True))
    feature_df = feature_df.fillna(0)

    scaler = StandardScaler()
    features_std = scaler.fit_transform(feature_df)
    return feature_df, features_std


def _diagram_lifetimes(diagram: Any) -> np.ndarray:
    if diagram is None:
        return np.array([])
    array = np.asarray(diagram, dtype=float)
    if array.size == 0:
        return np.array([])
    if array.ndim == 1:
        array = array.reshape(-1, 2)
    births = array[:, 0]
    deaths = array[:, 1]
    finite_mask = np.isfinite(births) & np.isfinite(deaths)
    if not finite_mask.any():
        return np.array([])
    lifetimes = deaths[finite_mask] - births[finite_mask]
    lifetimes = lifetimes[np.isfinite(lifetimes)]
    return lifetimes[lifetimes > 0]


def _sample_cluster_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    sample_index = np.linspace(0, len(points) - 1, max_points, dtype=int)
    return points[sample_index]


def _mapper_features(features_std: np.ndarray, km_module) -> Dict[str, np.ndarray]:
    n_samples = len(features_std)
    global_centroid = features_std.mean(axis=0)
    global_distance = np.linalg.norm(features_std - global_centroid, axis=1)

    mapper_connected_component_id = np.full(n_samples, -1, dtype=int)
    mapper_overlap_count = np.zeros(n_samples, dtype=int)
    mapper_node_degree = np.zeros(n_samples, dtype=int)
    mapper_component_size = np.zeros(n_samples, dtype=int)
    mapper_distance_to_core = np.full(n_samples, np.nan, dtype=float)

    if n_samples < max(12, config.TDA_MAPPER_DBSCAN_MIN_SAMPLES * 3):
        mapper_distance_to_core = normalize_to_01(global_distance)
        return {
            "mapper_connected_component_id": mapper_connected_component_id,
            "mapper_overlap_count": mapper_overlap_count,
            "mapper_node_degree": mapper_node_degree,
            "mapper_component_size": mapper_component_size,
            "mapper_distance_to_core": mapper_distance_to_core,
        }

    pca_components = max(1, min(config.TDA_PCA_COMPONENTS, features_std.shape[1], n_samples - 1))
    lens = PCA(n_components=pca_components, random_state=config.KMEANS_RANDOM_STATE).fit_transform(features_std)
    cover = km_module.Cover(n_cubes=config.TDA_MAPPER_N_CUBES, perc_overlap=config.TDA_MAPPER_OVERLAP)
    clusterer = DBSCAN(
        eps=config.TDA_MAPPER_DBSCAN_EPS,
        min_samples=min(config.TDA_MAPPER_DBSCAN_MIN_SAMPLES, max(2, n_samples // 20)),
    )

    mapper = km_module.KeplerMapper(verbose=0)
    graph = mapper.map(lens, features_std, cover=cover, clusterer=clusterer)
    nodes = graph.get("nodes", {}) or {}
    links = graph.get("links", {}) or {}
    if not nodes:
        mapper_distance_to_core = normalize_to_01(global_distance)
        return {
            "mapper_connected_component_id": mapper_connected_component_id,
            "mapper_overlap_count": mapper_overlap_count,
            "mapper_node_degree": mapper_node_degree,
            "mapper_component_size": mapper_component_size,
            "mapper_distance_to_core": mapper_distance_to_core,
        }

    node_graph = nx.Graph()
    node_graph.add_nodes_from(nodes.keys())
    for source, targets in links.items():
        for target in targets:
            node_graph.add_edge(source, target)

    component_lookup: Dict[str, tuple[int, int]] = {}
    for component_index, component_nodes in enumerate(nx.connected_components(node_graph)):
        component_nodes = list(component_nodes)
        member_indices: set[int] = set()
        for node_name in component_nodes:
            member_indices.update(int(index) for index in nodes.get(node_name, []))
        component_size = len(member_indices)
        for node_name in component_nodes:
            component_lookup[node_name] = (component_index, component_size)

    node_degrees = dict(node_graph.degree())
    node_centroids = {
        node_name: features_std[np.asarray(member_indices, dtype=int)].mean(axis=0)
        for node_name, member_indices in nodes.items()
        if len(member_indices) > 0
    }

    for node_name, member_indices in nodes.items():
        member_indices = np.asarray(member_indices, dtype=int)
        if member_indices.size == 0:
            continue
        centroid = node_centroids[node_name]
        component_index, component_size = component_lookup.get(node_name, (-1, int(member_indices.size)))
        node_degree = int(node_degrees.get(node_name, 0))
        distances = np.linalg.norm(features_std[member_indices] - centroid, axis=1)

        mapper_overlap_count[member_indices] += 1
        mapper_node_degree[member_indices] = np.maximum(mapper_node_degree[member_indices], node_degree)
        mapper_component_size[member_indices] = np.maximum(mapper_component_size[member_indices], component_size)

        for local_offset, row_index in enumerate(member_indices.tolist()):
            if mapper_connected_component_id[row_index] < 0:
                mapper_connected_component_id[row_index] = component_index
            distance_value = float(distances[local_offset])
            if np.isnan(mapper_distance_to_core[row_index]) or distance_value < mapper_distance_to_core[row_index]:
                mapper_distance_to_core[row_index] = distance_value

    mapper_distance_to_core = np.where(np.isnan(mapper_distance_to_core), global_distance, mapper_distance_to_core)
    return {
        "mapper_connected_component_id": mapper_connected_component_id,
        "mapper_overlap_count": mapper_overlap_count,
        "mapper_node_degree": mapper_node_degree,
        "mapper_component_size": mapper_component_size,
        "mapper_distance_to_core": mapper_distance_to_core,
    }


def _persistence_features(features_std: np.ndarray, ripser_parallel) -> Dict[str, np.ndarray]:
    n_samples = len(features_std)
    persistence_cluster_id = np.full(n_samples, -1, dtype=int)
    persistence_cluster_size = np.zeros(n_samples, dtype=int)
    persistence_cluster_distance = np.zeros(n_samples, dtype=float)
    persistence_h0_total = np.zeros(n_samples, dtype=float)
    persistence_h1_total = np.zeros(n_samples, dtype=float)
    persistence_h1_count = np.zeros(n_samples, dtype=float)
    persistence_h1_max = np.zeros(n_samples, dtype=float)

    if n_samples < 6:
        return {
            "persistence_cluster_id": persistence_cluster_id,
            "persistence_cluster_size": persistence_cluster_size,
            "persistence_cluster_distance": persistence_cluster_distance,
            "persistence_homology_feature_1": persistence_h0_total,
            "persistence_homology_feature_2": persistence_h1_total,
            "persistence_h1_feature_count": persistence_h1_count,
            "persistence_h1_max_persistence": persistence_h1_max,
        }

    cluster_count = min(config.TDA_PERSISTENCE_CLUSTER_COUNT, max(2, int(np.sqrt(max(n_samples, 4)))))
    kmeans = KMeans(
        n_clusters=cluster_count,
        random_state=config.KMEANS_RANDOM_STATE,
        n_init=10,
    )
    labels = kmeans.fit_predict(features_std)
    center_distance = kmeans.transform(features_std).min(axis=1)

    for cluster_id in np.unique(labels):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_points = features_std[cluster_indices]
        cluster_size = int(len(cluster_indices))
        sampled_points = _sample_cluster_points(cluster_points, config.TDA_PERSISTENCE_MAX_POINTS_PER_CLUSTER)

        h0_total = 0.0
        h1_total = 0.0
        h1_count = 0.0
        h1_max = 0.0

        if len(sampled_points) >= 4:
            try:
                persistence = ripser_parallel(sampled_points, maxdim=config.TDA_PERSISTENCE_MAXDIM)
                diagrams = persistence.get("dgms", []) if isinstance(persistence, dict) else []
                h0_lifetimes = _diagram_lifetimes(diagrams[0] if len(diagrams) > 0 else None)
                h1_lifetimes = _diagram_lifetimes(diagrams[1] if len(diagrams) > 1 else None)
                h0_total = float(h0_lifetimes.sum()) if h0_lifetimes.size else 0.0
                h1_total = float(h1_lifetimes.sum()) if h1_lifetimes.size else 0.0
                h1_count = float(h1_lifetimes.size)
                h1_max = float(h1_lifetimes.max()) if h1_lifetimes.size else 0.0
            except Exception as exc:
                LOGGER.warning("Persistent homology failed for cluster %s: %s", cluster_id, exc)

        persistence_cluster_id[cluster_indices] = int(cluster_id)
        persistence_cluster_size[cluster_indices] = cluster_size
        persistence_cluster_distance[cluster_indices] = center_distance[cluster_indices]
        persistence_h0_total[cluster_indices] = h0_total
        persistence_h1_total[cluster_indices] = h1_total
        persistence_h1_count[cluster_indices] = h1_count
        persistence_h1_max[cluster_indices] = h1_max

    return {
        "persistence_cluster_id": persistence_cluster_id,
        "persistence_cluster_size": persistence_cluster_size,
        "persistence_cluster_distance": persistence_cluster_distance,
        "persistence_homology_feature_1": persistence_h0_total,
        "persistence_homology_feature_2": persistence_h1_total,
        "persistence_h1_feature_count": persistence_h1_count,
        "persistence_h1_max_persistence": persistence_h1_max,
    }


def _build_tda_dataframe(df: pd.DataFrame, features_std: np.ndarray, *, km_module=None, ripser_parallel=None) -> pd.DataFrame:
    mapper = _mapper_features(features_std, km_module) if km_module is not None else {
        "mapper_connected_component_id": np.full(len(df), -1, dtype=int),
        "mapper_overlap_count": np.zeros(len(df), dtype=int),
        "mapper_node_degree": np.zeros(len(df), dtype=int),
        "mapper_component_size": np.zeros(len(df), dtype=int),
        "mapper_distance_to_core": normalize_to_01(np.linalg.norm(features_std - features_std.mean(axis=0), axis=1)),
    }
    persistence = _persistence_features(features_std, ripser_parallel) if ripser_parallel is not None else {
        "persistence_cluster_id": np.full(len(df), -1, dtype=int),
        "persistence_cluster_size": np.zeros(len(df), dtype=int),
        "persistence_cluster_distance": np.zeros(len(df), dtype=float),
        "persistence_homology_feature_1": np.zeros(len(df), dtype=float),
        "persistence_homology_feature_2": np.zeros(len(df), dtype=float),
        "persistence_h1_feature_count": np.zeros(len(df), dtype=float),
        "persistence_h1_max_persistence": np.zeros(len(df), dtype=float),
    }

    mapper_distance_risk = normalize_to_01(np.asarray(mapper["mapper_distance_to_core"], dtype=float))
    mapper_overlap_risk = normalize_to_01(np.asarray(mapper["mapper_overlap_count"], dtype=float))
    mapper_component_size = np.asarray(mapper["mapper_component_size"], dtype=float)
    mapper_small_component_risk = 1 - normalize_to_01(mapper_component_size)

    persistence_h0_risk = normalize_to_01(np.asarray(persistence["persistence_homology_feature_1"], dtype=float))
    persistence_h1_risk = normalize_to_01(np.asarray(persistence["persistence_homology_feature_2"], dtype=float))
    persistence_distance_risk = normalize_to_01(np.asarray(persistence["persistence_cluster_distance"], dtype=float))
    persistence_h1_count_risk = normalize_to_01(np.asarray(persistence["persistence_h1_feature_count"], dtype=float))

    tda_risk_score = normalize_to_01(
        0.24 * mapper_distance_risk
        + 0.16 * mapper_overlap_risk
        + 0.20 * mapper_small_component_risk
        + 0.14 * persistence_h0_risk
        + 0.16 * persistence_h1_risk
        + 0.10 * persistence_distance_risk
        + 0.00 * persistence_h1_count_risk
    )

    transaction_ids = df["transactionid"].values
    if getattr(transaction_ids, "ndim", 1) > 1:
        transaction_ids = transaction_ids[:, 0]

    return pd.DataFrame(
        {
            "transactionid": transaction_ids,
            "mapper_connected_component_id": mapper["mapper_connected_component_id"],
            "mapper_overlap_count": mapper["mapper_overlap_count"],
            "mapper_node_degree": mapper["mapper_node_degree"],
            "mapper_component_size": mapper["mapper_component_size"],
            "mapper_distance_to_core": mapper["mapper_distance_to_core"],
            "persistence_cluster_id": persistence["persistence_cluster_id"],
            "persistence_cluster_size": persistence["persistence_cluster_size"],
            "persistence_cluster_distance": persistence["persistence_cluster_distance"],
            "persistence_homology_feature_1": persistence["persistence_homology_feature_1"],
            "persistence_homology_feature_2": persistence["persistence_homology_feature_2"],
            "persistence_h1_feature_count": persistence["persistence_h1_feature_count"],
            "persistence_h1_max_persistence": persistence["persistence_h1_max_persistence"],
            "tda_risk_score": tda_risk_score,
        }
    )


def tda_analysis(df: pd.DataFrame, save_output: bool = True) -> pd.DataFrame:
    """
    Compute Mapper-style and persistent-homology-derived fraud features.
    """
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("STAGE 3b: TOPOLOGICAL DATA ANALYSIS")
    LOGGER.info("=" * 60)

    _, features_std = _prepare_tda_matrix(df)
    km_module, second_value = _import_tda_dependencies()

    if km_module is None:
        LOGGER.warning(
            "Real TDA dependencies are unavailable (%s). Falling back to lightweight topological approximations. "
            "Install `kmapper` and `giotto-ph` from requirements.txt to enable full Mapper and persistent homology.",
            second_value,
        )
        tda_df = _build_tda_dataframe(df, features_std)
    else:
        ripser_parallel = second_value
        LOGGER.info(
            "Running real TDA with Mapper (n_cubes=%s, overlap=%.2f) and persistent homology (clusters=%s, maxdim=%s)...",
            config.TDA_MAPPER_N_CUBES,
            config.TDA_MAPPER_OVERLAP,
            config.TDA_PERSISTENCE_CLUSTER_COUNT,
            config.TDA_PERSISTENCE_MAXDIM,
        )
        tda_df = _build_tda_dataframe(df, features_std, km_module=km_module, ripser_parallel=ripser_parallel)

    if save_output:
        save_csv(tda_df, config.TDA_FEATURES_FILE)
        LOGGER.info("\nStage 3b complete. TDA features saved to %s\n", config.TDA_FEATURES_FILE)
    else:
        LOGGER.info("\nStage 3b complete. TDA features computed in memory.\n")

    return tda_df


if __name__ == "__main__":
    from .ingest_clean import load_and_clean

    transactions = load_and_clean()
    print(tda_analysis(transactions).head())
