"""
Graph-based analysis module.
Builds a multi-entity transaction graph and computes interpretable graph features.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import networkx as nx

from . import config
from .utils import LOGGER, save_csv, normalize_to_01


def build_transaction_graph(df: pd.DataFrame) -> nx.MultiDiGraph:
    """
    Build a multi-entity transaction graph.

    Entities: AccountID, MerchantID, DeviceID, IP Address, Location
    Edges represent transactions between entities.

    Returns:
        NetworkX MultiDiGraph with transactions as edges and metadata
    """
    LOGGER.info("Building transaction graph...")

    G = nx.MultiDiGraph()

    # Add nodes for each entity type
    for account_id in df["accountid"].unique():
        G.add_node(f"acc_{account_id}", entity_type="account")

    for merchant_id in df["merchantid"].unique():
        G.add_node(f"mer_{merchant_id}", entity_type="merchant")

    for device_id in df["deviceid"].unique():
        G.add_node(f"dev_{device_id}", entity_type="device")

    for ip in df["ip_address"].unique():
        G.add_node(f"ip_{ip}", entity_type="ip")

    for location in df["location"].unique():
        G.add_node(f"loc_{location}", entity_type="location")

    # Add edges (transactions)
    for idx, row in df.iterrows():
        acc_node = f"acc_{row['accountid']}"
        mer_node = f"mer_{row['merchantid']}"
        dev_node = f"dev_{row['deviceid']}"
        ip_node = f"ip_{row['ip_address']}"
        loc_node = f"loc_{row['location']}"

        # Connect account -> merchant, device, IP, location
        G.add_edge(acc_node, mer_node, weight=row["transactionamount"], tx_id=row["transactionid"])
        G.add_edge(acc_node, dev_node, weight=row["transactionamount"], tx_id=row["transactionid"])
        G.add_edge(acc_node, ip_node, weight=row["transactionamount"], tx_id=row["transactionid"])
        G.add_edge(acc_node, loc_node, weight=row["transactionamount"], tx_id=row["transactionid"])

        # Device -> IP (co-location)
        G.add_edge(dev_node, ip_node, weight=row["transactionamount"], tx_id=row["transactionid"])

    LOGGER.info(f"  Created graph: {len(G.nodes())} nodes, {len(G.edges())} edges")

    return G


def compute_graph_features(df: pd.DataFrame, G: nx.MultiDiGraph) -> pd.DataFrame:
    """
    Compute per-transaction graph features.

    Features:
    - account_degree: Number of unique entities (merchants, devices, IPs, locations) per account
    - account_centrality: Betweenness centrality of account node
    - account_component_size: Size of connected component containing account
    - shared_device_count: How many other accounts share this device
    - shared_ip_count: How many other accounts share this IP
    - shared_merchant_count: How many other accounts share this merchant

    Returns:
        DataFrame with transactionid and graph features
    """
    LOGGER.info("Computing graph features...")

    graph_features = []

    # Precompute centrality
    try:
        centrality = nx.betweenness_centrality(G, normalized=True)
    except Exception as e:
        LOGGER.warning(f"  Could not compute centrality: {e}; using zeros")
        centrality = {node: 0 for node in G.nodes()}

    # Component info
    components = list(nx.weakly_connected_components(G))
    component_map = {}
    for comp_id, comp_nodes in enumerate(components):
        for node in comp_nodes:
            component_map[node] = len(comp_nodes)

    for idx, row in df.iterrows():
        acc_node = f"acc_{row['accountid']}"

        # Account degree (unique connections)
        account_degree = G.degree(acc_node) if acc_node in G else 0

        # Account centrality
        account_centrality = centrality.get(acc_node, 0)

        # Component size
        component_size = component_map.get(acc_node, 1)

        # Count shared entities
        mer_node = f"mer_{row['merchantid']}"
        dev_node = f"dev_{row['deviceid']}"
        ip_node = f"ip_{row['ip_address']}"

        # Shared device: count other accounts connected to this device
        shared_device_count = 0
        for node in G.predecessors(dev_node):
            if node != acc_node and node.startswith("acc_"):
                shared_device_count += 1

        # Shared IP: count other accounts connected to this IP
        shared_ip_count = 0
        for node in G.predecessors(ip_node):
            if node != acc_node and node.startswith("acc_"):
                shared_ip_count += 1

        # Shared merchant: count other accounts connected to this merchant
        shared_merchant_count = 0
        for node in G.predecessors(mer_node):
            if node != acc_node and node.startswith("acc_"):
                shared_merchant_count += 1

        graph_features.append({
            "transactionid": row["transactionid"],
            "accountid": row["accountid"],
            "account_degree": account_degree,
            "account_centrality": account_centrality,
            "component_size": component_size,
            "shared_device_count": shared_device_count,
            "shared_ip_count": shared_ip_count,
            "shared_merchant_count": shared_merchant_count,
        })

    feature_df = pd.DataFrame(graph_features)

    LOGGER.info(f"  Computed {len(feature_df)} transaction-level graph features")

    return feature_df


def compute_graph_risk_score(graph_features: pd.DataFrame) -> np.ndarray:
    """
    Compute a combined graph-based risk score.

    Risk factors:
    - High account centrality (bridge in network)
    - Large connected component (mainstream)
    - High shared device/IP/merchant counts (suspicious co-location)

    Returns:
        Risk scores normalized to [0, 1]
    """
    LOGGER.info("Computing graph-based risk score...")

    # Normalize each feature to [0, 1]
    scores = pd.DataFrame()

    # High centrality = suspicious (hub/bridge)
    scores["centrality_risk"] = normalize_to_01(graph_features["account_centrality"].values)

    # High degree = more connected (could be suspicious or normal high-volume)
    scores["degree_risk"] = normalize_to_01(graph_features["account_degree"].values)

    # Shared device/IP/merchant = suspicious
    scores["shared_device_risk"] = normalize_to_01(graph_features["shared_device_count"].values)
    scores["shared_ip_risk"] = normalize_to_01(graph_features["shared_ip_count"].values)
    scores["shared_merchant_risk"] = normalize_to_01(graph_features["shared_merchant_count"].values)

    # Combined risk (weighted average)
    risk_score = (
        0.25 * scores["centrality_risk"]
        + 0.15 * scores["degree_risk"]
        + 0.25 * scores["shared_device_risk"]
        + 0.20 * scores["shared_ip_risk"]
        + 0.15 * scores["shared_merchant_risk"]
    )

    risk_score = normalize_to_01(risk_score.values)

    return risk_score


def graph_analysis(df: pd.DataFrame) -> Tuple[pd.DataFrame, nx.MultiDiGraph]:
    """
    Run full graph-based analysis pipeline.

    Returns:
        Tuple of (graph_features_df, NetworkX graph)
    """
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("STAGE 4: GRAPH-BASED ANALYSIS")
    LOGGER.info("=" * 60)

    # Build graph
    G = build_transaction_graph(df)

    # Compute features
    graph_features = compute_graph_features(df, G)

    # Compute risk score
    graph_risk_score = compute_graph_risk_score(graph_features)
    graph_features["graph_risk_score"] = graph_risk_score

    # Save
    output_file = config.GRAPH_FEATURES_FILE
    save_csv(graph_features, output_file)

    LOGGER.info(f"\nStage 4 complete. Graph features saved to {output_file}\n")

    return graph_features, G


if __name__ == "__main__":
    # For testing
    from .ingest_clean import load_and_clean

    df = load_and_clean()
    graph_features, G = graph_analysis(df)
    print(f"\nGraph nodes: {len(G.nodes())}, edges: {len(G.edges())}")
    print(graph_features.head())
