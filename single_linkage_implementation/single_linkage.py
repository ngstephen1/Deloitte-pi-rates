"""
TDA Fraud Detection: Mapper + Single Linkage Clustering
========================================================
Following methodology from:
  - BitcoinHeist (DFRWS 2021) — Single Linkage inside Mapper
  - ETH/IMTF (2025) — Ensemble Mapper for robustness

Pipeline:
  1. Feature engineering (transaction-level + account-level)
  2. Standardize features (StandardScaler)
  3. PCA projection as Mapper filter function
  4. Build Mapper graph with Single Linkage clustering
  5. Score transactions by topological isolation
  6. Ensemble over multiple parameter configurations
  7. Validate against behavioral proxies
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import kmapper as km
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. Load Data
# ============================================================

print("=" * 65)
print("  TDA Fraud Detection: Mapper + Single Linkage")
print("=" * 65)

df = pd.read_csv('bank_transactions_data_2.csv')
print(f"\nLoaded {len(df)} transactions, {df['AccountID'].nunique()} accounts")

# ============================================================
# 2. Feature Engineering
# ============================================================

# Parse dates
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
df['PreviousTransactionDate'] = pd.to_datetime(df['PreviousTransactionDate'])

# Encode categoricals
df['Channel_num'] = df['Channel'].map({'ATM': 0, 'Online': 1, 'Branch': 2})
df['Type_num'] = df['TransactionType'].map({'Debit': 0, 'Credit': 1})

# Per-account behavioral stats
account_stats = df.groupby('AccountID').agg(
    tx_count=('TransactionAmount', 'count'),
    mean_amount=('TransactionAmount', 'mean'),
    std_amount=('TransactionAmount', 'std'),
    max_amount=('TransactionAmount', 'max'),
    mean_duration=('TransactionDuration', 'mean'),
    total_logins=('LoginAttempts', 'sum'),
    n_devices=('DeviceID', 'nunique'),
    n_ips=('IP Address', 'nunique'),
    n_merchants=('MerchantID', 'nunique'),
    n_locations=('Location', 'nunique'),
).fillna(0)

df = df.merge(account_stats, on='AccountID', how='left')

# Deviation from account norm
df['amount_zscore'] = (df['TransactionAmount'] - df['mean_amount']) / (df['std_amount'] + 1e-10)
df['duration_deviation'] = (df['TransactionDuration'] - df['mean_duration']).abs()

# Final feature list
feature_cols = [
    # Transaction-level
    'TransactionAmount', 'CustomerAge', 'TransactionDuration',
    'LoginAttempts', 'AccountBalance', 'Channel_num', 'Type_num',
    # Account-level
    'tx_count', 'mean_amount', 'std_amount', 'max_amount',
    'mean_duration', 'total_logins', 'n_devices', 'n_ips',
    'n_merchants', 'n_locations',
    # Deviation
    'amount_zscore', 'duration_deviation',
]

X = np.nan_to_num(df[feature_cols].values.astype(float))
print(f"Features: {len(feature_cols)}")

# ============================================================
# 3. Normalize (StandardScaler)
# ============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# 4. PCA Filter Function
# ============================================================

pca = PCA(n_components=2)
lens = pca.fit_transform(X_scaled)
print(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.1%}")

# ============================================================
# 5. Mapper Anomaly Scoring Function
# ============================================================

def mapper_anomaly_scores(X_scaled, lens, cover_params, dist_threshold):
    """
    Build Mapper graph with Single Linkage, return per-transaction
    anomaly scores based on topological isolation.
    """
    mapper = km.KeplerMapper(verbose=0)

    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=dist_threshold,
        linkage='single',
    )

    graph = mapper.map(
        lens, X_scaled,
        cover=km.Cover(**cover_params),
        clusterer=clusterer,
    )

    # Coverage check
    covered = set()
    for mems in graph['nodes'].values():
        covered.update(mems)

    # Build NetworkX graph
    G = nx.Graph()
    for nid, mems in graph['nodes'].items():
        G.add_node(nid, members=mems)
    for nid, conns in graph['links'].items():
        for t in conns:
            G.add_edge(nid, t)

    max_deg = max((G.degree(n) for n in G.nodes()), default=1)

    # Connected component sizes
    components = list(nx.connected_components(G))
    comp_sizes = {}
    for comp in components:
        size = sum(len(graph['nodes'][n]) for n in comp)
        for n in comp:
            comp_sizes[n] = size
    max_comp = max(comp_sizes.values()) if comp_sizes else 1

    # Score each transaction by its node's isolation
    scores = np.zeros(len(X_scaled))
    counts = np.zeros(len(X_scaled))

    for nid in G.nodes():
        deg = G.degree(nid)
        cc = nx.clustering(G, nid)
        comp_score = 1 - comp_sizes.get(nid, 1) / max_comp

        # Isolation = weighted combination of:
        #   - low degree (peripheral node)
        #   - low clustering coefficient (not well-connected)
        #   - small connected component (isolated cluster)
        isolation = (0.4 * (1 - deg / (max_deg + 1e-10)) +
                     0.3 * (1 - cc) +
                     0.3 * comp_score)

        for idx in graph['nodes'][nid]:
            scores[idx] += isolation
            counts[idx] += 1

    # Average for transactions in multiple nodes
    counts[counts == 0] = 1
    scores /= counts

    # Uncovered transactions get high (but not max) score
    uncov = set(range(len(X_scaled))) - covered
    if uncov and np.any(scores > 0):
        p95 = np.percentile(scores[scores > 0], 95)
        for idx in uncov:
            scores[idx] = p95

    return scores, graph, len(graph['nodes']), len(graph['links']), len(covered)

# ============================================================
# 6. Ensemble: Multiple Parameter Configurations
# ============================================================

print("\nRunning ensemble (5 configurations)...")

configs = [
    {'cover': {'n_cubes': 10, 'perc_overlap': 0.20}, 'dist': 2.0},
    {'cover': {'n_cubes': 12, 'perc_overlap': 0.25}, 'dist': 2.3},
    {'cover': {'n_cubes': 15, 'perc_overlap': 0.30}, 'dist': 2.5},
    {'cover': {'n_cubes': 18, 'perc_overlap': 0.35}, 'dist': 2.8},
    {'cover': {'n_cubes': 20, 'perc_overlap': 0.40}, 'dist': 3.0},
]

ensemble_scores = np.zeros((len(configs), len(df)))
main_graph = None

for ci, cfg in enumerate(configs):
    scores, graph, nn, ne, nc = mapper_anomaly_scores(
        X_scaled, lens, cfg['cover'], cfg['dist']
    )
    ensemble_scores[ci] = scores
    cov = nc / len(df) * 100
    print(f"  Config {ci+1}: cubes={cfg['cover']['n_cubes']}, "
          f"overlap={cfg['cover']['perc_overlap']}, dist={cfg['dist']} "
          f"→ {nn} nodes, {ne} edges, {cov:.0f}% coverage")

    # Keep the middle config's graph for visualization
    if ci == 2:
        main_graph = graph

# Final score: average across all configurations
df['anomaly_score'] = ensemble_scores.mean(axis=0)

# ============================================================
# 7. Classification
# ============================================================

threshold = np.percentile(df['anomaly_score'], 95)
df['is_anomaly'] = (df['anomaly_score'] >= threshold).astype(int)

n_anomalies = df['is_anomaly'].sum()
print(f"\nThreshold: {threshold:.4f}")
print(f"Anomalies flagged: {n_anomalies} ({n_anomalies/len(df)*100:.1f}%)")

# ============================================================
# 8. Validation
# ============================================================

anomalies = df[df['is_anomaly'] == 1]
normal = df[df['is_anomaly'] == 0]

print(f"\n{'─'*55}")
print(f"  {'Metric':<25s} {'Anomaly':>10s} {'Normal':>10s} {'Ratio':>8s}")
print(f"{'─'*55}")

for col, label in [
    ('LoginAttempts', 'Login Attempts'),
    ('TransactionAmount', 'Amount ($)'),
    ('TransactionDuration', 'Duration (s)'),
    ('AccountBalance', 'Balance ($)'),
    ('amount_zscore', 'Amount Z-Score'),
    ('n_devices', 'Devices/Account'),
    ('n_ips', 'IPs/Account'),
    ('n_locations', 'Locations/Account'),
]:
    a = anomalies[col].mean()
    n = normal[col].mean()
    r = a / (n + 1e-10)
    print(f"  {label:<25s} {a:>10.2f} {n:>10.2f} {r:>7.2f}x")

# High login capture
high_login = df['LoginAttempts'] >= 3
hl_captured = (df['is_anomaly'] == 1) & high_login
print(f"\n  High login attempts (≥3): {high_login.sum()} total, "
      f"{hl_captured.sum()} captured ({hl_captured.sum()/high_login.sum()*100:.1f}%)")

# Large transaction capture
large_tx = df['TransactionAmount'] >= df['TransactionAmount'].quantile(0.95)
lt_captured = (df['is_anomaly'] == 1) & large_tx
print(f"  Large transactions (top 5%): {large_tx.sum()} total, "
      f"{lt_captured.sum()} captured ({lt_captured.sum()/large_tx.sum()*100:.1f}%)")

# ============================================================
# 9. Visualization
# ============================================================

print(f"\nGenerating charts...")

fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=150)
fig.suptitle('TDA Mapper Fraud Detection (Single Linkage)',
             fontsize=16, fontweight='bold')

# 9a. Score distribution
ax = axes[0, 0]
ax.hist(df['anomaly_score'], bins=50, color='#2ca02c', edgecolor='white', alpha=0.8)
ax.axvline(threshold, color='red', ls='--', lw=2, label=f'Threshold ({threshold:.3f})')
ax.set_xlabel('Anomaly Score')
ax.set_ylabel('Count')
ax.set_title('Anomaly Score Distribution')
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 9b. PCA colored by score
ax = axes[0, 1]
sc = ax.scatter(lens[:, 0], lens[:, 1], c=df['anomaly_score'],
                cmap='YlOrRd', s=8, alpha=0.6)
plt.colorbar(sc, ax=ax, label='Anomaly Score')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('PCA Projection (by anomaly score)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 9c. Amount distribution
ax = axes[0, 2]
ax.hist(normal['TransactionAmount'], bins=40, alpha=0.7, color='#2ca02c',
        label='Normal', density=True, edgecolor='white')
ax.hist(anomalies['TransactionAmount'], bins=40, alpha=0.7, color='#d62728',
        label='Anomaly', density=True, edgecolor='white')
ax.set_xlabel('Transaction Amount ($)')
ax.set_ylabel('Density')
ax.set_title('Amount: Anomaly vs Normal')
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 9d. Login attempts boxplot
ax = axes[1, 0]
bp = ax.boxplot(
    [normal['LoginAttempts'].values, anomalies['LoginAttempts'].values],
    labels=['Normal', 'Anomaly'], patch_artist=True, widths=0.5,
)
bp['boxes'][0].set_facecolor('#2ca02c')
bp['boxes'][1].set_facecolor('#d62728')
for b in bp['boxes']:
    b.set_alpha(0.7)
ax.set_ylabel('Login Attempts')
ax.set_title('Login Attempts by Classification')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 9e. Feature comparison
ax = axes[1, 1]
compare = ['TransactionAmount', 'TransactionDuration', 'LoginAttempts',
           'AccountBalance', 'amount_zscore']
short_labels = ['Amount', 'Duration', 'Logins', 'Balance', 'Z-Score']
a_vals = [anomalies[f].mean() for f in compare]
n_vals = [normal[f].mean() for f in compare]
mx = [max(a, n) for a, n in zip(a_vals, n_vals)]
a_norm = [a / (m + 1e-10) for a, m in zip(a_vals, mx)]
n_norm = [n / (m + 1e-10) for n, m in zip(n_vals, mx)]
x = np.arange(len(short_labels))
w = 0.35
ax.bar(x - w/2, n_norm, w, label='Normal', color='#2ca02c', alpha=0.8)
ax.bar(x + w/2, a_norm, w, label='Anomaly', color='#d62728', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(short_labels, fontsize=9)
ax.set_ylabel('Normalized Value')
ax.set_title('Feature Profile Comparison')
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 9f. Top anomalous accounts
ax = axes[1, 2]
acct_scores = df.groupby('AccountID').agg(
    mean_score=('anomaly_score', 'mean'),
    anomaly_count=('is_anomaly', 'sum'),
).sort_values('mean_score', ascending=False)
top = acct_scores.head(15)
colors = ['#d62728' if v > 0 else '#2ca02c' for v in top['anomaly_count']]
ax.barh(range(len(top)), top['mean_score'], color=colors, alpha=0.8)
ax.set_yticks(range(len(top)))
ax.set_yticklabels(top.index, fontsize=8)
ax.set_xlabel('Mean Anomaly Score')
ax.set_title('Top 15 Anomalous Accounts')
ax.invert_yaxis()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
fig.savefig('fraud_single_linkage.png', bbox_inches='tight')
print("  Saved fraud_single_linkage.png")

# ============================================================
# 10. Interactive Mapper Graph
# ============================================================

mapper = km.KeplerMapper(verbose=0)
html = mapper.visualize(
    main_graph,
    path_html="mapper_single_linkage.html",
    title="Mapper Graph (Single Linkage)",
    custom_tooltips=df['TransactionID'].values,
    color_values=df['anomaly_score'].values,
    color_function_name="Anomaly Score",
)
print("  Saved mapper_single_linkage.html")

# ============================================================
# 11. Save Results
# ============================================================

output_cols = [
    'TransactionID', 'AccountID', 'TransactionAmount', 'TransactionDate',
    'TransactionType', 'Channel', 'LoginAttempts', 'CustomerAge',
    'CustomerOccupation', 'AccountBalance', 'anomaly_score', 'is_anomaly',
]
df[output_cols].to_csv('fraud_results_single_linkage.csv', index=False)
print("  Saved fraud_results_single_linkage.csv")

# ============================================================
# 12. Top Anomalies
# ============================================================

print(f"\n{'='*80}")
print(f"  Top 20 Anomalous Transactions")
print(f"{'='*80}")
top20 = df.nlargest(20, 'anomaly_score')[
    ['TransactionID', 'AccountID', 'TransactionAmount', 'LoginAttempts',
     'Channel', 'CustomerOccupation', 'AccountBalance', 'anomaly_score']
]
print(top20.to_string(index=False))

print(f"\n✅ Done! {n_anomalies} anomalies flagged ({n_anomalies/len(df)*100:.1f}%)")