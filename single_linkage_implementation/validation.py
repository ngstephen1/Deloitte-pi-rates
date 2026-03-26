"""
TDA Mapper Validation on Kaggle Credit Card Fraud Dataset
==========================================================
Compares 4 methods on labeled data:
  1. Mapper + Single Linkage (our TDA approach)
  2. Isolation Forest (unsupervised baseline)
  3. Local Outlier Factor (unsupervised baseline)
  4. XGBoost (supervised upper bound)

Dataset: creditcard.csv from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
  - 284,807 transactions, 492 fraud (0.17%)
  - Features: V1-V28 (PCA), Time, Amount, Class

Usage:
  1. Download creditcard.csv from Kaggle
  2. Place it in the same directory as this script
  3. Run: python validate_mapper.py

Output:
  - Comparison table with Precision, Recall, F1, ROC-AUC
  - validation_results.png (6-panel chart)
  - validation_results.csv (per-transaction scores)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, roc_curve,
    confusion_matrix, average_precision_score,
)
from sklearn.model_selection import train_test_split
import kmapper as km
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. Load Data
# ============================================================

print("=" * 70)
print("  Mapper Validation on Kaggle Credit Card Fraud Dataset")
print("=" * 70)

df = pd.read_csv('creditcard.csv')
print(f"\nLoaded: {len(df)} transactions")
print(f"Fraud:  {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")
print(f"Normal: {(df['Class']==0).sum()}")

# ============================================================
# 2. Prepare Features
# ============================================================

# V1-V28 are already PCA-transformed
# Scale Time and Amount to match
feature_cols = [f'V{i}' for i in range(1, 29)] + ['Amount']

X = df[feature_cols].values
y = df['Class'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Features: {len(feature_cols)}")

# ============================================================
# 3. Subsample for Mapper (full dataset too large for Mapper)
# ============================================================

# Mapper is O(n^2) so we subsample: keep ALL fraud + random normal
# This preserves the detection task while making Mapper feasible

np.random.seed(42)
fraud_idx = np.where(y == 1)[0]
normal_idx = np.where(y == 0)[0]

# Keep all fraud + 10,000 random normal
n_normal_sample = 10000
normal_sample = np.random.choice(normal_idx, n_normal_sample, replace=False)
subset_idx = np.concatenate([fraud_idx, normal_sample])
np.random.shuffle(subset_idx)

X_sub = X_scaled[subset_idx]
y_sub = y[subset_idx]

print(f"\nMapper subsample: {len(X_sub)} transactions "
      f"({y_sub.sum()} fraud, {(y_sub==0).sum()} normal)")

# PCA lens for Mapper
pca = PCA(n_components=2)
lens_sub = pca.fit_transform(X_sub)

# ============================================================
# 4. Method 1: Mapper + Single Linkage
# ============================================================

print("\n" + "─" * 50)
print("  Method 1: Mapper + Single Linkage")
print("─" * 50)

def mapper_scores(X, lens, cover_params, dist_threshold):
    mapper = km.KeplerMapper(verbose=0)
    clusterer = AgglomerativeClustering(
        n_clusters=None, distance_threshold=dist_threshold, linkage='single'
    )
    graph = mapper.map(
        lens, X,
        cover=km.Cover(**cover_params),
        clusterer=clusterer,
    )

    covered = set()
    for mems in graph['nodes'].values():
        covered.update(mems)

    G = nx.Graph()
    for nid, mems in graph['nodes'].items():
        G.add_node(nid, members=mems)
    for nid, conns in graph['links'].items():
        for t in conns:
            G.add_edge(nid, t)

    max_deg = max((G.degree(n) for n in G.nodes()), default=1)

    components = list(nx.connected_components(G))
    comp_sizes = {}
    for comp in components:
        size = sum(len(graph['nodes'][n]) for n in comp)
        for n in comp:
            comp_sizes[n] = size
    max_comp = max(comp_sizes.values()) if comp_sizes else 1

    scores = np.zeros(len(X))
    counts = np.zeros(len(X))

    for nid in G.nodes():
        deg = G.degree(nid)
        cc = nx.clustering(G, nid)
        comp_s = 1 - comp_sizes.get(nid, 1) / max_comp
        iso = 0.4*(1-deg/(max_deg+1e-10)) + 0.3*(1-cc) + 0.3*comp_s
        for idx in graph['nodes'][nid]:
            scores[idx] += iso
            counts[idx] += 1

    counts[counts == 0] = 1
    scores /= counts

    uncov = set(range(len(X))) - covered
    if uncov and np.any(scores > 0):
        p95 = np.percentile(scores[scores > 0], 95)
        for idx in uncov:
            scores[idx] = p95

    return scores

# Ensemble over 5 configurations
configs = [
    {'cover': {'n_cubes': 10, 'perc_overlap': 0.20}, 'dist': 2.0},
    {'cover': {'n_cubes': 12, 'perc_overlap': 0.25}, 'dist': 2.3},
    {'cover': {'n_cubes': 15, 'perc_overlap': 0.30}, 'dist': 2.5},
    {'cover': {'n_cubes': 18, 'perc_overlap': 0.35}, 'dist': 2.8},
    {'cover': {'n_cubes': 20, 'perc_overlap': 0.40}, 'dist': 3.0},
]

ensemble = np.zeros((len(configs), len(X_sub)))
for ci, cfg in enumerate(configs):
    print(f"  Config {ci+1}/5...", end=" ", flush=True)
    ensemble[ci] = mapper_scores(X_sub, lens_sub, cfg['cover'], cfg['dist'])
    print("done")

mapper_anomaly_scores = ensemble.mean(axis=0)
print(f"  Score range: [{mapper_anomaly_scores.min():.3f}, {mapper_anomaly_scores.max():.3f}]")

# ============================================================
# 5. Method 2: Isolation Forest
# ============================================================

print("\n" + "─" * 50)
print("  Method 2: Isolation Forest")
print("─" * 50)

iso_forest = IsolationForest(
    n_estimators=200, contamination=0.01, random_state=42, n_jobs=-1
)
iso_forest.fit(X_sub)
# score_samples returns negative scores; more negative = more anomalous
# Negate and shift so higher = more anomalous
iso_raw = -iso_forest.score_samples(X_sub)
iso_scores = (iso_raw - iso_raw.min()) / (iso_raw.max() - iso_raw.min())
print(f"  Done. Score range: [{iso_scores.min():.3f}, {iso_scores.max():.3f}]")

# ============================================================
# 6. Method 3: Local Outlier Factor
# ============================================================

print("\n" + "─" * 50)
print("  Method 3: Local Outlier Factor")
print("─" * 50)

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01, novelty=False)
lof_pred = lof.fit_predict(X_sub)
lof_raw = -lof.negative_outlier_factor_
lof_scores = (lof_raw - lof_raw.min()) / (lof_raw.max() - lof_raw.min())
print(f"  Done. Score range: [{lof_scores.min():.3f}, {lof_scores.max():.3f}]")

# ============================================================
# 7. Method 4: XGBoost (supervised upper bound)
# ============================================================

print("\n" + "─" * 50)
print("  Method 4: XGBoost (supervised)")
print("─" * 50)

try:
    from xgboost import XGBClassifier
    has_xgb = True
except ImportError:
    print("  XGBoost not installed. pip install xgboost")
    print("  Skipping XGBoost — will compare unsupervised methods only.")
    has_xgb = False

if has_xgb:
    # Train/test split on the subsample
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_sub, y_sub, np.arange(len(X_sub)),
        test_size=0.3, random_state=42, stratify=y_sub
    )

    xgb = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        random_state=42, eval_metric='logloss', use_label_encoder=False,
    )
    xgb.fit(X_train, y_train)

    # Score ALL subsample points (use predict_proba for fair comparison)
    xgb_scores = xgb.predict_proba(X_sub)[:, 1]
    print(f"  Done. Score range: [{xgb_scores.min():.3f}, {xgb_scores.max():.3f}]")

# ============================================================
# 8. Evaluation
# ============================================================

print("\n" + "=" * 70)
print("  EVALUATION")
print("=" * 70)

def evaluate_method(name, scores, y_true, threshold_pct=None):
    """
    Evaluate anomaly scores against true labels.
    For unsupervised methods, we sweep thresholds and report best F1.
    """
    # ROC-AUC and Average Precision (threshold-independent)
    roc = roc_auc_score(y_true, scores)
    ap = average_precision_score(y_true, scores)

    # Find best F1 threshold
    prec_curve, rec_curve, thresholds = precision_recall_curve(y_true, scores)
    f1_curve = 2 * prec_curve * rec_curve / (prec_curve + rec_curve + 1e-10)
    best_idx = np.argmax(f1_curve)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    y_pred = (scores >= best_threshold).astype(int)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Also report at fixed top-k (flag top 5%)
    top5_threshold = np.percentile(scores, 95)
    y_pred_5 = (scores >= top5_threshold).astype(int)
    prec_5 = precision_score(y_true, y_pred_5, zero_division=0)
    rec_5 = recall_score(y_true, y_pred_5, zero_division=0)
    f1_5 = f1_score(y_true, y_pred_5, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        'method': name,
        'roc_auc': roc,
        'avg_precision': ap,
        'best_f1': f1,
        'best_precision': prec,
        'best_recall': rec,
        'best_threshold': best_threshold,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'f1_at_5pct': f1_5,
        'prec_at_5pct': prec_5,
        'rec_at_5pct': rec_5,
        'scores': scores,
    }

methods = {
    'Mapper (Single Linkage)': mapper_anomaly_scores,
    'Isolation Forest': iso_scores,
    'LOF': lof_scores,
}
if has_xgb:
    methods['XGBoost (supervised)'] = xgb_scores

results = {}
for name, scores in methods.items():
    results[name] = evaluate_method(name, scores, y_sub)

# ============================================================
# 9. Print Results
# ============================================================

print(f"\n{'─'*90}")
print(f"{'Method':<28s} {'ROC-AUC':>8s} {'Avg-PR':>8s} "
      f"{'Best-F1':>8s} {'Prec':>8s} {'Recall':>8s} "
      f"{'F1@5%':>8s}")
print(f"{'─'*90}")

for name in methods:
    r = results[name]
    print(f"  {name:<26s} {r['roc_auc']:>8.4f} {r['avg_precision']:>8.4f} "
          f"{r['best_f1']:>8.4f} {r['best_precision']:>8.4f} {r['best_recall']:>8.4f} "
          f"{r['f1_at_5pct']:>8.4f}")

print(f"{'─'*90}")

print(f"\n  Confusion matrices (at best F1 threshold):")
for name in methods:
    r = results[name]
    print(f"\n  {name}:")
    print(f"    TP={r['tp']:>5d}  FP={r['fp']:>5d}")
    print(f"    FN={r['fn']:>5d}  TN={r['tn']:>5d}")

# ============================================================
# 10. Visualization
# ============================================================

print(f"\nGenerating charts...")

n_methods = len(methods)
fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=150)
fig.suptitle('TDA Mapper Validation: Credit Card Fraud Detection',
             fontsize=16, fontweight='bold')

colors = {
    'Mapper (Single Linkage)': '#2ca02c',
    'Isolation Forest': '#1f77b4',
    'LOF': '#ff7f0e',
    'XGBoost (supervised)': '#d62728',
}

# 10a. ROC Curves
ax = axes[0, 0]
for name in methods:
    fpr, tpr, _ = roc_curve(y_sub, results[name]['scores'])
    ax.plot(fpr, tpr, color=colors[name], lw=2,
            label=f"{name} (AUC={results[name]['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves')
ax.legend(fontsize=8, loc='lower right')
ax.grid(alpha=0.2)

# 10b. Precision-Recall Curves
ax = axes[0, 1]
for name in methods:
    prec, rec, _ = precision_recall_curve(y_sub, results[name]['scores'])
    ax.plot(rec, prec, color=colors[name], lw=2,
            label=f"{name} (AP={results[name]['avg_precision']:.3f})")
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves')
ax.legend(fontsize=8, loc='upper right')
ax.grid(alpha=0.2)

# 10c. Bar chart: Best F1
ax = axes[0, 2]
names = list(methods.keys())
f1s = [results[n]['best_f1'] for n in names]
bars = ax.bar(range(len(names)), f1s, color=[colors[n] for n in names], alpha=0.85)
for b, v in zip(bars, f1s):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
            f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
ax.set_xticks(range(len(names)))
ax.set_xticklabels([n.replace(' (', '\n(') for n in names], fontsize=8)
ax.set_ylabel('Best F1 Score')
ax.set_title('Best F1 Comparison')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 10d. Score distributions for fraud vs normal (Mapper)
ax = axes[1, 0]
ax.hist(mapper_anomaly_scores[y_sub == 0], bins=50, alpha=0.7,
        color='steelblue', label='Normal', density=True, edgecolor='white')
ax.hist(mapper_anomaly_scores[y_sub == 1], bins=50, alpha=0.7,
        color='#d62728', label='Fraud', density=True, edgecolor='white')
ax.set_xlabel('Mapper Anomaly Score')
ax.set_ylabel('Density')
ax.set_title('Mapper Score Distribution (Fraud vs Normal)')
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 10e. Precision and Recall at different thresholds (Mapper)
ax = axes[1, 1]
percentiles = [90, 92, 94, 95, 96, 97, 98, 99]
precs, recs = [], []
for p in percentiles:
    t = np.percentile(mapper_anomaly_scores, p)
    pred = (mapper_anomaly_scores >= t).astype(int)
    precs.append(precision_score(y_sub, pred, zero_division=0))
    recs.append(recall_score(y_sub, pred, zero_division=0))
ax.plot(percentiles, precs, 'o-', color='#2ca02c', lw=2, label='Precision')
ax.plot(percentiles, recs, 's-', color='#d62728', lw=2, label='Recall')
ax.set_xlabel('Threshold Percentile')
ax.set_ylabel('Score')
ax.set_title('Mapper: Precision/Recall vs Threshold')
ax.legend()
ax.grid(alpha=0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 10f. Summary metrics bar
ax = axes[1, 2]
x = np.arange(len(names))
w = 0.25
roc_vals = [results[n]['roc_auc'] for n in names]
ap_vals = [results[n]['avg_precision'] for n in names]
f1_vals = [results[n]['best_f1'] for n in names]
ax.bar(x - w, roc_vals, w, label='ROC-AUC', color='steelblue', alpha=0.8)
ax.bar(x, ap_vals, w, label='Avg Precision', color='#2ca02c', alpha=0.8)
ax.bar(x + w, f1_vals, w, label='Best F1', color='#d62728', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([n.replace(' (', '\n(') for n in names], fontsize=8)
ax.set_ylabel('Score')
ax.set_title('All Metrics Comparison')
ax.legend(fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
fig.savefig('validation_results.png', bbox_inches='tight')
print("  Saved validation_results.png")

# ============================================================
# 11. Save detailed results
# ============================================================

output = pd.DataFrame({
    'index': subset_idx,
    'true_label': y_sub,
    'mapper_score': mapper_anomaly_scores,
    'iforest_score': iso_scores,
    'lof_score': lof_scores,
})
if has_xgb:
    output['xgb_score'] = xgb_scores
output.to_csv('validation_results.csv', index=False)
print("  Saved validation_results.csv")

# ============================================================
# 12. Final Summary
# ============================================================

print(f"\n{'='*70}")
print(f"  FINAL SUMMARY")
print(f"{'='*70}")
print(f"\n  Dataset: {len(df)} transactions ({df['Class'].sum()} fraud)")
print(f"  Subsample: {len(X_sub)} ({y_sub.sum()} fraud, {(y_sub==0).sum()} normal)")
print(f"\n  Best performing method by metric:")
for metric in ['roc_auc', 'avg_precision', 'best_f1']:
    best = max(results.keys(), key=lambda n: results[n][metric])
    print(f"    {metric:<20s}: {best} ({results[best][metric]:.4f})")

print(f"\n  Mapper vs baselines:")
m = results['Mapper (Single Linkage)']
for baseline in ['Isolation Forest', 'LOF']:
    b = results[baseline]
    for metric in ['roc_auc', 'best_f1']:
        diff = m[metric] - b[metric]
        better = "BETTER" if diff > 0 else "WORSE"
        print(f"    vs {baseline:<20s} {metric}: {diff:+.4f} ({better})")

print(f"\n✅ Done!")