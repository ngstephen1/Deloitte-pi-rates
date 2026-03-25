"""
Configuration and constants for the fraud detection pipeline.
Centralized weights, thresholds, and parameters for easy tuning.
"""

from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Ensure directories exist
for d in [PROCESSED_DATA_DIR, FIGURES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Input/Output file paths
RAW_DATA_FILE = RAW_DATA_DIR / "bank_transactions_data.csv"
CLEANED_DATA_FILE = PROCESSED_DATA_DIR / "transactions_cleaned.csv"
ANOMALY_SCORES_FILE = REPORTS_DIR / "anomaly_scores.csv"
GRAPH_FEATURES_FILE = REPORTS_DIR / "graph_features.csv"
RISK_TRANSACTIONS_FILE = REPORTS_DIR / "risk_ranked_transactions.csv"
RISK_ACCOUNTS_FILE = REPORTS_DIR / "risk_ranked_accounts.csv"
ANALYST_DECISIONS_FILE = REPORTS_DIR / "analyst_decisions.csv"

# ============================================================================
# DATA CLEANING PARAMETERS
# ============================================================================
# If PreviousTransactionDate is unrealistic, regenerate synthetic dates
# with random intervals. Set to False to keep original.
REGENERATE_PREVIOUS_DATES = True
MIN_DAYS_SINCE_PREVIOUS = 1  # Minimum days between transactions
MAX_DAYS_SINCE_PREVIOUS = 180  # Maximum days between transactions (for synthetic)

# ============================================================================
# ANOMALY DETECTION PARAMETERS
# ============================================================================
# Isolation Forest
ISOLATION_FOREST_CONTAMINATION = 0.05  # Assume ~5% are anomalies
ISOLATION_FOREST_RANDOM_STATE = 42

# Local Outlier Factor
LOF_N_NEIGHBORS = 20
LOF_CONTAMINATION = 0.05

# K-Means clustering for anomaly detection
KMEANS_N_CLUSTERS = 10
KMEANS_RANDOM_STATE = 42
KMEANS_CONTAMINATION = 0.05  # Treat smallest clusters as anomalies

# Numeric features for anomaly detection
ANOMALY_FEATURES = [
    "transactionamount",
    "transactionduration",
    "loginattempts",
    "account_transaction_count",
    "merchant_transaction_count",
    "location_transaction_count",
    "transaction_amount_to_balance_ratio",
    "time_since_previous_transaction",
]

# ============================================================================
# GRAPH ANALYSIS PARAMETERS
# ============================================================================
# Entity types to include in transaction graph
GRAPH_ENTITIES = ["account_id", "merchant_id", "device_id", "ip_address", "location"]

# Thresholds for suspicious graph patterns
SUSPICIOUS_DEGREE_PERCENTILE = 90  # Nodes in top 10% by degree are suspicious
SUSPICIOUS_CENTRALITY_PERCENTILE = 85  # High centrality = bridge in network

# ============================================================================
# RISK SCORING WEIGHTS
# ============================================================================
# These are transparent weights for the composite risk score.
# Each component is normalized to [0, 1] before weighting.
# Adjust these to emphasize different signals.

RISK_WEIGHTS = {
    # Anomaly detection signals
    "isolation_forest_score": 0.25,
    "lof_score": 0.20,
    "kmeans_anomaly_score": 0.15,

    # Graph-based signals
    "graph_risk_score": 0.15,

    # Transaction metadata signals
    "login_attempt_risk": 0.10,
    "amount_outlier_risk": 0.10,

    # Synthetic/data quality signals
    "previous_date_regenerated": 0.05,  # Small penalty for synthetic previous dates
}

# Ensure weights sum to ~1.0 for interpretability
assert 0.99 <= sum(RISK_WEIGHTS.values()) <= 1.01, "Risk weights must sum to 1.0"

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================
# DPI for saved figures
FIGURE_DPI = 150

# Default color scheme
COLOR_NORMAL = "#1f77b4"  # Blue
COLOR_ANOMALOUS = "#d62728"  # Red
COLOR_SUSPICIOUS = "#ff7f0e"  # Orange

# ============================================================================
# ANALYST REVIEW PARAMETERS (Streamlit)
# ============================================================================
# Risk level thresholds for filtering
RISK_LEVEL_LOW = 0.33
RISK_LEVEL_MEDIUM = 0.66
RISK_LEVEL_HIGH = 1.0

# Decision options
DECISION_OPTIONS = ["Approve", "Dismiss", "Needs Review"]

# ============================================================================
# OPENAI EXPLANATIONS (Step 7)
# ============================================================================
# Set to True to enable natural-language explanations via OpenAI API
# Requires OPENAI_API_KEY environment variable
# If disabled or API key missing, pipeline still runs without explanations
USE_OPENAI_EXPLANATIONS = False  # Change to True to enable

# ============================================================================
# LOGGING
# ============================================================================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
