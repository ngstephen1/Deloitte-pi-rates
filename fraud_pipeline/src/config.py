"""
Configuration and constants for the fraud detection pipeline.
Centralized weights, thresholds, and parameters for easy tuning.
"""

import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency during install/bootstrap
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT.parent / ".env", override=False)
    load_dotenv(PROJECT_ROOT / ".env", override=False)

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"
CHATOPS_DIR = OUTPUT_DIR / "chatops"
CHATOPS_ACTIVE_DIR = CHATOPS_DIR / "active_context"
CHATOPS_UPLOADS_DIR = CHATOPS_DIR / "uploads"
CHATOPS_EXPORTS_DIR = CHATOPS_DIR / "exports"
CHATOPS_IMAGE_UPLOADS_DIR = CHATOPS_UPLOADS_DIR / "images"
CHATOPS_IMAGE_EXPORTS_DIR = CHATOPS_EXPORTS_DIR / "image_reviews"

# Ensure directories exist
for d in [
    PROCESSED_DATA_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    CHATOPS_DIR,
    CHATOPS_ACTIVE_DIR,
    CHATOPS_UPLOADS_DIR,
    CHATOPS_EXPORTS_DIR,
    CHATOPS_IMAGE_UPLOADS_DIR,
    CHATOPS_IMAGE_EXPORTS_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

# Input/Output file paths
RAW_DATA_FILE = RAW_DATA_DIR / "bank_transactions_data.csv"
CLEANED_DATA_FILE = PROCESSED_DATA_DIR / "transactions_cleaned.csv"
ANOMALY_SCORES_FILE = REPORTS_DIR / "anomaly_scores.csv"
GRAPH_FEATURES_FILE = REPORTS_DIR / "graph_features.csv"
RISK_TRANSACTIONS_FILE = REPORTS_DIR / "risk_ranked_transactions.csv"
RISK_ACCOUNTS_FILE = REPORTS_DIR / "risk_ranked_accounts.csv"
RISK_MERCHANTS_FILE = REPORTS_DIR / "risk_ranked_merchants.csv"
RISK_DEVICES_FILE = REPORTS_DIR / "risk_ranked_devices.csv"
RISK_IPS_FILE = REPORTS_DIR / "risk_ranked_ips.csv"
TOP_LOCATIONS_FILE = REPORTS_DIR / "top_locations.csv"
EXECUTIVE_SUMMARY_FILE = REPORTS_DIR / "executive_summary.json"
ANALYST_DECISIONS_FILE = REPORTS_DIR / "analyst_decisions.csv"
CHATOPS_MANIFEST_FILE = CHATOPS_ACTIVE_DIR / "manifest.json"
CHATOPS_ALERT_STATE_FILE = CHATOPS_DIR / "alert_state.json"
CHATOPS_DISCORD_STATE_FILE = CHATOPS_DIR / "discord_bot_state.json"

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
DECISION_OPTIONS = ["Approve Flag", "Dismiss", "Needs Review"]

# ============================================================================
# OPENAI EXPLANATIONS (Step 7)
# ============================================================================
# Set to True to enable natural-language explanations via OpenAI API
# Requires OPENAI_API_KEY environment variable
# If disabled or API key missing, pipeline still runs without explanations
USE_OPENAI_EXPLANATIONS = False  # Pipeline-time explanation generation

# Streamlit AI assistant controls
ENABLE_AI_FEATURES = True
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
OPENAI_REASONING_EFFORT = os.environ.get("OPENAI_REASONING_EFFORT", "none")
OPENAI_API_BASE_URL = os.environ.get("OPENAI_API_BASE_URL", "https://api.openai.com/v1")
OPENAI_REQUEST_TIMEOUT_SECONDS = int(os.environ.get("OPENAI_REQUEST_TIMEOUT_SECONDS", "45"))
OPENAI_TRANSPORT_PREFERENCE = os.environ.get("OPENAI_TRANSPORT_PREFERENCE", "http")
AI_MAX_CONTEXT_ROWS = 5
AI_MAX_OUTPUT_TOKENS = 700

# ============================================================================
# OPENCLAW / CHATOPS
# ============================================================================
def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _clean_env_text(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default)).strip().strip('"').strip("'")


OPENCLAW_ENABLED = _env_bool("OPENCLAW_ENABLED", True)
OPENCLAW_STREAMLIT_AUTO_SEND = _env_bool("OPENCLAW_STREAMLIT_AUTO_SEND", True)
OPENCLAW_PUBLIC_BASE_URL = _clean_env_text("OPENCLAW_PUBLIC_BASE_URL", "http://localhost:3002").rstrip("/")
OPENCLAW_AGENT = _clean_env_text("OPENCLAW_AGENT", "main")
OPENCLAW_WEBHOOK_FORMAT = _clean_env_text("OPENCLAW_WEBHOOK_FORMAT", "discord").lower()
OPENCLAW_DEFAULT_CHANNEL_ID = _clean_env_text("OPENCLAW_DEFAULT_CHANNEL_ID")
OPENCLAW_DEFAULT_CONVERSATION_ID = _clean_env_text("OPENCLAW_DEFAULT_CONVERSATION_ID")
OPENCLAW_DEFAULT_ACTOR_ID = _clean_env_text("OPENCLAW_DEFAULT_ACTOR_ID")
OPENCLAW_DISCORD_WEBHOOK_URL = _clean_env_text("OPENCLAW_DISCORD_WEBHOOK_URL")
OPENCLAW_WEBHOOK_URL = _clean_env_text("OPENCLAW_WEBHOOK_URL") or OPENCLAW_DISCORD_WEBHOOK_URL
OPENCLAW_DEFAULT_WEBHOOK_URL = _clean_env_text("OPENCLAW_DEFAULT_WEBHOOK_URL") or OPENCLAW_WEBHOOK_URL
OPENCLAW_REPORT_TOP_N = int(os.environ.get("OPENCLAW_REPORT_TOP_N", "5"))
OPENCLAW_ALERT_DEDUPE_HOURS = int(os.environ.get("OPENCLAW_ALERT_DEDUPE_HOURS", "8"))
OPENCLAW_ALERT_TRANSACTION_CRITICAL_THRESHOLD = float(
    os.environ.get("OPENCLAW_ALERT_TRANSACTION_CRITICAL_THRESHOLD", "0.8")
)
OPENCLAW_ALERT_TRANSACTION_WARNING_THRESHOLD = float(
    os.environ.get("OPENCLAW_ALERT_TRANSACTION_WARNING_THRESHOLD", str(RISK_LEVEL_MEDIUM))
)
OPENCLAW_ALERT_ACCOUNT_THRESHOLD = float(os.environ.get("OPENCLAW_ALERT_ACCOUNT_THRESHOLD", "0.75"))
OPENCLAW_ALERT_MERCHANT_HIGH_RISK_COUNT = int(os.environ.get("OPENCLAW_ALERT_MERCHANT_HIGH_RISK_COUNT", "2"))
OPENCLAW_ALERT_DEVICE_HIGH_RISK_COUNT = int(os.environ.get("OPENCLAW_ALERT_DEVICE_HIGH_RISK_COUNT", "1"))
OPENCLAW_ALERT_LOCATION_FLAGGED_COUNT = int(os.environ.get("OPENCLAW_ALERT_LOCATION_FLAGGED_COUNT", "8"))
OPENCLAW_ALERT_PENDING_REVIEW_THRESHOLD = int(os.environ.get("OPENCLAW_ALERT_PENDING_REVIEW_THRESHOLD", "12"))
OPENCLAW_ALERT_MAX_ITEMS = int(os.environ.get("OPENCLAW_ALERT_MAX_ITEMS", "6"))

# ChatOps model defaults stay separate from the lighter dashboard defaults.
OPENCLAW_OPENAI_MODEL = _clean_env_text("OPENCLAW_OPENAI_MODEL", "gpt-5.4")
OPENCLAW_OPENAI_REASONING_EFFORT = _clean_env_text("OPENCLAW_OPENAI_REASONING_EFFORT", "medium").lower()
OPENCLAW_OPENAI_MAX_OUTPUT_TOKENS = int(os.environ.get("OPENCLAW_OPENAI_MAX_OUTPUT_TOKENS", "600"))
OPENCLAW_IMAGE_MAX_OUTPUT_TOKENS = int(os.environ.get("OPENCLAW_IMAGE_MAX_OUTPUT_TOKENS", "950"))

# Discord companion bot controls
DISCORD_REPLY_ONLY_ON_MENTION = _env_bool("DISCORD_REPLY_ONLY_ON_MENTION", False)
DISCORD_ALLOW_DMS = _env_bool("DISCORD_ALLOW_DMS", True)
OPENCLAW_DISCORD_PROACTIVE_ENABLED = _env_bool("OPENCLAW_DISCORD_PROACTIVE_ENABLED", False)
OPENCLAW_DISCORD_PROACTIVE_IDLE_MINUTES = int(os.environ.get("OPENCLAW_DISCORD_PROACTIVE_IDLE_MINUTES", "180"))
OPENCLAW_DISCORD_PROACTIVE_MIN_INTERVAL_MINUTES = int(
    os.environ.get("OPENCLAW_DISCORD_PROACTIVE_MIN_INTERVAL_MINUTES", "360")
)
OPENCLAW_DISCORD_PROACTIVE_MAX_PER_DAY = int(os.environ.get("OPENCLAW_DISCORD_PROACTIVE_MAX_PER_DAY", "3"))
OPENCLAW_DISCORD_PROACTIVE_POLL_MINUTES = int(os.environ.get("OPENCLAW_DISCORD_PROACTIVE_POLL_MINUTES", "10"))
OPENCLAW_DISCORD_MAX_CONTEXT_MESSAGES = int(os.environ.get("OPENCLAW_DISCORD_MAX_CONTEXT_MESSAGES", "8"))
OPENCLAW_DISCORD_UPLOAD_PREVIEW_ROWS = int(os.environ.get("OPENCLAW_DISCORD_UPLOAD_PREVIEW_ROWS", "5"))
OPENCLAW_DISCORD_THREAD_AUTO_ARCHIVE_MINUTES = int(os.environ.get("OPENCLAW_DISCORD_THREAD_AUTO_ARCHIVE_MINUTES", "1440"))
OPENCLAW_DISCORD_CASE_ESCALATION_HOURS = int(os.environ.get("OPENCLAW_DISCORD_CASE_ESCALATION_HOURS", "6"))

# ============================================================================
# LOGGING
# ============================================================================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
