# Bank Transaction Fraud Detection Pipeline - Setup & Run Guide

## Quick Start (5 minutes)

```bash
# 1. Navigate to project
cd /Users/macbook/Hack/Deloitte-pi-rates/fraud_pipeline

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Validate setup (optional)
python validate.py

# 5. Run full pipeline
python run_pipeline.py

# 6. Launch Streamlit review app
streamlit run app/streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

Important:
- Use the project virtual environment, not the system Python install.
- The macOS system interpreter on this machine has a broken `httpcore` / `litellm` stack that can interfere with OpenAI SDK imports.
- The app now prefers direct HTTPS calls to the OpenAI Responses API, but the cleanest setup is still the isolated project venv above.

---

## Detailed Installation

### Prerequisites
- Python 3.8+
- ~500MB free disk space
- 2GB RAM (minimum; 4GB+ recommended)

### Step-by-Step Setup

#### 1. Create Virtual Environment

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

#### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Optional but recommended for AI features:

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-5-mini"
export OPENAI_REASONING_EFFORT="medium"
```

This installs:
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - Machine learning (anomaly detection, clustering)
- `matplotlib` & `plotly` - Visualizations
- `networkx` - Graph analysis
- `streamlit` - Interactive web app
- `scipy` - Scientific computing

#### 3. Verify Installation

```bash
python validate.py
```

Expected output:
```
======================================================================
PIPELINE VALIDATION
======================================================================

✓ Testing config module...
  Project root: /Users/macbook/Hack/Deloitte-pi-rates/fraud_pipeline
  Raw data file: .../data/raw/bank_transactions_data.csv
  Risk weights sum: 1.00
  ✓ Config loaded

✓ Testing utils module...
  ...

✓ ALL VALIDATION CHECKS PASSED
======================================================================
```

---

## Running the Pipeline

### Full Pipeline Run

```bash
python run_pipeline.py
```

**Expected runtime:** 30-60 seconds on standard laptop

**Output:**
```
======================================================================
FRAUD DETECTION PIPELINE - FULL RUN
======================================================================
Started at: 2026-03-24 XX:XX:XX

============================================================
STAGE 1: DATA INGESTION & CLEANING
============================================================
Loading raw data from .../data/raw/bank_transactions_data.csv
Loaded 2512 rows, 16 columns
Parsing date columns...
Inspecting PreviousTransactionDate values...
  Found 2512 rows with future PreviousTransactionDate
  Regenerating 2512 synthetic previous dates...
Engineering features...
  Engineered 8 features
Removed X duplicate transactions

=== DATA QUALITY REPORT ===
Rows: 2512
Columns: 24
Missing values:
  No missing values
...

Cleaned data saved to .../data/processed/transactions_cleaned.csv

...

============================================================
STAGE 5: RISK SCORING & RANKING
============================================================
Combining risk signals...
  Composite score range: [0.001, 0.989]
  Risk level distribution:
    Low: 2387 (95.0%)
    Medium: 100 (4.0%)
    High: 25 (1.0%)
...

======================================================================
PIPELINE EXECUTION COMPLETE ✓
======================================================================

📊 Output Files Created:
  Cleaned Data:           .../data/processed/transactions_cleaned.csv
  Anomaly Scores:         .../outputs/reports/anomaly_scores.csv
  Graph Features:         .../outputs/reports/graph_features.csv
  Ranked Transactions:    .../outputs/reports/risk_ranked_transactions.csv
  Ranked Accounts:        .../outputs/reports/risk_ranked_accounts.csv
  Visualizations:         .../outputs/figures/
  Summary Tables:         .../outputs/reports/

📈 Risk Distribution:
  High Risk:      25 (  1.0%)
  Medium Risk:   100 (  4.0%)
  Low Risk:     2387 ( 95.0%)

🔴 Top 5 Highest Risk Accounts:
  AC00128    - Risk: 0.897 (3 high-risk transactions)
  AC00455    - Risk: 0.825 (2 high-risk transactions)
  ...

✅ Next Steps:
  1. Review outputs in .../outputs/reports/
  2. Launch Streamlit app: streamlit run app/streamlit_app.py
  3. Import data into Tableau: Use CSV files from .../outputs/reports/

Completed at: 2026-03-24 XX:XX:XX
======================================================================
```

### Launch Streamlit Review Interface

```bash
streamlit run app/streamlit_app.py
```

This opens your browser to `http://localhost:8501` with an interactive analyst review interface.

---

## Output Files

After running the pipeline, check `outputs/reports/` for:

| File | Purpose |
|------|---------|
| `transactions_cleaned.csv` | Cleaned + engineered feature dataset |
| `anomaly_scores.csv` | Per-transaction anomaly detection scores (IF, LOF, K-Means) |
| `graph_features.csv` | Per-transaction network analysis metrics |
| `risk_ranked_transactions.csv` | **Main output**: All transactions ranked by composite risk |
| `risk_ranked_accounts.csv` | Account-level risk aggregation |
| `risk_ranked_merchants.csv` | Merchant-level risk summary |
| `risk_ranked_devices.csv` | Device-level risk summary |
| `risk_ranked_ips.csv` | IP address-level risk summary |
| `numeric_summary_statistics.csv` | EDA summary statistics |
| `outliers_summary.csv` | Detected outliers by column |
| `benford_law_analysis.png` | Benford's Law visualization |
| `transaction_amount_distribution.png` | Distribution plot |
| `customer_demographics.png` | Age and occupation distributions |
| `login_attempts_distribution.png` | Login attempt anomalies |

### For Tableau Import

Use these CSV files in Tableau:
- `risk_ranked_transactions.csv` - Main fact table
- `risk_ranked_accounts.csv` - Account dimension
- `risk_ranked_merchants.csv` - Merchant dimension
- `numeric_summary_statistics.csv` - Reference metrics

---

## Streamlit App Features

### 🔍 Suspicious Transactions View
- Filter by risk level (Low/Medium/High)
- Filter by risk score range
- Filter by account, merchant, or channel
- Select transactions to review
- View anomaly detection scores
- Record analyst decision (Approve/Dismiss/Needs Review)
- Add analyst notes

### 👥 Account Summary View
- View top N highest-risk accounts
- See account-level metrics (avg risk, high-risk %)
- Inspect all transactions for selected account

### 📋 Decision Log View
- View all recorded analyst decisions
- See decision breakdown (Approve/Dismiss/Needs Review)
- Export decision log to CSV

### ℹ️ Pipeline Info View
- Dataset overview statistics
- Risk distribution chart
- Score distribution histogram
- Pipeline configuration (weights, parameters)

---

## Configuration & Tuning

All pipeline parameters are in `src/config.py`. Key settings:

### Risk Weights (adjust to emphasize different signals)

```python
RISK_WEIGHTS = {
    "isolation_forest_score": 0.25,      # Isolation Forest contribution
    "lof_score": 0.20,                   # LOF contribution
    "kmeans_anomaly_score": 0.15,        # K-Means contribution
    "graph_risk_score": 0.15,            # Graph network signals
    "login_attempt_risk": 0.10,          # Login attempt anomalies
    "amount_outlier_risk": 0.10,         # Transaction amount outliers
    "previous_date_regenerated": 0.05,   # Synthetic date penalties
}
```

To emphasize graph-based anomalies, increase `graph_risk_score` weight and decrease others.

### Anomaly Detection Parameters

```python
# Isolation Forest
ISOLATION_FOREST_CONTAMINATION = 0.05  # Expected % of anomalies

# LOF
LOF_N_NEIGHBORS = 20
LOF_CONTAMINATION = 0.05

# K-Means
KMEANS_N_CLUSTERS = 10
KMEANS_CONTAMINATION = 0.05
```

### Risk Level Thresholds

```python
RISK_LEVEL_LOW = 0.33      # Below this = Low
RISK_LEVEL_MEDIUM = 0.66   # Below this = Medium, above = High
```

To run pipeline with custom configuration:
```python
# Edit src/config.py with your values
python run_pipeline.py
```

---

## Python API Usage

You can also import and use the pipeline modules directly:

```python
from src.ingest_clean import load_and_clean
from src.eda_profile import eda_and_profile
from src.anomaly_detection import run_anomaly_detection
from src.graph_analysis import graph_analysis
from src.risk_scoring import risk_scoring

# Load and clean data
df = load_and_clean()

# Run EDA
eda_results = eda_and_profile(df)

# Run anomaly detection
anomalies = run_anomaly_detection(df)

# Run graph analysis
graph_features, graph = graph_analysis(df)

# Compute risk scores
rankings = risk_scoring(df, anomalies, graph_features)

# Access results
high_risk_txs = rankings["transactions_ranked"][
    rankings["transactions_ranked"]["risk_level"] == "High"
]
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'pandas'"

**Solution:** Make sure your virtual environment is activated:
```bash
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows
```

Then reinstall requirements:
```bash
pip install -r requirements.txt
```

### Issue: "FileNotFoundError: Raw data not found"

**Solution:** Ensure raw data is in the correct location:
```bash
ls data/raw/bank_transactions_data.csv
```

If not found, copy it:
```bash
cp /path/to/bank_transactions_data.csv data/raw/
```

### Issue: Streamlit won't launch

**Solution:** Check that streamlit is installed:
```bash
pip install streamlit
streamlit run app/streamlit_app.py
```

If still failing, try:
```bash
python -m streamlit run app/streamlit_app.py
```

### Issue: Pipeline runs slow

**Reasons:**
- Disk I/O intensive: ensure fast SSD
- K-Means clustering: increase `KMEANS_N_CLUSTERS` to speed up
- Large dataset: consider sampling first 1000 rows for testing

### Issue: "Previous date regenerated" warning

This is expected! The raw dataset has PreviousTransactionDate in the future (2024) relative to TransactionDate (2023). The pipeline regenerates synthetic reasonable previous dates and flags them. This is documented in the code and in the "previous_date_regenerated" column.

---

## Performance Notes

| Stage | Time | Memory |
|-------|------|--------|
| Data Cleaning | ~2s | ~50MB |
| EDA | ~5s | ~100MB |
| Anomaly Detection | ~15s | ~150MB |
| Graph Analysis | ~10s | ~120MB |
| Risk Scoring | ~3s | ~80MB |
| **Total** | **~35s** | **~500MB** |

Timings are for 2,512 transactions on a standard 2026 MacBook.

---

## Next Steps

1. **Explore outputs:**
   ```bash
   ls -lh outputs/reports/
   head outputs/reports/risk_ranked_transactions.csv
   ```

2. **Review in Streamlit:**
   ```bash
   streamlit run app/streamlit_app.py
   ```

3. **Import to Tableau:**
   - Open Tableau
   - Connect to CSV files in `outputs/reports/`
   - Build dashboards for stakeholders

4. **Adjust configuration:**
   - Edit `src/config.py` risk weights
   - Rerun pipeline: `python run_pipeline.py`
   - Compare results

5. **Extend pipeline:**
   - Add supervised classification if fraud labels become available
   - Integrate real-time streaming mode
   - Add GNN for graph embeddings
   - Implement full TDA (currently stubbed)

---

## Project Structure

```
fraud_pipeline/
├── data/
│   ├── raw/                    ← Original CSV input
│   └── processed/              ← Cleaned dataset
├── outputs/
│   ├── figures/                ← EDA visualizations
│   └── reports/                ← CSV outputs for Tableau
├── src/
│   ├── config.py               ← Configuration & weights
│   ├── ingest_clean.py         ← Data loading & cleaning
│   ├── eda_profile.py          ← Exploratory analysis
│   ├── benford.py              ← Benford's Law analysis
│   ├── anomaly_detection.py    ← Anomaly detection
│   ├── tda_analysis.py         ← TDA (stub)
│   ├── graph_analysis.py       ← Graph-based features
│   ├── risk_scoring.py         ← Risk aggregation
│   ├── review_store.py         ← Analyst decisions
│   └── utils.py                ← Helpers
├── app/
│   └── streamlit_app.py        ← Streamlit interface
├── notebooks/                  ← Optional exploratory notebooks
├── run_pipeline.py             ← Main orchestrator
├── validate.py                 ← Validation script
├── requirements.txt            ← Dependencies
├── README.md                   ← Project overview
├── SETUP_GUIDE.md              ← This file
└── .gitignore
```

---

## Support & Questions

For issues or questions:
1. Check logs: `python run_pipeline.py 2>&1 | tee pipeline.log`
2. Review README.md for architecture details
3. Check code comments in `src/*.py`
4. Edit `src/config.py` to adjust parameters

---

**Last Updated:** 2026-03-24  
**Status:** Production-ready for demo and competition  
**Maintainer:** Fraud Pipeline Team
