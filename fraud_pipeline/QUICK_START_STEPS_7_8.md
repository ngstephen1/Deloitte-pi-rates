# Final Checklist & Commands

## ✅ Implementation Complete

Steps 7-8 are fully implemented and ready to run. Here are the **exact commands** for your terminal.

## Pre-Run Checklist

- [x] Step 1-6 pipeline implemented and working
- [x] Step 7 reporting module created (`src/reporting.py`)
- [x] Step 7 OpenAI utilities created (`src/openai_explanations.py`)
- [x] Step 8 Streamlit app upgraded (`app/streamlit_app.py`)
- [x] Configuration updated (`src/config.py`)
- [x] Pipeline orchestrator updated (`run_pipeline.py`)
- [x] Requirements updated (`requirements.txt`)
- [x] README updated with full documentation
- [x] Verification script created and passing
- [x] Raw data file exists in `data/raw/`

## Terminal Commands (Copy-Paste Ready)

### 1. First Time Setup (One-time)

```bash
cd /Users/macbook/Hack/Deloitte-pi-rates/fraud_pipeline

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify everything is set up
python3 verify_steps_7_8.py
```

Expected output:
```
✅ ALL CHECKS PASSED

Next steps:
  1. Run pipeline:        python3 run_pipeline.py
  2. Launch Streamlit:    streamlit run app/streamlit_app.py
```

### 2. Run Full Pipeline (Steps 1-7)

```bash
cd /Users/macbook/Hack/Deloitte-pi-rates/fraud_pipeline
source venv/bin/activate  # if using venv

python3 run_pipeline.py
```

**Expected output:**
- Logs showing progress through each stage
- Stage 1: Data cleaning ✓
- Stage 2: EDA & profiling ✓
- Stage 3: Anomaly detection ✓
- Stage 4: Graph analysis ✓
- Stage 5: Risk scoring ✓
- Stage 6: Reporting ✓
- Final message: ✅ PIPELINE EXECUTION COMPLETE
- Runtime: ~40-60 seconds

**Files created:**
```
outputs/
├── figures/
│   ├── risk_by_account.html
│   ├── risk_by_merchant.html
│   ├── risk_by_location.html
│   ├── amount_vs_risk_scatter.html
│   ├── risk_distribution.html
│   ├── risk_by_channel.html
│   └── risk_components_heatmap.html
└── reports/
    ├── anomaly_scores.csv
    ├── graph_features.csv
    ├── risk_ranked_transactions.csv
    ├── risk_ranked_accounts.csv
    ├── risk_ranked_merchants.csv
    ├── risk_ranked_devices.csv
    ├── risk_ranked_ips.csv
    ├── tableau_transactions.csv
    ├── tableau_accounts.csv
    ├── tableau_merchants.csv
    ├── tableau_devices.csv
    ├── tableau_ips.csv
    └── executive_summary.json
```

### 3. Launch Interactive App (Step 8)

After pipeline completes, in a new terminal (or same terminal after pipeline done):

```bash
cd /Users/macbook/Hack/Deloitte-pi-rates/fraud_pipeline
source venv/bin/activate  # if using venv

streamlit run app/streamlit_app.py
```

**Expected output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://xxx.xxx.xxx.xxx:8501
```

**Then:**
- Browser opens automatically to `http://localhost:8501`
- If not, manually open that URL
- Start exploring the app!

### 4. (Optional) Enable OpenAI Explanations

If you want natural-language risk explanations:

```bash
# Set your API key (get from https://platform.openai.com/api-keys)
export OPENAI_API_KEY="sk-proj-xxxxxxxxxxxx"

# Edit config to enable
# File: src/config.py
# Line ~135: Change USE_OPENAI_EXPLANATIONS = False  →  USE_OPENAI_EXPLANATIONS = True

# Re-run pipeline
python3 run_pipeline.py
```

If API key fails, pipeline continues without explanations (safe degradation).

## Interactive App Guide

### Page 1: Overview Dashboard
- View KPI metrics (total transactions, high-risk count, volume)
- See risk distribution charts
- Scroll to see pre-generated visualizations

### Page 2: Suspicious Transactions
1. Use filters at top (risk level, score, account, merchant)
2. Click on a transaction in the table to view details
3. Scroll down to see:
   - Amount and risk metrics
   - Risk component breakdown (bar chart)
   - AI explanation (if enabled)
   - **Analyst Review** section
4. Select decision (Approve/Dismiss/Needs Review)
5. Add optional notes
6. Click "💾 Save Decision"

### Page 3: Account Risk
1. Select account from dropdown
2. View account KPIs
3. Scroll to see transactions for that account
4. Optionally, read AI explanation

### Page 4: Merchants & Locations
- View top merchants/locations by risk
- See transaction counts and high-risk percentages

### Page 5: Review Log
- View all analyst decisions made
- Click "📥 Download Review Log" to export as CSV

### Page 6: Pipeline Info
- See configuration (weights, thresholds)
- Check data paths
- Learn about features used

## Troubleshooting

### Problem: "Pipeline outputs not found"
**Solution:**
```bash
python3 run_pipeline.py  # Run the pipeline first
# Wait for completion (look for ✅ at end of logs)
# Then launch Streamlit
```

### Problem: "Module 'openai' not found"
**Solution:**
```bash
pip install openai==0.27.8
# Or just disable OpenAI:
# In src/config.py: USE_OPENAI_EXPLANATIONS = False
```

### Problem: "streamlit: command not found"
**Solution:**
```bash
source venv/bin/activate  # Activate virtual environment
pip install streamlit==1.28.1
streamlit run app/streamlit_app.py
```

### Problem: "Port 8501 already in use"
**Solution:**
```bash
# Kill the other process or use different port:
streamlit run app/streamlit_app.py --server.port 8502
```

### Problem: Pipeline runs but Streamlit shows no data
**Solution:**
1. Check that `outputs/reports/risk_ranked_transactions.csv` exists:
   ```bash
   ls -lh outputs/reports/
   ```
2. If missing, re-run pipeline
3. If exists but Streamlit not loading: refresh browser (Ctrl+R)

## Configuration Adjustments

To change model behavior, edit `src/config.py`:

```python
# Lines 60-77: Risk scoring weights
RISK_WEIGHTS = {
    "isolation_forest_score": 0.25,  # ← increase to emphasize anomaly detection
    "lof_score": 0.20,
    "kmeans_anomaly_score": 0.15,
    "graph_risk_score": 0.15,        # ← increase to emphasize network patterns
    "login_attempt_risk": 0.10,
    "amount_outlier_risk": 0.10,
    "previous_date_regenerated": 0.05,
}

# Lines 48-50: Anomaly detection sensitivity
ISOLATION_FOREST_CONTAMINATION = 0.05  # Higher = more anomalies detected
LOF_N_NEIGHBORS = 20                    # Higher = larger neighborhoods
KMEANS_N_CLUSTERS = 10                  # Adjust based on data patterns

# Lines 91-93: Risk level cutoffs
RISK_LEVEL_LOW = 0.33
RISK_LEVEL_MEDIUM = 0.66
```

After editing, re-run:
```bash
python3 run_pipeline.py
# Streamlit will auto-reload
```

## File Structure Reference

```
fraud_pipeline/
├── README.md                          # Main documentation
├── STEPS_7_8_GUIDE.md                # Step 7-8 detailed guide
├── IMPLEMENTATION_STEPS_7_8.md       # Implementation details
├── requirements.txt                   # Dependencies
├── run_pipeline.py                   # Main orchestrator
├── validate.py                       # Pre-flight checks
├── verify_steps_7_8.py              # Step 7-8 verification
│
├── data/
│   ├── raw/bank_transactions_data.csv
│   └── processed/                    # Cleaned data (created by pipeline)
│
├── outputs/
│   ├── figures/                      # HTML visualizations (created by Step 7)
│   └── reports/                      # CSV exports (created by Steps 5-7)
│
├── src/
│   ├── config.py                     # Configuration & weights ⚙️
│   ├── utils.py                      # Helper functions
│   ├── ingest_clean.py              # Step 1: Data loading
│   ├── eda_profile.py               # Step 2: Profiling
│   ├── benford.py                   # Step 2b: Benford's Law
│   ├── anomaly_detection.py         # Step 3: Anomaly detection
│   ├── tda_analysis.py              # Step 3b: TDA
│   ├── graph_analysis.py            # Step 4: Graph analysis
│   ├── risk_scoring.py              # Step 5: Scoring
│   ├── reporting.py                 # Step 7: Visualizations & summaries ✨ NEW
│   ├── openai_explanations.py       # Step 7: AI explanations ✨ NEW
│   ├── review_store.py              # Step 6/8: Decision storage
│   └── __init__.py
│
└── app/
    └── streamlit_app.py             # Step 8: Interactive UI ✨ UPGRADED
```

## Performance Expectations

| Operation | Time | Memory |
|-----------|------|--------|
| Full Pipeline (Steps 1-7) | 40-60 sec | <500MB |
| Streamlit First Load | 2-3 sec | 100MB |
| Streamlit Subsequent Navigation | <100ms | - |
| OpenAI Batch (if enabled) | 5-10 sec | - |
| Filtering/Sorting | <100ms | - |

## Example Session

```bash
# Terminal 1: Setup & Pipeline
cd /Users/macbook/Hack/Deloitte-pi-rates/fraud_pipeline
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python3 verify_steps_7_8.py
# ✅ ALL CHECKS PASSED

python3 run_pipeline.py
# ... 50 seconds of processing logs ...
# ✅ PIPELINE EXECUTION COMPLETE

# Terminal 2: Launch App (while Terminal 1 running or after)
cd /Users/macbook/Hack/Deloitte-pi-rates/fraud_pipeline
source venv/bin/activate
streamlit run app/streamlit_app.py

# Browser opens at http://localhost:8501
# Now explore: Overview → Suspicious Transactions → Account Risk → Review Log → Export decisions
```

## Key Files to Remember

| File | Purpose | When To Edit |
|------|---------|--------------|
| `src/config.py` | All tunable parameters | When adjusting sensitivity/weights |
| `run_pipeline.py` | Orchestration logic | If adding new stages |
| `app/streamlit_app.py` | UI/UX | If changing app behavior |
| `outputs/reports/*.csv` | Final results | Never (auto-generated) |

## Support

If something breaks:

1. **Check logs**: Look for ✓ or ✗ in console output
2. **Run verification**: `python3 verify_steps_7_8.py`
3. **Check data**: `ls -lh outputs/reports/` and `outputs/figures/`
4. **Re-run pipeline**: `python3 run_pipeline.py`
5. **Refresh Streamlit**: Ctrl+R in browser

## Success Criteria

You'll know everything is working when:

- [x] `python3 run_pipeline.py` completes with ✅ PIPELINE EXECUTION COMPLETE
- [x] `outputs/reports/risk_ranked_transactions.csv` exists and has data
- [x] `outputs/figures/risk_by_account.html` exists
- [x] `streamlit run app/streamlit_app.py` launches without errors
- [x] Browser shows interactive Streamlit UI with 6 navigation options
- [x] Clicking on a transaction shows risk breakdown chart
- [x] Recording a decision saves it
- [x] Can download decision log as CSV

## Next Steps After Success

1. **Explore Results**: Review transactions in Streamlit
2. **Record Decisions**: Mark transactions as Approve/Dismiss/Needs Review
3. **Export for Tableau**: Use CSVs from `outputs/reports/tableau_*.csv`
4. **Adjust Weights**: If results not satisfactory, tune `src/config.py` and re-run
5. **Enable AI**: Set `USE_OPENAI_EXPLANATIONS = True` and re-run if you want
6. **Share Results**: Show the Streamlit app to stakeholders!

---

**All Commands Ready to Copy-Paste!** 🚀  
**Expected Total Time**: ~2 minutes setup + ~1 minute pipeline + instant Streamlit  
**Status**: ✅ Ready to Demo
