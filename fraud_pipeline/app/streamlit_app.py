"""
Enhanced Streamlit app for Step 8: Interactive fraud detection demo.

Features:
  - Executive summary dashboard
  - CSV upload and scoring
  - Suspicious transactions filter
  - Account/merchant/location risk views
  - Graph & TDA insights
  - Analyst review and approval workflow

Run with: streamlit run app/streamlit_app.py
"""

import sys
import os
import json
import io
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src import config
from src.utils import LOGGER, save_csv
from src.review_store import ReviewStore
from src.openai_explanations import explain_transaction, explain_account_risk


# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .high-risk { color: #d62728; font-weight: bold; }
    .medium-risk { color: #ff7f0e; font-weight: bold; }
    .low-risk { color: #2ca02c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CACHE & DATA LOADING
# ============================================================================

@st.cache_data
def load_pipeline_outputs():
    """Load all pre-computed pipeline outputs."""
    try:
        risk_transactions = pd.read_csv(config.RISK_TRANSACTIONS_FILE)
        risk_accounts = pd.read_csv(config.RISK_ACCOUNTS_FILE)
        risk_merchants = pd.read_csv(config.REPORTS_DIR / "risk_ranked_merchants.csv")
        risk_devices = pd.read_csv(config.REPORTS_DIR / "risk_ranked_devices.csv")
        risk_ips = pd.read_csv(config.REPORTS_DIR / "risk_ranked_ips.csv")
        
        # Load executive summary
        summary = {}
        summary_path = config.REPORTS_DIR / "executive_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
        
        # Load explanations if available
        explanations = {}
        exp_path = config.REPORTS_DIR / "openai_explanations.json"
        if exp_path.exists():
            with open(exp_path) as f:
                explanations = json.load(f)
        
        return {
            "transactions": risk_transactions,
            "accounts": risk_accounts,
            "merchants": risk_merchants,
            "devices": risk_devices,
            "ips": risk_ips,
            "summary": summary,
            "explanations": explanations,
        }
    except FileNotFoundError as e:
        st.error(f"❌ Pipeline outputs not found: {e}")
        st.info("💡 Run the backend pipeline first: `python run_pipeline.py`")
        st.stop()


@st.cache_data
def load_figure(figure_name: str):
    """Load pre-generated Plotly HTML figures."""
    figure_path = config.FIGURES_DIR / f"{figure_name}.html"
    if figure_path.exists():
        with open(figure_path) as f:
            return f.read()
    return None


# ============================================================================
# PAGE: EXECUTIVE SUMMARY
# ============================================================================

def page_overview():
    st.title("📊 Executive Summary")
    
    data = load_pipeline_outputs()
    summary = data["summary"]
    transactions = data["transactions"]
    
    if not summary:
        st.warning("No summary data available. Run pipeline first.")
        return
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Transactions",
            f"{summary.get('total_transactions', 0):,}",
            help="Total transactions analyzed"
        )
    
    with col2:
        high_risk = summary.get("high_risk_count", 0)
        pct = summary.get("high_risk_pct", 0)
        st.metric(
            "🔴 High Risk",
            f"{high_risk:,}",
            f"{pct:.1f}%",
            delta_color="inverse"
        )
    
    with col3:
        total_volume = summary.get("total_transaction_volume", 0)
        st.metric(
            "Total Volume",
            f"${total_volume:,.0f}",
            help="Sum of all transaction amounts"
        )
    
    with col4:
        high_risk_vol = summary.get("high_risk_transaction_volume", 0)
        st.metric(
            "🔴 High Risk Volume",
            f"${high_risk_vol:,.0f}",
            delta_color="inverse"
        )
    
    # Risk distribution
    st.subheader("Risk Level Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        risk_counts = transactions["risk_level"].value_counts()
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Transactions by Risk Level",
            color_discrete_map={
                "Low": "#2ca02c",
                "Medium": "#ff7f0e",
                "High": "#d62728",
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk score distribution
        fig = px.histogram(
            transactions,
            x="composite_risk_score",
            nbins=50,
            title="Risk Score Distribution",
            color_discrete_sequence=["#1f77b4"],
        )
        fig.update_xaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    
    # Entity counts
    st.subheader("Dataset Coverage")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accounts", f"{summary.get('total_accounts', 0):,}")
    with col2:
        st.metric("Merchants", f"{summary.get('total_merchants', 0):,}")
    with col3:
        st.metric("Locations", f"{summary.get('total_locations', 0):,}")
    with col4:
        st.metric("Devices", len(data["devices"]))
    
    # Pre-generated figures
    st.subheader("Key Visualizations")
    
    tab1, tab2, tab3, tab4 = st.tabs(["By Account", "By Merchant", "By Location", "Amount vs Risk"])
    
    with tab1:
        fig_html = load_figure("risk_by_account")
        if fig_html:
            st.components.v1.html(fig_html, height=700)
        else:
            st.info("Figure not available")
    
    with tab2:
        fig_html = load_figure("risk_by_merchant")
        if fig_html:
            st.components.v1.html(fig_html, height=700)
        else:
            st.info("Figure not available")
    
    with tab3:
        fig_html = load_figure("risk_by_location")
        if fig_html:
            st.components.v1.html(fig_html, height=700)
        else:
            st.info("Figure not available")
    
    with tab4:
        fig_html = load_figure("amount_vs_risk_scatter")
        if fig_html:
            st.components.v1.html(fig_html, height=600)
        else:
            st.info("Figure not available")


# ============================================================================
# PAGE: SUSPICIOUS TRANSACTIONS
# ============================================================================

def page_transactions():
    st.title("🚨 Suspicious Transactions")
    
    data = load_pipeline_outputs()
    transactions = data["transactions"]
    explanations = data["explanations"]
    store = ReviewStore()
    
    # Filters
    st.subheader("Filters")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        risk_level_filter = st.multiselect(
            "Risk Level",
            options=["High", "Medium", "Low"],
            default=["High", "Medium"],
        )
    
    with col2:
        min_score = st.slider("Min Risk Score", 0.0, 1.0, 0.5)
    
    with col3:
        account_filter = st.text_input("Account ID (partial)", "")
    
    with col4:
        merchant_filter = st.text_input("Merchant ID (partial)", "")
    
    # Apply filters
    filtered = transactions[transactions["risk_level"].isin(risk_level_filter)]
    filtered = filtered[filtered["composite_risk_score"] >= min_score]
    
    if account_filter:
        filtered = filtered[filtered["accountid"].str.contains(account_filter, na=False, case=False)]
    
    if merchant_filter:
        filtered = filtered[filtered["merchantid"].str.contains(merchant_filter, na=False, case=False)]
    
    st.info(f"Showing {len(filtered)} of {len(transactions)} transactions")
    
    # Display transactions
    if len(filtered) > 0:
        st.subheader("Transactions")
        
        display_cols = [
            "transactionid", "accountid", "merchantid", "location", "channel",
            "transactionamount", "composite_risk_score", "risk_level"
        ]
        display_cols = [c for c in display_cols if c in filtered.columns]
        
        st.dataframe(filtered[display_cols], use_container_width=True)
        
        # Detailed view
        st.subheader("Transaction Details")
        
        selected_tx_id = st.selectbox(
            "Select transaction ID to view details",
            options=filtered["transactionid"].tolist()
        )
        
        if selected_tx_id:
            tx = filtered[filtered["transactionid"] == selected_tx_id].iloc[0]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Amount", f"${tx.get('transactionamount', 0):.2f}")
            with col2:
                st.metric("Risk Score", f"{tx.get('composite_risk_score', 0):.3f}")
            with col3:
                st.metric("Risk Level", tx.get("risk_level", "N/A"))
            with col4:
                st.metric("Channel", tx.get("channel", "N/A"))
            
            # Risk component breakdown
            st.subheader("Risk Component Breakdown")
            
            components = {
                "Isolation Forest": tx.get("isolation_forest_score", 0),
                "Local Outlier Factor": tx.get("lof_score", 0),
                "K-Means": tx.get("kmeans_anomaly_score", 0),
                "Graph Risk": tx.get("graph_risk_score", 0),
            }
            
            fig = px.bar(
                x=list(components.keys()),
                y=list(components.values()),
                title="Risk Score Components",
                labels={"x": "Component", "y": "Score"},
                color_discrete_sequence=["#1f77b4"],
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # OpenAI explanation if available
            exp_key = f"transaction_{selected_tx_id}"
            if exp_key in explanations:
                st.info(f"📝 Analysis: {explanations[exp_key]}")
            else:
                st.text("(No AI explanation available)")
            
            # Analyst decision
            st.subheader("Analyst Review")
            
            current_decision = store.get_decision(selected_tx_id)
            decision = st.radio(
                "Decision",
                options=config.DECISION_OPTIONS,
                index=config.DECISION_OPTIONS.index(current_decision) if current_decision else 0,
                horizontal=True,
            )
            
            notes = st.text_area("Notes (optional)", value="")
            
            if st.button("💾 Save Decision", key=f"save_{selected_tx_id}"):
                store.record_decision(
                    transaction_id=selected_tx_id,
                    decision=decision,
                    notes=notes,
                )
                st.success("✅ Decision saved")
    
    else:
        st.warning("No transactions match the selected filters")


# ============================================================================
# PAGE: ACCOUNT RISK
# ============================================================================

def page_accounts():
    st.title("💰 Account Risk Analysis")
    
    data = load_pipeline_outputs()
    accounts = data["accounts"]
    transactions = data["transactions"]
    explanations = data["explanations"]
    
    # Top accounts
    st.subheader("Top 20 High-Risk Accounts")
    
    top_accounts = accounts.head(20).copy()
    
    display_cols = [
        "accountid", "account_risk_score", "max_risk_score",
        "transaction_count", "high_risk_transaction_count"
    ]
    display_cols = [c for c in display_cols if c in top_accounts.columns]
    
    st.dataframe(top_accounts[display_cols], use_container_width=True)
    
    # Selected account detail
    st.subheader("Account Details")
    
    selected_account = st.selectbox(
        "Select account",
        options=accounts["accountid"].tolist()
    )
    
    if selected_account:
        account = accounts[accounts["accountid"] == selected_account].iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Risk Score", f"{account.get('account_risk_score', 0):.3f}")
        with col2:
            st.metric("Transaction Count", int(account.get("transaction_count", 0)))
        with col3:
            st.metric("High Risk Count", int(account.get("high_risk_transaction_count", 0)))
        with col4:
            pct = account.get("high_risk_transaction_pct", 0)
            st.metric("High Risk %", f"{pct:.1f}%")
        
        # Account's transactions
        account_txs = transactions[transactions["accountid"] == selected_account]
        
        if len(account_txs) > 0:
            st.subheader(f"Transactions for {selected_account}")
            
            display_cols = [
                "transactionid", "merchantid", "transactionamount",
                "composite_risk_score", "risk_level", "channel"
            ]
            display_cols = [c for c in display_cols if c in account_txs.columns]
            
            st.dataframe(account_txs[display_cols].head(50), use_container_width=True)
        
        # OpenAI explanation if available
        exp_key = f"account_{selected_account}"
        if exp_key in explanations:
            st.info(f"📝 Analysis: {explanations[exp_key]}")


# ============================================================================
# PAGE: MERCHANT & LOCATION RISK
# ============================================================================

def page_entities():
    st.title("🏪 Merchant & Location Risk")
    
    data = load_pipeline_outputs()
    merchants = data["merchants"]
    transactions = data["transactions"]
    
    tab1, tab2 = st.tabs(["Merchants", "Locations"])
    
    with tab1:
        st.subheader("Top Merchants by Risk")
        
        top_merchants = merchants.head(20)
        display_cols = [c for c in ["merchantid", "avg_risk_score", "max_risk_score", "transaction_count", "high_risk_count"] if c in top_merchants.columns]
        st.dataframe(top_merchants[display_cols], use_container_width=True)
    
    with tab2:
        st.subheader("Top Locations by Risk")
        
        location_risk = transactions.groupby("location").agg({
            "composite_risk_score": ["mean", "max", "count"],
            "transactionid": "nunique",
        }).reset_index()
        location_risk.columns = ["location", "avg_risk_score", "max_risk_score", "transaction_count", "unique_tx"]
        location_risk = location_risk.sort_values("avg_risk_score", ascending=False).head(20)
        
        st.dataframe(location_risk, use_container_width=True)


# ============================================================================
# PAGE: REVIEW LOG
# ============================================================================

def page_decisions():
    st.title("📋 Analyst Review Log")
    
    store = ReviewStore()
    
    # Get all decisions from ReviewStore
    all_decisions_raw = store.get_all_decisions()
    
    if all_decisions_raw:
        # all_decisions_raw is a dict {tx_id: decision}, convert to dataframe
        decisions_list = [
            {"Transaction ID": tx_id, "Decision": decision}
            for tx_id, decision in all_decisions_raw.items()
        ]
        decisions_df = pd.DataFrame(decisions_list)
        
        # Summary stats
        col1, col2, col3 = st.columns(3)
        
        decision_counts = decisions_df["Decision"].value_counts()
        
        with col1:
            approve_count = decision_counts.get("Approve", 0)
            st.metric("Approved", approve_count)
        with col2:
            dismiss_count = decision_counts.get("Dismiss", 0)
            st.metric("Dismissed", dismiss_count)
        with col3:
            review_count = decision_counts.get("Needs Review", 0)
            st.metric("Needs Review", review_count)
        
        # Decision log
        st.subheader("Review Log")
        
        st.dataframe(decisions_df, use_container_width=True)
        
        # Export
        csv_buffer = io.StringIO()
        decisions_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="📥 Download Review Log",
            data=csv_data,
            file_name=f"review_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    else:
        st.info("No decisions recorded yet")


# ============================================================================
# PAGE: PIPELINE INFO
# ============================================================================

def page_info():
    st.title("ℹ️ Pipeline Information")
    
    st.subheader("Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Risk Scoring Weights:**")
        weights_df = pd.DataFrame([
            {"Component": k, "Weight": f"{v:.2f}"} 
            for k, v in config.RISK_WEIGHTS.items()
        ])
        st.dataframe(weights_df, use_container_width=True)
    
    with col2:
        st.write("**Thresholds:**")
        thresholds = {
            "Low Risk (0 to)": config.RISK_LEVEL_LOW,
            "Medium Risk (to)": config.RISK_LEVEL_MEDIUM,
            "High Risk (to)": config.RISK_LEVEL_HIGH,
        }
        thresholds_df = pd.DataFrame([
            {"Threshold": k, "Value": f"{v:.2f}"}
            for k, v in thresholds.items()
        ])
        st.dataframe(thresholds_df, use_container_width=True)
    
    st.subheader("Dataset Paths")
    
    paths = {
        "Raw Data": str(config.RAW_DATA_FILE),
        "Cleaned Data": str(config.CLEANED_DATA_FILE),
        "Risk Transactions": str(config.RISK_TRANSACTIONS_FILE),
        "Risk Accounts": str(config.RISK_ACCOUNTS_FILE),
        "Figures Directory": str(config.FIGURES_DIR),
        "Reports Directory": str(config.REPORTS_DIR),
    }
    
    for label, path in paths.items():
        st.text(f"**{label}:** {path}")
    
    st.subheader("Features")
    
    features_text = """
    - **Unsupervised anomaly detection:** Isolation Forest, LOF, K-Means
    - **Graph-based analysis:** NetworkX transaction networks
    - **Transparent risk scoring:** Configurable weighted combination
    - **Analyst workflow:** Review and approval tracking
    - **Tableau export:** All results as CSV
    - **OpenAI integration:** Optional natural-language explanations
    """
    st.markdown(features_text)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Sidebar navigation
    st.sidebar.title("🔍 Fraud Detection System")
    
    page = st.sidebar.radio(
        "Navigation",
        options=[
            "📊 Overview",
            "🚨 Suspicious Transactions",
            "💰 Account Risk",
            "🏪 Merchants & Locations",
            "📋 Review Log",
            "ℹ️ Pipeline Info",
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Fraud Detection Pipeline - Step 8**")
    st.sidebar.markdown("Unsupervised anomaly detection + analyst review")
    
    # Route pages
    if page == "📊 Overview":
        page_overview()
    elif page == "🚨 Suspicious Transactions":
        page_transactions()
    elif page == "💰 Account Risk":
        page_accounts()
    elif page == "🏪 Merchants & Locations":
        page_entities()
    elif page == "📋 Review Log":
        page_decisions()
    elif page == "ℹ️ Pipeline Info":
        page_info()


if __name__ == "__main__":
    main()
