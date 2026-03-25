"""
Reporting module for Step 7: Risk reporting and visualization.

Creates risk summaries, visualizations, and generates natural-language
explanations for flagged transactions using OpenAI API (optional).

Outputs:
  - Risk visualizations to outputs/figures/
  - Risk summary CSVs to outputs/reports/
  - OpenAI-generated explanations (optional) to outputs/reports/explanations.json
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

from . import config
from .utils import LOGGER, save_csv


# ============================================================================
# OPENAI-BASED EXPLANATIONS (Optional)
# ============================================================================

def generate_openai_explanation(prompt: str) -> Optional[str]:
    """
    Generate a natural-language explanation using OpenAI API.
    
    Args:
        prompt: Question/request for OpenAI
        
    Returns:
        Generated explanation, or None if API fails or is disabled
    """
    if not config.USE_OPENAI_EXPLANATIONS:
        return None
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        LOGGER.debug("OPENAI_API_KEY not set; skipping OpenAI explanations")
        return None
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a fraud detection analyst. Provide concise, factual explanations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.3,
        )
        
        explanation = response.choices[0].message.content.strip()
        return explanation
    except Exception as e:
        LOGGER.warning(f"OpenAI API call failed: {e}; continuing without explanations")
        return None


def explain_high_risk_transaction(tx_row: pd.Series, risk_breakdown: Dict) -> Optional[str]:
    """
    Generate explanation for why a transaction is high-risk.
    
    Args:
        tx_row: Transaction row with all features
        risk_breakdown: Dict of risk component contributions
        
    Returns:
        Natural language explanation or None
    """
    top_factors = sorted(risk_breakdown.items(), key=lambda x: x[1], reverse=True)[:3]
    top_factors_str = ", ".join([f"{k}={v:.2f}" for k, v in top_factors])
    
    prompt = f"""
Analyze this high-risk transaction:
- Amount: ${tx_row.get('transactionamount', 'N/A'):.2f}
- Account: {tx_row.get('accountid', 'N/A')}
- Merchant: {tx_row.get('merchantid', 'N/A')}
- Channel: {tx_row.get('channel', 'N/A')}
- Top risk factors: {top_factors_str}

Briefly explain why this is flagged (1-2 sentences).
"""
    
    return generate_openai_explanation(prompt)


def explain_top_accounts(accounts_df: pd.DataFrame, top_n: int = 3) -> Optional[Dict]:
    """
    Generate explanations for top N high-risk accounts.
    
    Returns:
        Dict mapping account_id to explanation, or None if API fails
    """
    explanations = {}
    
    for _, row in accounts_df.head(top_n).iterrows():
        account_id = row.get("accountid", "unknown")
        prompt = f"""
Account {account_id} has high fraud risk:
- Risk Score: {row.get('account_risk_score', 0):.3f}
- Transaction Count: {row.get('transaction_count', 0)}
- High Risk Transaction %: {row.get('high_risk_transaction_pct', 0):.1f}%

Briefly explain the risk profile (1-2 sentences).
"""
        explanation = generate_openai_explanation(prompt)
        if explanation:
            explanations[account_id] = explanation
    
    return explanations if explanations else None


# ============================================================================
# VISUALIZATIONS
# ============================================================================

def plot_risk_by_account(risk_df: pd.DataFrame, accounts_df: pd.DataFrame, top_n: int = 20) -> Path:
    """
    Create bar chart of risk scores by account (top N).
    
    Returns:
        Path to saved figure
    """
    LOGGER.info(f"Generating risk by account visualization (top {top_n})...")
    
    top_accounts = accounts_df.head(top_n).copy()
    top_accounts = top_accounts.sort_values("account_risk_score")
    
    fig = go.Figure(data=[
        go.Bar(
            y=top_accounts["accountid"],
            x=top_accounts["account_risk_score"],
            marker=dict(
                color=top_accounts["account_risk_score"],
                colorscale="Reds",
                showscale=True,
            ),
            orientation="h",
        )
    ])
    
    fig.update_layout(
        title=f"Top {top_n} Accounts by Risk Score",
        xaxis_title="Risk Score",
        yaxis_title="Account ID",
        height=600,
        hovermode="y unified",
    )
    
    path = config.FIGURES_DIR / "risk_by_account.html"
    fig.write_html(str(path))
    LOGGER.info(f"  Saved: {path}")
    return path


def plot_risk_by_merchant(risk_df: pd.DataFrame, merchants_df: pd.DataFrame, top_n: int = 20) -> Path:
    """Create bar chart of risk scores by merchant (top N)."""
    LOGGER.info(f"Generating risk by merchant visualization (top {top_n})...")
    
    top_merchants = merchants_df.head(top_n).copy()
    top_merchants = top_merchants.sort_values("avg_risk_score")
    
    fig = go.Figure(data=[
        go.Bar(
            y=top_merchants["merchantid"],
            x=top_merchants["avg_risk_score"],
            marker=dict(
                color=top_merchants["avg_risk_score"],
                colorscale="Oranges",
                showscale=True,
            ),
            orientation="h",
        )
    ])
    
    fig.update_layout(
        title=f"Top {top_n} Merchants by Avg Risk Score",
        xaxis_title="Average Risk Score",
        yaxis_title="Merchant ID",
        height=600,
        hovermode="y unified",
    )
    
    path = config.FIGURES_DIR / "risk_by_merchant.html"
    fig.write_html(str(path))
    LOGGER.info(f"  Saved: {path}")
    return path


def plot_amount_vs_risk(risk_df: pd.DataFrame) -> Path:
    """Create scatter plot: transaction amount vs composite risk score."""
    LOGGER.info("Generating amount vs risk scatter plot...")
    
    fig = px.scatter(
        risk_df,
        x="transactionamount",
        y="composite_risk_score",
        color="risk_level",
        color_discrete_map={
            "Low": "#2ca02c",
            "Medium": "#ff7f0e",
            "High": "#d62728",
        },
        title="Transaction Amount vs Composite Risk Score",
        labels={
            "transactionamount": "Transaction Amount ($)",
            "composite_risk_score": "Risk Score",
        },
        opacity=0.6,
        height=500,
    )
    
    fig.update_layout(hovermode="closest")
    path = config.FIGURES_DIR / "amount_vs_risk_scatter.html"
    fig.write_html(str(path))
    LOGGER.info(f"  Saved: {path}")
    return path


def plot_risk_distribution(risk_df: pd.DataFrame) -> Path:
    """Create histogram of risk score distribution."""
    LOGGER.info("Generating risk score distribution plot...")
    
    fig = px.histogram(
        risk_df,
        x="composite_risk_score",
        nbins=50,
        title="Risk Score Distribution",
        labels={"composite_risk_score": "Risk Score"},
        color_discrete_sequence=["#1f77b4"],
        height=400,
    )
    
    fig.update_xaxes(range=[0, 1])
    path = config.FIGURES_DIR / "risk_distribution.html"
    fig.write_html(str(path))
    LOGGER.info(f"  Saved: {path}")
    return path


def plot_risk_components_breakdown(risk_df: pd.DataFrame) -> Path:
    """
    Create heatmap showing contribution of each risk component
    for top N high-risk transactions.
    """
    LOGGER.info("Generating risk components breakdown heatmap...")
    
    top_n = 50
    components = [
        "isolation_forest_score",
        "lof_score",
        "kmeans_anomaly_score",
        "graph_risk_score",
        "login_attempt_risk",
        "amount_outlier_risk",
    ]
    
    # Get top N high-risk transactions
    top_transactions = risk_df.nlargest(top_n, "composite_risk_score")
    
    # Filter components to only those present in the data
    available_components = [c for c in components if c in top_transactions.columns]
    
    if not available_components:
        LOGGER.warning("No risk components available for heatmap.")
        available_components = ["composite_risk_score"]
    
    # Extract component scores
    component_matrix = top_transactions[available_components].fillna(0)
    
    fig = go.Figure(data=go.Heatmap(
        z=component_matrix.values,
        x=available_components,
        y=[f"Tx {i}" for i in range(len(component_matrix))],
        colorscale="RdYlGn_r",
    ))
    
    fig.update_layout(
        title=f"Risk Components Breakdown (Top {top_n} Transactions)",
        xaxis_title="Risk Component",
        yaxis_title="Transaction",
        height=800,
    )
    
    path = config.FIGURES_DIR / "risk_components_heatmap.html"
    fig.write_html(str(path))
    LOGGER.info(f"  Saved: {path}")
    return path


def plot_location_risk(risk_df: pd.DataFrame) -> Path:
    """Create bar chart of risk by location."""
    LOGGER.info("Generating risk by location visualization...")
    
    location_risk = risk_df.groupby("location").agg({
        "composite_risk_score": ["mean", "count"],
    }).reset_index()
    location_risk.columns = ["location", "avg_risk_score", "count"]
    location_risk = location_risk.sort_values("avg_risk_score", ascending=False).head(15)
    
    fig = px.bar(
        location_risk,
        x="avg_risk_score",
        y="location",
        title="Top 15 Locations by Average Risk Score",
        labels={"avg_risk_score": "Average Risk Score"},
        color="avg_risk_score",
        color_continuous_scale="Reds",
        height=500,
        orientation="h",
    )
    
    path = config.FIGURES_DIR / "risk_by_location.html"
    fig.write_html(str(path))
    LOGGER.info(f"  Saved: {path}")
    return path


def plot_channel_risk(risk_df: pd.DataFrame) -> Path:
    """Create chart of risk by channel."""
    LOGGER.info("Generating risk by channel visualization...")
    
    channel_risk = risk_df.groupby("channel", observed=False).agg({
        "composite_risk_score": ["mean", "max", "count"],
    }).reset_index()
    channel_risk.columns = ["channel", "avg_risk_score", "max_risk_score", "count"]
    
    fig = go.Figure(data=[
        go.Bar(
            x=channel_risk["channel"],
            y=channel_risk["avg_risk_score"],
            name="Avg Risk",
            marker_color="steelblue",
        ),
    ])
    
    fig.update_layout(
        title="Risk Score by Channel",
        xaxis_title="Channel",
        yaxis_title="Average Risk Score",
        height=400,
    )
    
    path = config.FIGURES_DIR / "risk_by_channel.html"
    fig.write_html(str(path))
    LOGGER.info(f"  Saved: {path}")
    return path


# ============================================================================
# SUMMARY REPORTS
# ============================================================================

def create_executive_summary(risk_df: pd.DataFrame, risk_results: Dict) -> Dict:
    """
    Create high-level executive summary statistics.
    
    Returns:
        Dictionary with summary metrics
    """
    LOGGER.info("Generating executive summary...")
    
    summary = {
        "total_transactions": int(len(risk_df)),
        "total_accounts": int(risk_df["accountid"].nunique()),
        "total_merchants": int(risk_df["merchantid"].nunique()),
        "total_locations": int(risk_df["location"].nunique()),
        
        # Risk distribution
        "high_risk_count": int((risk_df["risk_level"] == "High").sum()),
        "high_risk_pct": float(100 * (risk_df["risk_level"] == "High").sum() / len(risk_df)),
        
        "medium_risk_count": int((risk_df["risk_level"] == "Medium").sum()),
        "medium_risk_pct": float(100 * (risk_df["risk_level"] == "Medium").sum() / len(risk_df)),
        
        # Score statistics
        "avg_composite_score": float(risk_df["composite_risk_score"].mean()),
        "max_composite_score": float(risk_df["composite_risk_score"].max()),
        "median_composite_score": float(risk_df["composite_risk_score"].median()),
        
        # Total exposure
        "total_transaction_volume": float(risk_df["transactionamount"].sum()),
        "high_risk_transaction_volume": float(
            risk_df[risk_df["risk_level"] == "High"]["transactionamount"].sum()
        ),
    }
    
    # Save as JSON
    summary_path = config.REPORTS_DIR / "executive_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    LOGGER.info(f"  Saved: {summary_path}")
    
    return summary


def save_summary_for_tableau(risk_df: pd.DataFrame, risk_results: Dict) -> None:
    """
    Create clean, minimal CSV exports optimized for Tableau.
    
    Tableau expects:
      - Simple column names (no spaces, no special chars)
      - Denormalized data where useful
      - Key metrics in all exports
    """
    LOGGER.info("Creating Tableau-optimized exports...")
    
    # Transactions with full context - select only available columns
    desired_cols = [
        "transactionid",
        "accountid",
        "merchantid",
        "deviceid",
        "ip_address",
        "location",
        "channel",
        "transaction_type",
        "transactionamount",
        "transactionduration",
        "login_attempts",
        "composite_risk_score",
        "risk_level",
        "isolation_forest_score",
        "lof_score",
        "kmeans_anomaly_score",
        "graph_risk_score",
    ]
    
    available_cols = [c for c in desired_cols if c in risk_df.columns]
    tableau_transactions = risk_df[available_cols].copy()
    
    save_csv(tableau_transactions, config.REPORTS_DIR / "tableau_transactions.csv")
    
    # Accounts
    accounts_df = risk_results.get("accounts_ranked", pd.DataFrame())
    if not accounts_df.empty:
        save_csv(accounts_df, config.REPORTS_DIR / "tableau_accounts.csv")
    
    # Merchants
    merchants_df = risk_results.get("merchants_ranked", pd.DataFrame())
    if not merchants_df.empty:
        save_csv(merchants_df, config.REPORTS_DIR / "tableau_merchants.csv")
    
    # Devices
    devices_df = risk_results.get("devices_ranked", pd.DataFrame())
    if not devices_df.empty:
        save_csv(devices_df, config.REPORTS_DIR / "tableau_devices.csv")
    
    # IPs
    ips_df = risk_results.get("ips_ranked", pd.DataFrame())
    if not ips_df.empty:
        save_csv(ips_df, config.REPORTS_DIR / "tableau_ips.csv")
    
    LOGGER.info(f"  Tableau exports saved to {config.REPORTS_DIR}")


# ============================================================================
# MAIN REPORTING ORCHESTRATION
# ============================================================================

def generate_report(
    risk_df: pd.DataFrame,
    risk_results: Dict,
) -> Dict:
    """
    Run full reporting pipeline:
    - Generate visualizations
    - Create summaries
    - Generate OpenAI explanations (if enabled)
    - Export for Tableau
    
    Args:
        risk_df: Full risk-scored transaction dataframe
        risk_results: Dictionary with ranked results (accounts, merchants, etc.)
        
    Returns:
        Dictionary with paths to all generated artifacts
    """
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("STEP 7: REPORTING & VISUALIZATIONS")
    LOGGER.info("=" * 60)
    
    artifacts = {}
    
    # Unpack results
    accounts_df = risk_results.get("accounts_ranked", pd.DataFrame())
    merchants_df = risk_results.get("merchants_ranked", pd.DataFrame())
    devices_df = risk_results.get("devices_ranked", pd.DataFrame())
    ips_df = risk_results.get("ips_ranked", pd.DataFrame())
    
    # ====================================================================
    # VISUALIZATIONS
    # ====================================================================
    artifacts["figures"] = {}
    artifacts["figures"]["account_risk"] = plot_risk_by_account(risk_df, accounts_df)
    artifacts["figures"]["merchant_risk"] = plot_risk_by_merchant(risk_df, merchants_df)
    artifacts["figures"]["amount_vs_risk"] = plot_amount_vs_risk(risk_df)
    artifacts["figures"]["risk_distribution"] = plot_risk_distribution(risk_df)
    artifacts["figures"]["components"] = plot_risk_components_breakdown(risk_df)
    artifacts["figures"]["location_risk"] = plot_location_risk(risk_df)
    artifacts["figures"]["channel_risk"] = plot_channel_risk(risk_df)
    
    # ====================================================================
    # EXECUTIVE SUMMARY
    # ====================================================================
    artifacts["summary"] = create_executive_summary(risk_df, risk_results)
    
    # ====================================================================
    # TABLEAU EXPORTS
    # ====================================================================
    save_summary_for_tableau(risk_df, risk_results)
    
    # ====================================================================
    # OPENAI EXPLANATIONS (Optional)
    # ====================================================================
    if config.USE_OPENAI_EXPLANATIONS and os.environ.get("OPENAI_API_KEY"):
        LOGGER.info("Generating OpenAI explanations...")
        explanations = {}
        
        # Explain top 10 high-risk transactions
        high_risk = risk_df[risk_df["risk_level"] == "High"].head(10)
        for idx, tx_row in high_risk.iterrows():
            tx_id = tx_row.get("transactionid", f"tx_{idx}")
            risk_breakdown = {
                "isolation_forest": tx_row.get("isolation_forest_score", 0),
                "lof": tx_row.get("lof_score", 0),
                "kmeans": tx_row.get("kmeans_anomaly_score", 0),
                "graph": tx_row.get("graph_risk_score", 0),
            }
            explanation = explain_high_risk_transaction(tx_row, risk_breakdown)
            if explanation:
                explanations[f"transaction_{tx_id}"] = explanation
        
        # Explain top high-risk accounts
        account_explanations = explain_top_accounts(accounts_df, top_n=5)
        if account_explanations:
            explanations.update({f"account_{k}": v for k, v in account_explanations.items()})
        
        # Save explanations
        if explanations:
            exp_path = config.REPORTS_DIR / "openai_explanations.json"
            with open(exp_path, "w") as f:
                json.dump(explanations, f, indent=2)
            artifacts["explanations"] = exp_path
            LOGGER.info(f"  Saved {len(explanations)} explanations to {exp_path}")
    else:
        LOGGER.info("OpenAI explanations disabled (set USE_OPENAI_EXPLANATIONS=True to enable)")
    
    LOGGER.info("\n✅ Reporting complete.")
    LOGGER.info(f"Figures saved to:   {config.FIGURES_DIR}")
    LOGGER.info(f"Reports saved to:   {config.REPORTS_DIR}")
    
    return artifacts
