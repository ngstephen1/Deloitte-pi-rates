"""
Executive Streamlit app for the fraud detection pipeline.

Focus:
  - Premium executive dashboard styling
  - Analyst review workflow
  - Robust fallbacks when branding or partial outputs are missing

Run with: streamlit run app/streamlit_app.py
"""

import io
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config
from src.review_store import ReviewStore
from styles import (
    BRAND,
    apply_chart_theme,
    badge,
    find_logo_path,
    inject_global_styles,
    render_app_header,
    render_detail_card,
    render_insight,
    render_metric_card,
    render_section_header,
)


st.set_page_config(
    page_title="Executive Fraud Intelligence Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_global_styles()

PROJECT_ROOT = Path(__file__).parent.parent
LOGO_PATH = find_logo_path(PROJECT_ROOT)

NAV_ITEMS = [
    "Overview",
    "Suspicious Transactions",
    "Account Risk",
    "Merchants, Devices & Locations",
    "Review Log",
    "Pipeline Info",
]
RISK_ORDER = ["High", "Medium", "Low"]


@st.cache_data
def safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_figure(figure_name: str):
    figure_path = config.FIGURES_DIR / f"{figure_name}.html"
    if figure_path.exists():
        return figure_path.read_text()
    return None


def build_entity_summary(transactions: pd.DataFrame, entity_col: str) -> pd.DataFrame:
    if entity_col not in transactions.columns or transactions.empty:
        return pd.DataFrame()

    valid = transactions.dropna(subset=[entity_col]).copy()
    if valid.empty:
        return pd.DataFrame()

    summary = valid.groupby(entity_col).agg(
        avg_risk_score=("composite_risk_score", "mean"),
        max_risk_score=("composite_risk_score", "max"),
        transaction_count=("transactionid", "count"),
        high_risk_count=("risk_level", lambda values: (values == "High").sum()),
    ).reset_index()
    summary["high_risk_pct"] = 100 * summary["high_risk_count"] / summary["transaction_count"]
    return summary.sort_values(["max_risk_score", "avg_risk_score"], ascending=False).reset_index(drop=True)


def build_location_summary(transactions: pd.DataFrame) -> pd.DataFrame:
    return build_entity_summary(transactions, "location")


def build_summary_snapshot(
    transactions: pd.DataFrame,
    accounts: pd.DataFrame,
    merchants: pd.DataFrame,
    devices: pd.DataFrame,
    locations: pd.DataFrame,
    file_summary: dict,
) -> dict:
    if transactions.empty:
        return file_summary or {}

    summary = dict(file_summary or {})
    summary.update(
        {
            "total_transactions": int(len(transactions)),
            "total_accounts": int(transactions["accountid"].nunique(dropna=True)) if "accountid" in transactions else 0,
            "total_merchants": int(transactions["merchantid"].nunique(dropna=True)) if "merchantid" in transactions else 0,
            "total_locations": int(transactions["location"].nunique(dropna=True)) if "location" in transactions else 0,
            "high_risk_count": int((transactions["risk_level"] == "High").sum()),
            "high_risk_pct": float(100 * (transactions["risk_level"] == "High").mean()),
            "medium_risk_count": int((transactions["risk_level"] == "Medium").sum()),
            "medium_risk_pct": float(100 * (transactions["risk_level"] == "Medium").mean()),
            "avg_composite_score": float(transactions["composite_risk_score"].mean()),
            "max_composite_score": float(transactions["composite_risk_score"].max()),
            "median_composite_score": float(transactions["composite_risk_score"].median()),
            "total_transaction_volume": float(transactions["transactionamount"].sum()),
            "high_risk_transaction_volume": float(
                transactions.loc[transactions["risk_level"] == "High", "transactionamount"].sum()
            ),
            "flagged_transactions": int(transactions["risk_level"].isin(["High", "Medium"]).sum()),
            "high_risk_accounts": int(
                (accounts["account_risk_score"] >= config.RISK_LEVEL_MEDIUM).sum()
            )
            if "account_risk_score" in accounts
            else 0,
            "high_risk_merchants": int(
                (merchants["max_risk_score"] >= config.RISK_LEVEL_MEDIUM).sum()
            )
            if "max_risk_score" in merchants
            else 0,
            "high_risk_devices": int(
                (devices["max_risk_score"] >= config.RISK_LEVEL_MEDIUM).sum()
            )
            if "max_risk_score" in devices
            else 0,
            "high_risk_locations": int(
                (locations["max_risk_score"] >= config.RISK_LEVEL_MEDIUM).sum()
            )
            if "max_risk_score" in locations
            else 0,
        }
    )
    return summary


def _enrich_transaction_context(transactions: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty or "transactionid" not in transactions.columns:
        return transactions

    transactions = transactions.copy()
    transactions["transactionid"] = transactions["transactionid"].astype(str)

    cleaned = safe_read_csv(config.CLEANED_DATA_FILE)
    if cleaned.empty or "transactionid" not in cleaned.columns:
        return transactions

    cleaned = cleaned.copy()
    cleaned["transactionid"] = cleaned["transactionid"].astype(str)
    cleaned = cleaned.drop_duplicates(subset="transactionid")
    cleaned_lookup = cleaned.set_index("transactionid")

    context_columns = [
        "accountid",
        "merchantid",
        "deviceid",
        "ip_address",
        "location",
        "channel",
        "loginattempts",
        "transactionduration",
        "transactiontype",
    ]

    for column in context_columns:
        if column not in cleaned_lookup.columns:
            continue
        if column not in transactions.columns or transactions[column].isna().all():
            transactions[column] = transactions["transactionid"].map(cleaned_lookup[column])

    graph_features = safe_read_csv(config.GRAPH_FEATURES_FILE)
    if "graph_risk_score" not in transactions.columns and not graph_features.empty:
        graph_features = graph_features.drop_duplicates(subset="transactionid")
        graph_features["transactionid"] = graph_features["transactionid"].astype(str)
        transactions = transactions.merge(
            graph_features[["transactionid", "graph_risk_score"]],
            on="transactionid",
            how="left",
        )

    return transactions


@st.cache_data
def load_pipeline_outputs():
    try:
        transactions = safe_read_csv(config.RISK_TRANSACTIONS_FILE)
        if transactions.empty:
            raise FileNotFoundError(config.RISK_TRANSACTIONS_FILE)

        transactions = _enrich_transaction_context(transactions)
        transactions = transactions.sort_values("composite_risk_score", ascending=False).reset_index(drop=True)

        accounts = safe_read_csv(config.RISK_ACCOUNTS_FILE)
        merchants = safe_read_csv(config.REPORTS_DIR / "risk_ranked_merchants.csv")
        devices = safe_read_csv(config.REPORTS_DIR / "risk_ranked_devices.csv")
        ips = safe_read_csv(config.REPORTS_DIR / "risk_ranked_ips.csv")

        if accounts.empty and {"accountid", "composite_risk_score", "risk_level"}.issubset(transactions.columns):
            accounts = (
                transactions.groupby("accountid")
                .agg(
                    avg_risk_score=("composite_risk_score", "mean"),
                    max_risk_score=("composite_risk_score", "max"),
                    risk_score_std=("composite_risk_score", "std"),
                    transaction_count=("transactionid", "count"),
                    high_risk_transaction_count=("risk_level", lambda values: (values == "High").sum()),
                )
                .reset_index()
            )
            accounts["account_risk_score"] = accounts["max_risk_score"]
            accounts["high_risk_transaction_pct"] = (
                100 * accounts["high_risk_transaction_count"] / accounts["transaction_count"]
            )
            accounts["account_risk_rank"] = range(1, len(accounts) + 1)
            accounts = accounts.sort_values("account_risk_score", ascending=False).reset_index(drop=True)

        if merchants.empty:
            merchants = build_entity_summary(transactions, "merchantid")
        if devices.empty:
            devices = build_entity_summary(transactions, "deviceid")
        if ips.empty:
            ips = build_entity_summary(transactions, "ip_address")

        locations = build_location_summary(transactions)

        summary_path = config.REPORTS_DIR / "executive_summary.json"
        file_summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        explanations_path = config.REPORTS_DIR / "openai_explanations.json"
        explanations = json.loads(explanations_path.read_text()) if explanations_path.exists() else {}

        summary = build_summary_snapshot(transactions, accounts, merchants, devices, locations, file_summary)

        return {
            "transactions": transactions,
            "accounts": accounts,
            "merchants": merchants,
            "devices": devices,
            "ips": ips,
            "locations": locations,
            "summary": summary,
            "explanations": explanations,
        }
    except FileNotFoundError as exc:
        st.error(f"Pipeline output missing: {exc}")
        st.info("Run `python3 run_pipeline.py` from `fraud_pipeline/` before opening the dashboard.")
        st.stop()


def format_currency(value) -> str:
    return f"${value:,.0f}" if pd.notna(value) else "N/A"


def format_score(value) -> str:
    return f"{value:.3f}" if pd.notna(value) else "N/A"


def format_percent(value) -> str:
    return f"{value:.1f}%" if pd.notna(value) else "N/A"


def render_sidebar(page_name: str, data: dict) -> str:
    summary = data["summary"]
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-panel">
                <div style="font-size:0.78rem; letter-spacing:0.14em; text-transform:uppercase; color:#b7c6bc; font-weight:800;">
                    Navigation
                </div>
                <div style="font-size:1.25rem; font-weight:800; margin-top:0.45rem;">Fraud Intelligence</div>
                <div style="color:#dce5df; margin-top:0.3rem; line-height:1.5;">
                    Executive monitoring and analyst triage across transactions, entities, and review outcomes.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        selected = st.radio("Go to", NAV_ITEMS, index=NAV_ITEMS.index(page_name), label_visibility="collapsed")

        st.markdown(
            f"""
            <div class="sidebar-panel">
                <div style="font-size:0.78rem; letter-spacing:0.14em; text-transform:uppercase; color:#b7c6bc; font-weight:800;">
                    Snapshot
                </div>
                <div style="margin-top:0.7rem;">{badge('High Risk')}</div>
                <div style="margin-top:0.5rem; color:#eef3ef;">{summary.get('high_risk_count', 0):,} transactions</div>
                <div style="margin-top:0.85rem;">{badge('Needs Review')}</div>
                <div style="margin-top:0.5rem; color:#eef3ef;">{summary.get('flagged_transactions', 0):,} flagged items</div>
                <div style="margin-top:0.85rem; color:#b7c6bc;">Coverage</div>
                <div style="margin-top:0.35rem; color:#eef3ef;">{summary.get('total_transactions', 0):,} transactions</div>
                <div style="margin-top:0.25rem; color:#eef3ef;">{summary.get('total_accounts', 0):,} accounts</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption("Wide layout tuned for live presentation and laptop screens.")
    return selected


def render_transactions_filters(transactions: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-panel">
                <div style="font-size:0.78rem; letter-spacing:0.14em; text-transform:uppercase; color:#b7c6bc; font-weight:800;">
                    Transaction Filters
                </div>
            """,
            unsafe_allow_html=True,
        )
        risk_levels = st.multiselect(
            "Risk Level",
            options=RISK_ORDER,
            default=["High", "Medium"],
        )
        min_score = st.slider("Minimum Risk Score", 0.0, 1.0, 0.45, 0.01)
        account_filter = st.text_input("Account ID contains", "")
        merchant_filter = st.text_input("Merchant ID contains", "")
        channel_options = sorted(transactions["channel"].dropna().astype(str).unique().tolist()) if "channel" in transactions else []
        channel_filter = st.multiselect("Channel", options=channel_options, default=channel_options)
        st.markdown("</div>", unsafe_allow_html=True)

    filtered = transactions[transactions["risk_level"].isin(risk_levels)].copy()
    filtered = filtered[filtered["composite_risk_score"] >= min_score]

    if account_filter and "accountid" in filtered:
        filtered = filtered[filtered["accountid"].astype(str).str.contains(account_filter, case=False, na=False)]
    if merchant_filter and "merchantid" in filtered:
        filtered = filtered[filtered["merchantid"].astype(str).str.contains(merchant_filter, case=False, na=False)]
    if channel_filter and "channel" in filtered:
        filtered = filtered[filtered["channel"].astype(str).isin(channel_filter)]

    return filtered


def display_dataframe(df: pd.DataFrame, formatters: dict | None = None, height: int = 360) -> None:
    view = df.copy()
    if formatters:
        for column, formatter in formatters.items():
            if column in view.columns:
                view[column] = view[column].map(lambda value: formatter(value) if pd.notna(value) else "N/A")
    st.dataframe(view, use_container_width=True, hide_index=True, height=height)


def page_overview(data: dict) -> None:
    transactions = data["transactions"]
    summary = data["summary"]
    accounts = data["accounts"]
    merchants = data["merchants"]
    devices = data["devices"]
    locations = data["locations"]

    render_section_header(
        "Executive Summary",
        "A business-facing snapshot of flagged exposure, entity concentration, and portfolio-wide risk posture.",
        kicker="Overview",
    )

    kpi_specs = [
        ("Flagged Transactions", f"{summary.get('flagged_transactions', 0):,}", "Medium and high risk transactions requiring attention."),
        ("High-Risk Accounts", f"{summary.get('high_risk_accounts', 0):,}", "Accounts with score at or above the medium-risk threshold."),
        ("High-Risk Merchants", f"{summary.get('high_risk_merchants', 0):,}", "Merchants touching elevated-risk transaction clusters."),
        ("High-Risk Devices", f"{summary.get('high_risk_devices', 0):,}", "Devices recurring in elevated-risk transaction activity."),
        ("High-Risk Locations", f"{summary.get('high_risk_locations', 0):,}", "Locations with high-risk transaction exposure."),
    ]
    for column, (label, value, footnote) in zip(st.columns(5), kpi_specs):
        with column:
            render_metric_card(label, value, footnote)

    render_section_header(
        "Risk Posture",
        "Current transaction mix, score dispersion, and transaction-channel concentration.",
        kicker="Monitoring",
    )
    col1, col2 = st.columns([1.05, 1], gap="large")

    with col1:
        risk_mix = (
            transactions["risk_level"]
            .value_counts()
            .reindex(RISK_ORDER)
            .dropna()
        )
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=risk_mix.index.tolist(),
                    values=risk_mix.values.tolist(),
                    hole=0.58,
                    marker=dict(colors=[BRAND["danger"], BRAND["warning"], BRAND["green"]]),
                    textinfo="label+percent",
                )
            ]
        )
        fig.update_layout(title="Transactions by Risk Tier")
        st.plotly_chart(apply_chart_theme(fig, 420), use_container_width=True)

    with col2:
        fig = px.histogram(
            transactions,
            x="composite_risk_score",
            nbins=35,
            color="risk_level",
            category_orders={"risk_level": RISK_ORDER},
            color_discrete_map={"High": BRAND["danger"], "Medium": BRAND["warning"], "Low": BRAND["green"]},
            title="Composite Risk Score Distribution",
        )
        fig.update_layout(barmode="overlay")
        fig.update_traces(opacity=0.7)
        fig.update_xaxes(range=[0, 1])
        st.plotly_chart(apply_chart_theme(fig, 420), use_container_width=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        channel_risk = (
            transactions.groupby("channel")
            .agg(avg_risk_score=("composite_risk_score", "mean"), transaction_count=("transactionid", "count"))
            .reset_index()
            .sort_values("avg_risk_score", ascending=False)
        )
        fig = px.bar(
            channel_risk,
            x="channel",
            y="avg_risk_score",
            color="avg_risk_score",
            color_continuous_scale=[[0, BRAND["green_soft"]], [0.55, BRAND["green"]], [1, BRAND["charcoal"]]],
            title="Average Risk by Channel",
        )
        st.plotly_chart(apply_chart_theme(fig, 380), use_container_width=True)

    with col2:
        fig = px.scatter(
            transactions,
            x="transactionamount",
            y="composite_risk_score",
            color="risk_level",
            size_max=12,
            opacity=0.72,
            category_orders={"risk_level": RISK_ORDER},
            color_discrete_map={"High": BRAND["danger"], "Medium": BRAND["warning"], "Low": BRAND["green"]},
            title="Transaction Amount vs Risk Score",
            hover_data=["transactionid", "accountid", "merchantid"],
        )
        st.plotly_chart(apply_chart_theme(fig, 380), use_container_width=True)

    render_section_header(
        "Entity Hotspots",
        "Quick view of the most exposed accounts, merchants, devices, and locations in the current run.",
        kicker="Concentration",
    )
    tab1, tab2, tab3, tab4 = st.tabs(["Accounts", "Merchants", "Devices", "Locations"])

    with tab1:
        top_accounts = accounts.head(12)[
            [c for c in ["accountid", "account_risk_score", "high_risk_transaction_count", "transaction_count"] if c in accounts]
        ]
        display_dataframe(
            top_accounts,
            formatters={"account_risk_score": format_score},
            height=330,
        )

    with tab2:
        top_merchants = merchants.head(12)[
            [c for c in ["merchantid", "avg_risk_score", "max_risk_score", "transaction_count", "high_risk_count"] if c in merchants]
        ]
        display_dataframe(
            top_merchants,
            formatters={"avg_risk_score": format_score, "max_risk_score": format_score},
            height=330,
        )

    with tab3:
        top_devices = devices.head(12)[
            [c for c in ["deviceid", "avg_risk_score", "max_risk_score", "transaction_count", "high_risk_count"] if c in devices]
        ]
        display_dataframe(
            top_devices,
            formatters={"avg_risk_score": format_score, "max_risk_score": format_score},
            height=330,
        )

    with tab4:
        top_locations = locations.head(12)
        display_dataframe(
            top_locations,
            formatters={"avg_risk_score": format_score, "max_risk_score": format_score, "high_risk_pct": format_percent},
            height=330,
        )

    render_section_header(
        "Embedded Reporting Views",
        "Pipeline-generated Plotly artifacts remain available for drill-down during the presentation.",
        kicker="Artifacts",
    )
    fig_tabs = st.tabs(["Account View", "Merchant View", "Location View", "Component Heatmap"])
    for tab, figure_name, height in [
        (fig_tabs[0], "risk_by_account", 650),
        (fig_tabs[1], "risk_by_merchant", 650),
        (fig_tabs[2], "risk_by_location", 650),
        (fig_tabs[3], "risk_components_heatmap", 780),
    ]:
        with tab:
            fig_html = load_figure(figure_name)
            if fig_html:
                components.html(fig_html, height=height)
            else:
                st.info("Figure not available in the current output bundle.")


def page_transactions(data: dict) -> None:
    transactions = data["transactions"]
    explanations = data["explanations"]
    store = ReviewStore()

    render_section_header(
        "Suspicious Transactions",
        "Triage flagged transactions, inspect risk drivers, and record analyst actions without leaving the dashboard.",
        kicker="Analyst Queue",
    )

    filtered = render_transactions_filters(transactions)
    st.markdown(
        f"""
        <div class="insight-card">
            {badge('Needs Review')}<span style="margin-left:0.65rem;"></span>
            Showing <strong>{len(filtered):,}</strong> transactions from a portfolio of <strong>{len(transactions):,}</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if filtered.empty:
        st.warning("No transactions match the selected filters.")
        return

    queue_view = filtered[
        [c for c in ["transactionid", "accountid", "merchantid", "location", "channel", "transactionamount", "composite_risk_score", "risk_level"] if c in filtered]
    ].copy()
    display_dataframe(
        queue_view,
        formatters={"transactionamount": format_currency, "composite_risk_score": format_score},
        height=350,
    )

    selected_tx_id = st.selectbox(
        "Select transaction for investigation",
        options=filtered["transactionid"].astype(str).tolist(),
    )
    tx = filtered.loc[filtered["transactionid"].astype(str) == str(selected_tx_id)].iloc[0]

    col1, col2, col3, col4 = st.columns(4, gap="medium")
    with col1:
        render_detail_card("Transaction Amount", format_currency(tx.get("transactionamount")))
    with col2:
        render_detail_card("Composite Risk Score", format_score(tx.get("composite_risk_score")))
    with col3:
        st.markdown(
            f"""
            <div class="detail-card">
                <div class="detail-label">Risk Level</div>
                <div class="detail-value">{badge(f"{tx.get('risk_level', 'Needs Review')} Risk" if tx.get('risk_level') in ['High', 'Medium', 'Low'] else tx.get('risk_level', 'Needs Review'))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        render_detail_card("Channel", str(tx.get("channel", "N/A")))

    render_section_header(
        "Risk Component Breakdown",
        "Underlying model contributions for the selected transaction.",
        kicker="Diagnostics",
    )
    components_map = {
        "Isolation Forest": float(tx.get("isolation_forest_score", 0) or 0),
        "Local Outlier Factor": float(tx.get("lof_score", 0) or 0),
        "K-Means": float(tx.get("kmeans_anomaly_score", 0) or 0),
        "Graph Risk": float(tx.get("graph_risk_score", 0) or 0),
        "Amount Outlier": float(tx.get("amount_outlier_risk", 0) or 0),
    }
    breakdown = pd.DataFrame({"component": list(components_map.keys()), "score": list(components_map.values())})
    fig = px.bar(
        breakdown.sort_values("score", ascending=True),
        x="score",
        y="component",
        orientation="h",
        color="score",
        color_continuous_scale=[[0, BRAND["green_soft"]], [0.55, BRAND["green"]], [1, BRAND["danger"]]],
        title=f"Risk Drivers for {selected_tx_id}",
    )
    fig.update_xaxes(range=[0, 1])
    st.plotly_chart(apply_chart_theme(fig, 360), use_container_width=True)

    st.markdown(
        f"""
        <div class="insight-card">
            <strong>Transaction context</strong><br>
            Account: {tx.get('accountid', 'N/A')}<br>
            Merchant: {tx.get('merchantid', 'N/A')}<br>
            Location: {tx.get('location', 'N/A')}
        </div>
        """,
        unsafe_allow_html=True,
    )

    explanation_key = f"transaction_{selected_tx_id}"
    if explanation_key in explanations:
        render_insight(f"<strong>AI Explanation</strong><br>{explanations[explanation_key]}")

    render_section_header(
        "Analyst Decision",
        "Capture disposition, notes, and review rationale for the selected case.",
        kicker="Workflow",
    )
    current_decision = store.get_decision(str(selected_tx_id))
    default_decision = current_decision["decision"] if current_decision else "Needs Review"
    default_notes = current_decision["analyst_notes"] if current_decision else ""

    decision_col, note_col = st.columns([0.9, 1.7], gap="large")
    with decision_col:
        decision = st.radio(
            "Decision",
            options=config.DECISION_OPTIONS,
            index=config.DECISION_OPTIONS.index(default_decision),
        )
        st.markdown(badge("Approved Flag" if decision == "Approve" else decision), unsafe_allow_html=True)
    with note_col:
        notes = st.text_area("Analyst Notes", value=default_notes, height=140)

    if st.button("Save Analyst Decision", key=f"save_{selected_tx_id}"):
        store.record_decision(
            transaction_id=str(selected_tx_id),
            account_id=str(tx.get("accountid", "")),
            decision=decision,
            notes=notes,
        )
        st.success("Decision recorded successfully.")


def page_accounts(data: dict) -> None:
    accounts = data["accounts"]
    transactions = data["transactions"]
    explanations = data["explanations"]

    render_section_header(
        "Account Risk Analysis",
        "Inspect concentration, transaction patterns, and risk severity at the account level.",
        kicker="Account View",
    )

    display_cols = [c for c in ["accountid", "account_risk_score", "max_risk_score", "transaction_count", "high_risk_transaction_count", "high_risk_transaction_pct"] if c in accounts]
    display_dataframe(
        accounts.head(20)[display_cols],
        formatters={
            "account_risk_score": format_score,
            "max_risk_score": format_score,
            "high_risk_transaction_pct": format_percent,
        },
        height=360,
    )

    account_options = accounts["accountid"].dropna().astype(str).tolist()
    if not account_options:
        st.warning("No account-level outputs are available.")
        return

    selected_account = st.selectbox("Select account for detail", options=account_options)
    account = accounts.loc[accounts["accountid"].astype(str) == str(selected_account)].iloc[0]
    account_txs = transactions.loc[transactions["accountid"].astype(str) == str(selected_account)].copy()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_detail_card("Account Risk Score", format_score(account.get("account_risk_score")))
    with col2:
        render_detail_card("Transaction Count", f"{int(account.get('transaction_count', 0)):,}")
    with col3:
        render_detail_card("High-Risk Count", f"{int(account.get('high_risk_transaction_count', 0)):,}")
    with col4:
        render_detail_card("High-Risk Mix", format_percent(account.get("high_risk_transaction_pct")))

    if not account_txs.empty:
        fig = px.bar(
            account_txs.head(25).sort_values("composite_risk_score"),
            x="composite_risk_score",
            y="transactionid",
            orientation="h",
            color="risk_level",
            category_orders={"risk_level": RISK_ORDER},
            color_discrete_map={"High": BRAND["danger"], "Medium": BRAND["warning"], "Low": BRAND["green"]},
            title=f"Highest-Risk Transactions for {selected_account}",
            hover_data=["merchantid", "transactionamount", "channel"],
        )
        st.plotly_chart(apply_chart_theme(fig, 520), use_container_width=True)

        display_dataframe(
            account_txs[
                [c for c in ["transactionid", "merchantid", "transactionamount", "composite_risk_score", "risk_level", "channel"] if c in account_txs]
            ].head(50),
            formatters={"transactionamount": format_currency, "composite_risk_score": format_score},
            height=360,
        )

    explanation_key = f"account_{selected_account}"
    if explanation_key in explanations:
        render_insight(f"<strong>AI Explanation</strong><br>{explanations[explanation_key]}")


def page_entities(data: dict) -> None:
    merchants = data["merchants"]
    devices = data["devices"]
    locations = data["locations"]

    render_section_header(
        "Merchant, Device, and Location Exposure",
        "Cross-entity risk concentration for merchant networks, reused devices, and geographic hotspots.",
        kicker="Entity View",
    )

    tabs = st.tabs(["Merchants", "Devices", "Locations"])

    with tabs[0]:
        if merchants.empty:
            st.info("Merchant summary is not available.")
        else:
            fig = px.bar(
                merchants.head(15).sort_values("avg_risk_score"),
                x="avg_risk_score",
                y="merchantid",
                orientation="h",
                color="max_risk_score",
                color_continuous_scale=[[0, BRAND["green_soft"]], [0.55, BRAND["green"]], [1, BRAND["danger"]]],
                title="Top Merchants by Average Risk Score",
            )
            st.plotly_chart(apply_chart_theme(fig, 500), use_container_width=True)
            display_dataframe(
                merchants.head(25),
                formatters={"avg_risk_score": format_score, "max_risk_score": format_score, "high_risk_pct": format_percent},
                height=360,
            )

    with tabs[1]:
        if devices.empty:
            st.info("Device summary is not available.")
        else:
            fig = px.bar(
                devices.head(15).sort_values("avg_risk_score"),
                x="avg_risk_score",
                y="deviceid",
                orientation="h",
                color="max_risk_score",
                color_continuous_scale=[[0, BRAND["green_soft"]], [0.55, BRAND["green"]], [1, BRAND["danger"]]],
                title="Top Devices by Average Risk Score",
            )
            st.plotly_chart(apply_chart_theme(fig, 500), use_container_width=True)
            display_dataframe(
                devices.head(25),
                formatters={"avg_risk_score": format_score, "max_risk_score": format_score, "high_risk_pct": format_percent},
                height=360,
            )

    with tabs[2]:
        if locations.empty:
            st.info("Location summary is not available.")
        else:
            fig = px.bar(
                locations.head(15).sort_values("avg_risk_score"),
                x="avg_risk_score",
                y="location",
                orientation="h",
                color="max_risk_score",
                color_continuous_scale=[[0, BRAND["green_soft"]], [0.55, BRAND["green"]], [1, BRAND["danger"]]],
                title="Top Locations by Average Risk Score",
            )
            st.plotly_chart(apply_chart_theme(fig, 500), use_container_width=True)
            display_dataframe(
                locations.head(25),
                formatters={"avg_risk_score": format_score, "max_risk_score": format_score, "high_risk_pct": format_percent},
                height=360,
            )


def page_decisions() -> None:
    store = ReviewStore()
    decisions = store.get_all_decisions()

    render_section_header(
        "Analyst Review Log",
        "Disposition history for case review, approvals, dismissals, and items still in queue.",
        kicker="Governance",
    )

    if decisions.empty:
        st.info("No analyst decisions recorded yet.")
        return

    counts = decisions["decision"].value_counts()
    metrics = [
        ("Approved Flags", f"{int(counts.get('Approve', 0)):,}", "Transactions confirmed for escalation or retention."),
        ("Dismissed", f"{int(counts.get('Dismiss', 0)):,}", "Transactions reviewed and dismissed."),
        ("Needs Review", f"{int(counts.get('Needs Review', 0)):,}", "Transactions still awaiting final analyst disposition."),
    ]
    for column, (label, value, note) in zip(st.columns(3), metrics):
        with column:
            render_metric_card(label, value, note)

    log_view = decisions.copy()
    log_view["status"] = log_view["decision"]
    display_dataframe(log_view[["transactionid", "accountid", "status", "analyst_notes", "timestamp"]], height=380)

    csv_buffer = io.StringIO()
    decisions.to_csv(csv_buffer, index=False)
    st.download_button(
        "Download Review Log",
        data=csv_buffer.getvalue(),
        file_name=f"review_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )


def page_info(data: dict) -> None:
    summary = data["summary"]
    render_section_header(
        "Pipeline Information",
        "Configuration, thresholds, and output paths backing the current executive demo.",
        kicker="Reference",
    )

    col1, col2 = st.columns(2, gap="large")
    with col1:
        weights_df = pd.DataFrame(
            [{"Component": component, "Weight": weight} for component, weight in config.RISK_WEIGHTS.items()]
        )
        display_dataframe(weights_df, formatters={"Weight": lambda value: f"{value:.2f}"}, height=320)

    with col2:
        thresholds_df = pd.DataFrame(
            [
                {"Threshold": "Low Risk Upper Bound", "Value": config.RISK_LEVEL_LOW},
                {"Threshold": "Medium Risk Upper Bound", "Value": config.RISK_LEVEL_MEDIUM},
                {"Threshold": "High Risk Upper Bound", "Value": config.RISK_LEVEL_HIGH},
                {"Threshold": "High-Risk Share", "Value": summary.get("high_risk_pct", 0) / 100},
            ]
        )
        display_dataframe(thresholds_df, formatters={"Value": lambda value: f"{value:.2f}"}, height=320)

    st.markdown(
        f"""
        <div class="insight-card">
            <strong>Data Paths</strong><br>
            Raw Data: {config.RAW_DATA_FILE}<br>
            Cleaned Data: {config.CLEANED_DATA_FILE}<br>
            Ranked Transactions: {config.RISK_TRANSACTIONS_FILE}<br>
            Ranked Accounts: {config.RISK_ACCOUNTS_FILE}<br>
            Figures Directory: {config.FIGURES_DIR}<br>
            Reports Directory: {config.REPORTS_DIR}
        </div>
        """,
        unsafe_allow_html=True,
    )

    features_df = pd.DataFrame(
        {
            "Capability": [
                "Unsupervised anomaly detection",
                "Graph-based transaction analysis",
                "Composite transparent risk scoring",
                "Analyst review and decision tracking",
                "Executive exports and embedded charts",
                "Optional OpenAI narrative explanations",
            ]
        }
    )
    display_dataframe(features_df, height=280)


def main() -> None:
    data = load_pipeline_outputs()
    page = st.session_state.get("selected_page", "Overview")
    page = render_sidebar(page, data)
    st.session_state["selected_page"] = page

    render_app_header(
        "Fraud Intelligence Dashboard",
        "A polished executive monitoring layer for transaction surveillance, entity exposure analysis, and analyst review decisions.",
        logo_path=LOGO_PATH,
    )

    if page == "Overview":
        page_overview(data)
    elif page == "Suspicious Transactions":
        page_transactions(data)
    elif page == "Account Risk":
        page_accounts(data)
    elif page == "Merchants, Devices & Locations":
        page_entities(data)
    elif page == "Review Log":
        page_decisions()
    elif page == "Pipeline Info":
        page_info(data)


if __name__ == "__main__":
    main()
