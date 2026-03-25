"""
Executive Streamlit app for the fraud detection pipeline.

Focus:
  - Premium executive dashboard styling
  - Upload-driven analysis for demos
  - AI recommendations, Q&A, and case explanations
  - Reliable analyst review workflow

Run with: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config
from src.ai_assistant import (
    ai_availability_message,
    answer_data_question,
    bundle_context_summary,
    explain_case,
    generate_ai_recommendations,
    is_ai_enabled,
    rule_based_recommendations,
    rule_based_reminders,
)
from src.chatops import publish_and_send_report, publish_bundle_context, send_alert_notifications
from src.chatops.context_loader import build_review_summary
from src.dashboard_data import (
    CSV_UPLOAD_OPTIONS,
    bundle_from_transactions,
    bundle_from_uploaded_csv,
    validate_uploaded_csv,
)
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
    "Executive Summary",
    "Upload Data",
    "Suspicious Transactions",
    "Risky Entities",
    "AI Recommendations",
    "Ask Questions About Data",
    "Analyst Review Log",
    "OOF Controls",
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
        "login_attempt_risk",
        "device_change_flag",
        "ip_change_flag",
        "time_since_previous_transaction",
        "previous_date_regenerated",
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
def load_pipeline_outputs() -> Dict[str, Any]:
    transactions = safe_read_csv(config.RISK_TRANSACTIONS_FILE)
    if transactions.empty:
        raise FileNotFoundError(config.RISK_TRANSACTIONS_FILE)

    transactions = _enrich_transaction_context(transactions)
    transactions = transactions.sort_values("composite_risk_score", ascending=False).reset_index(drop=True)
    bundle = bundle_from_transactions(transactions, "Pipeline outputs")

    accounts = safe_read_csv(config.RISK_ACCOUNTS_FILE)
    merchants = safe_read_csv(config.REPORTS_DIR / "risk_ranked_merchants.csv")
    devices = safe_read_csv(config.REPORTS_DIR / "risk_ranked_devices.csv")
    ips = safe_read_csv(config.REPORTS_DIR / "risk_ranked_ips.csv")

    if not accounts.empty:
        bundle["accounts"] = accounts
    if not merchants.empty:
        bundle["merchants"] = merchants
    if not devices.empty:
        bundle["devices"] = devices
    if not ips.empty:
        bundle["ips"] = ips

    summary_path = config.REPORTS_DIR / "executive_summary.json"
    review_log = ReviewStore().get_all_decisions()
    bundle["summary"] = {
        **bundle["summary"],
        **build_review_summary(bundle["transactions"], review_log),
    }
    if summary_path.exists():
        file_summary = json.loads(summary_path.read_text())
        bundle["summary"] = {
            **bundle["summary"],
            **file_summary,
        }

    explanations_path = config.REPORTS_DIR / "openai_explanations.json"
    bundle["explanations"] = json.loads(explanations_path.read_text()) if explanations_path.exists() else {}
    bundle["review_log"] = review_log
    bundle["source_label"] = "Pipeline outputs"
    bundle["uploaded_type"] = "pipeline_outputs"
    return bundle


def get_active_bundle() -> Dict[str, Any]:
    try:
        return st.session_state.get("uploaded_bundle") or load_pipeline_outputs()
    except FileNotFoundError as exc:
        st.error(f"Pipeline output missing: {exc}")
        st.info("Run `python3 run_pipeline.py` from `fraud_pipeline/` before opening the dashboard.")
        st.stop()


def current_source_label(bundle: Dict[str, Any]) -> str:
    return bundle.get("source_label", "Pipeline outputs")


def publish_bundle_for_chatops(bundle: Dict[str, Any], *, publish_reason: str, headline: str | None = None) -> Dict[str, Any]:
    if config.OPENCLAW_STREAMLIT_AUTO_SEND:
        return publish_and_send_report(bundle, headline=headline, publish_reason=publish_reason)
    manifest = publish_bundle_context(bundle, publish_reason=publish_reason)
    return {"manifest": manifest, "delivery": None, "message": None}


def render_chatops_status(result: Dict[str, Any] | None) -> None:
    if not result:
        return
    delivery = result.get("delivery")
    manifest = result.get("manifest") or {}
    if delivery is None:
        st.caption(
            f"Published active context for ChatOps from `{manifest.get('source_label', 'active dashboard context')}`. "
            "Automatic webhook delivery is disabled."
        )
        return
    if delivery.delivered:
        st.caption("ChatOps delivery completed successfully.")
    elif delivery.delivery_error:
        st.info(f"ChatOps delivery did not complete: {delivery.delivery_error}")


def bundle_signature(bundle: Dict[str, Any]) -> str:
    context = bundle_context_summary(bundle)
    return hashlib.md5(context.encode("utf-8")).hexdigest()


def format_currency(value) -> str:
    return f"${value:,.0f}" if pd.notna(value) else "N/A"


def format_score(value) -> str:
    return f"{value:.3f}" if pd.notna(value) else "N/A"


def format_percent(value) -> str:
    return f"{value:.1f}%" if pd.notna(value) else "N/A"


def render_sidebar(page_name: str, data: Dict[str, Any]) -> str:
    summary = data.get("summary", {}) or {}
    active_source = current_source_label(data)
    with st.sidebar:
        st.markdown(
            f"""
            <div class="sidebar-panel">
                <div style="font-size:0.78rem; letter-spacing:0.14em; text-transform:uppercase; color:#d5e1da; font-weight:800;">
                    Navigation
                </div>
                <div style="font-size:1.25rem; font-weight:800; margin-top:0.45rem;">Fraud Intelligence</div>
                <div style="color:#eef4ef; margin-top:0.35rem; line-height:1.5;">
                    Executive monitoring, upload triage, AI guidance, and analyst review in one workspace.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        selected = st.radio("Go to", NAV_ITEMS, index=NAV_ITEMS.index(page_name), label_visibility="collapsed")

        st.markdown(
            f"""
            <div class="sidebar-panel">
                <div style="font-size:0.78rem; letter-spacing:0.14em; text-transform:uppercase; color:#d5e1da; font-weight:800;">
                    Active Source
                </div>
                <div style="margin-top:0.55rem; color:#ffffff; font-weight:700;">{active_source}</div>
                <div style="margin-top:0.85rem;">{badge('High Risk')}</div>
                <div style="margin-top:0.45rem; color:#eef4ef;">{summary.get('high_risk_count', 0):,} transactions</div>
                <div style="margin-top:0.85rem;">{badge('Needs Review')}</div>
                <div style="margin-top:0.45rem; color:#eef4ef;">{summary.get('flagged_transactions', 0):,} flagged items</div>
                <div style="margin-top:0.85rem; color:#d5e1da;">Coverage</div>
                <div style="margin-top:0.35rem; color:#eef4ef;">{summary.get('total_transactions', 0):,} transactions</div>
                <div style="margin-top:0.25rem; color:#eef4ef;">{summary.get('total_accounts', 0):,} accounts</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="sidebar-panel">
                <div style="font-size:0.78rem; letter-spacing:0.14em; text-transform:uppercase; color:#d5e1da; font-weight:800;">
                    AI Status
                </div>
                <div style="margin-top:0.55rem;">{badge('Approved Flag' if is_ai_enabled() else 'Needs Review')}</div>
                <div style="margin-top:0.5rem; color:#eef4ef; line-height:1.45;">{ai_availability_message()}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.session_state.get("uploaded_bundle") is not None:
            if st.button("Revert to Pipeline Outputs", use_container_width=True):
                st.session_state.pop("uploaded_bundle", None)
                st.session_state.pop("qa_history", None)
                st.rerun()

        st.caption("Designed for laptop presentation and live executive walkthroughs.")
    return selected


def display_dataframe(df: pd.DataFrame, formatters: dict | None = None, height: int = 360) -> None:
    if df.empty:
        st.info("No data available for this view.")
        return
    view = df.copy()
    if formatters:
        for column, formatter in formatters.items():
            if column in view.columns:
                view[column] = view[column].map(lambda value: formatter(value) if pd.notna(value) else "N/A")
    st.dataframe(view, use_container_width=True, hide_index=True, height=height)


def render_case_explanation(entity_type: str, case_summary: Dict[str, Any], bundle: Dict[str, Any], case_key: str) -> None:
    cache = st.session_state.setdefault("case_explanations", {})
    explanation = cache.get(case_key)
    if explanation is None:
        explanation = explain_case(entity_type=entity_type, case_summary=case_summary, bundle=bundle)
        cache[case_key] = explanation

    render_insight(f"<strong>Recommended review focus</strong><br>{explanation['baseline']}")
    if explanation.get("ai_text"):
        render_insight(f"<strong>AI Case Explanation</strong><br>{explanation['ai_text']}")
    elif not explanation.get("ai_available"):
        st.info(ai_availability_message())


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(uploaded_file.getvalue()))


def get_cached_ai_recommendations(bundle: Dict[str, Any]) -> Dict[str, Any]:
    cache = st.session_state.setdefault("ai_recommendation_cache", {})
    signature = bundle_signature(bundle)
    if signature not in cache:
        cache[signature] = generate_ai_recommendations(bundle)
    return cache[signature]


def render_upload_validation(validation: Dict[str, Any], upload_type: str) -> None:
    expected = validation["expected_columns"]
    missing = validation["missing_columns"]
    if missing:
        st.error(
            f"Validation failed for `{upload_type}`. Missing required columns: {', '.join(missing)}"
        )
    else:
        st.success(f"Validation passed for `{upload_type}`.")

    st.markdown(
        f"""
        <div class="insight-card">
            <strong>Validation result</strong><br>
            Selected type: {upload_type}<br>
            Required columns: {', '.join(expected)}<br>
            Status: {'Valid upload' if not missing else 'Missing required columns'}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_transaction_snapshot(bundle: Dict[str, Any], prefix: str = "upload") -> None:
    transactions = bundle.get("transactions", pd.DataFrame())
    accounts = bundle.get("accounts", pd.DataFrame())
    merchants = bundle.get("merchants", pd.DataFrame())
    devices = bundle.get("devices", pd.DataFrame())
    locations = bundle.get("locations", pd.DataFrame())
    summary = bundle.get("summary", {}) or {}

    if transactions.empty:
        st.info("No transaction-level results are available for this uploaded file.")
        return

    render_section_header(
        "Processed Outputs",
        "Suspicious transaction ranking, entity hotspots, anomaly signals, and risk score visuals generated from the uploaded file.",
        kicker="Results",
    )

    kpi_specs = [
        ("Flagged Transactions", f"{summary.get('flagged_transactions', 0):,}", "Medium and high risk transactions."),
        ("High-Risk Accounts", f"{summary.get('high_risk_accounts', 0):,}", "Accounts above the medium-risk threshold."),
        ("High-Risk Merchants", f"{summary.get('high_risk_merchants', 0):,}", "Merchants concentrated in suspicious activity."),
        ("High-Risk Devices", f"{summary.get('high_risk_devices', 0):,}", "Devices linked to elevated-risk cases."),
        ("High-Risk Locations", f"{summary.get('high_risk_locations', 0):,}", "Locations showing the strongest risk clustering."),
    ]
    for column, (label, value, note) in zip(st.columns(5), kpi_specs):
        with column:
            render_metric_card(label, value, note)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        risk_mix = transactions["risk_level"].value_counts().reindex(RISK_ORDER).dropna()
        if not risk_mix.empty:
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
            fig.update_layout(title="Risk Tier Mix")
            st.plotly_chart(apply_chart_theme(fig, 360), use_container_width=True)

    with col2:
        if {"channel", "composite_risk_score", "transactionid"}.issubset(transactions.columns):
            channel_risk = (
                transactions.groupby("channel", dropna=False)
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
            st.plotly_chart(apply_chart_theme(fig, 360), use_container_width=True)

    filter_col1, filter_col2 = st.columns([1.1, 1], gap="large")
    available_levels = [level for level in RISK_ORDER if level in transactions["risk_level"].dropna().unique().tolist()]
    with filter_col1:
        risk_levels = st.multiselect(
            "Filter risk levels",
            options=available_levels,
            default=available_levels[:2] or available_levels,
            key=f"{prefix}_risk_levels",
        )
    with filter_col2:
        min_score = st.slider(
            "Minimum composite risk score",
            min_value=0.0,
            max_value=1.0,
            value=0.45,
            step=0.01,
            key=f"{prefix}_min_score",
        )

    filtered = transactions.copy()
    if risk_levels:
        filtered = filtered[filtered["risk_level"].isin(risk_levels)]
    filtered = filtered[filtered["composite_risk_score"] >= min_score]

    render_section_header(
        "Suspicious Transaction Summary",
        "Filtered suspicious transactions with anomaly factors and composite risk scores for immediate review.",
        kicker="Ranked Queue",
    )
    suspicious_columns = [
        column for column in [
            "transactionid",
            "accountid",
            "merchantid",
            "location",
            "deviceid",
            "channel",
            "transactionamount",
            "isolation_forest_score",
            "lof_score",
            "kmeans_anomaly_score",
            "graph_risk_score",
            "composite_risk_score",
            "risk_level",
        ] if column in filtered.columns
    ]
    display_dataframe(
        filtered[suspicious_columns].head(25),
        formatters={
            "transactionamount": format_currency,
            "isolation_forest_score": format_score,
            "lof_score": format_score,
            "kmeans_anomaly_score": format_score,
            "graph_risk_score": format_score,
            "composite_risk_score": format_score,
        },
        height=360,
    )

    if not filtered.empty:
        selected_tx_id = st.selectbox(
            "Select suspicious transaction for explanation",
            options=filtered["transactionid"].astype(str).tolist(),
            key=f"{prefix}_transaction_explanation",
        )
        tx = filtered.loc[filtered["transactionid"].astype(str) == str(selected_tx_id)].iloc[0]
        render_case_explanation(
            "transaction",
            {
                "transactionid": tx.get("transactionid"),
                "accountid": tx.get("accountid"),
                "merchantid": tx.get("merchantid"),
                "location": tx.get("location"),
                "channel": tx.get("channel"),
                "transactionamount": tx.get("transactionamount"),
                "composite_risk_score": tx.get("composite_risk_score"),
            },
            bundle,
            case_key=f"{prefix}::transaction::{selected_tx_id}",
        )

    render_section_header(
        "High-Risk Accounts, Merchants, Locations, and Devices",
        "Top-ranked entity views produced from the uploaded file for a fast executive walkthrough.",
        kicker="Entity Hotspots",
    )
    tabs = st.tabs(["Accounts", "Merchants", "Locations", "Devices"])
    with tabs[0]:
        display_dataframe(
            accounts.head(15)[[column for column in ["accountid", "account_risk_score", "transaction_count", "high_risk_transaction_count"] if column in accounts.columns]],
            formatters={"account_risk_score": format_score},
            height=320,
        )
    with tabs[1]:
        display_dataframe(
            merchants.head(15)[[column for column in ["merchantid", "avg_risk_score", "max_risk_score", "transaction_count", "high_risk_count"] if column in merchants.columns]],
            formatters={"avg_risk_score": format_score, "max_risk_score": format_score},
            height=320,
        )
    with tabs[2]:
        display_dataframe(
            locations.head(15),
            formatters={"avg_risk_score": format_score, "max_risk_score": format_score, "high_risk_pct": format_percent},
            height=320,
        )
    with tabs[3]:
        display_dataframe(
            devices.head(15)[[column for column in ["deviceid", "avg_risk_score", "max_risk_score", "transaction_count", "high_risk_count"] if column in devices.columns]],
            formatters={"avg_risk_score": format_score, "max_risk_score": format_score},
            height=320,
        )


def render_inline_ai_guidance(bundle: Dict[str, Any], prefix: str = "upload") -> None:
    render_section_header(
        "AI Recommendations",
        "Grounded monitoring guidance, reminders, and business-facing recommendations for the current uploaded dataset.",
        kicker="AI Guidance",
    )

    recommendations = get_cached_ai_recommendations(bundle)
    reminder_columns = st.columns(max(1, len(recommendations["reminders"])))
    for column, reminder in zip(reminder_columns, recommendations["reminders"]):
        with column:
            render_insight(reminder)

    for item in recommendations["baseline_recommendations"]:
        render_insight(item)

    if recommendations["ai_recommendations"]:
        for item in recommendations["ai_recommendations"]:
            render_insight(item)
    else:
        st.info(recommendations["availability_message"])

    render_section_header(
        "Ask Questions About Data",
        "Ask live questions about the uploaded dataset and get grounded responses tied to the current results.",
        kicker="Q&A",
    )
    question = st.text_input(
        "Ask about the uploaded data",
        placeholder="Example: Which merchants appear most often in flagged transactions?",
        key=f"{prefix}_question_input",
    )
    if st.button("Ask About Uploaded Data", key=f"{prefix}_ask_button") and question.strip():
        response = answer_data_question(question.strip(), bundle)
        history = st.session_state.setdefault(f"{prefix}_qa_history", [])
        history.append(
            {
                "question": question.strip(),
                "answer": response["ai_answer"] or response["heuristic_answer"],
                "used_ai": bool(response["ai_answer"]),
            }
        )

    history = st.session_state.get(f"{prefix}_qa_history", [])
    if history:
        for item in reversed(history):
            render_insight(f"<strong>Question</strong><br>{item['question']}")
            render_insight(f"<strong>Answer</strong><br>{item['answer']}")
    else:
        st.info("Ask a question after processing a valid CSV to open a live investigation thread.")


def render_uploaded_review_log(bundle: Dict[str, Any]) -> None:
    review_log = bundle.get("review_log", pd.DataFrame())
    if review_log.empty:
        st.info("No analyst review rows were loaded from this upload.")
        return

    render_section_header(
        "Review History",
        "Uploaded analyst decisions, timestamps, and notes for operational review history.",
        kicker="Review Log",
    )
    counts = review_log["decision"].value_counts() if "decision" in review_log.columns else pd.Series(dtype=int)
    metrics = [
        ("Approved", f"{int(counts.get('Approve Flag', 0)):,}", "Uploaded cases approved for escalation."),
        ("Dismissed", f"{int(counts.get('Dismiss', 0)):,}", "Uploaded cases dismissed after review."),
        ("Needs Review", f"{int(counts.get('Needs Review', 0)):,}", "Uploaded cases still awaiting closure."),
        ("Rows", f"{len(review_log):,}", "Total records loaded from the review log."),
    ]
    for column, (label, value, note) in zip(st.columns(4), metrics):
        with column:
            render_metric_card(label, value, note)

    display_dataframe(
        review_log[[column for column in ["case_id", "transactionid", "accountid", "decision", "analyst_notes", "created_at", "updated_at", "review_version"] if column in review_log.columns]],
        height=380,
    )


def render_upload_outputs(bundle: Dict[str, Any]) -> None:
    uploaded_type = bundle.get("uploaded_type")
    if uploaded_type in {"raw_transaction_dataset", "transactions"}:
        render_transaction_snapshot(bundle)
        render_inline_ai_guidance(bundle)
        return

    if uploaded_type == "review_log":
        render_uploaded_review_log(bundle)
        return

    st.info("Uploaded data was loaded successfully, but there is no dedicated dashboard renderer for this type yet.")


def page_overview(bundle: Dict[str, Any]) -> None:
    transactions = bundle.get("transactions", pd.DataFrame())
    summary = bundle.get("summary", {}) or {}
    accounts = bundle.get("accounts", pd.DataFrame())
    merchants = bundle.get("merchants", pd.DataFrame())
    devices = bundle.get("devices", pd.DataFrame())
    locations = bundle.get("locations", pd.DataFrame())

    render_section_header(
        "Executive Summary",
        "A business-facing snapshot of flagged exposure, entity concentration, and portfolio-wide risk posture.",
        kicker="Overview",
    )

    kpi_specs = [
        ("Flagged Transactions", f"{summary.get('flagged_transactions', 0):,}", "Medium and high risk transactions requiring attention."),
        ("High-Risk Accounts", f"{summary.get('high_risk_accounts', 0):,}", "Accounts with elevated account-level risk scores."),
        ("High-Risk Merchants", f"{summary.get('high_risk_merchants', 0):,}", "Merchants most associated with suspicious activity."),
        ("High-Risk Devices", f"{summary.get('high_risk_devices', 0):,}", "Devices recurring in elevated-risk transactions."),
        ("High-Risk Locations", f"{summary.get('high_risk_locations', 0):,}", "Locations with concentrated suspicious activity."),
    ]
    for column, (label, value, footnote) in zip(st.columns(5), kpi_specs):
        with column:
            render_metric_card(label, value, footnote)

    reminder_cards = rule_based_reminders(bundle)
    if reminder_cards:
        reminder_columns = st.columns(len(reminder_cards))
        for column, message in zip(reminder_columns, reminder_cards):
            with column:
                render_insight(message)

    if transactions.empty:
        st.warning("No transaction-level data is available in the active source.")
        return

    render_section_header(
        "Risk Posture",
        "Current transaction mix, score dispersion, and transaction-channel concentration.",
        kicker="Monitoring",
    )
    col1, col2 = st.columns([1.05, 1], gap="large")

    with col1:
        risk_mix = transactions["risk_level"].value_counts().reindex(RISK_ORDER).dropna()
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
        fig.update_traces(opacity=0.74)
        fig.update_xaxes(range=[0, 1])
        st.plotly_chart(apply_chart_theme(fig, 420), use_container_width=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        channel_risk = (
            transactions.groupby("channel", dropna=False)
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
            opacity=0.72,
            category_orders={"risk_level": RISK_ORDER},
            color_discrete_map={"High": BRAND["danger"], "Medium": BRAND["warning"], "Low": BRAND["green"]},
            title="Transaction Amount vs Risk Score",
            hover_data=[column for column in ["transactionid", "accountid", "merchantid"] if column in transactions.columns],
        )
        st.plotly_chart(apply_chart_theme(fig, 380), use_container_width=True)

    render_section_header(
        "Entity Hotspots",
        "Quick view of the most exposed accounts, merchants, devices, and locations in the current run.",
        kicker="Concentration",
    )
    tab1, tab2, tab3, tab4 = st.tabs(["Accounts", "Merchants", "Devices", "Locations"])

    with tab1:
        display_dataframe(
            accounts.head(12)[[column for column in ["accountid", "account_risk_score", "high_risk_transaction_count", "transaction_count"] if column in accounts.columns]],
            formatters={"account_risk_score": format_score},
            height=330,
        )

    with tab2:
        display_dataframe(
            merchants.head(12)[[column for column in ["merchantid", "avg_risk_score", "max_risk_score", "transaction_count", "high_risk_count"] if column in merchants.columns]],
            formatters={"avg_risk_score": format_score, "max_risk_score": format_score},
            height=330,
        )

    with tab3:
        display_dataframe(
            devices.head(12)[[column for column in ["deviceid", "avg_risk_score", "max_risk_score", "transaction_count", "high_risk_count"] if column in devices.columns]],
            formatters={"avg_risk_score": format_score, "max_risk_score": format_score},
            height=330,
        )

    with tab4:
        display_dataframe(
            locations.head(12),
            formatters={"avg_risk_score": format_score, "max_risk_score": format_score, "high_risk_pct": format_percent},
            height=330,
        )

    render_section_header(
        "Embedded Reporting Views",
        "Existing Plotly artifacts remain available for live drill-down during the presentation.",
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


def page_upload_data() -> None:
    render_section_header(
        "Upload Data",
        "Choose one of the approved CSV presets first, then upload and validate the file before the dashboard processes it.",
        kicker="Upload Workflow",
    )

    upload_type = st.radio(
        "CSV Type",
        options=CSV_UPLOAD_OPTIONS,
        index=None,
        horizontal=True,
        key="upload_csv_type",
    )

    if not upload_type:
        st.info("Choose one CSV type first. The file uploader appears after the selection is made.")
        return

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key=f"uploader_{upload_type}")

    if uploaded_file is None:
        st.info("Upload a CSV after choosing the dataset type.")
        return

    raw_df = read_uploaded_csv(uploaded_file)
    validation = validate_uploaded_csv(raw_df, upload_type)
    normalized_df = validation["normalized_df"]

    st.success(f"Loaded `{uploaded_file.name}` for review.")
    st.markdown(
        f"""
        <div class="insight-card">
            <strong>Dataset summary</strong><br>
            File name: {uploaded_file.name}<br>
            Selected type: {upload_type}<br>
            Row count: {len(normalized_df):,}<br>
            Detected columns: {len(normalized_df.columns):,}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("Detected columns")
    st.code(", ".join(normalized_df.columns.tolist()) or "(none)")
    render_upload_validation(validation, upload_type)

    render_section_header(
        "Preview",
        "Use the preview to confirm the file structure before loading it into the active dashboard context.",
        kicker="Sanity Check",
    )
    display_dataframe(normalized_df.head(25), height=320)

    action_label = "Run Fraud Analysis" if upload_type == "Raw transaction dataset" else "Load Uploaded File"

    if st.button(action_label, disabled=not validation["is_valid"]):
        try:
            with st.spinner("Processing uploaded data..."):
                bundle = bundle_from_uploaded_csv(raw_df, upload_type, f"{upload_type}: {uploaded_file.name}")
                st.session_state["uploaded_bundle"] = bundle
                st.session_state.pop("qa_history", None)
                st.session_state.pop("ai_recommendation_cache", None)
                st.session_state.pop("upload_qa_history", None)
                if isinstance(bundle.get("transactions"), pd.DataFrame) and not bundle.get("transactions").empty:
                    chatops_result = publish_bundle_for_chatops(
                        bundle,
                        publish_reason="streamlit_upload",
                        headline=f"Fraud analysis completed for {uploaded_file.name}.",
                    )
                else:
                    manifest = publish_bundle_context(bundle, publish_reason="streamlit_upload")
                    chatops_result = {"manifest": manifest, "delivery": None, "message": None}
                st.session_state["last_chatops_delivery"] = chatops_result
            st.success(f"Loaded `{uploaded_file.name}` as the active dashboard source.")
            st.toast("Fraud Analysis Report Sent to Chat")
            render_chatops_status(chatops_result)
        except Exception as exc:
            st.error(f"Processing failed: {exc}")

    if st.session_state.get("uploaded_bundle") is not None:
        active_bundle = st.session_state["uploaded_bundle"]
        active_summary = active_bundle.get("summary", {}) or {}
        st.markdown(
            f"""
            <div class="insight-card">
                <strong>Current active source</strong><br>
                {current_source_label(active_bundle)}<br>
                Flagged transactions: {active_summary.get('flagged_transactions', 0):,}
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_upload_outputs(active_bundle)


def render_transactions_filters(transactions: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty:
        return transactions
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-panel">
                <div style="font-size:0.78rem; letter-spacing:0.14em; text-transform:uppercase; color:#d5e1da; font-weight:800;">
                    Transaction Filters
                </div>
            """,
            unsafe_allow_html=True,
        )
        available_levels = [level for level in RISK_ORDER if level in transactions["risk_level"].dropna().unique().tolist()]
        risk_levels = st.multiselect("Risk Level", options=available_levels, default=available_levels[:2] or available_levels)
        min_score = st.slider("Minimum Risk Score", 0.0, 1.0, 0.45, 0.01)
        account_filter = st.text_input("Account ID contains", "")
        merchant_filter = st.text_input("Merchant ID contains", "")
        channel_options = sorted(transactions["channel"].dropna().astype(str).unique().tolist()) if "channel" in transactions else []
        channel_filter = st.multiselect("Channel", options=channel_options, default=channel_options)
        st.markdown("</div>", unsafe_allow_html=True)

    filtered = transactions.copy()
    if risk_levels:
        filtered = filtered[filtered["risk_level"].isin(risk_levels)]
    filtered = filtered[filtered["composite_risk_score"] >= min_score]

    if account_filter and "accountid" in filtered:
        filtered = filtered[filtered["accountid"].astype(str).str.contains(account_filter, case=False, na=False)]
    if merchant_filter and "merchantid" in filtered:
        filtered = filtered[filtered["merchantid"].astype(str).str.contains(merchant_filter, case=False, na=False)]
    if channel_filter and "channel" in filtered:
        filtered = filtered[filtered["channel"].astype(str).isin(channel_filter)]
    return filtered


def page_transactions(bundle: Dict[str, Any]) -> None:
    transactions = bundle.get("transactions", pd.DataFrame())
    explanations = bundle.get("explanations", {}) or {}
    store = ReviewStore()

    render_section_header(
        "Suspicious Transactions",
        "Triage flagged transactions, inspect model drivers, ask for case explanations, and record analyst actions without leaving the dashboard.",
        kicker="Analyst Queue",
    )

    if transactions.empty:
        st.warning("The active source does not include ranked transaction-level data.")
        return

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

    queue_view = filtered[[column for column in [
        "transactionid",
        "accountid",
        "merchantid",
        "location",
        "channel",
        "transactionamount",
        "composite_risk_score",
        "risk_level",
    ] if column in filtered.columns]].copy()
    display_dataframe(queue_view, formatters={"transactionamount": format_currency, "composite_risk_score": format_score}, height=350)

    selected_tx_id = st.selectbox("Select transaction for investigation", options=filtered["transactionid"].astype(str).tolist())
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
        "Login Attempt": float(tx.get("login_attempt_risk", 0) or 0),
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

    render_case_explanation(
        "transaction",
        {
            "transactionid": tx.get("transactionid"),
            "accountid": tx.get("accountid"),
            "merchantid": tx.get("merchantid"),
            "location": tx.get("location"),
            "channel": tx.get("channel"),
            "transactionamount": tx.get("transactionamount"),
            "composite_risk_score": tx.get("composite_risk_score"),
        },
        bundle,
        case_key=f"transaction::{selected_tx_id}",
    )

    explanation_key = f"transaction_{selected_tx_id}"
    if explanation_key in explanations:
        render_insight(f"<strong>Saved pipeline explanation</strong><br>{explanations[explanation_key]}")

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
        st.markdown(badge(decision), unsafe_allow_html=True)
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


def page_entities(bundle: Dict[str, Any]) -> None:
    accounts = bundle.get("accounts", pd.DataFrame())
    merchants = bundle.get("merchants", pd.DataFrame())
    devices = bundle.get("devices", pd.DataFrame())
    locations = bundle.get("locations", pd.DataFrame())
    transactions = bundle.get("transactions", pd.DataFrame())

    render_section_header(
        "Risky Accounts, Merchants, Devices, and Locations",
        "Cross-entity exposure analysis for the highest-risk accounts, merchants, devices, and geographies.",
        kicker="Entity View",
    )

    tabs = st.tabs(["Accounts", "Merchants", "Devices", "Locations"])

    with tabs[0]:
        display_dataframe(
            accounts.head(25)[[column for column in ["accountid", "account_risk_score", "max_risk_score", "transaction_count", "high_risk_transaction_count", "high_risk_transaction_pct"] if column in accounts.columns]],
            formatters={"account_risk_score": format_score, "max_risk_score": format_score, "high_risk_transaction_pct": format_percent},
            height=360,
        )
        if not accounts.empty and not transactions.empty:
            selected_account = st.selectbox("Select account", options=accounts["accountid"].astype(str).tolist(), key="account_select")
            account_row = accounts.loc[accounts["accountid"].astype(str) == selected_account].iloc[0]
            render_case_explanation("account", account_row.to_dict(), bundle, case_key=f"account::{selected_account}")
            account_txs = transactions.loc[transactions["accountid"].astype(str) == selected_account]
            if not account_txs.empty:
                display_dataframe(
                    account_txs.head(20)[[column for column in ["transactionid", "merchantid", "transactionamount", "composite_risk_score", "risk_level", "channel"] if column in account_txs.columns]],
                    formatters={"transactionamount": format_currency, "composite_risk_score": format_score},
                    height=280,
                )

    with tabs[1]:
        display_dataframe(
            merchants.head(25),
            formatters={"avg_risk_score": format_score, "max_risk_score": format_score, "high_risk_pct": format_percent},
            height=360,
        )
        if not merchants.empty:
            selected_merchant = st.selectbox("Select merchant", options=merchants["merchantid"].astype(str).tolist(), key="merchant_select")
            merchant_row = merchants.loc[merchants["merchantid"].astype(str) == selected_merchant].iloc[0]
            render_case_explanation("merchant", merchant_row.to_dict(), bundle, case_key=f"merchant::{selected_merchant}")

    with tabs[2]:
        display_dataframe(
            devices.head(25),
            formatters={"avg_risk_score": format_score, "max_risk_score": format_score, "high_risk_pct": format_percent},
            height=360,
        )
        if not devices.empty:
            top_device = devices.iloc[0]
            render_insight(
                f"<strong>Operational note</strong><br>Device {top_device.get('deviceid', 'N/A')} currently leads the device-risk ranking and should be reviewed alongside shared-account patterns."
            )

    with tabs[3]:
        display_dataframe(
            locations.head(25),
            formatters={"avg_risk_score": format_score, "max_risk_score": format_score, "high_risk_pct": format_percent},
            height=360,
        )
        if not locations.empty:
            selected_location = st.selectbox("Select location", options=locations["location"].astype(str).tolist(), key="location_select")
            location_row = locations.loc[locations["location"].astype(str) == selected_location].iloc[0]
            render_case_explanation("location", location_row.to_dict(), bundle, case_key=f"location::{selected_location}")


def page_ai_recommendations(bundle: Dict[str, Any]) -> None:
    render_section_header(
        "AI Recommendations",
        "Dynamic recommendations, reminders, and monitoring insights generated from the active dataset.",
        kicker="AI Guidance",
    )

    recommendations = get_cached_ai_recommendations(bundle)

    reminder_columns = st.columns(max(1, len(recommendations["reminders"])))
    for column, reminder in zip(reminder_columns, recommendations["reminders"]):
        with column:
            render_insight(reminder)

    st.markdown("### Rule-Based Recommendations")
    for item in recommendations["baseline_recommendations"]:
        render_insight(item)

    st.markdown("### AI Recommendations")
    if recommendations["ai_recommendations"]:
        for item in recommendations["ai_recommendations"]:
            render_insight(item)
    else:
        st.info(recommendations["availability_message"])


def page_questions(bundle: Dict[str, Any]) -> None:
    render_section_header(
        "Ask Questions About Data",
        "Ask high-level or investigation-specific questions about the active dataset. Answers stay grounded in the ranked outputs and compact data summaries.",
        kicker="Q&A",
    )

    st.caption(f"Active data source: {current_source_label(bundle)}")
    st.caption(
        "Best results come from investigation questions such as: "
        "`What are the riskiest merchants?`, "
        "`Why was TX000275 flagged?`, "
        "`Which location has the most high-risk activity?`, "
        "`What should OOF review first?`"
    )
    question = st.text_input(
        "Ask about the current data",
        placeholder="Example: Which merchants appear most often in flagged transactions?",
    )

    if st.button("Ask", key="ask_data_question") and question.strip():
        response = answer_data_question(question.strip(), bundle)
        history = st.session_state.setdefault("qa_history", [])
        history.append(
            {
                "question": question.strip(),
                "answer": response["ai_answer"] or response["heuristic_answer"],
                "used_ai": bool(response["ai_answer"]),
            }
        )

    if not st.session_state.get("qa_history"):
        st.info("Ask a question to start an investigation-oriented Q&A thread.")
        return

    for item in reversed(st.session_state["qa_history"]):
        st.markdown(
            f"""
            <div class="detail-card" style="margin-bottom:0.9rem;">
                <div class="detail-label">Question</div>
                <div class="detail-value" style="font-size:1rem;">{item['question']}</div>
                <div class="detail-label" style="margin-top:0.9rem;">Answer</div>
                <div style="margin-top:0.35rem; color:{BRAND['text']}; line-height:1.6;">{item['answer']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def page_decisions(bundle: Dict[str, Any]) -> None:
    store = ReviewStore()
    decisions = store.get_all_decisions()
    transactions = bundle.get("transactions", pd.DataFrame())
    uploaded_review_log = bundle.get("review_log", pd.DataFrame())

    render_section_header(
        "Analyst Review Log",
        "Disposition history for case review, approvals, dismissals, and items still awaiting analyst action.",
        kicker="Governance",
    )

    reviewed_ids = set(decisions["transactionid"].astype(str)) if not decisions.empty and "transactionid" in decisions else set()
    active_case_ids = set(transactions["transactionid"].astype(str)) if not transactions.empty and "transactionid" in transactions else set()
    unreviewed_count = len(active_case_ids - reviewed_ids) if active_case_ids else 0

    counts = decisions["decision"].value_counts() if not decisions.empty else pd.Series(dtype=int)
    metrics = [
        ("Approved", f"{int(counts.get('Approve Flag', 0)):,}", "Transactions approved for escalation or retention."),
        ("Dismissed", f"{int(counts.get('Dismiss', 0)):,}", "Transactions reviewed and dismissed."),
        ("Needs Review", f"{int(counts.get('Needs Review', 0)):,}", "Transactions explicitly left in review status."),
        ("Unreviewed", f"{unreviewed_count:,}", "Transactions in the active dataset with no analyst decision yet."),
    ]
    for column, (label, value, note) in zip(st.columns(4), metrics):
        with column:
            render_metric_card(label, value, note)

    if not decisions.empty:
        log_view = decisions.copy()
        display_dataframe(
            log_view[[column for column in ["case_id", "transactionid", "accountid", "decision", "analyst_notes", "created_at", "updated_at", "review_version"] if column in log_view.columns]],
            height=380,
        )

        csv_buffer = io.StringIO()
        decisions.to_csv(csv_buffer, index=False)
        st.download_button(
            "Download Review Log",
            data=csv_buffer.getvalue(),
            file_name=f"review_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    else:
        st.info("No stored analyst decisions yet. Saving a decision from the Suspicious Transactions page will create the log automatically.")

    if isinstance(uploaded_review_log, pd.DataFrame) and not uploaded_review_log.empty:
        render_section_header(
            "Uploaded Review Log Preview",
            "This is the currently loaded uploaded review log. It does not overwrite the persistent analyst decision store unless you explicitly use it operationally.",
            kicker="Uploaded Context",
        )
        display_dataframe(uploaded_review_log, height=320)


def page_controls(bundle: Dict[str, Any]) -> None:
    summary = bundle.get("summary", {}) or {}
    transactions = bundle.get("transactions", pd.DataFrame())
    recommendations = rule_based_recommendations(bundle)

    render_section_header(
        "Controls and Recommendations for OOF",
        "Operational control priorities, monitoring focus, and dashboard reference information for the Office of Oversight and Finance.",
        kicker="Controls",
    )

    for recommendation in recommendations:
        render_insight(recommendation)

    if not transactions.empty and "channel" in transactions.columns:
        channel_risk = (
            transactions.groupby("channel", dropna=False)
            .agg(avg_risk_score=("composite_risk_score", "mean"), transaction_count=("transactionid", "count"))
            .reset_index()
            .sort_values("avg_risk_score", ascending=False)
        )
        fig = px.bar(
            channel_risk,
            x="channel",
            y="avg_risk_score",
            color="avg_risk_score",
            color_continuous_scale=[[0, BRAND["green_soft"]], [0.55, BRAND["green"]], [1, BRAND["danger"]]],
            title="Control Priorities by Channel",
        )
        st.plotly_chart(apply_chart_theme(fig, 360), use_container_width=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        weights_df = pd.DataFrame(
            [{"Component": component, "Weight": weight} for component, weight in config.RISK_WEIGHTS.items()]
        )
        display_dataframe(weights_df, formatters={"Weight": lambda value: f"{value:.2f}"}, height=300)
    with col2:
        thresholds_df = pd.DataFrame(
            [
                {"Threshold": "Low Risk Upper Bound", "Value": config.RISK_LEVEL_LOW},
                {"Threshold": "Medium Risk Upper Bound", "Value": config.RISK_LEVEL_MEDIUM},
                {"Threshold": "High Risk Upper Bound", "Value": config.RISK_LEVEL_HIGH},
                {"Threshold": "Current High-Risk Share", "Value": summary.get("high_risk_pct", 0) / 100},
            ]
        )
        display_dataframe(thresholds_df, formatters={"Value": lambda value: f"{value:.2f}"}, height=300)

    st.markdown(
        f"""
        <div class="insight-card">
            <strong>Runbook</strong><br>
            1. Triage top-ranked transactions and accounts first.<br>
            2. Focus controls on the highest-risk channel, merchant, and location clusters.<br>
            3. Persist analyst decisions in the review log for repeatable governance.<br>
            4. Use the AI Q&amp;A section for concise executive summaries tied to the active dataset.
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_section_header(
        "ChatOps Delivery",
        "Publish the active fraud context for Discord/OpenClaw, send the latest report digest, or push threshold-based alerts.",
        kicker="ChatOps",
    )
    report_col, alert_col = st.columns(2, gap="large")
    with report_col:
        if st.button("Send Latest Report to Chat"):
            result = publish_bundle_for_chatops(
                bundle,
                publish_reason="streamlit_manual_send",
                headline="Manual fraud report dispatch from the executive dashboard.",
            )
            st.session_state["last_chatops_delivery"] = result
            if result.get("delivery") and result["delivery"].delivered:
                st.success("Latest fraud report sent to ChatOps.")
            else:
                st.info("Report context published. Review the ChatOps status note below for delivery details.")
    with alert_col:
        if st.button("Send Active Alerts to Chat"):
            publish_bundle_context(bundle, publish_reason="streamlit_manual_alerts")
            alert_result = send_alert_notifications(bundle)
            st.session_state["last_chatops_alerts"] = alert_result
            sent_count = sum(
                1
                for item in alert_result["deliveries"]
                if not item.get("skipped") and item.get("delivery") and item["delivery"].delivered
            )
            skipped_count = sum(1 for item in alert_result["deliveries"] if item.get("skipped"))
            st.success(f"Alert dispatch completed. Sent {sent_count} alert(s); skipped {skipped_count} deduped alert(s).")

    render_chatops_status(st.session_state.get("last_chatops_delivery"))
    if st.session_state.get("last_chatops_alerts"):
        deliveries = st.session_state["last_chatops_alerts"]["deliveries"]
        if deliveries:
            lines = []
            for item in deliveries:
                alert = item["alert"]
                if item.get("skipped"):
                    lines.append(f"{alert['title']}: skipped ({item.get('reason')})")
                    continue
                delivery = item["delivery"]
                lines.append(
                    f"{alert['title']}: {'delivered' if delivery.delivered else f'not delivered ({delivery.delivery_error})'}"
                )
            st.caption(" | ".join(lines))


def main() -> None:
    bundle = get_active_bundle()
    page = st.session_state.get("selected_page", "Executive Summary")
    if page not in NAV_ITEMS:
        page = "Executive Summary"
    page = render_sidebar(page, bundle)
    st.session_state["selected_page"] = page

    render_app_header(
        "Fraud Intelligence Dashboard",
        "A polished executive monitoring layer for transaction surveillance, upload-driven analysis, AI guidance, and analyst review decisions.",
        logo_path=LOGO_PATH,
    )

    render_insight(
        f"<strong>Active dataset</strong><br>{current_source_label(bundle)}"
    )

    if page == "Executive Summary":
        page_overview(bundle)
    elif page == "Upload Data":
        page_upload_data()
    elif page == "Suspicious Transactions":
        page_transactions(bundle)
    elif page == "Risky Entities":
        page_entities(bundle)
    elif page == "AI Recommendations":
        page_ai_recommendations(bundle)
    elif page == "Ask Questions About Data":
        page_questions(bundle)
    elif page == "Analyst Review Log":
        page_decisions(bundle)
    elif page == "OOF Controls":
        page_controls(bundle)


if __name__ == "__main__":
    main()
