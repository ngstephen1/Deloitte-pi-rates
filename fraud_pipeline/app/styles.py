from pathlib import Path
from typing import Optional

import streamlit as st


BRAND = {
    "green": "#86BC25",
    "green_dark": "#5D8C14",
    "green_soft": "#EAF5D2",
    "charcoal": "#1B1F1E",
    "charcoal_soft": "#252B29",
    "ink": "#111715",
    "panel": "#FFFFFF",
    "panel_alt": "#F4F7F5",
    "line": "#D7E0DA",
    "text": "#24302B",
    "muted": "#66736D",
    "warning": "#C98512",
    "danger": "#C23B32",
}


STATUS_CLASS = {
    "High Risk": "badge-high",
    "Medium Risk": "badge-medium",
    "Low Risk": "badge-low",
    "Approved Flag": "badge-approved",
    "Dismissed": "badge-dismissed",
    "Needs Review": "badge-review",
    "Approve": "badge-approved",
    "Dismiss": "badge-dismissed",
}


def inject_global_styles() -> None:
    st.markdown(
        f"""
        <style>
            :root {{
                --brand-green: {BRAND["green"]};
                --brand-green-dark: {BRAND["green_dark"]};
                --brand-green-soft: {BRAND["green_soft"]};
                --brand-charcoal: {BRAND["charcoal"]};
                --brand-charcoal-soft: {BRAND["charcoal_soft"]};
                --brand-panel: {BRAND["panel"]};
                --brand-panel-alt: {BRAND["panel_alt"]};
                --brand-line: {BRAND["line"]};
                --brand-text: {BRAND["text"]};
                --brand-muted: {BRAND["muted"]};
                --brand-warning: {BRAND["warning"]};
                --brand-danger: {BRAND["danger"]};
                --shadow-soft: 0 18px 45px rgba(17, 23, 21, 0.08);
                --shadow-card: 0 10px 26px rgba(17, 23, 21, 0.07);
                --radius-lg: 24px;
                --radius-md: 18px;
                --radius-sm: 999px;
            }}

            .stApp {{
                background:
                    radial-gradient(circle at top right, rgba(134, 188, 37, 0.10), transparent 22%),
                    linear-gradient(180deg, #f8faf8 0%, #f1f5f2 100%);
                color: var(--brand-text);
            }}

            div[data-testid="stAppViewContainer"] .main,
            div[data-testid="stAppViewContainer"] .main p,
            div[data-testid="stAppViewContainer"] .main li,
            div[data-testid="stAppViewContainer"] .main label,
            div[data-testid="stAppViewContainer"] .main span,
            div[data-testid="stAppViewContainer"] .main div,
            div[data-testid="stAppViewContainer"] .main h1,
            div[data-testid="stAppViewContainer"] .main h2,
            div[data-testid="stAppViewContainer"] .main h3,
            div[data-testid="stAppViewContainer"] .main h4 {{
                color: var(--brand-text);
            }}

            body, p, li, label, span, div {{
                color: inherit;
            }}

            div[data-testid="stAppViewContainer"] > .main {{
                padding-top: 1.25rem;
            }}

            .block-container {{
                padding-top: 0.5rem;
                padding-bottom: 2rem;
                max-width: 1440px;
            }}

            section[data-testid="stSidebar"] {{
                background:
                    linear-gradient(180deg, rgba(27, 31, 30, 0.98) 0%, rgba(20, 24, 23, 0.98) 100%);
                border-right: 1px solid rgba(255, 255, 255, 0.08);
            }}

            section[data-testid="stSidebar"] * {{
                color: #f4f8f5;
            }}

            section[data-testid="stSidebar"] .stRadio > label,
            section[data-testid="stSidebar"] .stMultiSelect > label,
            section[data-testid="stSidebar"] .stSlider > label,
            section[data-testid="stSidebar"] .stTextInput > label,
            section[data-testid="stSidebar"] .stSelectbox > label,
            section[data-testid="stSidebar"] .stFileUploader > label {{
                color: #dfe8e2;
                font-weight: 600;
            }}

            section[data-testid="stSidebar"] .stRadio [role="radiogroup"] {{
                gap: 0.35rem;
            }}

            section[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {{
                font-size: 0.96rem;
            }}

            .sidebar-panel {{
                background: rgba(255, 255, 255, 0.06);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 18px;
                padding: 1rem 1rem 0.85rem 1rem;
                margin-bottom: 1rem;
                box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
            }}

            .pipeline-frame {{
                background: linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(244,247,245,0.98) 100%);
                border: 1px solid rgba(27, 31, 30, 0.08);
                border-radius: 26px;
                padding: 1.1rem 1.1rem 0.9rem 1.1rem;
                box-shadow: var(--shadow-card);
                margin-bottom: 1.15rem;
            }}

            .pipeline-frame-title {{
                color: var(--brand-charcoal);
                font-weight: 800;
                font-size: 1.05rem;
                margin-bottom: 0.25rem;
            }}

            .pipeline-frame-copy {{
                color: var(--brand-muted);
                font-size: 0.96rem;
                line-height: 1.55;
                margin-bottom: 0.9rem;
            }}

            .pipeline-frame-note {{
                color: var(--brand-muted);
                font-size: 0.88rem;
                margin-top: 0.65rem;
            }}

            div[data-testid="stMetric"] {{
                background: var(--brand-panel);
                border: 1px solid rgba(27, 31, 30, 0.06);
                border-radius: var(--radius-md);
                padding: 1rem 1rem 0.9rem 1rem;
                box-shadow: var(--shadow-card);
            }}

            div[data-testid="stMetric"] label {{
                color: var(--brand-muted);
                font-weight: 600;
            }}

            div[data-testid="stMetricValue"] {{
                color: var(--brand-charcoal);
            }}

            .hero-card {{
                position: relative;
                overflow: hidden;
                border-radius: 30px;
                padding: 2rem 2rem 1.6rem 2rem;
                margin-bottom: 1.4rem;
                background:
                    linear-gradient(135deg, rgba(27,31,30,0.98) 0%, rgba(37,43,41,0.96) 62%, rgba(64,84,41,0.96) 100%);
                box-shadow: var(--shadow-soft);
                border: 1px solid rgba(255,255,255,0.06);
            }}

            .hero-card::after {{
                content: "";
                position: absolute;
                width: 320px;
                height: 320px;
                right: -120px;
                top: -140px;
                border-radius: 50%;
                background: radial-gradient(circle, rgba(134,188,37,0.35) 0%, rgba(134,188,37,0.02) 70%);
            }}

            .hero-row {{
                display: flex;
                justify-content: space-between;
                gap: 1rem;
                align-items: center;
                position: relative;
                z-index: 1;
            }}

            .hero-copy {{
                max-width: 78%;
            }}

            .hero-kicker {{
                font-size: 0.82rem;
                text-transform: uppercase;
                letter-spacing: 0.18em;
                color: rgba(234, 245, 210, 0.78);
                margin-bottom: 0.65rem;
                font-weight: 700;
            }}

            .hero-title {{
                font-size: clamp(1.9rem, 3vw, 3rem);
                font-weight: 800;
                line-height: 1.08;
                color: #f6faf4 !important;
                margin: 0;
                text-shadow: 0 2px 14px rgba(0, 0, 0, 0.28);
            }}

            .hero-subtitle {{
                color: rgba(244, 248, 245, 0.94) !important;
                margin-top: 0.7rem;
                max-width: 860px;
                font-size: 1rem;
                line-height: 1.6;
                text-shadow: 0 1px 8px rgba(0, 0, 0, 0.16);
            }}

            .hero-card .hero-kicker,
            .hero-card .hero-title,
            .hero-card .hero-subtitle,
            .hero-card .hero-logo-fallback,
            .hero-card .hero-logo-fallback span {{
                color: #f6faf4 !important;
            }}

            .hero-logo,
            .hero-logo-fallback {{
                min-width: 180px;
                text-align: right;
            }}

            .hero-logo img {{
                max-height: 58px;
                object-fit: contain;
                filter: drop-shadow(0 10px 22px rgba(0, 0, 0, 0.16));
            }}

            .hero-logo-fallback {{
                display: inline-flex;
                align-items: center;
                justify-content: flex-end;
                gap: 0.55rem;
                font-size: 1.15rem;
                font-weight: 700;
                color: #ffffff;
            }}

            .hero-logo-fallback .logo-dot {{
                width: 14px;
                height: 14px;
                border-radius: 50%;
                background: var(--brand-green);
                display: inline-block;
                box-shadow: 0 0 20px rgba(134, 188, 37, 0.45);
            }}

            .kpi-card {{
                background: linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(248,250,248,0.98) 100%);
                border: 1px solid rgba(27, 31, 30, 0.07);
                border-top: 4px solid var(--brand-green);
                border-radius: var(--radius-md);
                padding: 1.1rem 1.15rem 1rem 1.15rem;
                min-height: 138px;
                box-shadow: var(--shadow-card);
            }}

            .kpi-label {{
                color: var(--brand-muted);
                font-size: 0.86rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }}

            .kpi-value {{
                color: var(--brand-charcoal);
                font-size: clamp(1.7rem, 2vw, 2.35rem);
                line-height: 1.05;
                margin: 0.7rem 0 0.45rem 0;
                font-weight: 800;
            }}

            .kpi-footnote {{
                color: var(--brand-muted);
                font-size: 0.92rem;
                line-height: 1.45;
            }}

            .section-banner {{
                display: flex;
                justify-content: space-between;
                align-items: flex-end;
                gap: 1rem;
                margin: 1.45rem 0 0.85rem 0;
            }}

            .section-kicker {{
                color: var(--brand-green-dark);
                text-transform: uppercase;
                font-size: 0.78rem;
                font-weight: 800;
                letter-spacing: 0.14em;
                margin-bottom: 0.3rem;
            }}

            .section-title {{
                margin: 0;
                color: var(--brand-charcoal);
                font-size: 1.35rem;
                line-height: 1.25;
            }}

            .section-description {{
                color: var(--brand-muted);
                margin-top: 0.32rem;
                max-width: 820px;
                line-height: 1.55;
            }}

            .panel-card {{
                background: rgba(255,255,255,0.98);
                border: 1px solid rgba(27,31,30,0.07);
                border-radius: 24px;
                padding: 1rem 1rem 0.5rem 1rem;
                box-shadow: var(--shadow-card);
                margin-bottom: 1rem;
            }}

            .detail-card {{
                background: linear-gradient(180deg, #ffffff 0%, #f8faf8 100%);
                border: 1px solid rgba(27,31,30,0.07);
                border-radius: 20px;
                padding: 1rem 1.1rem;
                box-shadow: var(--shadow-card);
            }}

            .detail-label {{
                color: var(--brand-muted);
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-size: 0.78rem;
                font-weight: 700;
            }}

            .detail-value {{
                margin-top: 0.35rem;
                color: var(--brand-charcoal);
                font-size: 1.1rem;
                font-weight: 700;
            }}

            .insight-card {{
                background: linear-gradient(135deg, rgba(134,188,37,0.12), rgba(255,255,255,0.95));
                border: 1px solid rgba(134,188,37,0.22);
                border-radius: 18px;
                padding: 0.95rem 1rem;
                color: var(--brand-text);
                margin-bottom: 0.85rem;
            }}

            .badge {{
                display: inline-flex;
                align-items: center;
                gap: 0.35rem;
                padding: 0.33rem 0.72rem;
                border-radius: var(--radius-sm);
                font-weight: 700;
                font-size: 0.8rem;
                letter-spacing: 0.02em;
                border: 1px solid transparent;
            }}

            .badge::before {{
                content: "";
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: currentColor;
                opacity: 0.9;
            }}

            .badge-high {{
                background: rgba(194,59,50,0.12);
                color: var(--brand-danger);
                border-color: rgba(194,59,50,0.18);
            }}

            .badge-medium {{
                background: rgba(201,133,18,0.12);
                color: var(--brand-warning);
                border-color: rgba(201,133,18,0.2);
            }}

            .badge-low,
            .badge-approved {{
                background: rgba(134,188,37,0.14);
                color: var(--brand-green-dark);
                border-color: rgba(134,188,37,0.22);
            }}

            .badge-dismissed {{
                background: rgba(102,115,109,0.12);
                color: var(--brand-muted);
                border-color: rgba(102,115,109,0.18);
            }}

            .badge-review {{
                background: rgba(27,31,30,0.08);
                color: var(--brand-charcoal);
                border-color: rgba(27,31,30,0.12);
            }}

            .stat-strip {{
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.8rem;
                margin-top: 0.4rem;
            }}

            .stat-pill {{
                background: rgba(255,255,255,0.08);
                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 16px;
                padding: 0.85rem 0.95rem;
                color: #ffffff;
            }}

            .stat-pill-label {{
                display: block;
                color: rgba(240,245,241,0.72);
                font-size: 0.8rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                margin-bottom: 0.35rem;
            }}

            .stat-pill-value {{
                font-size: 1.15rem;
                font-weight: 700;
            }}

            .stTabs [data-baseweb="tab-list"] {{
                gap: 0.55rem;
                padding: 0.25rem;
                background: rgba(255,255,255,0.72);
                border-radius: 999px;
                border: 1px solid rgba(27,31,30,0.08);
                box-shadow: inset 0 1px 0 rgba(255,255,255,0.7);
            }}

            .stTabs [data-baseweb="tab"] {{
                border-radius: 999px;
                padding: 0.55rem 1rem;
                color: var(--brand-muted);
                font-weight: 700;
                border: none;
            }}

            .stTabs [aria-selected="true"] {{
                background: linear-gradient(135deg, var(--brand-green) 0%, var(--brand-green-dark) 100%);
                color: #0f140f !important;
            }}

            .stButton > button,
            .stDownloadButton > button {{
                background: linear-gradient(135deg, var(--brand-green) 0%, #9bd238 100%);
                color: #111715;
                border: none;
                border-radius: 999px;
                font-weight: 800;
                padding: 0.68rem 1.15rem;
                box-shadow: 0 14px 24px rgba(134,188,37,0.22);
            }}

            .stButton > button:hover,
            .stDownloadButton > button:hover {{
                background: linear-gradient(135deg, #93cc30 0%, #abd95a 100%);
            }}

            .stButton > button p,
            .stDownloadButton > button p {{
                color: #111715 !important;
                font-weight: 800 !important;
            }}

            .stTextInput input,
            .stNumberInput input,
            .stSelectbox [data-baseweb="select"] > div,
            .stMultiSelect [data-baseweb="select"] > div,
            .stTextArea textarea {{
                border-radius: 14px !important;
                border: 1px solid rgba(27,31,30,0.12) !important;
                background: rgba(255,255,255,0.94) !important;
                color: var(--brand-charcoal) !important;
            }}

            .stTextInput label,
            .stTextArea label,
            .stSelectbox label,
            .stMultiSelect label,
            .stSlider label,
            .stFileUploader label,
            .stRadio label,
            .stMarkdown,
            .stCaption,
            .stCodeBlock {{
                color: var(--brand-text) !important;
            }}

            div[data-testid="stAppViewContainer"] .main .stRadio > label,
            div[data-testid="stAppViewContainer"] .main .stFileUploader > label,
            div[data-testid="stAppViewContainer"] .main .stTextArea > label,
            div[data-testid="stAppViewContainer"] .main .stTextInput > label,
            div[data-testid="stAppViewContainer"] .main .stSelectbox > label {{
                color: #111715 !important;
                font-weight: 700 !important;
            }}

            div[data-testid="stAppViewContainer"] .main .stRadio [data-testid="stMarkdownContainer"] p,
            div[data-testid="stAppViewContainer"] .main .stRadio div[role="radiogroup"] label p {{
                color: #111715 !important;
                font-weight: 600 !important;
            }}

            .stSlider [data-baseweb="slider"] > div > div {{
                background: var(--brand-green);
            }}

            .stAlert {{
                border-radius: 18px;
                border: 1px solid rgba(27,31,30,0.08);
            }}

            .stAlert p,
            .stAlert div {{
                color: var(--brand-text) !important;
            }}

            [data-testid="stDataFrame"] {{
                border: 1px solid rgba(27,31,30,0.08);
                border-radius: 18px;
                overflow: hidden;
                box-shadow: var(--shadow-card);
            }}

            [data-testid="stDataFrame"] thead tr th {{
                background: #eff4f0 !important;
                color: var(--brand-charcoal) !important;
                font-weight: 700 !important;
            }}

            [data-testid="stDataFrame"] tbody tr td,
            [data-testid="stDataFrame"] tbody tr td div {{
                color: var(--brand-text) !important;
            }}

            @media (max-width: 980px) {{
                .hero-row {{
                    flex-direction: column;
                    align-items: flex-start;
                }}

                .hero-copy {{
                    max-width: 100%;
                }}

                .hero-logo,
                .hero-logo-fallback {{
                    text-align: left;
                    min-width: auto;
                }}

                .stat-strip {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def find_logo_path(project_root: Path) -> Optional[Path]:
    logo_candidates = [
        project_root / "app" / "assets" / "deloitte_logo.png",
        project_root / "app" / "assets" / "deloitte_logo.svg",
        project_root / "app" / "assets" / "logo.png",
        project_root / "app" / "assets" / "logo.svg",
        project_root / "logo.png",
        project_root / "logo.svg",
    ]
    for candidate in logo_candidates:
        if candidate.exists():
            return candidate
    return None


def render_app_header(title: str, subtitle: str, logo_path: Optional[Path] = None) -> None:
    logo_markup = (
        f'<div class="hero-logo"><img src="data:image/{logo_path.suffix[1:]};base64,{_to_base64(logo_path)}" alt="Brand logo"></div>'
        if logo_path
        else (
            '<div class="hero-logo-fallback">'
            '<span>Executive Fraud Intelligence</span>'
            '<span class="logo-dot"></span>'
            "</div>"
        )
    )
    st.markdown(
        f"""
        <section class="hero-card">
            <div class="hero-row">
                <div class="hero-copy">
                    <div class="hero-kicker" style="color:#f6faf4 !important;">Office of Oversight and Finance</div>
                    <h1 class="hero-title" style="color:#ffffff !important;">{title}</h1>
                    <div class="hero-subtitle" style="color:#f6faf4 !important;">{subtitle}</div>
                </div>
                {logo_markup}
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, description: str = "", kicker: str = "Executive View") -> None:
    description_html = f'<div class="section-description">{description}</div>' if description else ""
    st.markdown(
        f"""
        <div class="section-banner">
            <div>
                <div class="section-kicker">{kicker}</div>
                <h2 class="section-title">{title}</h2>
                {description_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, footnote: str = "") -> None:
    footnote_html = f'<div class="kpi-footnote">{footnote}</div>' if footnote else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {footnote_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_detail_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="detail-card">
            <div class="detail-label">{label}</div>
            <div class="detail-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_insight(message: str) -> None:
    st.markdown(f'<div class="insight-card">{message}</div>', unsafe_allow_html=True)


def badge(label: str) -> str:
    badge_class = STATUS_CLASS.get(label, "badge-review")
    return f'<span class="badge {badge_class}">{label}</span>'


def render_status_pill(label: str) -> str:
    return badge(label)


def apply_chart_theme(fig, height: Optional[int] = None):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=18, r=18, t=56, b=18),
        font=dict(color="#111715", family="Aptos, Segoe UI, sans-serif"),
        title_font=dict(color="#111715"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.72)",
            font=dict(color="#111715"),
            title=dict(font=dict(color="#111715")),
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(103, 115, 109, 0.16)",
        zeroline=False,
        title_font=dict(color="#111715"),
        tickfont=dict(color="#111715"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(103, 115, 109, 0.12)",
        zeroline=False,
        title_font=dict(color="#111715"),
        tickfont=dict(color="#111715"),
    )
    fig.update_coloraxes(
        colorbar=dict(
            title=dict(font=dict(color="#111715")),
            tickfont=dict(color="#111715"),
        )
    )
    if height is not None:
        fig.update_layout(height=height)
    return fig


def _to_base64(path: Path) -> str:
    import base64

    return base64.b64encode(path.read_bytes()).decode("utf-8")
