# =============================================================================
# ChargeGPT — Streamlit Dashboard (Dark Theme)
# =============================================================================
# Professional dark-themed UI for battery health prediction and analysis.
#
# Run:
#   conda activate nlm
#   streamlit run app.py
# =============================================================================

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import (
    predict_soh_internal,
    create_agent,
    run_agent_query,
)

# ─── Page Config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ChargeGPT | Battery Health Guardian",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Dark Theme CSS ─────────────────────────────────────────────────────────

# Color palette
BG_PRIMARY = "#0e1117"
BG_CARD = "#161b22"
BG_CARD_HOVER = "#1c2333"
BORDER = "#30363d"
TEXT_PRIMARY = "#e6edf3"
TEXT_SECONDARY = "#8b949e"
TEXT_MUTED = "#6e7681"
ACCENT = "#00d4aa"
ACCENT_DIM = "#00d4aa33"
RED = "#f85149"
ORANGE = "#d29922"
BLUE = "#58a6ff"
GREEN = "#00d4aa"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, sans-serif;
    }}

    /* Force dark backgrounds everywhere */
    .stApp {{
        background-color: {BG_PRIMARY};
    }}
    [data-testid="stSidebar"] {{
        background-color: {BG_CARD};
        border-right: 1px solid {BORDER};
    }}
    [data-testid="stHeader"] {{
        background-color: {BG_PRIMARY};
    }}

    /* Main header */
    .main-header {{
        font-size: 1.8rem;
        font-weight: 700;
        color: {TEXT_PRIMARY};
        margin-bottom: 0.1rem;
        letter-spacing: -0.03em;
    }}
    .sub-header {{
        font-size: 0.92rem;
        color: {TEXT_SECONDARY};
        margin-bottom: 1.8rem;
        font-weight: 400;
    }}

    /* Metric cards */
    .metric-card {{
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 1.3rem 1.5rem;
        text-align: center;
        transition: border-color 0.2s;
    }}
    .metric-card:hover {{
        border-color: {ACCENT};
    }}
    .metric-value {{
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0.2rem 0;
        letter-spacing: -0.02em;
    }}
    .metric-label {{
        font-size: 0.72rem;
        color: {TEXT_MUTED};
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }}
    .metric-sub {{
        font-size: 0.78rem;
        color: {TEXT_SECONDARY};
        margin-top: 0.2rem;
    }}

    /* Status colors */
    .status-excellent {{ color: {GREEN}; }}
    .status-good {{ color: {BLUE}; }}
    .status-fair {{ color: {ORANGE}; }}
    .status-poor {{ color: {RED}; }}
    .status-critical {{ color: #da3633; }}

    /* Section title */
    .section-title {{
        font-size: 1.1rem;
        font-weight: 600;
        color: {TEXT_PRIMARY};
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid {BORDER};
    }}

    /* Sidebar branding */
    .sidebar-brand {{
        font-size: 1.3rem;
        font-weight: 700;
        color: {TEXT_PRIMARY};
        letter-spacing: -0.02em;
    }}
    .sidebar-sub {{
        font-size: 0.78rem;
        color: {TEXT_MUTED};
        margin-top: -4px;
        margin-bottom: 1rem;
    }}
    .sidebar-info {{
        font-size: 0.75rem;
        color: {TEXT_SECONDARY};
        line-height: 1.9;
    }}
    .sidebar-info-label {{
        font-size: 0.68rem;
        color: {TEXT_MUTED};
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }}

    /* Info box override */
    .stAlert {{
        background-color: {BG_CARD};
        border: 1px solid {BORDER};
        color: {TEXT_SECONDARY};
    }}

    /* Chat styling */
    .chat-user {{
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-left: 3px solid {ACCENT};
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin: 0.5rem 0;
        color: {TEXT_PRIMARY};
    }}
    .chat-agent {{
        background: {BG_CARD_HOVER};
        border: 1px solid {BORDER};
        border-left: 3px solid {BLUE};
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin: 0.5rem 0;
        color: {TEXT_PRIMARY};
        white-space: pre-wrap;
        font-size: 0.9rem;
        line-height: 1.6;
    }}

    /* Hide branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    /* Dividers */
    hr {{
        border-color: {BORDER};
    }}

    /* Slider label fix */
    .stSlider label, .stSelectSlider label {{
        color: {TEXT_SECONDARY} !important;
    }}
</style>
""", unsafe_allow_html=True)


# ─── Chart Theme ────────────────────────────────────────────────────────────

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=BG_CARD,
    font=dict(family="Inter", color=TEXT_SECONDARY, size=12),
    margin=dict(l=50, r=20, t=50, b=50),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_SECONDARY, size=11),
    ),
    hoverlabel=dict(
        bgcolor=BG_CARD_HOVER,
        bordercolor=BORDER,
        font=dict(color=TEXT_PRIMARY, family="Inter"),
    ),
)

def apply_axis_style(fig):
    """Apply consistent dark grid styling to chart axes."""
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    return fig


# ─── Helpers ────────────────────────────────────────────────────────────────

def get_soh_status(soh_pct):
    if soh_pct >= 90:
        return "Excellent", "status-excellent", GREEN
    elif soh_pct >= 80:
        return "Good", "status-good", BLUE
    elif soh_pct >= 70:
        return "Fair", "status-fair", ORANGE
    elif soh_pct >= 60:
        return "Poor", "status-poor", RED
    else:
        return "Critical", "status-critical", "#da3633"


def create_soh_gauge(soh_value):
    soh_pct = soh_value * 100
    status, _, bar_color = get_soh_status(soh_pct)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=soh_pct,
        number={"suffix": "%", "font": {"size": 44, "color": TEXT_PRIMARY, "family": "Inter"}},
        title={"text": status, "font": {"size": 14, "color": TEXT_SECONDARY, "family": "Inter"}},
        delta={"reference": 100, "decreasing": {"color": RED}, "suffix": "%",
               "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": BORDER,
                     "tickfont": {"size": 10, "color": TEXT_MUTED}},
            "bar": {"color": bar_color, "thickness": 0.7},
            "bgcolor": BG_CARD_HOVER,
            "borderwidth": 0,
            "steps": [
                {"range": [0, 60], "color": "#1c1215"},
                {"range": [60, 80], "color": "#1c1a12"},
                {"range": [80, 100], "color": "#0d1f17"},
            ],
            "threshold": {
                "line": {"color": RED, "width": 2},
                "thickness": 0.8,
                "value": 80,
            },
        },
    ))
    fig.update_layout(CHART_LAYOUT)
    fig.update_layout(height=260, margin=dict(l=30, r=30, t=30, b=10))
    return fig


def create_degradation_chart(c_rates, max_cycles=500):
    fig = go.Figure()
    colors = {0.5: GREEN, 1.0: BLUE, 1.5: ORANGE, 2.0: RED, 3.0: "#da3633"}

    for rate in c_rates:
        cycles = list(range(1, max_cycles + 1, 5))
        sohs = [predict_soh_internal(rate, c) * 100 for c in cycles]
        fig.add_trace(go.Scatter(
            x=cycles, y=sohs, mode="lines", name=f"{rate}C",
            line=dict(color=colors.get(rate, TEXT_MUTED), width=2.5),
            hovertemplate=f"<b>{rate}C</b><br>Cycle: %{{x}}<br>SOH: %{{y:.1f}}%<extra></extra>",
        ))

    fig.add_hline(y=80, line_dash="dash", line_color=RED, line_width=1,
                  annotation_text="80% Warranty Threshold",
                  annotation_position="top left",
                  annotation_font_size=10, annotation_font_color=RED)

    fig.update_layout(CHART_LAYOUT)
    fig.update_layout(
        title=dict(text="Battery Degradation by Charging Speed",
                   font=dict(size=14, color=TEXT_PRIMARY)),
        xaxis_title="Cycle Number", yaxis_title="State of Health (%)",
        yaxis=dict(range=[40, 102]),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return apply_axis_style(fig)


def create_comparison_bar(c_rates, num_cycles):
    sohs = [predict_soh_internal(r, num_cycles) * 100 for r in c_rates]
    colors_list = [get_soh_status(s)[2] for s in sohs]

    fig = go.Figure(go.Bar(
        x=[f"{r}C" for r in c_rates], y=sohs,
        marker_color=colors_list,
        text=[f"{s:.1f}%" for s in sohs],
        textposition="outside",
        textfont=dict(size=12, family="Inter", color=TEXT_PRIMARY),
        hovertemplate="<b>%{x}</b><br>SOH: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=80, line_dash="dash", line_color=RED, line_width=1)
    fig.update_layout(CHART_LAYOUT)
    fig.update_layout(
        title=dict(text=f"SOH at {num_cycles} Cycles",
                   font=dict(size=14, color=TEXT_PRIMARY)),
        xaxis_title="Charging Rate", yaxis_title="SOH (%)",
        yaxis=dict(range=[0, 110]),
        height=360,
    )
    return apply_axis_style(fig)


def create_lifespan_bar(lifespans, target_soh):
    rates = [0.5, 1.0, 1.5, 2.0, 3.0]
    colors_list = [GREEN, BLUE, ORANGE, RED, "#da3633"]

    fig = go.Figure(go.Bar(
        x=[f"{r}C" for r in rates], y=lifespans,
        marker_color=colors_list,
        text=[f"{l:.1f} yrs" for l in lifespans],
        textposition="outside",
        textfont=dict(size=12, family="Inter", color=TEXT_PRIMARY),
    ))
    fig.update_layout(CHART_LAYOUT)
    fig.update_layout(
        title=dict(text=f"Estimated Lifespan to {target_soh}% SOH",
                   font=dict(size=14, color=TEXT_PRIMARY)),
        xaxis_title="Charging Rate", yaxis_title="Years",
        height=350,
    )
    return apply_axis_style(fig)


# ─── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<p class="sidebar-brand">ChargeGPT</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-sub">Battery Health Guardian</p>', unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "Navigation",
        ["Health Predictor", "Strategy Comparison", "Lifespan Estimator", "AI Assistant"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown('<p class="sidebar-info-label">Model Info</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <p class="sidebar-info">
        Architecture: Transformer Encoder<br>
        Parameters: 76,033<br>
        R-squared: 0.9971<br>
        MAE: 0.88%
    </p>
    """, unsafe_allow_html=True)


# ─── Page: Health Predictor ─────────────────────────────────────────────────

if page == "Health Predictor":
    st.markdown('<p class="main-header">Battery Health Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict battery State of Health based on charging behavior</p>', unsafe_allow_html=True)

    col_input, col_viz = st.columns([1, 2])

    with col_input:
        st.markdown('<p class="section-title">Parameters</p>', unsafe_allow_html=True)

        c_rate = st.select_slider(
            "Charging Rate (C-rate)",
            options=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            value=1.0,
        )
        num_cycles = st.slider("Number of Cycles", 10, 500, 300, 10)

        speed_desc = {
            0.5: "Slow — overnight, ~10 hours",
            1.0: "Normal — home charger, ~5 hours",
            1.5: "Moderate — Level 2, ~3 hours",
            2.0: "Fast — DC Fast, ~1.5 hours",
            2.5: "Very Fast — Supercharger, ~1 hour",
            3.0: "Ultra Fast — 350kW, ~40 min",
        }
        st.info(f"**{c_rate}C** | {speed_desc.get(c_rate, '')}")

    with col_viz:
        soh = predict_soh_internal(c_rate, num_cycles)
        soh_pct = soh * 100
        status, css_class, _ = get_soh_status(soh_pct)
        loss = (1 - soh) * 100

        gauge_c, metric_c = st.columns([1.3, 1])

        with gauge_c:
            st.plotly_chart(create_soh_gauge(soh), use_container_width=True)

        with metric_c:
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom: 1rem;">
                <p class="metric-label">Predicted SOH</p>
                <p class="metric-value {css_class}">{soh_pct:.1f}%</p>
                <p class="metric-sub">After {num_cycles} cycles at {c_rate}C</p>
            </div>
            <div class="metric-card">
                <p class="metric-label">Capacity Loss</p>
                <p class="metric-value" style="color: {RED};">{loss:.1f}%</p>
                <p class="metric-sub">Degradation from new</p>
            </div>
            """, unsafe_allow_html=True)


# ─── Page: Strategy Comparison ──────────────────────────────────────────────

elif page == "Strategy Comparison":
    st.markdown('<p class="main-header">Charging Strategy Comparison</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Compare degradation patterns across different charging speeds</p>', unsafe_allow_html=True)

    ctrl1, ctrl2 = st.columns([3, 1])
    with ctrl1:
        selected_rates = st.multiselect(
            "Select C-rates to compare",
            [0.5, 1.0, 1.5, 2.0, 3.0],
            default=[0.5, 1.0, 2.0, 3.0],
        )
    with ctrl2:
        compare_cycles = st.slider("Snapshot at cycle", 50, 500, 300, 50)

    if not selected_rates:
        st.warning("Select at least one C-rate.")
    else:
        st.plotly_chart(create_degradation_chart(selected_rates), use_container_width=True)
        st.plotly_chart(create_comparison_bar(selected_rates, compare_cycles), use_container_width=True)

        # Metric cards row
        st.markdown('<p class="section-title">Detailed Breakdown</p>', unsafe_allow_html=True)
        cols = st.columns(len(selected_rates))
        for i, rate in enumerate(sorted(selected_rates)):
            soh = predict_soh_internal(rate, compare_cycles)
            soh_pct = soh * 100
            status, css_class, _ = get_soh_status(soh_pct)
            with cols[i]:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">{rate}C</p>
                    <p class="metric-value {css_class}">{soh_pct:.1f}%</p>
                    <p class="metric-sub">{status} | {(1-soh)*100:.1f}% loss</p>
                </div>
                """, unsafe_allow_html=True)


# ─── Page: Lifespan Estimator ───────────────────────────────────────────────

elif page == "Lifespan Estimator":
    st.markdown('<p class="main-header">Battery Lifespan Estimator</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Estimate battery longevity under different charging conditions</p>', unsafe_allow_html=True)

    col_in, col_out = st.columns([1, 2])

    with col_in:
        st.markdown('<p class="section-title">Your Usage</p>', unsafe_allow_html=True)
        est_c_rate = st.select_slider("Primary C-rate", [0.5, 1.0, 1.5, 2.0, 3.0], 1.0, key="ls_cr")
        target_soh = st.slider("End-of-Life SOH (%)", 50, 90, 80, 5)
        cycles_per_year = st.slider("Cycles per Year", 100, 600, 300, 50)

    with col_out:
        # Binary search for lifespan
        target_frac = target_soh / 100.0

        def find_lifespan_cycles(rate, target):
            lo, hi = 1, 3000
            res = hi
            for _ in range(20):
                mid = (lo + hi) // 2
                if predict_soh_internal(rate, mid) > target:
                    lo = mid + 1
                else:
                    res = mid
                    hi = mid - 1
            return res

        result_cycles = find_lifespan_cycles(est_c_rate, target_frac)
        years = result_cycles / cycles_per_year
        soh_at_end = predict_soh_internal(est_c_rate, result_cycles)

        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Estimated Cycles</p>
                <p class="metric-value" style="color: {TEXT_PRIMARY};">~{result_cycles}</p>
                <p class="metric-sub">Until {target_soh}% SOH</p>
            </div>
            """, unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Estimated Lifespan</p>
                <p class="metric-value" style="color: {GREEN};">~{years:.1f} yrs</p>
                <p class="metric-sub">At {cycles_per_year} cycles/year</p>
            </div>
            """, unsafe_allow_html=True)
        with r3:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">SOH at End</p>
                <p class="metric-value" style="color: {ORANGE};">{soh_at_end*100:.1f}%</p>
                <p class="metric-sub">At ~{result_cycles} cycles</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # All C-rates comparison
        all_rates = [0.5, 1.0, 1.5, 2.0, 3.0]
        lifespans = [find_lifespan_cycles(r, target_frac) / cycles_per_year for r in all_rates]
        st.plotly_chart(create_lifespan_bar(lifespans, target_soh), use_container_width=True)


# ─── Page: AI Assistant ─────────────────────────────────────────────────────

elif page == "AI Assistant":
    st.markdown('<p class="main-header">AI Battery Assistant</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about battery health — powered by ChargeGPT Agent + Gemini</p>', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        with st.spinner("Initializing AI Agent..."):
            st.session_state.agent = create_agent()

    # Example queries
    st.markdown('<p class="section-title">Try asking</p>', unsafe_allow_html=True)
    examples = [
        "What happens if I fast charge at 3C daily for 2 years?",
        "Compare slow vs fast charging after 500 cycles",
        "How long will my battery last with 1C charging?",
        "Create a charging plan for 80km daily driving",
    ]
    ex_cols = st.columns(2)
    for i, ex in enumerate(examples):
        with ex_cols[i % 2]:
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": ex})
                with st.spinner("Analyzing..."):
                    response = run_agent_query(st.session_state.agent, ex)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

    st.divider()

    # Chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user"><b>You:</b> {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-agent"><b>ChargeGPT:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)

    user_input = st.chat_input("Ask about battery health...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Analyzing..."):
            try:
                response = run_agent_query(st.session_state.agent, user_input)
            except Exception as e:
                response = f"Error: {str(e)}. Please try rephrasing."
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

    if st.session_state.messages:
        if st.button("Clear conversation", type="secondary"):
            st.session_state.messages = []
            st.rerun()
