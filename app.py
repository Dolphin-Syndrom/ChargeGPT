import os
import sys
import streamlit as st
import plotly.graph_objects as go
import numpy as np

# Add root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from inference_engine import InferenceEngine
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ChargeGPT",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background-color: #0d1117; color: #e6edf3; }

    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #21262d;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background-color: #161b22;
        border: 1px solid #21262d;
        border-radius: 10px;
        margin-bottom: 8px;
    }

    /* Chat input */
    [data-testid="stChatInput"] textarea {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        color: #e6edf3 !important;
        border-radius: 8px !important;
    }

    /* Metric cards */
    .metric-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 12px;
    }
    .metric-value { font-size: 30px; font-weight: 700; color: #3b82f6; line-height: 1; }
    .metric-value-warn { font-size: 30px; font-weight: 700; color: #f59e0b; line-height: 1; }
    .metric-value-bad { font-size: 30px; font-weight: 700; color: #ef4444; line-height: 1; }
    .metric-label { font-size: 11px; color: #6b7280; margin-top: 6px; text-transform: uppercase; letter-spacing: 0.8px; }

    /* Page header */
    .page-header { border-bottom: 1px solid #21262d; padding-bottom: 14px; margin-bottom: 20px; }
    .page-title { font-size: 20px; font-weight: 700; color: #e6edf3; margin: 0; }
    .page-subtitle { font-size: 13px; color: #6b7280; margin-top: 4px; }

    /* Tool badge */
    .tool-badge {
        display: inline-block;
        background: #0f2a0f;
        border: 1px solid #22c55e;
        color: #86efac;
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 11px;
        font-family: monospace;
        margin: 2px 0 6px 0;
    }

    /* Status badges */
    .status-ok { display:inline-block; background:#1a2e1a; border:1px solid #22c55e; color:#86efac; border-radius:20px; padding:2px 10px; font-size:12px; }
    .status-bad { display:inline-block; background:#2d0f0f; border:1px solid #ef4444; color:#fca5a5; border-radius:20px; padding:2px 10px; font-size:12px; }

    hr { border-color: #21262d; }

    /* Buttons */
    .stButton > button {
        background-color: #1d4ed8; color: white; border: none;
        border-radius: 8px; font-weight: 500;
    }
    .stButton > button:hover { background-color: #2563eb; border: none; }
</style>
""", unsafe_allow_html=True)

# ─── Shared Resources ─────────────────────────────────────────────────────────

@st.cache_resource
def load_engine():
    return InferenceEngine()

engine = load_engine()

def make_llm():
    """Create a fresh LLM instance each call — avoids shared-state blocking."""
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# ─── Tool Definitions ─────────────────────────────────────────────────────────

@tool
def simulate_battery_health(c_rate: float, current_soh: float = 1.0):
    """Simulates battery degradation for a specific charging speed (C-rate)."""
    try:
        if c_rate <= 0 or c_rate > 10:
            return {"error": "C-rate must be between 0.1 and 5.0"}
        return engine.predict_future_soh(c_rate, current_soh)
    except Exception as e:
        return {"error": str(e)}

@tool
def get_battery_history(days: int = 7):
    """Retrieves simulated charging history for the last N days."""
    import random
    from datetime import datetime, timedelta
    history = []
    for i in range(days):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        is_fast = random.choice([True, True, False])
        if is_fast:
            history.append({"date": date, "type": "Fast Charge (DC)", "c_rate": 3.0,
                             "max_temp_reached": random.uniform(42.0, 55.0), "duration_minutes": 45})
        else:
            history.append({"date": date, "type": "Slow Charge (AC)", "c_rate": 0.5,
                             "max_temp_reached": random.uniform(25.0, 28.0), "duration_minutes": 300})
    return history

@tool
def simulate_counterfactual(c_rate: float, current_soh: float = 0.95):
    """Runs a what-if simulation for an alternative charging rate."""
    return engine.predict_future_soh(c_rate, current_soh)

@tool
def explain_physics_concept(concept: str):
    """Explains a battery physics concept."""
    kb = {
        "arrhenius": "The Arrhenius equation states degradation rates double for every 10°C rise. High temps cause exponentially more damage.",
        "sei": "SEI (Solid Electrolyte Interphase) growth is the main cause of capacity fade. High temps accelerate this layer's growth.",
        "plating": "Lithium Plating occurs at high C-rates. Ions pile up on the anode surface, forming dendrites that kill capacity.",
        "thermal": "Thermal Stress causes electrode particles to expand and crack, exposing fresh surface area for more SEI growth.",
    }
    concept = concept.lower()
    for key, val in kb.items():
        if key in concept:
            return val
    return "Heat and high voltage are the two main killers of Li-ion batteries."

# ─── Agent Turn Helpers ───────────────────────────────────────────────────────

def run_guardian_turn(lc_messages, user_input):
    llm = make_llm()
    tools = [simulate_battery_health]
    llm_with_tools = llm.bind_tools(tools)
    lc_messages.append(HumanMessage(content=user_input))
    tools_used = []

    # Agentic loop: keep going until we get a text response
    for _ in range(5):  # max 5 iterations to prevent infinite loops
        response = llm_with_tools.invoke(lc_messages)
        lc_messages.append(response)

        if response.tool_calls:
            for tc in response.tool_calls:
                tools_used.append(tc["name"])
                result = simulate_battery_health.invoke(tc)
                lc_messages.append(ToolMessage(
                    tool_call_id=tc["id"], name=tc["name"], content=str(result)
                ))
        else:
            # Got a text response — done
            content = response.content or "(No response)"
            return lc_messages, content, tools_used

    return lc_messages, "Analysis complete. Please ask a follow-up question.", tools_used

def run_diagnostics_turn(lc_messages, user_input):
    llm = make_llm()
    tools = [get_battery_history, simulate_counterfactual, explain_physics_concept]
    llm_with_tools = llm.bind_tools(tools)
    lc_messages.append(HumanMessage(content=user_input))
    tools_used = []
    tool_map = {
        "get_battery_history": get_battery_history,
        "simulate_counterfactual": simulate_counterfactual,
        "explain_physics_concept": explain_physics_concept,
    }

    # Agentic loop: keep going until we get a text response
    for _ in range(5):
        response = llm_with_tools.invoke(lc_messages)
        lc_messages.append(response)

        if response.tool_calls:
            for tc in response.tool_calls:
                tools_used.append(tc["name"])
                fn = tool_map.get(tc["name"])
                result = fn.invoke(tc) if fn else "Unknown tool"
                lc_messages.append(ToolMessage(
                    tool_call_id=tc["id"], name=tc["name"], content=str(result)
                ))
        else:
            content = response.content or "(No response)"
            return lc_messages, content, tools_used

    return lc_messages, "Diagnosis complete. Please ask a follow-up question.", tools_used

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='padding: 12px 0 20px 0;'>
        <div style='font-size: 20px; font-weight: 700; color: #e6edf3;'>ChargeGPT</div>
        <div style='font-size: 12px; color: #6b7280; margin-top: 4px;'>Battery Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        options=["Battery Guardian", "Diagnostics", "Simulator"],
        label_visibility="collapsed"
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    api_key = os.getenv("GROQ_API_KEY", "")
    if api_key:
        st.markdown('<span class="status-ok">API Connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-bad">API Key Missing</span>', unsafe_allow_html=True)
        st.caption("Set GROQ_API_KEY in your .env file")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size: 11px; color: #6b7280;'>"
        "Model: llama-3.3-70b-versatile<br>"
        "Inference: ChargeGPT Transformer (Epoch 32)"
        "</div>",
        unsafe_allow_html=True
    )

# ─── Page: Battery Guardian ───────────────────────────────────────────────────

if page == "Battery Guardian":
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Battery Guardian</div>
        <div class="page-subtitle">AI advisor that protects your battery health through physics-informed guidance</div>
    </div>
    """, unsafe_allow_html=True)

    # Session state
    if "guardian_history" not in st.session_state:
        st.session_state.guardian_history = []  # list of {"role", "content", "tools"}
    if "guardian_lc" not in st.session_state:
        st.session_state.guardian_lc = [
            SystemMessage(content="""You are the 'Battery Guardian', an AI assistant for EV owners.
Protect battery health by advising the user based on physics.
You have access to the 'simulate_battery_health' tool.
RULES:
1. ALWAYS call the tool before giving charging advice.
2. Warn about degradation when fast charging is requested.
3. Propose safer alternatives with quantified health savings.
4. Be scientific but accessible.""")
        ]

    # Render existing chat
    for msg in st.session_state.guardian_history:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
            if msg.get("tools"):
                for t in msg["tools"]:
                    st.markdown(f'<span class="tool-badge">model call: {t}</span>', unsafe_allow_html=True)
            st.markdown(msg["content"])

    # Chat input — non-blocking, triggers on Enter
    user_input = st.chat_input("Ask about charging speed, battery health, or degradation...")

    if user_input:
        # Show user message immediately
        st.session_state.guardian_history.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)

        # Run agent
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Analyzing..."):
                try:
                    lc_msgs, reply, tools_used = run_guardian_turn(
                        st.session_state.guardian_lc, user_input
                    )
                    st.session_state.guardian_lc = lc_msgs
                    if tools_used:
                        for t in tools_used:
                            st.markdown(f'<span class="tool-badge">model call: {t}</span>', unsafe_allow_html=True)
                    st.markdown(reply)
                    st.session_state.guardian_history.append({
                        "role": "assistant", "content": reply, "tools": tools_used
                    })
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.session_state.guardian_history:
        if st.button("Clear conversation", key="guardian_clear"):
            st.session_state.guardian_history = []
            st.session_state.guardian_lc = [st.session_state.guardian_lc[0]]
            st.rerun()

# ─── Page: Diagnostics ────────────────────────────────────────────────────────

elif page == "Diagnostics":
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Diagnostics Agent</div>
        <div class="page-subtitle">Battery detective that analyzes your charging history and diagnoses health issues</div>
    </div>
    """, unsafe_allow_html=True)

    if "diag_history" not in st.session_state:
        st.session_state.diag_history = []
    if "diag_lc" not in st.session_state:
        st.session_state.diag_lc = [
            SystemMessage(content="""You are the 'ChargeGPT Diagnostics Agent', a Battery Detective.
GOAL: Diagnose battery health issues using charging history and physics.
TOOLS: get_battery_history, simulate_counterfactual, explain_physics_concept.
FLOW:
1. Call get_battery_history when user reports health loss.
2. Identify high temps (>40C) or high C-rates (>2.0).
3. Call simulate_counterfactual(c_rate=0.5) to show potential savings.
4. Call explain_physics_concept for the scientific root cause.
TONE: Professional, scientific, insightful.""")
        ]

    for msg in st.session_state.diag_history:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🔬"):
            if msg.get("tools"):
                for t in msg["tools"]:
                    st.markdown(f'<span class="tool-badge">model call: {t}</span>', unsafe_allow_html=True)
            st.markdown(msg["content"])

    user_input = st.chat_input("Describe your battery issue...")

    if user_input:
        st.session_state.diag_history.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🔬"):
            with st.spinner("Diagnosing..."):
                try:
                    lc_msgs, reply, tools_used = run_diagnostics_turn(
                        st.session_state.diag_lc, user_input
                    )
                    st.session_state.diag_lc = lc_msgs
                    if tools_used:
                        for t in tools_used:
                            st.markdown(f'<span class="tool-badge">model call: {t}</span>', unsafe_allow_html=True)
                    st.markdown(reply)
                    st.session_state.diag_history.append({
                        "role": "assistant", "content": reply, "tools": tools_used
                    })
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.session_state.diag_history:
        if st.button("Clear conversation", key="diag_clear"):
            st.session_state.diag_history = []
            st.session_state.diag_lc = [st.session_state.diag_lc[0]]
            st.rerun()

# ─── Page: Simulator ──────────────────────────────────────────────────────────

elif page == "Simulator":
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Battery Simulator</div>
        <div class="page-subtitle">Explore how charging speed and current health affect predicted State of Health</div>
    </div>
    """, unsafe_allow_html=True)

    col_controls, col_chart = st.columns([1, 2])

    with col_controls:
        st.markdown(
            "<div style='font-size: 12px; font-weight: 600; color: #9ca3af; margin-bottom: 14px; "
            "text-transform: uppercase; letter-spacing: 0.8px;'>Parameters</div>",
            unsafe_allow_html=True
        )

        current_soh = st.slider("Current State of Health (%)", 50, 100, 95, 1) / 100.0
        c_rate = st.slider("Charging Rate (C-rate)", 0.1, 5.0, 1.0, 0.1)

        st.markdown("<br>", unsafe_allow_html=True)

        try:
            result = engine.predict_future_soh(c_rate, current_soh)
            soh_pct = result["predicted_soh"] * 100
            deg_pct = result["degradation_impact"] * 100
            sim_temp = result["simulated_temp"]

            soh_cls = "metric-value" if soh_pct >= 85 else ("metric-value-warn" if soh_pct >= 70 else "metric-value-bad")
            temp_cls = "metric-value" if sim_temp < 35 else ("metric-value-warn" if sim_temp < 45 else "metric-value-bad")

            st.markdown(f"""
            <div class="metric-card">
                <div class="{soh_cls}">{soh_pct:.1f}%</div>
                <div class="metric-label">Predicted SOH</div>
            </div>
            <div class="metric-card">
                <div class="metric-value-bad">-{deg_pct:.4f}%</div>
                <div class="metric-label">Degradation Impact</div>
            </div>
            <div class="metric-card">
                <div class="{temp_cls}">{sim_temp:.1f}°C</div>
                <div class="metric-label">Simulated Cell Temp</div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Simulation error: {e}")

    with col_chart:
        c_rates = np.arange(0.1, 5.1, 0.1)
        soh_values, temp_values = [], []

        for cr in c_rates:
            try:
                r = engine.predict_future_soh(cr, current_soh)
                soh_values.append(r["predicted_soh"] * 100)
                temp_values.append(r["simulated_temp"])
            except:
                soh_values.append(None)
                temp_values.append(None)

        chart_layout = dict(
            paper_bgcolor='#161b22',
            plot_bgcolor='#0d1117',
            font=dict(color='#8b949e', size=12),
            margin=dict(l=10, r=10, t=40, b=10),
            height=255
        )
        axis_style = dict(gridcolor='#21262d', zerolinecolor='#21262d', tickfont=dict(color='#6b7280'))

        # SOH chart
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=list(c_rates), y=soh_values, mode='lines', name='SOH',
            line=dict(color='#3b82f6', width=2.5),
            fill='tozeroy', fillcolor='rgba(59,130,246,0.08)'
        ))
        fig1.add_vline(x=c_rate, line_dash="dash", line_color="#6b7280", line_width=1.5,
                       annotation_text=f"  {c_rate:.1f}C", annotation_font_color="#9ca3af", annotation_font_size=11)
        fig1.update_layout(
            title=dict(text="Predicted SOH vs Charging Rate", font=dict(size=13, color="#e6edf3")),
            xaxis=dict(title="C-rate", **axis_style),
            yaxis=dict(title="SOH (%)", **axis_style,
                       range=[max(0, min(v for v in soh_values if v) - 2), max(v for v in soh_values if v) + 1]
                       if any(soh_values) else [0, 100]),
            **chart_layout
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Temperature chart
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=list(c_rates), y=temp_values, mode='lines', name='Temperature',
            line=dict(color='#f59e0b', width=2.5),
            fill='tozeroy', fillcolor='rgba(245,158,11,0.08)'
        ))
        fig2.add_hline(y=40, line_dash="dot", line_color="#ef4444", line_width=1.5,
                       annotation_text="  Danger threshold (40°C)",
                       annotation_font_color="#fca5a5", annotation_font_size=11)
        fig2.add_vline(x=c_rate, line_dash="dash", line_color="#6b7280", line_width=1.5)
        fig2.update_layout(
            title=dict(text="Simulated Cell Temperature vs Charging Rate", font=dict(size=13, color="#e6edf3")),
            xaxis=dict(title="C-rate", **axis_style),
            yaxis=dict(title="Temperature (°C)", **axis_style),
            **chart_layout
        )
        st.plotly_chart(fig2, use_container_width=True)
