# =============================================================================
# ChargeGPT — Multi-Tool ReAct AI Agent
# =============================================================================
# An autonomous Battery Health Guardian powered by LangChain + Google Gemini.
# Uses the trained Transformer model as a prediction tool.
#
# Tools:
#   1. predict_soh          — Predict SOH for a given C-rate and cycle count
#   2. compare_strategies   — Compare multiple charging strategies side-by-side
#   3. estimate_lifespan    — Estimate cycles/years until battery hits 80% SOH
#   4. recommend_plan       — Generate an optimal weekly charging schedule
#
# Usage:
#   conda activate nlm
#   python agent.py
# =============================================================================

import os
import sys
import json
import numpy as np
import pickle
import torch
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

# Local imports
from model import ChargeGPTModel, SEQ_LEN, NUM_FEATURES

# ─── Setup ──────────────────────────────────────────────────────────────────

load_dotenv()

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    print("❌ Error: GEMINI_API_KEY not found in .env file")
    sys.exit(1)

MODEL_PATH = "models/chargegpt_model.pth"
SCALER_PATH = "models/scaler.pkl"
DEVICE = torch.device("cpu")


# ─── Load Model & Scaler (Once) ────────────────────────────────────────────

def load_model_and_scaler():
    """Load the trained Transformer model and feature scaler."""
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    config = checkpoint.get("config", {})

    model = ChargeGPTModel(
        num_features=config.get("num_features", NUM_FEATURES),
        d_model=config.get("d_model", 64),
        n_heads=config.get("n_heads", 4),
        n_layers=config.get("n_layers", 2),
        dim_ff=config.get("dim_ff", 128),
        dropout=config.get("dropout", 0.3),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    print(f"✅ Model loaded from {MODEL_PATH}")
    print(f"✅ Scaler loaded from {SCALER_PATH}")
    return model, scaler


MODEL, SCALER = load_model_and_scaler()


# ─── Physics-Based Feature Generation ──────────────────────────────────────

BASE_DEGRADATION_RATE = 0.008
C_RATE_STRESS_EXPONENT = 1.5
AMBIENT_TEMP = 25.0
THERMAL_RESISTANCE = 3.5
ACTIVATION_ENERGY = 0.3
NOMINAL_CAPACITY = 5.0
NOMINAL_VOLTAGE = 3.7


def generate_cycle_features(c_rate, cycle_num):
    """Generate realistic features for a single cycle at a given C-rate."""
    c_rate_factor = c_rate ** C_RATE_STRESS_EXPONENT
    temp = AMBIENT_TEMP + THERMAL_RESISTANCE * (c_rate ** 1.8)
    temp_factor = np.exp(ACTIVATION_ENERGY * (temp - 25.0) / 10.0)
    k = BASE_DEGRADATION_RATE * c_rate_factor * temp_factor
    soh = max(1.0 - k * np.sqrt(cycle_num), 0.5)

    heating = THERMAL_RESISTANCE * (c_rate ** 1.8)
    aging_heat = 0.002 * cycle_num * c_rate
    temperature = AMBIENT_TEMP + heating + aging_heat

    degradation = 1.0 - soh
    voltage_mean = NOMINAL_VOLTAGE + 0.2 - (0.8 * degradation) - (0.05 * c_rate)
    voltage_std = 0.15 + (0.3 * degradation) + (0.02 * c_rate)
    current_mean = c_rate * NOMINAL_CAPACITY * soh
    discharge_capacity = NOMINAL_CAPACITY * soh

    return [voltage_mean, voltage_std, current_mean, temperature, discharge_capacity, c_rate]


def predict_soh_internal(c_rate, num_cycles):
    """Internal: create sliding window and run through model."""
    start_cycle = max(1, num_cycles - SEQ_LEN + 1)
    features = []
    for cycle in range(start_cycle, num_cycles + 1):
        features.append(generate_cycle_features(c_rate, cycle))

    while len(features) < SEQ_LEN:
        features.insert(0, features[0])

    features_array = np.array(features)
    features_scaled = SCALER.transform(features_array)

    input_tensor = torch.FloatTensor(features_scaled).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        soh_prediction = MODEL(input_tensor).item()

    return soh_prediction


# ─── Tool Definitions ──────────────────────────────────────────────────────

@tool
def predict_soh(c_rate: float, num_cycles: int) -> str:
    """Predict battery State of Health (SOH) after a given number of charge-discharge cycles at a specific C-rate (charging speed).

    Args:
        c_rate: Charging rate (e.g., 0.5 for slow, 1.0 for normal, 2.0 for fast, 3.0 for ultra-fast)
        num_cycles: Number of charge-discharge cycles (1-500)
    """
    c_rate = max(0.1, min(c_rate, 5.0))
    num_cycles = max(1, min(num_cycles, 500))

    soh = predict_soh_internal(c_rate, num_cycles)
    soh_pct = soh * 100

    if soh_pct >= 90:
        status, advice = "🟢 Excellent", "Battery is in great condition."
    elif soh_pct >= 80:
        status, advice = "🟡 Good", "Battery is healthy but showing some wear."
    elif soh_pct >= 70:
        status, advice = "🟠 Fair", "Noticeable degradation. Consider slower charging."
    elif soh_pct >= 60:
        status, advice = "🔴 Poor", "Significant degradation. Battery replacement may be needed soon."
    else:
        status, advice = "⛔ Critical", "Battery is heavily degraded. Replacement recommended."

    return (
        f"Battery SOH Prediction:\n"
        f"  C-rate: {c_rate}C\n"
        f"  Cycles: {num_cycles}\n"
        f"  Predicted SOH: {soh_pct:.1f}%\n"
        f"  Status: {status}\n"
        f"  Assessment: {advice}"
    )


@tool
def compare_strategies(c_rates: str, num_cycles: int) -> str:
    """Compare battery health across multiple charging strategies (C-rates) after the same number of cycles.

    Args:
        c_rates: Comma-separated C-rates to compare (e.g., "0.5,1.0,2.0,3.0")
        num_cycles: Number of charge-discharge cycles to simulate (1-500)
    """
    num_cycles = max(1, min(num_cycles, 500))
    try:
        rates = [float(r.strip()) for r in c_rates.split(",")]
    except ValueError:
        return "Error: c_rates must be comma-separated numbers, e.g., '0.5,1.0,2.0'"

    results = []
    for rate in rates:
        rate = max(0.1, min(rate, 5.0))
        soh = predict_soh_internal(rate, num_cycles)
        results.append((rate, soh))

    results.sort(key=lambda x: -x[1])

    output = f"Charging Strategy Comparison after {num_cycles} cycles:\n"
    output += f"{'─' * 50}\n"
    output += f"{'C-Rate':>8} | {'SOH':>8} | {'Status':>12} | {'Loss':>10}\n"
    output += f"{'─' * 50}\n"

    for rate, soh in results:
        soh_pct = soh * 100
        loss_pct = (1 - soh) * 100
        status = "Excellent" if soh_pct >= 90 else "Good" if soh_pct >= 80 else "Fair" if soh_pct >= 70 else "Poor" if soh_pct >= 60 else "Critical"
        output += f"  {rate:>5.1f}C | {soh_pct:>6.1f}% | {status:>12} | {loss_pct:>8.1f}%\n"

    output += f"{'─' * 50}\n"
    best, worst = results[0], results[-1]
    diff = (best[1] - worst[1]) * 100
    output += f"\nKey Insight: {best[0]}C preserves {diff:.1f}% more health than {worst[0]}C after {num_cycles} cycles."

    return output


@tool
def estimate_lifespan(c_rate: float, target_soh: float = 0.8) -> str:
    """Estimate how many charge-discharge cycles and approximate years until the battery reaches a target SOH threshold.

    Args:
        c_rate: Charging rate to simulate (e.g., 1.0 for normal, 2.0 for fast)
        target_soh: Target SOH threshold (default 0.8 = 80%, typical warranty limit)
    """
    c_rate = max(0.1, min(c_rate, 5.0))
    target_soh = max(0.5, min(target_soh, 0.99))

    low, high = 1, 2000
    result_cycles = high

    for _ in range(20):
        mid = (low + high) // 2
        soh = predict_soh_internal(c_rate, mid)
        if soh > target_soh:
            low = mid + 1
        else:
            result_cycles = mid
            high = mid - 1

    cycles_per_year = 300
    years = result_cycles / cycles_per_year
    soh_at_limit = predict_soh_internal(c_rate, result_cycles)

    return (
        f"Battery Lifespan Estimate:\n"
        f"  Charging at: {c_rate}C\n"
        f"  Target SOH: {target_soh * 100:.0f}%\n"
        f"  ─────────────────────────────\n"
        f"  Estimated cycles to {target_soh * 100:.0f}% SOH: ~{result_cycles} cycles\n"
        f"  Approximate lifespan: ~{years:.1f} years (at {cycles_per_year} cycles/year)\n"
        f"  SOH at that point: {soh_at_limit * 100:.1f}%\n"
        f"  ─────────────────────────────\n"
        f"  Note: 80% SOH is the typical EV warranty threshold."
    )


@tool
def recommend_plan(daily_km: float = 50, battery_kwh: float = 60) -> str:
    """Generate an optimal weekly charging recommendation based on driving habits and battery size.

    Args:
        daily_km: Average daily driving distance in km (default 50)
        battery_kwh: Battery pack capacity in kWh (default 60)
    """
    km_per_kwh = 6
    daily_kwh = daily_km / km_per_kwh
    days_per_full_charge = battery_kwh / daily_kwh

    soh_slow = predict_soh_internal(0.5, 300)
    soh_normal = predict_soh_internal(1.0, 300)
    soh_fast = predict_soh_internal(2.0, 300)

    output = f"🔋 Personalized Charging Plan\n"
    output += f"{'═' * 45}\n"
    output += f"  Daily driving: {daily_km:.0f} km\n"
    output += f"  Battery size:  {battery_kwh:.0f} kWh\n"
    output += f"  Daily energy:  {daily_kwh:.1f} kWh ({daily_kwh / battery_kwh * 100:.0f}% of battery)\n"
    output += f"  Full charge every: ~{days_per_full_charge:.1f} days\n"
    output += f"{'═' * 45}\n\n"
    output += f"📅 Recommended Weekly Schedule:\n"
    output += f"  Mon-Fri: Charge at home overnight (0.5C — slow)\n"
    output += f"           Charge to 80% max daily\n"
    output += f"  Sat:     Top to 90% if needed for weekend trip\n"
    output += f"  Sun:     Skip charging if above 40%\n\n"
    output += f"⚡ Fast Charging (2C+): Use only when necessary\n"
    output += f"  Limit to 1-2 times per week max\n\n"
    output += f"📊 Projected Battery Health After 1 Year (300 cycles):\n"
    output += f"  Slow charging (0.5C):   {soh_slow * 100:.1f}% SOH\n"
    output += f"  Normal charging (1.0C): {soh_normal * 100:.1f}% SOH\n"
    output += f"  Fast charging (2.0C):   {soh_fast * 100:.1f}% SOH\n\n"
    savings = (soh_slow - soh_fast) * 100
    output += f"💡 Following this plan saves ~{savings:.1f}% battery health per year\n"
    output += f"   compared to fast charging exclusively."

    return output


# ─── Agent Setup ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are ChargeGPT, an autonomous Battery Health Guardian for Electric Vehicles.

You have access to a Physics-Informed Transformer model trained on electrochemical
simulations of lithium-ion battery degradation. Your role is to help EV owners
understand and optimize their battery health.

When answering:
1. ALWAYS use your tools to provide data-backed answers — never guess SOH values
2. Compare at least 2 charging strategies when possible (slow vs fast)
3. Quantify the health impact with specific numbers
4. Give actionable, practical recommendations
5. Briefly explain the physics when relevant (SEI layer, lithium plating, thermal effects)
6. Be concise but thorough"""


def create_agent():
    """Create the ChargeGPT ReAct agent using langgraph."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY,
    )

    tools = [predict_soh, compare_strategies, estimate_lifespan, recommend_plan]

    agent = create_react_agent(
        llm,
        tools,
        prompt=SYSTEM_PROMPT,
    )

    return agent


# ─── Interactive CLI ────────────────────────────────────────────────────────

def run_agent_query(agent, query: str) -> str:
    """Run a single query and return the response."""
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    # Get the last AI message (skip tool-calling messages and non-AI messages)
    for msg in reversed(result["messages"]):
        if (hasattr(msg, "content")
            and msg.content
            and not isinstance(msg, HumanMessage)
            and not getattr(msg, "tool_calls", None)):
            return msg.content
    return "No response generated."


def main():
    """Run the agent in interactive mode."""
    print("=" * 60)
    print("  ⚡ ChargeGPT — Battery Health Guardian")
    print("  Type your question or 'quit' to exit")
    print("=" * 60)
    print()
    print("  Example questions:")
    print("  • What happens if I fast charge at 3C for 500 cycles?")
    print("  • Compare 0.5C vs 1C vs 2C charging after 300 cycles")
    print("  • How long will my battery last with normal 1C charging?")
    print("  • Create a charging plan for 80km daily driving")
    print()

    agent = create_agent()

    while True:
        try:
            user_input = input("\n🔋 You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("\n👋 Goodbye! Charge wisely!")
                break

            print("\n⏳ Thinking...")
            response = run_agent_query(agent, user_input)
            print(f"\n⚡ ChargeGPT:\n{response}")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try rephrasing your question.")


if __name__ == "__main__":
    main()
