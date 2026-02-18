import os
import sys
import random
from datetime import datetime, timedelta

# Add parent directory to sys.path to allow importing inference_engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference_engine import InferenceEngine

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

# ─── Configuration ──────────────────────────────────────────────────────────

engine = InferenceEngine()

# Load environment variables
from dotenv import load_dotenv
# Load .env from parent directory
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# ─── Tools ──────────────────────────────────────────────────────────────────

@tool
def get_battery_history(days: int = 7):
    """
    Retrieves the charging history for the last N days.
    Contains data about C-rates, Temperatures, and charging duration.
    
    Args:
        days (int): Number of past days to retrieve. Default is 7.
        
    Returns:
        list: A list of dictionaries representing charging sessions.
    """
    # Mocking data to demonstrate "Thermal" issues
    history = []
    
    # Simulate a "bad" usage pattern (High Heat)
    for i in range(days):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        
        # Randomly generate a session - mostly FAST charging recently
        is_fast_charge = random.choice([True, True, False]) # Bias towards fast charge
        
        if is_fast_charge:
            session = {
                "date": date,
                "type": "Fast Charge (DC)",
                "c_rate": 3.0,
                "max_temp_reached": random.uniform(42.0, 55.0), # Heat stress
                "duration_minutes": 45
            }
        else:
            session = {
                "date": date,
                "type": "Slow Charge (AC)",
                "c_rate": 0.5,
                "max_temp_reached": random.uniform(25.0, 28.0), # Cool
                "duration_minutes": 300
            }
        history.append(session)
        
    return history

@tool
def simulate_counterfactual(c_rate: float, current_soh: float = 0.95):
    """
    Runs a 'What-If' simulation to see what health WOULD have been if charged differently.
    Useful for proving to the user that fast charging caused the damage.
    
    Args:
        c_rate (float): The alternative charging rate to simulate (e.g. 0.5).
        current_soh (float): The battery's SOH at that time.
    """
    return engine.predict_future_soh(c_rate, current_soh)

@tool
def explain_physics_concept(concept: str):
    """
    Explains a battery physics concept (like 'SEI', 'Plating', 'Arrhenius').
    Useful for explaining WHY damage occurred.
    
    Args:
        concept (str): The scientific term to explain.
    """
    knowledge_base = {
        "arrhenius": "The Arrhenius equation (k = A * exp(-Ea/RT)) states that degradation rates double for every 10°C rise in temperature. Your 45°C charging caused 4x the damage of 25°C charging.",
        "sei": "Solid Electrolyte Interphase (SEI) growth is the main cause of capacity fade. High temperatures accelerate the chemical reactions that thicken this layer, clogging the anode.",
        "plating": "Lithium Plating happens when you charge too fast (high C-rate). Ions pile up on the surface instead of entering the anode, forming metallic dendrites that kill capacity and can cause shorts.",
        "thermal": "Thermal Stress causes electrode particles to expand and crack. These cracks expose fresh surface area for more SEI growth, creating a vicious cycle of degradation.",
    }
    
    concept = concept.lower()
    for key, value in knowledge_base.items():
        if key in concept:
            return value
            
    return "Heat and High Voltage are the two main killers of Li-ion batteries."

# ─── Manual Agent Loop ──────────────────────────────────────────────────────

def run_diagnostics_agent():
    print("⚡ ChargeGPT Diagnostics Agent (Refined)")
    print("=======================================")

    if "GROQ_API_KEY" not in os.environ:
        print("\n⚠️  GROQ_API_KEY not found.")
        print("   Please set it using the .env file")
        return

    try:
        # Groq Llama 3 70b
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    except Exception as e:
        print(f"❌ Failed to initialize Groq: {e}")
        return

    tools = [get_battery_history, simulate_counterfactual, explain_physics_concept]
    llm_with_tools = llm.bind_tools(tools)
    
    messages = [
        SystemMessage(content="""You are the 'ChargeGPT Diagnostics Agent', a specialized Battery Detective.
        
        GOAL: Diagnose battery health issues by analyzing history and proving root causes using physics.
        
        TOOLS:
        1. 'get_battery_history': See past charging habits (Heat, Speed).
        2. 'simulate_counterfactual': Run a simulation to show "What would have happened if they charged slower?".
        3. 'explain_physics_concept': Define scientific terms.
        
        DIAGNOSIS FLOW:
        1. User complains about health loss.
        2. Call `get_battery_history`.
        3. Identify the "Crime": High Temps (>40°C) or High C-rates (>2.0).
        4. Prove it: Call `simulate_counterfactual(c_rate=0.5)` to show how much health *could* have been saved.
        5. Explain it: Call `explain_physics_concept` to give the scientific reason (Arrhenius/SEI).
        
        TONE:
        - Professional, Scientific, Insightful.
        - Use phrases like "My analysis indicates...", "The root cause is...", "Counterfactual analysis shows..."
        """)
    ]
    
    print("\n💬 Report your battery issues... (Type 'quit' to exit)")
    print("   Example: 'I feel like I'm losing range too fast.'\n")
    
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit"]:
                break
            
            messages.append(HumanMessage(content=user_input))
            
            # First LLM call
            response = llm_with_tools.invoke(messages)
            messages.append(response)
            
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    print(f"   🔎 Analyzing... (Calling tool: {tool_call['name']})")
                    
                    if tool_call["name"] == "get_battery_history":
                        tool_result = get_battery_history.invoke(tool_call)
                    elif tool_call["name"] == "simulate_counterfactual":
                        tool_result = simulate_counterfactual.invoke(tool_call)
                    elif tool_call["name"] == "explain_physics_concept":
                        tool_result = explain_physics_concept.invoke(tool_call)
                    else:
                        tool_result = "Unknown tool"
                        
                    messages.append(ToolMessage(
                        tool_call_id=tool_call["id"],
                        name=tool_call["name"],
                        content=str(tool_result)
                    ))
                
                # Second LLM call
                final_response = llm_with_tools.invoke(messages)
                messages.append(final_response)
                print(f"\nDetective: {final_response.content}\n")
            else:
                print(f"\nDetective: {response.content}\n")
                
        except Exception as e:
            print(f"\n❌ Error: {e}")
            break

if __name__ == "__main__":
    run_diagnostics_agent()
