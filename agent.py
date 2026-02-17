import os
import sys
from inference_engine import InferenceEngine

# LangChain Imports - Core only (Robust to version changes)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

# ─── Configuration ──────────────────────────────────────────────────────────

# Initialize Inference Engine
engine = InferenceEngine()

# ─── Tool Definition ────────────────────────────────────────────────────────

@tool
def simulate_battery_health(c_rate: float, current_soh: float = 1.0):
    """
    Simulates battery degradation for a specific charging speed (C-rate).
    
    Args:
        c_rate (float): The charging rate (e.g. 0.5 for slow, 3.0 for fast).
                        Must be between 0.1 and 5.0.
        current_soh (float): Current State of Health (0.0 to 1.0). Default is 1.0.
        
    Returns:
        dict: Predicted SOH and degradation impact.
    """
    try:
        # Validate inputs
        if c_rate <= 0 or c_rate > 10:
             return {"error": "C-rate must be typically between 0.1 and 5.0"}
             
        # Run simulation
        result = engine.predict_future_soh(c_rate, current_soh)
        return result
    except Exception as e:
        return {"error": str(e)}

# ─── Manual Agent Loop ──────────────────────────────────────────────────────

def run_agent_loop():
    print("⚡ ChargeGPT Battery Guardian Agent")
    print("==================================")

    # 1. API Key Check
    if "GOOGLE_API_KEY" not in os.environ:
        print("\n⚠️  GOOGLE_API_KEY not found.")
        print("   Please set it using: $env:GOOGLE_API_KEY='your_key'")
        return

    # 2. Initialize LLM
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
    except Exception as e:
        print(f"❌ Failed to initialize Gemini: {e}")
        return

    # 3. Bind Tools
    tools = [simulate_battery_health]
    llm_with_tools = llm.bind_tools(tools)
    
    # 4. Initialize Chat History
    messages = [
        SystemMessage(content="""You are the 'Battery Guardian', an advanced AI assistant for Electric Vehicle owners.
        Your goal is to Protect the Battery Health by negotiating with the user.
        
        You have access to a Physics-Informed Battery Model via the 'simulate_battery_health' tool.
        
        RULES:
        1. ALWAYS use the tool to simulate the effect of a charging action before giving advice.
        2. When a user wants to Fast Charge (e.g. 3C), warning them about the degradation impact.
        3. Propose alternatives (e.g. "If you charge at 1.0C instead, you save 0.02% health").
        4. Be scientific but accessible. Explain WHY (e.g. "High C-rates cause heat").
        
        Current Context:
        - The user cares about range and longevity.
        """)
    ]
    
    print("\n💬 Chat with me! (Type 'quit' to exit)")
    print("   Example: 'Is it okay to fast charge at 3C today?'\n")
    
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit"]:
                break
            
            # Add user message
            messages.append(HumanMessage(content=user_input))
            
            # First LLM call
            # print("   (Thinking...)")
            response = llm_with_tools.invoke(messages)
            messages.append(response)
            
            # If tool calls are present, execute them
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    print(f"   🤖 Calling tool: {tool_call['name']}...")
                    
                    if tool_call["name"] == "simulate_battery_health":
                        # Execute tool
                        tool_result = simulate_battery_health.invoke(tool_call)
                        
                        # Add tool output to history
                        messages.append(ToolMessage(
                            tool_call_id=tool_call["id"],
                            name=tool_call["name"],
                            content=str(tool_result)
                        ))
                
                # Second LLM call (with tool outputs)
                final_response = llm_with_tools.invoke(messages)
                messages.append(final_response)
                print(f"\nGuardian: {final_response.content}\n")
            else:
                # No tool call needed (e.g. standard greeting)
                print(f"\nGuardian: {response.content}\n")
                
        except Exception as e:
            print(f"\n❌ Error: {e}")
            break

if __name__ == "__main__":
    run_agent_loop()
