import time
from datetime import timedelta
import asyncio
import json

import pandas as pd
import streamlit as st
import os
from io import BytesIO

# ──────────────────────────────────────────────────────────────────────────────
# Semantic Kernel and Azure OpenAI imports
# ──────────────────────────────────────────────────────────────────────────────
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions.kernel_function import KernelFunction
from semantic_kernel.contents import ChatHistory
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
from openai import AzureOpenAI

# ──────────────────────────────────────────────────────────────────────────────
# Azure OpenAI Configuration
# ──────────────────────────────────────────────────────────────────────────────
aoai = st.secrets["AZURE_OPENAI"]

endpoint = os.getenv("ENDPOINT_URL", "https://agent-ai-servicesqqx5.cognitiveservices.azure.com/")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", aoai["API_KEY"])

# Initialize Azure OpenAI client for direct calls
client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=endpoint,
    api_key=subscription_key
)

# Initialize Semantic Kernel
kernel = sk.Kernel()
kernel.add_service(
    AzureChatCompletion(
        deployment_name=deployment,
        endpoint=endpoint,
        api_key=subscription_key,
        api_version="2024-12-01-preview",
        service_id="AzureOpenAI",
    )
)

# ──────────────────────────────────────────────────────────────────────────────
# App Insights telemetry
# ──────────────────────────────────────────────────────────────────────────────
from applicationinsights import TelemetryClient
tc = TelemetryClient(st.secrets["APP_INSIGHTS"]["INSTRUMENTATION_KEY"])

def track_to_app_insights(rec: dict):
    tc.track_event(
        name="AgentInteraction",
        properties={
            "user":   rec["user"],
            "tools":  ",".join(rec["tools"]),
        },
        measurements={
            "latency": rec["latency"],
            "tokens":  rec["tokens"],
        }
    )
    tc.flush()

# ──────────────────────────────────────────────────────────────────────────────
# Azure Log Analytics client
# ──────────────────────────────────────────────────────────────────────────────
from azure.identity import DefaultAzureCredential
from azure.monitor.query import LogsQueryClient

credential    = DefaultAzureCredential()
la_client     = LogsQueryClient(credential)
WORKSPACE_ID  = st.secrets["AZURE"]["WORKSPACE_ID"]

# ──────────────────────────────────────────────────────────────────────────────
# Tool definitions and agent setup
# ──────────────────────────────────────────────────────────────────────────────
def call_tool(tool_name: str, prompt: str) -> str:
    time.sleep(0.4)
    return f"[{tool_name} result for '{prompt}']\n"

# Create a Tools plugin and add it to the kernel
class ToolsPlugin:
    def __init__(self):
        pass
        
    async def web_search(self, query: str) -> str:
        """
        Search the web for information.
        
        Args:
            query: The search query
            
        Returns:
            Search results
        """
        return call_tool("web_search", query)
    
    async def calculator(self, expression: str) -> str:
        """
        Perform calculations.
        
        Args:
            expression: The calculation to perform
            
        Returns:
            Calculation result
        """
        return call_tool("calculator", expression)

# Register the plugin with the kernel
tools_plugin = ToolsPlugin()
kernel.add_plugin(tools_plugin, plugin_name="Tools")

# Create chat histories for our agents
search_agent_history = ChatHistory()
search_agent_history.add_system_message("You are a specialized agent that searches the web for information. Use the web_search function when the user asks for information that might be found online.")

calculator_agent_history = ChatHistory()
calculator_agent_history.add_system_message("You are a specialized agent that performs calculations. Use the calculator function when the user asks for any mathematical operations.")

coordinator_agent_history = ChatHistory()
coordinator_agent_history.add_system_message("""You are a coordinator agent that determines which specialized agent to use based on the user's question.
Your job is to:
1. Analyze the user's request
2. Decide which specialized agent should handle the request: search agent, calculator agent, or both
3. Synthesize responses from multiple agents if needed
4. Provide a clear, helpful response to the user""")

# ──────────────────────────────────────────────────────────────────────────────
# Agent invocation
# ──────────────────────────────────────────────────────────────────────────────
async def query_agent_with_sk(user_message: str, selected_agent="auto") -> dict:
    start = time.time()
    used_tools = []
    
    # Determine which tools might be needed based on message content
    if "search" in user_message.lower():
        used_tools.append("web_search")
    if "calc" in user_message.lower():
        used_tools.append("calculator")
    
    # We no longer reset the history to maintain conversation context between queries
    # The agent will now have access to previous messages
    
    # Force specific agent if selected
    if selected_agent == "Search Agent":
        used_tools = ["web_search"]
    elif selected_agent == "Calculator Agent":
        used_tools = ["calculator"]
    
    chat_service = kernel.get_service(service_id="AzureOpenAI")
    execution_settings = OpenAIChatPromptExecutionSettings()
    
    # Use multi-agent approach if auto selected and multiple tools needed
    if selected_agent == "Auto (Coordinator)" and len(used_tools) > 1:
        # First, ask coordinator to plan
        coordinator_agent_history.add_user_message(f"""
        User question: {user_message}
        
        I need to coordinate between different specialized agents to answer this question.
        Which agents should handle this request and how should I break down the task?
        """)
        
        
        coordinator_response = await chat_service.get_chat_message_content(
            coordinator_agent_history,
            settings=execution_settings,
            max_tokens=500,
            temperature=0.5,
        )
        print(coordinator_response)
        coordinator_plan = coordinator_response.content
        coordinator_agent_history.add_assistant_message(coordinator_plan)
        
        # Execute search if needed
        search_result = ""
        if "web_search" in used_tools:
            search_agent_history.add_user_message(user_message)
            search_response =  await chat_service.get_chat_message_content(
                search_agent_history,
                settings=execution_settings,
                max_tokens=500,
                temperature=0.5,
            )
            search_result = search_response.content
            search_agent_history.add_assistant_message(search_result)
        
        # Execute calculation if needed
        calc_result = ""
        if "calculator" in used_tools:
            calculator_agent_history.add_user_message(user_message)
            calc_response =  await chat_service.get_chat_message_content(
                calculator_agent_history,
                settings=execution_settings,
                max_tokens=500,
                temperature=0.5,
            )
            calc_result = calc_response.content
            calculator_agent_history.add_assistant_message(calc_result)
        
        # Have coordinator synthesize the final answer
        synthesis_prompt = f"""
        User question: {user_message}
        
        Here are the results from specialized agents:
        
        Search agent: {search_result if search_result else "Not used"}
        
        Calculator agent: {calc_result if calc_result else "Not used"}
        
        Please synthesize a final, coherent answer for the user.
        """
        
        coordinator_agent_history.add_user_message(synthesis_prompt)
        final_synthesis = await kernel.chat(coordinator_agent_history)
        answer = final_synthesis.value
        
    else:
        # Use a single agent if only one tool is needed or specific agent selected
        if "web_search" in used_tools or selected_agent == "Search Agent":
            search_agent_history.add_user_message(user_message)
            agent_response =  await chat_service.get_chat_message_content(
                search_agent_history,
                settings=execution_settings,
                max_tokens=500,
                temperature=0.5,
            )
            answer = agent_response.content
            
            used_tools = ["web_search"]
        elif "calculator" in used_tools or selected_agent == "Calculator Agent":
            calculator_agent_history.add_user_message(user_message)
            agent_response = await chat_service.get_chat_message_content(
                calculator_agent_history,
                settings=execution_settings,
                max_tokens=500,
                temperature=0.5,
            )
            answer = agent_response.content
            used_tools = ["calculator"]
        else:
            # Use the coordinator as a general assistant if no specific tools needed
            coordinator_agent_history.add_user_message(user_message)
            agent_response =  await chat_service.get_chat_message_content(
                coordinator_agent_history,
                settings=execution_settings,
                max_tokens=500,
                temperature=0.5,
            )
            answer = agent_response.content
            used_tools = []
    
    # Calculate metrics
    latency = time.time() - start
    tokens = 0  # Semantic Kernel doesn't expose token usage directly
    
    # Create record for tracking
    rec = {
        "user":    user_message,
        "answer":  answer,
        "latency": latency,
        "tokens":  tokens,
        "tools":   used_tools,
        "timestamp": pd.Timestamp.utcnow()
    }
    
    # Track to App Insights
    track_to_app_insights(rec)
    
    return rec

# For backward compatibility, create a synchronous wrapper
def query_agent(user_message: str, selected_agent="auto") -> dict:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(query_agent_with_sk(user_message, selected_agent))
    loop.close()
    return result

# ──────────────────────────────────────────────────────────────────────────────
# Fetch history from App Insights
# ──────────────────────────────────────────────────────────────────────────────
from azure.core.exceptions import HttpResponseError

@st.cache_data(ttl=1)
def fetch_history(hours: int = 1) -> pd.DataFrame:
    timespan = timedelta(hours=hours)

    # 1) Try the "classic" App Insights table
    kusto_v1 = """
    AppEvents
    | where Name == "AgentInteraction"
    | extend 
        latency = todouble(todynamic(Measurements).latency),
        tokens  = toint(todynamic(Measurements).tokens),
        tools   = split(tostring(todynamic(Properties).tools), ","),
        user    = tostring(todynamic(Properties).user)
    | project Name, TimeGenerated, user, latency, tokens, tools
    | order by TimeGenerated desc
    """

    
    try:
        resp = la_client.query_workspace(WORKSPACE_ID, query=kusto_v1, timespan=timespan)
        table = resp.tables[0]
        df = pd.DataFrame(table.rows, columns=[c for c in table.columns])
        # normalize timestamp column
        if "timegenerated" in df.columns:
            df = df.rename(columns={"timegenerated": "timestamp"})
        return df
    except HttpResponseError as e:
        # if it was a missing-table error, try the next query
        print(f"Error querying logs: {e}")
        return pd.DataFrame(columns=["timestamp","user","latency","tokens","tools"])
        

    # if we got here, neither table existed
    

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("🕵️ Multi-Agent Observability (Semantic Kernel + App Insights)")

# Initialize session state for conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
    
# Sync the session conversation history with the agent chat histories
def sync_conversation_history_with_agents():
    """
    Synchronize the conversation history from session state with the agent chat histories.
    This ensures that agents have context from previous interactions.
    """
    # First, reset all agent histories to just their system messages
    search_agent_history.messages = search_agent_history.messages[:1]
    calculator_agent_history.messages = calculator_agent_history.messages[:1]
    coordinator_agent_history.messages = coordinator_agent_history.messages[:1]
    
    # Now replay the conversation history into the agent histories
    for msg in st.session_state.conversation_history:
        if msg["role"] == "user":
            # Add user messages to all agent histories
            search_agent_history.add_user_message(msg["content"])
            calculator_agent_history.add_user_message(msg["content"])
            coordinator_agent_history.add_user_message(msg["content"])
        elif msg["role"] == "assistant":
            # Add assistant messages to the appropriate agent history
            agent_type = msg.get("agent_type", "Coordinator")
            if agent_type == "Search":
                search_agent_history.add_assistant_message(msg["content"])
            elif agent_type == "Calculator":
                calculator_agent_history.add_assistant_message(msg["content"])
            else:  # Coordinator
                coordinator_agent_history.add_assistant_message(msg["content"])

# When the app starts, sync conversation history with agent chat histories
if 'app_initialized' not in st.session_state:
    sync_conversation_history_with_agents()
    st.session_state.app_initialized = True

col_q, col_dash = st.columns([1,2])

with col_q:
    st.header("Ask the Agents")
    
    # Add a button to clear conversation history
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear Conversation"):
            st.session_state.conversation_history = []
            # Reset agent histories to just their system messages
            search_agent_history.messages = search_agent_history.messages[:1]
            calculator_agent_history.messages = calculator_agent_history.messages[:1]
            coordinator_agent_history.messages = coordinator_agent_history.messages[:1]
            st.rerun()
    
    with col2:
        if st.session_state.conversation_history:
            # Convert conversation history to JSON
            conversation_json = json.dumps(
                st.session_state.conversation_history, 
                default=lambda x: x.isoformat() if hasattr(x, 'isoformat') else str(x),
                indent=2
            )
            
            # Create a download button
            st.download_button(
                label="Export Conversation",
                data=conversation_json,
                file_name="conversation_history.json",
                mime="application/json"
            )
    
    # Display conversation history
    st.subheader("Conversation History")
    
    # Add filter options
    filter_options = ["All", "User", "Assistant", "System"]
    selected_filter = st.radio("Filter by message type:", filter_options, horizontal=True)
    
    conversation_container = st.container()
    with conversation_container:
        for message in st.session_state.conversation_history:
            # Skip messages based on filter
            if (selected_filter == "User" and message["role"] != "user") or \
               (selected_filter == "Assistant" and message["role"] != "assistant") or \
               (selected_filter == "System" and message["role"] != "system"):
                continue
                
            # Format timestamp
            timestamp = message.get("timestamp", pd.Timestamp.utcnow())
            time_str = timestamp.strftime("%H:%M:%S")
            
            if message["role"] == "user":
                st.markdown(f"""
                <div style='background-color: #e6f7ff; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                    <div style='display: flex; justify-content: space-between;'>
                        <strong>User:</strong>
                        <span style='color: #666; font-size: 0.8em;'>{time_str}</span>
                    </div>
                    <div style='margin-top: 5px;'>{message['content']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            elif message["role"] == "assistant":
                agent_type = "Coordinator"
                if "agent_type" in message:
                    agent_type = message["agent_type"]
                
                # Different background colors for different agent types
                bg_color = "#f0f0f0"  # Default gray
                if agent_type == "Search":
                    bg_color = "#e6ffe6"  # Light green
                elif agent_type == "Calculator":
                    bg_color = "#e6e6ff"  # Light blue
                
                tools_used = ""
                if "tools" in message and message["tools"]:
                    tools_used = f" <span style='font-size: 0.9em; color: #666;'>(Tools: {', '.join(message['tools'])})</span>"
                
                st.markdown(f"""
                <div style='background-color: {bg_color}; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                    <div style='display: flex; justify-content: space-between;'>
                        <strong>{agent_type} Agent{tools_used}</strong>
                        <span style='color: #666; font-size: 0.8em;'>{time_str}</span>
                    </div>
                    <div style='margin-top: 5px;'>{message['content']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            elif message["role"] == "system":
                st.markdown(f"""
                <div style='background-color: #fff3cd; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                    <div style='display: flex; justify-content: space-between;'>
                        <strong>System:</strong>
                        <span style='color: #666; font-size: 0.8em;'>{time_str}</span>
                    </div>
                    <div style='margin-top: 5px;'>{message['content']}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Input area
    user_input = st.text_area("Your question:", height=120)
    
    # Agent selection
    agent_options = ["Auto (Coordinator)", "Search Agent", "Calculator Agent"]
    selected_agent = st.radio("Select Agent:", agent_options)
    
    if st.button("Submit"):
        if user_input:  # Only proceed if there's input
            # Add user message to conversation history
            st.session_state.conversation_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": pd.Timestamp.utcnow()
            })
            
            # Sync conversation history with agent chat histories
            sync_conversation_history_with_agents()
            
            with st.spinner("Agents working..."):
                rec = query_agent(user_input, selected_agent)
                
                # Determine agent type for the response
                agent_type = "Coordinator"
                if len(rec["tools"]) == 1:
                    if rec["tools"][0] == "web_search":
                        agent_type = "Search"
                    elif rec["tools"][0] == "calculator":
                        agent_type = "Calculator"
                
                # Add assistant response to conversation history
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": rec["answer"],
                    "agent_type": agent_type,
                    "tools": rec["tools"],
                    "latency": rec["latency"],
                    "timestamp": pd.Timestamp.utcnow()
                })
                
                # Add system message about performance metrics
                st.session_state.conversation_history.append({
                    "role": "system",
                    "content": f"Response generated in {rec['latency']:.2f}s using {len(rec['tools'])} tool{'s' if len(rec['tools']) > 1 else ''}{': ' + ', '.join(rec['tools']) if rec['tools'] else ''}",
                    "timestamp": pd.Timestamp.utcnow(),
                    "metrics": {
                        "latency": rec['latency'],
                        "tools_count": len(rec['tools'])
                    }
                })
                
                # Force a rerun to update the conversation display
                st.rerun()

with col_dash:
    st.header("📊 Live Dashboard")
    
    # Add conversation statistics section if there's conversation history
    if st.session_state.conversation_history:
        st.subheader("Current Conversation Stats")
        
        # Count message types
        user_messages = sum(1 for msg in st.session_state.conversation_history if msg["role"] == "user")
        assistant_messages = sum(1 for msg in st.session_state.conversation_history if msg["role"] == "assistant")
        system_messages = sum(1 for msg in st.session_state.conversation_history if msg["role"] == "system")
        
        # Calculate metrics from the conversation
        agent_types = {}
        tools_used = []
        total_latency = 0
        latency_count = 0
        
        for msg in st.session_state.conversation_history:
            if msg["role"] == "assistant" and "agent_type" in msg:
                agent_type = msg.get("agent_type", "Coordinator")
                agent_types[agent_type] = agent_types.get(agent_type, 0) + 1
                
                if "tools" in msg and msg["tools"]:
                    tools_used.extend(msg["tools"])
            
            if msg["role"] == "system" and "metrics" in msg and "latency" in msg["metrics"]:
                total_latency += msg["metrics"]["latency"]
                latency_count += 1
        
        # Display conversation stats
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("User Messages", user_messages)
        c2.metric("Assistant Responses", assistant_messages)
        c3.metric("System Messages", system_messages)
        c4.metric("Avg Response Time", f"{total_latency/latency_count:.2f}s" if latency_count > 0 else "N/A")
        
        # Show agent type distribution
        if agent_types:
            st.subheader("Agent Type Distribution")
            agent_df = pd.DataFrame({"Count": agent_types}).reset_index().rename(columns={"index": "Agent Type"})
            st.bar_chart(agent_df.set_index("Agent Type"))
        
        # Show tools usage in current conversation
        if tools_used:
            st.subheader("Tools Used in Current Conversation")
            tools_df = pd.DataFrame(pd.Series(tools_used).value_counts()).reset_index().rename(columns={"index": "Tool", 0: "Count"})
            st.bar_chart(tools_df.set_index("Tool"))
    
    # Load historical data
    df = fetch_history(hours=2)
    if df.empty:
        st.info("No interactions in the last 2 hours.")
    else:
        # KPIs
        avg_lat   = df.latency.mean()
        p95_lat   = df.latency.quantile(0.95)
        avg_tok   = df.tokens.mean() if 'tokens' in df.columns and not df.tokens.empty else 0
        tot_q     = len(df)
        
        c1,c2,c3 = st.columns(3)
        c1.metric("Avg Latency", f"{avg_lat:.2f}s", f"p95 {p95_lat:.2f}s")
        c2.metric("Avg Tokens", f"{avg_tok:.0f}")
        c3.metric("Total Queries", tot_q)

        # trends
        st.subheader("Latency Over Time")
        latency_df = df.copy()
        if "TimeGenerated" in latency_df.columns:
            st.line_chart(latency_df.set_index("TimeGenerated")[["latency"]])
        elif "timestamp" in latency_df.columns:
            st.line_chart(latency_df.set_index("timestamp")[["latency"]])

        # Tool usage
        st.subheader("Tool Usage")
        all_tools = []
        for tools_list in df.tools:
            if isinstance(tools_list, list):
                all_tools.extend(tools_list)
            elif isinstance(tools_list, str):
                # Handle case where tools might be a comma-separated string
                all_tools.extend([t.strip() for t in tools_list.split(',')])
        
        tool_counts = pd.Series(all_tools).value_counts()
        if not tool_counts.empty:
            st.bar_chart(tool_counts)
        else:
            st.info("No tool usage data available.")

        st.subheader("Recent Interactions")
        display_cols = ["timestamp", "user", "latency", "tools"]
        display_cols = [col for col in display_cols if col in df.columns]
        
        st.dataframe(
            df[display_cols],
            height=200,
            use_container_width=True
        )
