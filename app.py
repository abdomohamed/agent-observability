from importlib import metadata
from re import A
import time
from datetime import timedelta
import asyncio
import json

from httpx import Auth
import pandas as pd
import streamlit as st
import os
import plotly.express as px
from io import BytesIO

# ──────────────────────────────────────────────────────────────────────────────
# Azure AI Agents and Semantic Kernel imports
# ──────────────────────────────────────────────────────────────────────────────
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions import kernel_function, KernelArguments
from semantic_kernel.contents import ChatHistory
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings

# Azure AI Agents imports
from semantic_kernel.agents import ChatCompletionAgent
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import (   
    AzureChatCompletion,
    AzureChatPromptExecutionSettings
)
from semantic_kernel.contents import FunctionCallContent, FunctionResultContent, AuthorRole
from semantic_kernel.contents.chat_message_content import ChatMessageContent

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

class BasePlugin():
    """ A base class for defining function plugins that can be used by agents.
    """

    def call_tool(self, tool_name: str, prompt: str) -> str:
        time.sleep(0.4)
        return f"[{tool_name} result for '{prompt}']\n"

    # Define tools using the Azure AI Agents SDK
class SearchPlugin(BasePlugin):
    """ A plugin for web search functionality. """
    
    @kernel_function(description="Search the web for information")
    async def web_search(self, query: str) -> str:
        """
        Search the web for information.
        
        Args:
            query: The search query
            
        Returns:
            Search results
        """
        return self.call_tool("web_search", query)


class CalculatorPlugin(BasePlugin):
    @kernel_function(description="Provide web search capabilities")
    async def web_search(self, query: str) -> str:
        """
        Search the web for information.
        
        Args:
            query: The search query
                
        Returns:
            Search results
        """
        return self.call_tool("web_search", query)

# Create agents with different tools and system prompts

settings = AzureChatPromptExecutionSettings(service_id="AzureOpenAI")

def create_search_agent():
    agent = ChatCompletionAgent(
        kernel=kernel,
        name="search_agent",
        instructions="You are a specialized agent that searches the web for information. Use the web_search function when the user asks for information that might be found online.",
        plugins=[SearchPlugin()],
        arguments=KernelArguments(settings=settings),
    )
    return agent

def create_calculator_agent():
    agent = ChatCompletionAgent(
        kernel=kernel,
        name="calculator_agent",
        instructions="You are a specialized agent that performs calculations. Use the calculator function when the user asks for any mathematical operations.",
        plugins=[CalculatorPlugin()],
        arguments=KernelArguments(settings=settings),
    )
     
    return agent

def create_coordinator_agent():
    agent = ChatCompletionAgent(
        kernel=kernel,
        name="coordinator_agent",
        instructions="""You are a coordinator agent that determines which specialized agent to use based on the user's question.
Your job is to:
1. Analyze the user's request
2. Decide which specialized agent should handle the request: search agent, calculator agent, or both
3. Synthesize responses from multiple agents if needed
4. Provide a clear, helpful response to the user""",
        plugins=[SearchPlugin(), CalculatorPlugin()],
        arguments=KernelArguments(settings=settings),
    )
    return agent

# Initialize agents
search_agent = None
calculator_agent = None
coordinator_agent = None

def initialize_agents():
    """Initialize the agents if they haven't been created yet"""
    global search_agent, calculator_agent, coordinator_agent
    
    if search_agent is None:
        search_agent = create_search_agent()
    
    if calculator_agent is None:
        calculator_agent = create_calculator_agent()
    
    if coordinator_agent is None:
        coordinator_agent = create_coordinator_agent()

# Keep conversation histories separate for each agent
if 'search_agent_messages' not in st.session_state:
    st.session_state.search_agent_messages = []
    
if 'calculator_agent_messages' not in st.session_state:
    st.session_state.calculator_agent_messages = []
    
if 'coordinator_agent_messages' not in st.session_state:
    st.session_state.coordinator_agent_messages = []

# ──────────────────────────────────────────────────────────────────────────────
# Agent invocation
# ──────────────────────────────────────────────────────────────────────────────
async def query_agent_with_sk(user_message: str, selected_agent="auto") -> dict:
    start = time.time()
    used_tools = []
    answer = ""
    tokens = 0
    metadata = {"user_message": user_message, "selected_agent": selected_agent, "timestamp": pd.Timestamp.utcnow()}
    
    # Initialize agents if needed
    initialize_agents()
    
    # Determine which tools might be needed based on message content
    if "search" in user_message.lower():
        used_tools.append("web_search")
    if "calc" in user_message.lower():
        used_tools.append("calculator")
    
    # Force specific agent if selected
    if selected_agent == "Search Agent":
        used_tools = ["web_search"]
    elif selected_agent == "Calculator Agent":
        used_tools = ["calculator"]
    
    # Use multi-agent approach if auto selected and multiple tools needed
    if selected_agent == "Auto (Coordinator)" and len(used_tools) > 1:
        # Use coordinator agent with its history
        messages = st.session_state.coordinator_agent_messages.copy()
        messages.append(ChatMessageContent(role=AuthorRole.USER, content=user_message, metadata=metadata))

        agent_response = await coordinator_agent.get_response(
            messages=messages,
            temperature=0.5,
            max_tokens=500
        )
        
        # Update the message history
        st.session_state.coordinator_agent_messages = messages.copy()
        st.session_state.coordinator_agent_messages.append(
            ChatMessageContent(role=AuthorRole.ASSISTANT, content=agent_response.message.content, metadata=metadata)
        )
        
        answer = agent_response.message.content
        
        # Extract token usage if available
        if hasattr(agent_response, 'metadata') and 'usage' in agent_response.metadata:
            tokens = agent_response.metadata["usage"].completion_tokens or 0
            
        # Extract tool usage
        for tool_call in agent_response.message.tool_calls or []:
            tool_name = tool_call.function.name
            if tool_name not in used_tools:
                used_tools.append(tool_name)
        
    else:
        # Use a single agent if only one tool is needed or specific agent selected
        if "web_search" in used_tools or selected_agent == "Search Agent":
            messages = st.session_state.search_agent_messages.copy()
            messages.append(ChatMessageContent(
                role=AuthorRole.USER,
                content=user_message,
                metadata=metadata
            ))
              
            agent_response = await search_agent.get_response(
                messages=messages,
                temperature=0.5,
                max_tokens=500
            )
            
            # Update the message history
            st.session_state.search_agent_messages = messages.copy()
            st.session_state.search_agent_messages.append(
                ChatMessageContent(role=AuthorRole.ASSISTANT, content=agent_response.message.content, metadata=metadata)
            )
            
            answer = agent_response.message.content
            used_tools = ["web_search"]
            
            # Extract token usage if available
            if hasattr(agent_response, 'metadata') and 'usage' in agent_response.metadata:
                tokens = agent_response.metadata["usage"].completion_tokens or 0
                
        elif "calculator" in used_tools or selected_agent == "Calculator Agent":
            messages = st.session_state.calculator_agent_messages.copy()
            messages.append(ChatMessageContent(role=AuthorRole.USER, content=user_message, metadata=metadata))

            agent_response = await calculator_agent.get_response(
                messages=messages,
                temperature=0.5,
                max_tokens=500
            )
            
            # Update the message history
            st.session_state.calculator_agent_messages = messages.copy()
            st.session_state.calculator_agent_messages.append(
                ChatMessageContent(role=AuthorRole.ASSISTANT, content=agent_response.message.content, metadata=metadata)
            )
            
            answer = agent_response.message.content
            used_tools = ["calculator"]
            
            # Extract token usage if available
            if hasattr(agent_response, 'metadata') and 'usage' in agent_response.metadata:
                tokens = agent_response.metadata["usage"].completion_tokens or 0
                
        else:
            # Use the coordinator as a general assistant if no specific tools needed
            messages = st.session_state.coordinator_agent_messages.copy()
            messages.append(ChatMessageContent(role=AuthorRole.USER, content=user_message, metadata=metadata))

            agent_response = await coordinator_agent.get_response(
                messages=messages,
                temperature=0.5,
                max_tokens=500
            )
            
            # Update the message history
            st.session_state.coordinator_agent_messages = messages.copy()
            st.session_state.coordinator_agent_messages.append(
                ChatMessageContent(role=AuthorRole.ASSISTANT, content=agent_response.message.content, metadata=metadata)
            )
            
            answer = agent_response.message.content
            used_tools = []
            
            # Extract token usage if available
            if hasattr(agent_response, 'metadata') and 'usage' in agent_response.metadata:
                tokens = agent_response.metadata["usage"].completion_tokens or 0
            
    
    # Calculate metrics
    latency = time.time() - start
    
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
        

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agent Observability",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for better UI
st.markdown("""
<style>
    /* Main app styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Chat message styling */
    .user-message {
        background-color: #e6f7ff;
        border-left: 5px solid #1890ff;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .assistant-message {
        background-color: #f6f8fa;
        border-left: 5px solid #52c41a;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .system-message {
        background-color: #fff3cd;
        border-left: 5px solid #faad14;
        padding: 10px 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        font-size: 0.9em;
    }
    
    /* Headers styling */
    h1 {
        color: #1890ff;
        font-weight: 700;
    }
    
    h2 {
        color: #333;
        font-weight: 600;
        margin-top: 1.5rem;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f6f8fa;
        border-radius: 5px 5px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #e6f7ff;
        border-bottom: 2px solid #1890ff;
    }
</style>
""", unsafe_allow_html=True)

st.title("🕵️ Multi-Agent Observability")
st.caption("Powered by Semantic Kernel + App Insights")

# Initialize session state for conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
    
# Sync the session conversation history with the agent chat histories
def sync_conversation_history_with_agents():
    """
    Synchronize the conversation history from session state with the agent messages.
    This ensures that agents have context from previous interactions.
    """
    # First, reset all agent messages
    st.session_state.search_agent_messages = []
    st.session_state.calculator_agent_messages = []
    st.session_state.coordinator_agent_messages = []
    
    # Now replay the conversation history into the agent messages
    for msg in st.session_state.conversation_history:
        if msg.role == AuthorRole.USER:
            # Add user messages to all agent histories
            st.session_state.search_agent_messages.append(ChatMessageContent(role=AuthorRole.USER, content=msg.content))
            st.session_state.calculator_agent_messages.append(ChatMessageContent(role=AuthorRole.USER, content=msg.content))
            st.session_state.coordinator_agent_messages.append(ChatMessageContent(role=AuthorRole.USER, content=msg.content))
        elif msg.role == AuthorRole.ASSISTANT:
            # Add assistant messages to the appropriate agent history
            agent_type = msg.metadata.get("agent_type", "Coordinator")
            if agent_type == "Search":
                st.session_state.search_agent_messages.append(ChatMessageContent(role=AuthorRole.ASSISTANT, content=msg.content))
            elif agent_type == "Calculator":
                st.session_state.calculator_agent_messages.append(ChatMessageContent(role=AuthorRole.ASSISTANT, content=msg.content))
            else:  # Coordinator
                st.session_state.coordinator_agent_messages.append(ChatMessageContent(role=AuthorRole.ASSISTANT, content=msg.content))

# When the app starts, sync conversation history with agent chat histories
if 'app_initialized' not in st.session_state:
    sync_conversation_history_with_agents()
    st.session_state.app_initialized = True

# Create tabs for Chat and Dashboard
chat_tab, dashboard_tab, diagnostics_tab = st.tabs(["💬 Chat Interface", "📊 Dashboard", "🔍 Smart Diagnostics"])

# CHAT INTERFACE TAB
with chat_tab:
    # Create two columns for chat history and input
    chat_col, settings_col = st.columns([3, 1])
    
    with chat_col:
        # Chat container with custom styling
        chat_container = st.container()
        with chat_container:
            st.markdown("### Conversation")
            
            # Display chat messages with improved styling
            for message in st.session_state.conversation_history:
                # Format timestamp
                timestamp = message.metadata.get("timestamp", pd.Timestamp.utcnow())
                time_str = timestamp.strftime("%H:%M:%S")

                if message.role == AuthorRole.USER:
                    st.markdown(f"""
                    <div class="user-message">
                        <div style='display: flex; justify-content: space-between;'>
                            <strong>👤 You</strong>
                            <span style='color: #666; font-size: 0.8em;'>{time_str}</span>
                        </div>
                        <div style='margin-top: 8px;'>{message.content}</div>
                    </div>
                    """, unsafe_allow_html=True)

                elif message.role == AuthorRole.ASSISTANT:
                    agent_type = "Coordinator"
                    icon = "🤖"
                    if "agent_type" in message.metadata:
                        agent_type = message.metadata["agent_type"]
                        if agent_type == "Search":
                            icon = "🔍"
                        elif agent_type == "Calculator":
                            icon = "🧮"
                    
                    tools_used = ""
                    if "tools" in message.metadata and message.metadata["tools"]:
                        tools_used = f" <span style='font-size: 0.9em; color: #666;'>(Tools: {', '.join(message.metadata['tools'])})</span>"

                    st.markdown(f"""
                    <div class="assistant-message">
                        <div style='display: flex; justify-content: space-between;'>
                            <strong>{icon} {agent_type} Agent{tools_used}</strong>
                            <span style='color: #666; font-size: 0.8em;'>{time_str}</span>
                        </div>
                        <div style='margin-top: 8px;'>{message.content}</div>
                    </div>
                    """, unsafe_allow_html=True)

                elif message.role == AuthorRole.SYSTEM:
                    st.markdown(f"""
                    <div class="system-message">
                        <div style='display: flex; justify-content: space-between;'>
                            <strong>⚙️ System</strong>
                            <span style='color: #666; font-size: 0.8em;'>{time_str}</span>
                        </div>
                        <div style='margin-top: 5px;'>{message.content}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Input area with improved styling
        st.markdown("### Ask a Question")
        user_input = st.text_area("Type your message here...", height=100, 
                                  placeholder="Ask me anything...")
        
        # Submit button with better styling
        submit_col, clear_col = st.columns([1, 1])
        with submit_col:
            submit_button = st.button("Send Message 📤", use_container_width=True, type="primary")
        with clear_col:
            if st.button("Clear Chat 🗑️", use_container_width=True):
                st.session_state.conversation_history = []
                # Reset agent message histories
                st.session_state.search_agent_messages = []
                st.session_state.calculator_agent_messages = []
                st.session_state.coordinator_agent_messages = []
                st.rerun()
    
    with settings_col:
        st.markdown("### Settings")
        
        # Agent selection with better UI
        st.markdown("**Select Agent Type:**")
        agent_options = ["Auto (Coordinator)", "Search Agent", "Calculator Agent"]
        selected_agent = st.radio("", agent_options, index=0)
        
        # Export conversation option
        st.markdown("**Conversation Actions:**")
        if st.session_state.conversation_history:
            # Convert conversation history to JSON
            conversation_json = json.dumps(
                st.session_state.conversation_history, 
                default=lambda x: x.isoformat() if hasattr(x, 'isoformat') else str(x),
                indent=2
            )
            
            # Create a download button
            st.download_button(
                label="Export Conversation 💾",
                data=conversation_json,
                file_name="conversation_history.json",
                mime="application/json",
                use_container_width=True
            )
        
        # Display some info about the agents
        st.markdown("---")
        st.markdown("**Available Agents:**")
        
        st.markdown("""
        - **🤖 Coordinator:** Manages routing between specialized agents
        - **🔍 Search:** Specialized in web search tasks
        - **🧮 Calculator:** Specialized in math calculations
        """)
        
        # Add some usage tips
        st.markdown("---")
        st.markdown("**Tips:**")
        st.markdown("""
        - Use clear, specific questions
        - Check the dashboard for performance metrics
        - Try different agents for specialized tasks
        """)
    
    # Process form submission
    if submit_button and user_input:  # Only proceed if there's input
        # Add user message to conversation history
        st.session_state.conversation_history.append(
            ChatMessageContent(
                role=AuthorRole.USER,
                content=user_input,
                metadata= {
                    "timestamp": pd.Timestamp.utcnow()
                }
            )
        )

        # Sync conversation history with agent chat histories
        sync_conversation_history_with_agents()
        
        with st.spinner("🧠 Agents are thinking..."):
            rec = query_agent(user_input, selected_agent)
            
            # Determine agent type for the response
            agent_type = "Coordinator"
            if len(rec["tools"]) == 1:
                if rec["tools"][0] == "web_search":
                    agent_type = "Search"
                elif rec["tools"][0] == "calculator":
                    agent_type = "Calculator"
            
            # Add assistant response to conversation history
            st.session_state.conversation_history.append(
                ChatMessageContent(
                    role=AuthorRole.ASSISTANT,
                    content=rec["answer"],
                    metadata={
                        "user": rec["user"],
                        "tokens": rec["tokens"],
                        "agent_type": agent_type,
                        "tools": rec["tools"],
                        "timestamp": pd.Timestamp.utcnow(),
                        "latency": rec["latency"],
                    }
                )
            )

            # Add system message about performance metrics
            st.session_state.conversation_history.append(
                ChatMessageContent(
                    role=AuthorRole.SYSTEM,
                    content=f"Response generated in {rec['latency']:.2f}s using {len(rec['tools'])} tool{'s' if len(rec['tools']) > 1 else ''}{': ' + ', '.join(rec['tools']) if rec['tools'] else ''}",
                    metadata={
                        "metrics": {
                            "latency": rec['latency'],
                            "tools_count": len(rec['tools'])
                        },
                        "timestamp": pd.Timestamp.utcnow()
                    }
                )
            )
        
        # Rerun to refresh the UI
        st.rerun()

# DASHBOARD TAB
with dashboard_tab:
    st.header("📊 Live Dashboard")
    
    # Add conversation statistics section if there's conversation history
    if st.session_state.conversation_history:
        st.subheader("Current Conversation Stats")
        
        # Count message types
        user_messages = sum(1 for msg in st.session_state.conversation_history if msg.role == AuthorRole.USER)
        assistant_messages = sum(1 for msg in st.session_state.conversation_history if msg.role == AuthorRole.ASSISTANT)
        system_messages = sum(1 for msg in st.session_state.conversation_history if msg.role == AuthorRole.SYSTEM)

        # Calculate metrics from the conversation
        agent_types = {}
        tools_used = []
        total_latency = 0
        latency_count = 0
        
        for msg in st.session_state.conversation_history:
            if msg.role == AuthorRole.ASSISTANT and "agent_type" in msg.metadata:
                agent_type = msg.metadata.get("agent_type", "Coordinator")
                agent_types[agent_type] = agent_types.get(agent_type, 0) + 1

                if "tools" in msg.metadata and msg.metadata["tools"]:
                    tools_used.extend(msg.metadata["tools"])

            if msg.role == AuthorRole.SYSTEM and "metrics" in msg.metadata and "latency" in msg.metadata["metrics"]:
                total_latency += msg.metadata["metrics"]["latency"]
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
    
    # Historical Data Dashboard
    st.markdown("---")
    st.subheader("📈 Historical Performance")
    
    # Time range selection
    hours_options = [1, 2, 6, 12, 24]
    selected_hours = st.select_slider("Time Range (hours)", options=hours_options, value=2)
    
    # Load historical data
    df = fetch_history(hours=selected_hours)
    if df.empty:
        st.info(f"No interactions in the last {selected_hours} hour{'s' if selected_hours > 1 else ''}.")
    else:
        # Create dashboard layout
        metrics_col1, metrics_col2 = st.columns(2)
        
        with metrics_col1:
            # KPIs
            avg_lat = df.latency.mean()
            p95_lat = df.latency.quantile(0.95)
            avg_tok = df.tokens.mean() if 'tokens' in df.columns and not df.tokens.empty else 0
            tot_q = len(df)
            
            c1, c2 = st.columns(2)
            c1.metric("Avg Latency", f"{avg_lat:.2f}s", f"p95 {p95_lat:.2f}s")
            c2.metric("Total Queries", tot_q)
            
            # Latency trends with SLI/SLO
            st.subheader("Latency Over Time (SLI)")
            
            # Add SLO configuration
            slo_container = st.container()
            with slo_container:
                slo_col1, slo_col2 = st.columns([3, 1])
                with slo_col1:
                    st.caption("Service Level Objective (SLO) Configuration")
                with slo_col2:
                    slo_target = st.number_input("Target Latency (s)", 
                                                 min_value=0.1, 
                                                 max_value=10.0, 
                                                 value=2.0, 
                                                 step=0.1,
                                                 help="Target response time in seconds")
            
            # Prepare latency data with SLI/SLO indicators
            latency_df = df.copy()
            
            # Create a visualization of latency with SLO target line
            if "TimeGenerated" in latency_df.columns or "timestamp" in latency_df.columns:
                import altair as alt
                
                # Determine which timestamp column to use
                time_col = "timestamp" if "timestamp" in latency_df.columns else "TimeGenerated"
                
                # Calculate SLI compliance (percentage of requests meeting SLO)
                sli_compliance = (latency_df["latency"] <= slo_target).mean() * 100
                
                # Create base chart for latency values
                base = alt.Chart(latency_df).encode(
                    x=alt.X(f'{time_col}:T', title='Time'),
                    y=alt.Y('latency:Q', title='Latency (seconds)')
                )
                
                # Create the line chart for latency
                line = base.mark_line(color='#1890ff').encode(
                    tooltip=[
                        alt.Tooltip(f'{time_col}:T', title='Time'),
                        alt.Tooltip('latency:Q', title='Latency (s)')
                    ]
                )
                
                # Add points to highlight violations
                violations = base.transform_filter(
                    alt.datum.latency > slo_target
                ).mark_point(color='red', size=100).encode(
                    tooltip=[
                        alt.Tooltip(f'{time_col}:T', title='Time'),
                        alt.Tooltip('latency:Q', title='Latency (s)'),
                        alt.Tooltip('user:N', title='Query')
                    ]
                )
                
                # Add SLO target line
                target_line = alt.Chart(
                    pd.DataFrame({'threshold': [slo_target]})
                ).mark_rule(color='red', strokeDash=[3, 3]).encode(
                    y='threshold:Q'
                )
                
                # Combine charts
                chart = alt.layer(line, violations, target_line).properties(
                    height=250
                ).interactive()
                
                # Display the chart
                st.altair_chart(chart, use_container_width=True)
                
                # Display SLI compliance metric
                st.metric(
                    "SLI Compliance", 
                    f"{sli_compliance:.1f}%", 
                    f"{sli_compliance - 95:.1f}%" if sli_compliance != 95 else "On target",
                    help="Percentage of requests meeting the SLO target latency"
                )
        
        with metrics_col2:
            c1, c2 = st.columns(2)
            c1.metric("Avg Tokens", f"{avg_tok:.0f}")
            c2.metric("Unique Queries", df['user'].nunique() if 'user' in df.columns else "N/A")
            
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

        # Recent interactions table
        st.subheader("Recent Interactions")
        display_cols = ["timestamp", "user", "latency", "tools", "tokens"]
        display_cols = [col for col in display_cols if col in df.columns]
        
        st.dataframe(
            df[display_cols],
            height=300,
            use_container_width=True
        )
    
    # SLI/SLO Summary Section
    st.markdown("---")
    st.subheader("📏 Service Level Indicators (SLI) Summary")

    # Create summary layout
    sli_summary_cols = st.columns(2)

    with sli_summary_cols[0]:
        st.markdown("""
        ### Performance Metrics
        
        The following Service Level Indicators (SLIs) are being tracked:
        
        - **Latency**: Response time in seconds
        - **Token Usage**: Number of tokens used per request
        - **Tool Usage**: Distribution of tool types used
        
        The main Service Level Objective (SLO) is focused on latency.
        """)
        
        if not df.empty:
            # Calculate SLO achievement over time periods
            last_hour_data = df[df['timestamp'] > pd.Timedelta(hours=1)] if 'timestamp' in df.columns else pd.DataFrame()
            
            # Create summary metrics
            st.markdown("### Time-based Compliance")
            time_cols = st.columns(3)
            
            # Last hour
            if not last_hour_data.empty:
                last_hour_compliance = (last_hour_data['latency'] <= slo_target).mean() * 100
                time_cols[0].metric(
                    "Last Hour", 
                    f"{last_hour_compliance:.1f}%",
                    help="Percentage of requests in the last hour meeting the SLO"
                )
            else:
                time_cols[0].metric("Last Hour", "N/A", help="No data available for the last hour")
                
            # All time in current view
            all_time_compliance = (df['latency'] <= slo_target).mean() * 100
            time_cols[1].metric(
                f"Last {selected_hours}h", 
                f"{all_time_compliance:.1f}%",
                help=f"Percentage of requests in the last {selected_hours} hours meeting the SLO"
            )
            
            # Display error budget
            error_budget = 5.0  # 95% target means 5% error budget
            error_budget_used = 100 - all_time_compliance
            error_budget_remaining = error_budget - error_budget_used if error_budget_used <= error_budget else 0
            
            time_cols[2].metric(
                "Error Budget Remaining", 
                f"{error_budget_remaining:.1f}%",
                help=f"Remaining error budget (target is {100-error_budget}% compliance)"
            )

    with sli_summary_cols[1]:
        st.markdown("""
        ### SLO Definitions
        
        **Service Level Objective (SLO)**: Target level of reliability for the service.
        
        **Current SLO Targets**:
        - Latency: Responses should complete within target seconds
        - Reliability: 95% of requests should meet the latency target
        
        **Error Budget**: 5% of requests can exceed the latency target while still meeting the SLO.
        """)
        
        # Add a latency distribution chart if we have data
        if not df.empty:
            st.markdown("### Latency Distribution")
            
            # Create histogram for latency distribution
            import altair as alt
            
            # Calculate latency buckets
            hist_data = pd.DataFrame({
                'latency': df['latency'],
                'meets_slo': df['latency'] <= slo_target
            })
            
            latency_hist = alt.Chart(hist_data).mark_bar().encode(
                alt.X('latency:Q', bin=alt.Bin(maxbins=20), title='Latency (seconds)'),
                alt.Y('count()', title='Number of Requests'),
                alt.Color('meets_slo:N', 
                          scale=alt.Scale(domain=[True, False], range=['#52c41a', '#f5222d']),
                          legend=alt.Legend(title="Meets SLO"))
            ).properties(height=200)
            
            # Add a rule for the SLO target
            rule = alt.Chart(pd.DataFrame({'slo': [slo_target]})).mark_rule(
                color='red', 
                strokeDash=[3, 3]
            ).encode(x='slo:Q')
            
            # Display chart
            st.altair_chart(latency_hist + rule, use_container_width=True)
    
# SMART DIAGNOSTICS TAB
with diagnostics_tab:
    st.header("🔍 Smart Diagnostics")
    
    # Time range selection
    st.subheader("Select Time Range for Analysis")
    hours_options = [1, 2, 4, 6, 12, 24, 48]
    diagnostic_hours = st.select_slider("Time Range (hours)", options=hours_options, value=6)
    
    # Run diagnostics button
    run_diagnostics = st.button("Run Smart Diagnostics", type="primary", use_container_width=True)
    
    if run_diagnostics:
        with st.spinner("Fetching transaction data and analyzing patterns..."):
            # Fetch transaction data based on selected time range
            df_diagnostics = fetch_history(hours=diagnostic_hours)
            
            if df_diagnostics.empty:
                st.warning(f"No transaction data available for the last {diagnostic_hours} hours.")
            else:
                st.success(f"Found {len(df_diagnostics)} transactions in the last {diagnostic_hours} hours")
                
                # Display the raw data
                expander = st.expander("View Raw Transaction Data")
                with expander:
                    st.dataframe(df_diagnostics)
                
                # Data visualization section
                st.subheader("Transaction Visualizations")
                
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    # Latency distribution
                    st.markdown("#### Latency Distribution")
                    if 'latency' in df_diagnostics.columns:
                        fig = px.histogram(
                            df_diagnostics, 
                            x='latency',
                            nbins=20,
                            title="Response Latency Distribution (seconds)",
                            color_discrete_sequence=['#1890ff']
                        )
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                
                with viz_col2:
                    # Tool usage pie chart
                    st.markdown("#### Tool Usage Distribution")
                    all_tools = []
                    for tools_list in df_diagnostics.tools:
                        if isinstance(tools_list, list):
                            all_tools.extend(tools_list)
                        elif isinstance(tools_list, str):
                            # Handle case where tools might be a comma-separated string
                            all_tools.extend([t.strip() for t in tools_list.split(',')])
                    
                    tool_counts = pd.Series(all_tools).value_counts().reset_index()
                    tool_counts.columns = ['Tool', 'Count']
                    
                    if not tool_counts.empty:
                        fig = px.pie(
                            tool_counts, 
                            values='Count', 
                            names='Tool',
                            title="Tool Usage Distribution",
                            hole=0.4
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No tool usage data available.")
                
                # Time series analysis
                st.subheader("Temporal Analysis")
                if "timestamp" in df_diagnostics.columns:
                    # Ensure timestamp is datetime
                    if not pd.api.types.is_datetime64_any_dtype(df_diagnostics.timestamp):
                        df_diagnostics['timestamp'] = pd.to_datetime(df_diagnostics.timestamp)
                    
                    # Group by hour and count transactions
                    df_diagnostics['hour'] = df_diagnostics.timestamp.dt.floor('H')
                    hourly_counts = df_diagnostics.groupby('hour').size().reset_index(name='count')
                    
                    # Create time series chart
                    fig = px.line(
                        hourly_counts, 
                        x='hour', 
                        y='count', 
                        title="Transaction Volume by Hour",
                        markers=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # LLM Analysis section
                st.subheader("🧠 AI-Powered Diagnostic Analysis")
                
                # Prepare summary data for the LLM
                summary_data = {
                    "total_transactions": len(df_diagnostics),
                    "avg_latency": df_diagnostics.latency.mean() if 'latency' in df_diagnostics.columns else 'N/A',
                    "max_latency": df_diagnostics.latency.max() if 'latency' in df_diagnostics.columns else 'N/A',
                    "p95_latency": df_diagnostics.latency.quantile(0.95) if 'latency' in df_diagnostics.columns else 'N/A',
                    "tool_distribution": dict(pd.Series(all_tools).value_counts()) if all_tools else {},
                    "time_range_hours": diagnostic_hours
                }
                
                
                # Define hourly transaction counts if not already defined
                hourly_counts = pd.DataFrame()
                if "timestamp" in df_diagnostics.columns:
                    # Ensure timestamp is datetime if not already
                    if not pd.api.types.is_datetime64_any_dtype(df_diagnostics.timestamp):
                        df_diagnostics['timestamp'] = pd.to_datetime(df_diagnostics.timestamp)
                    
                    # Group by hour and count transactions
                    if 'hour' not in df_diagnostics.columns:
                        df_diagnostics['hour'] = df_diagnostics.timestamp.dt.floor('H')
                    
                    hourly_counts = df_diagnostics.groupby('hour').size().reset_index(name='count')
                
                # Create a prompt for the LLM analysis with tabular data
                prompt = f"""
                You are an expert AI system performance analyst. I will provide you with log data from an AI assistant system.

                Here's a summary of the data from the last {diagnostic_hours} hours:
                - Total transactions: {summary_data['total_transactions']}
                - Average latency: {summary_data['avg_latency']:.2f}s
                - P95 latency: {summary_data['p95_latency']:.2f}s
                - Maximum latency: {summary_data['max_latency']:.2f}s
                - Tool usage distribution: {summary_data['tool_distribution']}

                Here's a sample of the transaction logs (up to 40 records):
                ```
                {df_diagnostics.head(40).reset_index(drop=True).to_string()}
                ```

                Hourly transaction volume:
                ```
                {hourly_counts.to_string(index=False) if not hourly_counts.empty else "No hourly data available"}
                ```

                Additional metrics:
                - Number of unique queries: {df_diagnostics['user'].nunique() if 'user' in df_diagnostics.columns else 'N/A'}
                - High-latency transactions (>3s): {(df_diagnostics['latency'] > 3).sum() if 'latency' in df_diagnostics.columns else 0} ({(df_diagnostics['latency'] > 3).mean() * 100:.1f}% of total)
                - Tools used per transaction: {sum(len(t) if isinstance(t, list) else len(t.split(',')) if isinstance(t, str) else 0 for t in df_diagnostics.tools) / len(df_diagnostics):.2f} avg

                Please analyze the logs for potential performance issues by:

                1. Identifying patterns in latency spikes:
                   - Which types of queries consistently have higher latency?
                   - Are there specific patterns in high-latency transactions?

                2. Analyzing correlations between:
                   - Query complexity and latency
                   - Tool usage and latency (e.g., web_search vs. calculator vs. no tools)
                   - Query length/type and latency
                   - Time of day and performance

                3. Finding anomalies:
                   - Unusual latency outliers (significantly above average)
                   - Inconsistent latency for similar queries
                   - Any irregular patterns in system behavior

                4. Comparing performance:
                   - Queries with similar content but different latency
                   - Performance differences between tool-using vs. non-tool queries
                   - Variations in performance across different time periods

                5. Providing specific recommendations:
                   - Which query types should be optimized?
                   - Are there specific tools that need performance improvement?
                   - Is there a pattern of degraded performance at certain times?
                   - Are there specific user patterns that could be optimized?

                Focus on actionable insights that would help improve system performance.
                """
                
                # Call the Azure OpenAI API for analysis
                with st.spinner("AI is analyzing the transaction data..."):
                    try:
                        # Make the API call to Azure OpenAI
                        response = client.chat.completions.create(
                            model=deployment,
                            messages=[
                                {"role": "system", "content": "You are an expert operations analyst who specializes in analyzing agent observability data."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.3,
                            max_tokens=800
                        )
                        
                        # Extract the analysis
                        analysis = response.choices[0].message.content
                        
                        # Display the analysis in a nice card
                        st.markdown(
                        """<div style="background-color: #f0f8ff; padding: 20px; border-radius: 10px; border-left: 5px solid #1890ff;">
                            <h4 style="color: #1890ff;">AI Analysis</h4>
                            <div style="margin-top: 10px;">
                        """, 
                        unsafe_allow_html=True
                        )
                        
                        st.markdown(analysis)
                        
                        st.markdown(
                        """</div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                        )
                        
                    except Exception as e:
                        st.error(f"Error while generating AI analysis: {str(e)}")
                        st.info("Try adjusting the time range or try again later.")
    else:
        # Instructions when diagnostics haven't been run yet
        st.info("👆 Select a time range and click 'Run Smart Diagnostics' to analyze transaction data and get AI-powered insights.")
        
        st.markdown(
        """### Smart Diagnostics Features
        
        This tab provides advanced analytics on your agent interactions:
        
        - **Transaction Analysis**: Visualize response times and tool usage patterns
        - **Temporal Analysis**: See how traffic and performance vary over time
        - **AI-Powered Insights**: Get intelligent analysis of system behavior
        
        Use these insights to identify performance bottlenecks, unusual patterns, or opportunities for optimization.
        """)
