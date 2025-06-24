import streamlit as st
import time
import asyncio
import pandas as pd
from semantic_kernel.functions import kernel_function, KernelArguments
from semantic_kernel.contents import AuthorRole
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatPromptExecutionSettings
from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatCompletionAgent

from src.utils.telemetry import track_to_app_insights

# ──────────────────────────────────────────────────────────────────────────────
# Tool definitions and agent setup
# ──────────────────────────────────────────────────────────────────────────────

class BasePlugin():
    """ A base class for defining function plugins that can be used by agents.
    """

    def call_tool(self, tool_name: str, prompt: str) -> str:
        # Implementation not shown in the original file
        pass

# Define tools using the Azure AI Agents SDK
class SearchPlugin(BasePlugin):
    """ A plugin for web search functionality. """
    
    @kernel_function(description="Search the web for information")
    def web_search(self, query: str) -> str:
        # Implementation not shown in the original file
        return f"Search results for: {query}"

class CalculatorPlugin(BasePlugin):
    @kernel_function(description="Provide web search capabilities")
    def calculator(self, expression: str) -> str:
        # Implementation not shown in the original file
        return f"Calculation result for: {expression}"

# Create agents with different tools and system prompts
def create_search_agent(kernel):
    """Create and return a search agent"""
    settings = AzureChatPromptExecutionSettings(service_id="AzureOpenAI")
    agent = ChatCompletionAgent(
        kernel=kernel,
        name="search_agent",
        instructions="You are a specialized agent that searches the web for information. Use the web_search function when the user asks for information that might be found online.",
        plugins=[SearchPlugin()],
        arguments=KernelArguments(settings=settings),
    )
    return agent

def create_calculator_agent(kernel):
    """Create and return a calculator agent"""
    settings = AzureChatPromptExecutionSettings(service_id="AzureOpenAI")
    agent = ChatCompletionAgent(
        kernel=kernel,
        name="calculator_agent",
        instructions="You are a specialized agent that performs calculations. Use the calculator function when the user asks for any mathematical operations.",
        plugins=[CalculatorPlugin()],
        arguments=KernelArguments(settings=settings),
    )
    return agent

def create_coordinator_agent(kernel):
    """Create and return a coordinator agent"""
    settings = AzureChatPromptExecutionSettings(service_id="AzureOpenAI")
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

def initialize_agents(kernel):
    """Initialize the agents if they haven't been created yet"""
    if "search_agent" not in st.session_state:
        st.session_state.search_agent = create_search_agent(kernel)
    
    if "calculator_agent" not in st.session_state:
        st.session_state.calculator_agent = create_calculator_agent(kernel)
    
    if "coordinator_agent" not in st.session_state:
        st.session_state.coordinator_agent = create_coordinator_agent(kernel)

    # Initialize agent message histories if they don't exist
    if 'search_agent_messages' not in st.session_state:
        st.session_state.search_agent_messages = []
        
    if 'calculator_agent_messages' not in st.session_state:
        st.session_state.calculator_agent_messages = []
        
    if 'coordinator_agent_messages' not in st.session_state:
        st.session_state.coordinator_agent_messages = []

# ──────────────────────────────────────────────────────────────────────────────
# Agent invocation
# ──────────────────────────────────────────────────────────────────────────────
async def query_agent_with_sk(user_message: str, selected_agent="auto", telemetry_client=None) -> dict:
    start = time.time()
    used_tools = []
    answer = ""
    tokens = 0
    metadata = {"user_message": user_message, "selected_agent": selected_agent, "timestamp": pd.Timestamp.utcnow()}
    
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

        agent_response = await st.session_state.coordinator_agent.get_response(
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
            tokens = agent_response.metadata['usage'].get('total_tokens', 0)
            
        # Extract tool usage
        for tool_call in agent_response.message.tool_calls or []:
            tool_name = tool_call.get('name', '')
            if tool_name not in used_tools:
                used_tools.append(tool_name)
        
    else:
        # Use a single agent if only one tool is needed or specific agent selected
        if "web_search" in used_tools or selected_agent == "Search Agent":
            messages = st.session_state.search_agent_messages.copy()
            messages.append(ChatMessageContent(role=AuthorRole.USER, content=user_message, metadata=metadata))
            
            agent_response = await st.session_state.search_agent.get_response(
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
                
        elif "calculator" in used_tools or selected_agent == "Calculator Agent":
            messages = st.session_state.calculator_agent_messages.copy()
            messages.append(ChatMessageContent(role=AuthorRole.USER, content=user_message, metadata=metadata))
            
            agent_response = await st.session_state.calculator_agent.get_response(
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
                tokens = agent_response.metadata['usage'].get('total_tokens', 0)
                
        else:
            # Use the coordinator as a general assistant if no specific tools needed
            messages = st.session_state.coordinator_agent_messages.copy()
            messages.append(ChatMessageContent(role=AuthorRole.USER, content=user_message, metadata=metadata))

            agent_response = await st.session_state.coordinator_agent.get_response(
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
                tokens = agent_response.metadata['usage'].get('total_tokens', 0)
            
    
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
    if telemetry_client:
        track_to_app_insights(telemetry_client, rec)
    
    return rec

# For backward compatibility, create a synchronous wrapper
def query_agent(user_message: str, selected_agent="auto", telemetry_client=None) -> dict:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(query_agent_with_sk(user_message, selected_agent, telemetry_client))
    loop.close()
    return result

# Function to sync conversation history with agent chat histories
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
