import streamlit as st
from semantic_kernel.contents import AuthorRole
from semantic_kernel.contents.chat_message_content import ChatMessageContent

# Import modules from the src package
from src.utils.styling import apply_custom_styling
from src.utils.azure_config import initialize_azure_openai
from src.utils.telemetry import initialize_telemetry, initialize_log_analytics
from src.agents.agent_manager import initialize_agents, sync_conversation_history_with_agents
from src.tabs.chat_tab import render_chat_tab
from src.tabs.dashboard_tab import render_dashboard_tab
from src.tabs.diagnostics_tab import render_diagnostics_tab

# ──────────────────────────────────────────────────────────────────────────────
# App Initialization
# ──────────────────────────────────────────────────────────────────────────────
def initialize_app():
    """Initialize the application state and resources"""
    # Set up the page
    st.set_page_config(
        page_title="Agent Observability",
        page_icon="🕵️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom styling
    apply_custom_styling()
    
    # Initialize Azure OpenAI
    client, kernel, deployment, endpoint, subscription_key = initialize_azure_openai()
    
    # Initialize telemetry
    telemetry_client = initialize_telemetry()
    
    # Initialize Log Analytics
    la_client, workspace_id = initialize_log_analytics()
    
    # Initialize agents
    initialize_agents(kernel)
    
    # Initialize session state for conversation history
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    # When the app starts, sync conversation history with agent chat histories
    if 'app_initialized' not in st.session_state:
        sync_conversation_history_with_agents()
        st.session_state.app_initialized = True
    
    return client, kernel, deployment, telemetry_client, la_client, workspace_id

# ──────────────────────────────────────────────────────────────────────────────
# Main App
# ──────────────────────────────────────────────────────────────────────────────
def main():
    """Main application function"""
    # Initialize the app
    client, kernel, deployment, telemetry_client, la_client, workspace_id = initialize_app()
    
    # App title and description
    st.title("🕵️ Multi-Agent Observability")
    st.caption("Powered by Semantic Kernel + App Insights")
    
    # Create tabs for Chat, Dashboard, and Diagnostics
    chat_tab, dashboard_tab, diagnostics_tab = st.tabs(["💬 Chat Interface", "📊 Dashboard", "🔍 Smart Diagnostics"])
    
    # Render each tab
    with chat_tab:
        render_chat_tab(telemetry_client)
    
    with dashboard_tab:
        render_dashboard_tab(la_client, workspace_id)
    
    with diagnostics_tab:
        render_diagnostics_tab(la_client, workspace_id, client, deployment)

if __name__ == "__main__":
    main()
