import streamlit as st
import json
import pandas as pd
from semantic_kernel.contents import AuthorRole
from semantic_kernel.contents.chat_message_content import ChatMessageContent

from src.agents.agent_manager import query_agent, sync_conversation_history_with_agents

def render_chat_tab(telemetry_client):
    """Render the chat interface tab"""
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
                    # Determine icon and agent type
                    agent_type = message.metadata.get("agent_type", "Coordinator")
                    icon = "🔍" if agent_type == "Search" else "🧮" if agent_type == "Calculator" else "🤖"
                    
                    # Show tools used if any
                    tools_used = ""
                    if "tools" in message.metadata and message.metadata["tools"]:
                        tools_list = message.metadata["tools"]
                        tools_used = f" (using {', '.join(tools_list)})"
                    
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
                [{"role": msg.role, "content": msg.content} for msg in st.session_state.conversation_history], 
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
            rec = query_agent(user_input, selected_agent, telemetry_client)
            
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
