import streamlit as st
import pandas as pd
from semantic_kernel.contents import AuthorRole
import altair as alt

from src.utils.telemetry import fetch_history

def render_dashboard_tab(la_client, workspace_id):
    """Render the dashboard tab"""
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
    df = fetch_history(la_client, workspace_id, hours=selected_hours)
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
            last_hour_data = df[df['timestamp'] > pd.Timestamp.utcnow() - pd.Timedelta(hours=1)] if 'timestamp' in df.columns else pd.DataFrame()
            
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
