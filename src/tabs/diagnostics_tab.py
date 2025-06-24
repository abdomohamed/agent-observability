import streamlit as st
import pandas as pd
import plotly.express as px

from src.utils.telemetry import fetch_history

def render_diagnostics_tab(la_client, workspace_id, client, deployment):
    """Render the diagnostics tab"""
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
            df_diagnostics = fetch_history(la_client, workspace_id, hours=diagnostic_hours)
            
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
                
                # Create a prompt for the LLM analysis
                prompt = f"""
                You are an AI operations analyst examining transaction data from an agent-based system. 
                
                Here's a summary of the data from the last {diagnostic_hours} hours:
                - Total transactions: {summary_data['total_transactions']}
                - Average latency: {summary_data['avg_latency']:.2f}s
                - P95 latency: {summary_data['p95_latency']:.2f}s
                - Maximum latency: {summary_data['max_latency']:.2f}s
                - Tool usage distribution: {summary_data['tool_distribution']}
                
                Based on this data, please provide:
                1. A brief summary of the system's performance
                2. Any notable patterns or anomalies
                3. Recommendations for improving performance
                4. Potential issues that might need attention
                
                Focus on operational insights that would be valuable for a system administrator.
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
