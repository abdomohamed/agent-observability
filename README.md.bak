# Agent Observability Dashboard

## Overview
This application provides a multi-agent observability dashboard that demonstrates the use of Semantic Kernel for building and orchestrating AI agents with telemetry tracking. It allows users to interact with different specialized agents (Search, Calculator, and Coordinator) while monitoring performance metrics and conversation history.

## Key Features
- **Multi-Agent System**: Utilizes a coordinator agent that delegates to specialized agents based on the query type
- **Live Conversation Tracking**: Maintains and displays the conversation history with message type identification
- **Performance Monitoring**: Tracks and visualizes metrics like latency, token usage, and agent round trips
- **Telemetry Integration**: Uses Azure Application Insights to log agent interactions for historical analysis
- **Interactive Dashboard**: Real-time visualization of agent performance and usage patterns
- **Smart Diagnostics**: AI-powered analysis of historical interactions and performance patterns

## Architecture
The application consists of:
- A Streamlit-based web UI for user interaction
- Semantic Kernel for agent orchestration and tool integration
- Azure AI Agents for tool and plugin definitions
- Azure OpenAI Service for LLM capabilities (using GPT-4o)
- Azure Application Insights for telemetry collection
- Azure Log Analytics for historical data querying

## Setup Instructions

### Prerequisites
- Python 3.12+ (recommended)
- Azure OpenAI API access with GPT-4o deployment
- Azure Application Insights resource
- Azure Log Analytics workspace

### Environment Configuration
Create a Streamlit secrets file at `.streamlit/secrets.toml` with the following structure:

```toml
[AZURE_OPENAI]
API_KEY = "your-azure-openai-api-key"
# Set the endpoint URL and deployment name in environment variables or here
# ENDPOINT_URL = "your-azure-openai-endpoint"
# DEPLOYMENT_NAME = "gpt-4o"

[APP_INSIGHTS]
INSTRUMENTATION_KEY = "your-app-insights-instrumentation-key"

[AZURE]
WORKSPACE_ID = "your-log-analytics-workspace-id"
```

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd agent-observability
```

2. Create and activate a virtual environment:
```bash
python -m venv agent_env
source agent_env/bin/activate  # On Windows: agent_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

Start the Streamlit application:
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`.

## Usage Guide

1. **Asking Questions**: Enter your question in the text area and click "Submit"
2. **Selecting Agents**: Choose between:
   - **Auto (Coordinator)**: Automatically selects the most appropriate agent(s)
   - **Search Agent**: Specialized for information retrieval
   - **Calculator Agent**: Specialized for mathematical operations

3. **Conversation Features**:
   - View the complete conversation history with message type identification
   - Filter messages by type (User, Assistant, System)
   - Clear the conversation history
   - Export the conversation as a JSON file

4. **Dashboard**:
   - Monitor real-time performance metrics
   - View historical interaction data
   - Analyze tool usage patterns
   - Track conversation statistics
   
5. **Smart Diagnostics**:
   - View in-depth analysis of historical interactions
   - Get AI-powered insights about performance patterns
   - Analyze response time distributions
   - Identify potential system bottlenecks

## Dependencies
- `streamlit`: Web application framework
- `semantic-kernel>=1.33.0`: Agent orchestration and tool integration
- `openai>=1.67`: Azure OpenAI API integration
- `azure-identity`: Azure authentication
- `azure-monitor-query`: Log Analytics queries
- `azure-ai-agents>=1.1.0b1`: Azure AI Agents SDK
- `pandas`: Data manipulation and analysis
- `applicationinsights`: Telemetry collection
- `plotly`: Data visualization

## Troubleshooting

- **Authentication Issues**: Ensure your Azure credentials are correctly configured
- **API Limits**: Check for rate limiting if the application becomes unresponsive
- **Missing Data**: Verify App Insights is properly collecting telemetry
- **GPT-4o Availability**: Make sure your Azure OpenAI resource has access to the GPT-4o model
- **Python Environment**: Use Python 3.12+ for optimal compatibility

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
MIT License
