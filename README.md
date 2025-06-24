# Agent Observability App

A Streamlit application for monitoring and observing multi-agent interactions using Azure OpenAI and Azure Application Insights.

## Project Structure

```
agent-observability/
├── app.py                # Main application entry point
├── requirements.txt      # Python package dependencies
├── src/                  # Source code package
│   ├── __init__.py       # Package initialization
│   ├── agents/           # Agent-related functionality
│   │   ├── __init__.py
│   │   └── agent_manager.py  # Agent creation and management
│   ├── tabs/             # UI tabs
│   │   ├── __init__.py
│   │   ├── chat_tab.py       # Chat interface tab
│   │   ├── dashboard_tab.py  # Dashboard and metrics tab
│   │   └── diagnostics_tab.py # Diagnostics and analysis tab
│   └── utils/            # Utility functions and services
│       ├── __init__.py
│       ├── azure_config.py    # Azure OpenAI configuration
│       ├── styling.py         # UI styling
│       └── telemetry.py       # Telemetry and logging functions
```

## Features

- Multi-agent chat interface with specialized agents
- Real-time performance dashboard
- Historical metrics and SLO tracking
- Smart diagnostics with AI-powered analysis

## Running the Application

To run the application:

```bash
streamlit run app.py
```

## Dependencies

The application requires the following main dependencies:
- streamlit
- semantic-kernel
- azure-openai
- plotly
- pandas
- applicationinsights
- azure-monitor-query

For a complete list, see the `requirements.txt` file.

## Configuration

The application requires the following secrets to be configured in Streamlit:

```
[AZURE_OPENAI]
API_KEY = "your-api-key"

[APP_INSIGHTS]
INSTRUMENTATION_KEY = "your-instrumentation-key"

[AZURE]
WORKSPACE_ID = "your-workspace-id"
```

You can set these in `.streamlit/secrets.toml` or using environment variables.
