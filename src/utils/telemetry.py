import streamlit as st
from applicationinsights import TelemetryClient
from azure.identity import DefaultAzureCredential
from azure.monitor.query import LogsQueryClient
from azure.core.exceptions import HttpResponseError
import pandas as pd
from datetime import timedelta

# ──────────────────────────────────────────────────────────────────────────────
# App Insights telemetry
# ──────────────────────────────────────────────────────────────────────────────
def initialize_telemetry():
    """Initialize and return the telemetry client"""
    tc = TelemetryClient(st.secrets["APP_INSIGHTS"]["INSTRUMENTATION_KEY"])
    return tc

def track_to_app_insights(tc, rec: dict):
    """Track an agent interaction to App Insights"""
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
def initialize_log_analytics():
    """Initialize and return the Log Analytics client"""
    credential = DefaultAzureCredential()
    la_client = LogsQueryClient(credential)
    WORKSPACE_ID = st.secrets["AZURE"]["WORKSPACE_ID"]
    return la_client, WORKSPACE_ID

@st.cache_data(ttl=1)
def fetch_history(la_client, workspace_id, hours: int = 1) -> pd.DataFrame:
    """Fetch agent interaction history from App Insights"""
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
        resp = la_client.query_workspace(workspace_id, query=kusto_v1, timespan=timespan)
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
