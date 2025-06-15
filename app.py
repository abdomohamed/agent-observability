import time
from datetime import timedelta

import pandas as pd
import streamlit as st
import os

# ──────────────────────────────────────────────────────────────────────────────
# Azure OpenAI Client via openai-python
# ──────────────────────────────────────────────────────────────────────────────

from openai import AzureOpenAI


aoai = st.secrets["AZURE_OPENAI"]

endpoint = os.getenv("ENDPOINT_URL", "https://agent-ai-servicesqqx5.cognitiveservices.azure.com/")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", aoai["API_KEY"])

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=endpoint,
    api_key=subscription_key
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
            "rounds": rec["rounds"],
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
# Tool stubs
# ──────────────────────────────────────────────────────────────────────────────
def call_tool(tool_name: str, prompt: str) -> str:
    time.sleep(0.4)
    return f"[{tool_name} result for “{prompt}”]\n"

# ──────────────────────────────────────────────────────────────────────────────
# Agent invocation
# ──────────────────────────────────────────────────────────────────────────────
def query_agent(user_message: str) -> dict:
    start = time.time()
    used_tools = []
    if "search" in user_message.lower():
        used_tools.append("web_search")
    if "calc" in user_message.lower():
        used_tools.append("calculator")

    # run our stubs
    tool_blob = "".join(call_tool(t, user_message) for t in used_tools)

    # Azure OpenAI call
    resp = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": user_message + "\n" + tool_blob}
        ],
        max_tokens=800,
        temperature=0.7,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False
    )

    answer  = resp.choices[0].message.content
    latency = time.time() - start
    tokens  = resp.usage.total_tokens

    rec = {
        "user":    user_message,
        "answer":  answer,
        "latency": latency,
        "tokens":  tokens,
        "tools":   used_tools,
        "rounds":  1,
        "timestamp": pd.Timestamp.utcnow()
    }
    track_to_app_insights(rec)
    return rec

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
        user    = tostring(todynamic(Properties).user),
        rounds  = toint(todynamic(Properties).rounds)
    | project Name, TimeGenerated, user, latency, tokens, tools, rounds
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
        return pd.DataFrame(columns=["timestamp","user","latency","tokens","tools","rounds"])
        

    # if we got here, neither table existed
    

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("🕵️ AI Agent Observability (Azure OpenAI + App Insights)")

col_q, col_dash = st.columns([1,2])

with col_q:
    st.header("Ask the Agent")
    user_input = st.text_area("Your question:", height=120)
    if st.button("Submit"):
        rec = query_agent(user_input)
        st.markdown("**Agent**: " + rec["answer"])
        st.write(f"⏱ {rec['latency']:.2f}s · 🧮 {rec['tokens']} · 🛠 {rec['tools']}")

with col_dash:
    st.header("📊 Live Dashboard")
    df = fetch_history(hours=2)
    if df.empty:
        st.info("No interactions in the last 2 hours.")
    else:
        # KPIs
        avg_lat   = df.latency.mean()
        p95_lat   = df.latency.quantile(0.95)
        avg_tok   = df.tokens.mean()
        tot_q     = len(df)
        # tools_cnt = pd.Series(sum(df.tools.tolist(), [])).value_counts()
        rounds_pct= df.rounds.value_counts(normalize=True) * 100

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Avg Latency", f"{avg_lat:.2f}s", f"p95 {p95_lat:.2f}s")
        c2.metric("Avg Tokens", f"{avg_tok:.0f}")
        c3.metric("Total Queries", tot_q)
        # c4.metric("Distinct Tools", len(tools_cnt))

        # trends
        st.subheader("Latency & Tokens Over Time")
        st.line_chart(df.set_index("TimeGenerated")[["latency","tokens"]])

        st.subheader("Tool Usage Frequency")
        # st.bar_chart(tools_cnt)

        st.subheader("Rounds per Query (%)")
        st.bar_chart(rounds_pct)

        st.subheader("Recent Interactions")
        st.dataframe(
            df[["TimeGenerated","user","latency","tokens","tools"]],
            height=200,
            use_container_width=True
        )