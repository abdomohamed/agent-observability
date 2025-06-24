import os
import streamlit as st
from openai import AzureOpenAI
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

# ──────────────────────────────────────────────────────────────────────────────
# Azure OpenAI Configuration
# ──────────────────────────────────────────────────────────────────────────────

def initialize_azure_openai():
    """Initialize and return Azure OpenAI client and Semantic Kernel"""
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
    
    return client, kernel, deployment, endpoint, subscription_key
