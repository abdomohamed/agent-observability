import asyncio
from re import search
import sys

from httpx import get
from semantic_kernel.agents import Agent, ChatCompletionAgent, GroupChatOrchestration, AzureAIAgent
from semantic_kernel.agents.orchestration.group_chat import BooleanResult, GroupChatManager, MessageResult, StringResult
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.contents import AuthorRole, ChatHistory, ChatMessageContent
from semantic_kernel.functions import KernelArguments
from semantic_kernel.kernel import Kernel
from semantic_kernel.prompt_template import KernelPromptTemplate, PromptTemplateConfig
from azure.identity import DefaultAzureCredential
import streamlit as st
import os

from agents.group_chat_manager import ChatCompletionGroupChatManager
from agents.plugins import CalculatorPlugin, SearchPlugin

from semantic_kernel.connectors.ai import FunctionChoiceBehavior

from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import (   
    AzureChatCompletion,
    AzureChatPromptExecutionSettings
)

def create_ai_project_client():
    """Create an Azure AI Agent client using the DefaultAzureCredential."""
    creds = DefaultAzureCredential()
    return AzureAIAgent.create_client(credential=creds, endpoint="https://agent-ai-servicesqqx5.services.ai.azure.com/api/projects/agent-ai-servicesqqx5-project")

async def create_search_agent():
    client = create_ai_project_client()
    
    agent_definition = await client.agents.get_agent(agent_id="asst_H9YF4Ux61nJriR9zdAFG278r")
    
    agent_definition.description = "A specialized agent for web search tasks."
    
    kernel, _ = get_kernel_settings()
    
    agent = AzureAIAgent(
        kernel=kernel,
        client=client,
        definition=agent_definition,
    )
   
    return agent


async def create_calculator_agent():
    kernel, settings = get_kernel_settings()
    
    agent = ChatCompletionAgent(
        kernel=kernel,
        name="calculator_agent",
        description="A specialized agent for performing calculations.",
        instructions="You are a specialized agent that performs calculations. Use the calculator function when the user asks for any mathematical operations.",
        plugins=[CalculatorPlugin()],
        arguments=KernelArguments(settings=settings),
    )
     
    return agent



async def get_coordinator_agent() -> Agent:
    """Return the coordinator agent that manages the group style discussion."""
    kernel, settings = get_kernel_settings()
    return await create_coordinator_agent()

def get_kernel_settings() -> {Kernel, PromptExecutionSettings}:
    """Return the kernel and settings for the agents."""
    
    aoai = st.secrets["AZURE_OPENAI"]
    endpoint = os.getenv("ENDPOINT_URL", "https://agent-ai-servicesqqx5.cognitiveservices.azure.com/")
    deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
    subscription_key = os.getenv("AZURE_OPENAI_API_KEY", aoai["API_KEY"])

    kernel = Kernel()
    
    # Add plugins or other services as needed
    kernel.add_plugin(CalculatorPlugin(), "CalculatorPlugin")
    kernel.add_plugin(SearchPlugin(), "SearchPlugin")
    kernel.add_service(
        AzureChatCompletion(
            deployment_name=deployment,
            endpoint=endpoint,
            api_key=subscription_key,
            api_version="2024-12-01-preview",
            service_id="AzureOpenAI",
        )
    )
    
    
    settings = AzureChatPromptExecutionSettings(service_id="AzureOpenAI")
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
    
    return kernel, settings

async def get_agents() -> list[Agent]:
    """Return a list of agents that will participate in the group style discussion.

    Feel free to add or remove agents.
    """
    kernel, settings = get_kernel_settings()
    search_agent = await create_search_agent()
    calculator_agent = await create_calculator_agent()
    
    return [search_agent, calculator_agent]


async def create_coordinator_agent(callback) -> GroupChatOrchestration:
    kernel, _ = get_kernel_settings()
    return GroupChatOrchestration(
        description="A coordinator agent that manages multiple specialized agents. You can use this agent to route tasks to the appropriate specialized agents based on the user's request.",
        members=await get_agents(),
        manager=ChatCompletionGroupChatManager(service=kernel.get_service(type=AzureChatCompletion, service_id="AzureOpenAI")),
        agent_response_callback=callback
    )
