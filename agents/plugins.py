from semantic_kernel.functions import kernel_function, KernelArguments

# ──────────────────────────────────────────────────────────────────────────────
# Tool definitions and agent setup
# ──────────────────────────────────────────────────────────────────────────────

class BasePlugin():
    """ A base class for defining function plugins that can be used by agents.
    """

    def call_tool(self, tool_name: str, prompt: str) -> str:
        time.sleep(0.4)
        return f"[{tool_name} result for '{prompt}']\n"

    # Define tools using the Azure AI Agents SDK
class SearchPlugin(BasePlugin):
    """ A plugin for web search functionality. """
    
    @kernel_function(description="Search the web for information")
    async def web_search(self, query: str) -> str:
        """
        Search the web for information.
        
        Args:
            query: The search query
            
        Returns:
            Search results
        """
        return self.call_tool("web_search", query)


class CalculatorPlugin(BasePlugin):
    @kernel_function(description="Provide web search capabilities")
    async def web_search(self, query: str) -> str:
        """
        Search the web for information.
        
        Args:
            query: The search query
                
        Returns:
            Search results
        """
        return self.call_tool("web_search", query)