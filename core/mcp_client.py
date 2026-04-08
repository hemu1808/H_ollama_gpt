import asyncio
import logging
from typing import Callable, Dict, Any, List

logger = logging.getLogger(__name__)

class MCPToolRegistry:
    """
    Extensible registry for Agentic Tools (Model Context Protocol).
    Serves as the central hub for local functions and remote HTTP/MCP tools.
    """
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        
    def register(self, name: str, func: Callable):
        """Register a new tool dynamically."""
        self.tools[name] = func
        logger.info(f"Registered MCP Tool: {name}")
        
    async def execute_tool(self, name: str, *args, **kwargs) -> Any:
        """Route and execute a specific tool asynchronously."""
        if name not in self.tools:
            return f"Error: Tool {name} not found in MCP protocol."
        try:
            func = self.tools[name]
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                # Offload synchronous tools to thread pool to prevent blocking
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
        except Exception as e:
            logger.error(f"MCP Executor Error ({name}): {e}")
            return f"Error executing {name}: {str(e)}"

    async def execute_parallel(self, tool_calls: List[Dict[str, Any]]) -> List[Any]:
        """
        Concurrent execution of multiple tools.
        Expects a list of dicts: [{'name': 'tool_name', 'args': [], 'kwargs': {}}]
        """
        tasks = []
        for call in tool_calls:
            name = call.get('name')
            args = call.get('args', [])
            kwargs = call.get('kwargs', {})
            tasks.append(self.execute_tool(name, *args, **kwargs))
            
        logger.info(f"Executing {len(tasks)} MCP tasks concurrently...")
        return await asyncio.gather(*tasks, return_exceptions=True)

# ---------------------------------------------------------
# Global Instance
# ---------------------------------------------------------
mcp_registry = MCPToolRegistry()

# ---------------------------------------------------------
# Default Tools Implementation
# ---------------------------------------------------------
import wikipedia
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for information."""
    try:
        return wikipedia.summary(query, sentences=3)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Ambiguous query. Options: {e.options[:5]}"
    except Exception:
        return "Could not find information on Wikipedia."

def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Input must be a valid Python expression."""
    try:
        allowed_names = {"__builtins__": None}
        return str(eval(expression, allowed_names, {}))
    except Exception as e:
        return f"Error evaluating expression: {e}"

# Register default local tools
mcp_registry.register("wikipedia_search", wikipedia_search)
mcp_registry.register("calculator", calculator)

# Future: Register remote tools from actual MCP HTTP Servers
# mcp_registry.register("weather_api", remote_weather_fetcher_async)
