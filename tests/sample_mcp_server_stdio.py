#Reference: https://github.com/openai/openai-agents-python/blob/main/examples/mcp/sse_example/server.py
from mcp.server.fastmcp import FastMCP
## Create an MCP server
mcp = FastMCP("Demo")

## mcp.tool is tool calling
@mcp.tool()
def calculator_tool(expression: str) -> str:
    """Securely evaluates an arithmetic expression in python.
    Parameters:
        expression (str): A string containing a basic arithmetic expression valid in python.
    Returns:
        str: The result of the expression or an error message.
    """
    try:
        # Evaluate with restricted built-ins for basic arithmetic.
        result = eval(expression, {"__builtins__": {}}, {})
        print(f"The result of {expression} is {str(result)}")
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    # Initialize and run the server as stdio or fastapi service
    ## stdio provides is local mcp service through stdio communication
    # transport='sse' for fast api service, or transport='stdio' or local IO communication
    mcp.run(transport='stdio')
