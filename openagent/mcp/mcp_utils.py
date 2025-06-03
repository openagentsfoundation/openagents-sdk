import functools
import json
from typing import Any, Optional
from mcp.types import Tool as MCPTool
from .ensure_strict_json_schema import ensure_strict_json_schema
from .mcp_client import MCPClient
from ..tools import FunctionTool
from ..exceptions import *

class MCPAccessUtil:
    """Set of utilities for interop between MCP and OpenAgents SDK tools."""
    # get function tools from all MCP client connections
    @classmethod
    async def get_all_function_tools(
        cls, mcp_clients: list[MCPClient], convert_schemas_to_strict: bool | None = False
    ) -> list[FunctionTool]:
        """Get all function tools from a list of MCP clients."""
        tools = []
        tool_names: set[str] = set()
        for mcp_client in mcp_clients:
            mcp_tools = await cls.get_function_tools(mcp_client, convert_schemas_to_strict)
            server_tool_names = {tool.name for tool in mcp_tools}
            if len(server_tool_names & tool_names) > 0:
                raise AgentException(
                    f"Duplicate tool names found across MCP clients: {server_tool_names & tool_names}",
                    error_code = ErrorCode.ModelBehaviorError,
                    module=mcp_client.name or "mcp"
                )
            tool_names.update(server_tool_names)
            tools.extend(mcp_tools)

        return tools

    # get function tools from a single MCP client
    @classmethod
    async def get_function_tools(
        cls, mcp_client: MCPClient, convert_schemas_to_strict: bool | None = False
    ) -> list[FunctionTool]:
        """Get all function tools from a single MCP client."""
        await mcp_client.connect()
        tools = await mcp_client.list_tools()
        return [cls.to_function_tool(tool, mcp_client, convert_schemas_to_strict) for tool in tools]

    @classmethod
    def to_function_tool(
        cls, mcp_tool: MCPTool, mcp_client: MCPClient, convert_schemas_to_strict: bool | None = False
    ) -> FunctionTool:
        """Convert an MCP tool to an Agents SDK function tool."""
        invoke_func = functools.partial(cls.invoke_mcp_tool, mcp_client, mcp_tool)
        input_schema, is_strict = mcp_tool.inputSchema, False

        # MCP spec doesn't require the inputSchema to have `properties`, but OpenAI spec does.
        if "properties" not in input_schema:
            input_schema["properties"] = {}

        if convert_schemas_to_strict:
            try:
                input_schema = ensure_strict_json_schema(input_schema)
                is_strict = True
            except Exception as e:
                mcp_client.get_logger().warning(f"Error converting MCP schema to strict mode: {e}")

        tool_name = mcp_tool.name
        if mcp_client.mcp_tool_prefix: # if there's prefix such as "mcp_"
            tool_name = mcp_client.mcp_tool_prefix + tool_name

        json_schema = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": mcp_tool.description or "",
                "parameters": input_schema,
            },
        }
        
        function_tool = FunctionTool(
            name = tool_name,
            description = mcp_tool.description or "",
            json_schema = json_schema,
            callback = invoke_func,
            is_async = True # it's async function
            )

        function_tool.input_arguments = {}
        for k,v in input_schema["properties"].items():
            function_tool.input_arguments[k] = v["type"]

        return function_tool

    @classmethod
    async def invoke_mcp_tool(
        cls, mcp_client: MCPClient, mcp_tool: MCPTool, **args
    ) -> str:
        """Invoke an MCP tool and return the result as a string."""
        try:
            json_data: dict[str, Any] = args
        except Exception as e:
            mcp_client.get_logger().debug(f"Invalid JSON input for tool {mcp_tool.name}: {args}")
            raise AgentException(
                f"Invalid JSON input for MCP tool {mcp_tool.name}: {args}",
                error_code=ErrorCode.ModelBehaviorError,
                module= mcp_client.name or "mcp"
            ) from e

        mcp_client.get_logger().debug(f"Invoking MCP tool {mcp_tool.name} with input {args}")

        try:
            result = await mcp_client.call_tool(mcp_tool.name, json_data)
        except Exception as e:
            mcp_client.get_logger().error(f"Error invoking MCP tool {mcp_tool.name}: {e}")
            raise AgentException(
                f"Error invoking MCP tool {mcp_tool.name}: {e}",
                error_code=ErrorCode.ModelBehaviorError,
                module=mcp_client.name or "mcp"
                ) from e

        mcp_client.get_logger().debug(f"MCP tool {mcp_tool.name} returned {result}")

        # The MCP tool result is a list of content items, whereas OpenAI tool outputs are a single
        # string. We'll try to convert.
        if len(result.content) == 1:
            tool_output = result.content[0].model_dump_json()
        elif len(result.content) > 1:
            tool_output = json.dumps([item.model_dump() for item in result.content])
        else:
            mcp_client.get_logger().error(f"Errored MCP tool result: {result}")
            tool_output = "Error running tool."

        return tool_output
