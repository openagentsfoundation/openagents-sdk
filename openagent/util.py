import functools
import json
from typing import TYPE_CHECKING, Any
from ..tools import FunctionTool, FunctionTool as Tool
from ..utils.logger import logger
from .ensure_strict_json_schema import ensure_strict_json_schema

if TYPE_CHECKING:
    from mcp.types import Tool as MCPTool
    from .server import MCPServer

class MCPUtil:
    """Set of utilities for interop between MCP and Agents SDK tools."""
    logger = None

    @classmethod
    def initialize(cls, logger):
        cls.logger = logger
        pass

    @classmethod
    async def get_all_function_tools(
        cls, servers: list["MCPServer"], convert_schemas_to_strict: bool
    ) -> list[Tool]:
        """Get all function tools from a list of MCP servers."""
        tools = []
        tool_names: set[str] = set()
        for server in servers:
            server_tools = await cls.get_function_tools(server, convert_schemas_to_strict)
            server_tool_names = {tool.name for tool in server_tools}
            if len(server_tool_names & tool_names) > 0:
                raise Exception(
                    f"Duplicate tool names found across MCP servers: "
                    f"{server_tool_names & tool_names}"
                )
            tool_names.update(server_tool_names)
            tools.extend(server_tools)

        return tools

    @classmethod
    async def get_function_tools(
        cls, server: "MCPServer", convert_schemas_to_strict: bool
    ) -> list[Tool]:
        """Get all function tools from a single MCP server."""
        tools = await server.list_tools()
        return [cls.to_function_tool(tool, server, convert_schemas_to_strict) for tool in tools]

    @classmethod
    def to_function_tool(
        cls, tool: "MCPTool", server: "MCPServer", convert_schemas_to_strict: bool
    ) -> FunctionTool:
        """Convert an MCP tool to an Agents SDK function tool."""
        invoke_func = functools.partial(cls.invoke_mcp_tool, server, tool)
        schema, is_strict = tool.inputSchema, False

        # MCP spec doesn't require the inputSchema to have `properties`, but OpenAI spec does.
        if "properties" not in schema:
            schema["properties"] = {}

        if convert_schemas_to_strict:
            try:
                schema = ensure_strict_json_schema(schema)
                is_strict = True
            except Exception as e:
                logger.warning(f"Error converting MCP schema to strict mode: {e}")

        return FunctionTool(
            name=tool.name,
            description=tool.description or "",
            params_json_schema=schema,
            callback=invoke_func,
            strict_json_schema=is_strict,
        )

    @classmethod
    async def invoke_mcp_tool(
        cls, server: "MCPServer", tool: "MCPTool", input_json: str
    ) -> str:
        """Invoke an MCP tool and return the result as a string."""
        try:
            json_data: dict[str, Any] = json.loads(input_json) if input_json else {}
        except Exception as e:
            logger.debug(f"Invalid JSON input for tool {tool.name}: {input_json}")
            raise Exception(
                f"Invalid JSON input for tool {tool.name}: {input_json}"
            ) from e

        logger.debug(f"Invoking MCP tool {tool.name} with input {input_json}")

        try:
            result = await server.call_tool(tool.name, json_data)
        except Exception as e:
            logger.error(f"Error invoking MCP tool {tool.name}: {e}")
            raise Exception(f"Error invoking MCP tool {tool.name}: {e}") from e

        logger.debug(f"MCP tool {tool.name} returned {result}")

        # The MCP tool result is a list of content items, whereas OpenAI tool outputs are a single
        # string. We'll try to convert.
        if len(result.content) == 1:
            tool_output = result.content[0].model_dump_json()
        elif len(result.content) > 1:
            tool_output = json.dumps([item.model_dump() for item in result.content])
        else:
            logger.error(f"Errored MCP tool result: {result}")
            tool_output = "Error running tool."

        return tool_output
