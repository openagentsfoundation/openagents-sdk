# https://github.com/openai/openai-agents-python/blob/main/examples/mcp/sse_example/server.py
import asyncio
import os, shutil, subprocess, time, socket
from typing import Any, Literal
from openagent import AssistantAgent, AgentStream, function_tool, get_logger, logging
from openagent import MCPClient, MCPClientSse, MCPClientStdio

# Start SSE MCP server implemented in python, for Demo purpose
## use `async with PythonMCPServerRunner(...)`, or 'await server.start()' to start the server
class PythonSseMCPServerHoster:
    def __init__(self, file:str, port:int|None=8000):
        directory = os.path.dirname(file)
        if not directory:
            directory = os.getcwd()
        self.file_path = os.path.join(directory, file)
        self.process:subprocess.Popen[Any] | None = None
        self.port = port

    async def start(self, timeout:int = 10):
        # Run `uv run server.py` to start the SSE server
        #process = subprocess.Popen(["uv", "run", server_file])
        print(f"Starting MCP server from file: {self.file_path}")
        env = os.environ.copy()
        env["FASTMCP_PORT"] = str(self.port) # override the listen port
        self.process = subprocess.Popen(["python", self.file_path],
                                        env=env)
        # ping the listen port until timeout
        start_time = asyncio.get_event_loop().time()
        while True:
            if self.process.poll() is not None:
                raise RuntimeError("MCP server process terminated unexpectedly.")
            try:
                with socket.create_connection(("localhost", self.port), timeout=1):
                    return
            except (ConnectionRefusedError, OSError):
                await asyncio.sleep(0.1)
            if asyncio.get_event_loop().time() - start_time > timeout:
                raise TimeoutError("Timed out waiting for MCP server to start.")
            
        
    async def shutdown(self):
        if self.process:
            self.process.terminate()
            self.process = None
        
    # context management
    async def __aenter__(self):
        await self.start()
        print("PythonSseMCPServerHoster started")

    async def __aexit__(self, exc_type, exc, tb):
        await self.shutdown()
        print("PythonSseMCPServerHoster exited")

async def chat_loop_with_mcp(mcp_clients: list[MCPClient]):
    app_logger = get_logger("assistant_agent_app", level=logging.INFO)
    agent = AssistantAgent(
        name="PersonalAssisant",
        instructions="You are helpful assistant",
        mcps = mcp_clients,
        logger = app_logger,
        verbose = True)
    
    print(agent.get_tool_list_str())
    
    print("Welcome to MCP Test. Start with something like 'what can you do?'")
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            print("Please Enter Something")
            continue
        if user_input.lower() == "exit":
            break
        # request stream response
        response_stream:AgentStream = await agent.run(user_input, stream=True)
        async for chunk in response_stream:
            if not chunk.done: # token by token chunks of reasoning steps and answer
                print(chunk.text, end="", flush=True) 
            else:              # the final answer
                print(f"\n\033[1;32;40mAI: {chunk.text}\033[0m")

async def async_main() -> None:
    # Remote MCP via SSE
    # name a prefix to avoid duplicated tool names from multiplte mcp clients
    mcp_sse_client = MCPClientSse(
        name = "SSE MCP Server, via Python",
        mcp_tool_prefix = "sse_",
        params = {
            "url": "http://localhost:9000/sse",
        })
    # Local MCP through StdIO
    mcp_stdio_client = MCPClientStdio(
        name = "StdIO MCP Server, via Python",
        mcp_tool_prefix = "stdio_",
        params = {
            "command": "python",
            "args": ["sample_mcp_server_stdio.py"],
        })
    
    mcp_server = PythonSseMCPServerHoster(file="sample_mcp_server_sse.py", port=9000)

    # Use async with to start all mcp resources before agent can access mcp
    async with mcp_server, mcp_sse_client, mcp_stdio_client:
        await chat_loop_with_mcp([mcp_sse_client, mcp_stdio_client]) # can be mutiple mcp clients, but tools name cannot be duplicated.

if __name__ == "__main__":
    asyncio.run(async_main())