import asyncio
from typing import Dict, Optional, Any, ClassVar, List, cast
from collections.abc import AsyncGenerator, AsyncIterator
from abc import ABC, abstractmethod
from pydantic import BaseModel
from enum import Enum
import traceback
from .chatclient import (
    BaseChatClient,
    OpenAIChatClient,
    Prompt,
    Options,
    FunctionTool,
    function_tool,
    Chunk as RawChunk,
    Response,
    ResponseStream,
    ChatContext,
    MessageType,
    prompt_to_message
    )
from .tools import transform_string_function_style, __CTX_NAME__
from .utils.logger import logging, get_global_logger
from .utils import multitask, identity, formatter
from .default import *
from .exceptions import *
from .chatclient import APIConnectionError, BadRequestError, RateLimitError, OpenAIError
from .mcp import *

MAX_STEPS:Optional[int] = OPENAGENT_MAX_STEPS
# =============================================================================
# Memory Management Classes
# TO-DO: persist in DB
# =============================================================================
MAX_IN_MEMORY_MESSAGES = 50
class Memory:
    """
    Implements a In-cahe memory for agents.
    """
    def __init__(self, 
                 messages:Optional[List[MessageType]] = None) -> None:
        """
        cache: the memory in cache
        persist_id: the persist ID (datatable ID) in DB
        """
        # make a copy of message list
        self.cache: Optional[List[MessageType]] = list(messages) if messages else []

    def add(self, message: MessageType) -> None:
        """
        Add a new message to memory. If short-term exceeds 10 entries,
        archive the oldest messages.
        """
        self.cache.append(message)
        # only keep MAX_IN_MEMORY_MESSAGES items
        if len(self.cache) > MAX_IN_MEMORY_MESSAGES:
            self.cache.pop(0)

    def get_messages(self) -> List[MessageType]:
        """
        Returns the full conversation context as a list as:
        [
            {
                "role": ...
                "content":...
            }
        ]
        """
        all_messages = list(self.cache) # make a copy of current memory in cache
        return all_messages
    
    def get_messages_str(self) -> str:
        """
        Returns the full conversation context as a formatted string as:
        '''
        [user]:%content%,
        [assistant]:%content%
        '''
        """
        all_messages = self.cache
        return "\n".join([f"[{msg['role']}]: {msg['content']}" for msg in all_messages])
    
    # Search context from memory (from both cache and persistence)
    def search_context(term:str, top_k:Optional[int]=5)->List[MessageType]:
        pass

class AgentContext(ChatContext):
    @property
    def context(self)->dict:
        return self.model_dump(exclude=["messages"])

class Rejection(BaseModel):
    error_code:Optional[ErrorCode] = ErrorCode.Unknown
    text:Optional[str] = None # rejection reason
    def __str__(self)->str:
        return self.text or ""

class Chunk(BaseModel):
    class Type(Enum):
        Step  = 1 # reasoning step
        Answer = 2 # final answer
        Reject = 3 # rejected by agent, check text for rejection reason, error_code for rejection code.
    type:Optional[Type] = Type.Step
    text:Optional[str] = None # step, answer or rejection reason
    error_code:Optional[ErrorCode] = ErrorCode.NoError
    done:Optional[bool] = False # done==False means step, True means final answer

AgentStream = AsyncGenerator[Chunk, None]
class Answer(BaseModel):
    think:Optional[str] = None
    final:Optional[str] = None
    # return final answer by string formatting
    def __str__(self)->str:
        return self.final or ""

class AgentResponse(BaseModel):
    class Type(Enum):
        Answer = 1 # answer consists of thinking step (response.answer.think) and final answer (response.answer.final).
        Reject = 2 # rejected by agent

    type:Optional[Type] = Type.Answer
    answer:Optional[Answer] = None
    rejection:Optional[Rejection] = None # if not None, it's means the requst is rejected
    # return final answer or rejection details by string formatting
    def __str__(self)->str:
        return str(self.answer) if self.answer is not None else str(self.rejection)

# Base class of Agent
"""Initialize with arguments:
- llm_client (BaseChatClient): the api client of LLM model, support Openai/Azure OpenAI, Ollama (OpenAI compatible), Vllm (OpenAI compatible)
  -- Currently, use OpenAIChatClient for OpenAI/AzureOpenAI/Ollama/Vllm LLM models.
- name (str): the agent name or the agent role. 
  -- Recommand to use a meaningful English characters without space to represent the agent role, e.g., CustomerServiceAgent, SalesAgent.
  -- It will be used in agentic workflows and and agent tools.
- description (str): describe what does the agent do in one line. It will be used in agentic workflows and agent tools.
- tools (list[FunctionTool]): user defined or pre-defined the tool functions called by LLM.
  -- use decorator @function_tool to define a function, or make_function_tool to define it. refer package to openagent.tools
- max_steps (int=30): max reasoning steps to complete a task in one agent.run(...) to avoid dead loop.
  -- note: agents typically need take multiple reasoning steps, such as mutiple tool calls step by step or reasoning in a loop until meet a exit criteria.
- logger (logging.Logger=None): log to a "xxx.log" file or on screen. If not set, only log to screen. 
- verbose (bool=False): turn on detailed logging on screen (for debugging).
"""
class BaseAgent(ABC):
    def __init__(self,
                 llm_client:BaseChatClient,
                 name:Optional[str] = "Agent",
                 description:Optional[str] = "", # describe what the agent do, for agent routing
                 instructions:Optional[str] = "You are a helpful agent that assist user.", # the rules (system prompt of the agent
                 tools: Optional[list[FunctionTool]] = [],
                 mcps: Optional[list[MCPClient]] = [], 
                 max_steps:int|None = OPENAGENT_MAX_STEPS,
                 logger:Optional[logging.Logger] = None,
                 verbose:bool|None=False) -> None:
        self.llm_client = llm_client
        self.name = name
        self.description = description
        self.instructions = instructions
        self.max_steps = max_steps
        self.tools:Optional[list[FunctionTool]] = []
        self.tools_map:Optional[Dict[str, FunctionTool]] = {}
        self.register_tools(tools)
        self.mcps = mcps
        self.mcp_tools:Optional[list[FunctionTool]] = []
        self.memory = Memory()
        self.logger:Optional[logging.Logger] = logger or get_global_logger()
        self.verbose = verbose
        self.interrupted = False
        self.is_executing = False
        if self.llm_client: # None for workflow agents
            self.llm_client.verbose = verbose

        self.log(f"{self.name} is initialized", logging.DEBUG)
        # for context manager
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()

    # to be overriden by subclass
    @property
    def system_prompt(self)->str:
        return self.instructions

    # Run agent with user input,context(Optional) and return response in one-shot or in stream
    async def run(self,
                  input:str|Prompt|List[MessageType],
                  context:Optional[AgentContext] = None,
                  stream:Optional[bool]=False,
                  throw_exception:Optional[bool]=False) -> AgentResponse|AgentStream:
        """
        Processes the user's input by iteratively building a prompt from memory,
        sending it to the LLM asynchronously, and checking for tool actions or a final answer.
        Parameters:
            - task (str): The user's task.
            - context (AgentContext): the context in the run
            - stream (bool): use non-stream mode no stream mode
            - throw_exception (bool): whether throw AgentException, default is true
        Returns:
            - str: The final answer extracted from the LLM's response.
            - AsyncGenerator[Chunk]: the chunk of intermidiate step or final answer
        """
        # handle LLM exception for streamed response and relay the chunks
        async def _generate_rejection(ae:AgentException)->AgentStream:
            yield Chunk(type=Chunk.Type.Reject, done=True, text = str(ae), error_code=ae.error_code)

        async def _generate_stream(response_stream:AgentStream)->AgentStream:
            try:
                async for chunk in response_stream:
                    yield chunk
            except Exception as e:
                ae = self._preprocess_exception(e) # convert to agent exception
                if throw_exception:
                    raise ae from e
                else:
                    yield Chunk(type=Chunk.Type.Reject, done=True, text = str(ae), error_code=ae.error_code)
        
        # context is required to store in-run conversations for function calls
        if context is None:
            context = AgentContext()
        # if the input is a list of messages, use the message list as memory
        if isinstance(input, list):
            self.memory = Memory(messages = input)
            self.log(f'run agent with message list as ipnut. #items:{len(input)}, context:{context.context}', logging.DEBUG)
            input = None # empty the input, leverage all messages in memory
        elif isinstance(input, str):
            input = Prompt(text=input)
            self.log(f'run agent with text input. text:"{input.text}", context:{context.context}', logging.DEBUG)
        elif isinstance(input, Prompt):
            text = input.text or ""
            images_count = len(input.images) if input.images else ""
            audio_size = len(input.audio) if input.audio else ""
            self.log(f'run agent with prompt input. text:"{text}", #images:{images_count}, audio_size:{audio_size}; context:{context.context}', logging.DEBUG)

        try:
            # delay loading mcp tools
            await self._register_all_mcp_tools_async() # load all mcp tools at the runing phase
            rejection:Optional[Rejection] = None
            if not stream:
                think_steps:list[str] = []
                final_answer = ""
                response_stream:AgentStream = self._run_impl(input, context=context)
                async for chunk in response_stream:
                    if not chunk.done:
                        think_steps.append(chunk.text)
                    if chunk.done:
                        final_answer = chunk.text
                return AgentResponse(type = AgentResponse.Type.Answer,
                                     answer = Answer(think = '\n'.join(think_steps), final=final_answer))
            else:
                return _generate_stream(self._run_impl(input, context=context))
        except Exception as e: # unexpected exception
            ae = self._preprocess_exception(e)
            # throw exception to upstream if throw_exception is True
            if throw_exception: 
                raise ae from e
            else: # otherwise, return AgentResponse. note, only the exception of non-streamed response come to here
                if not stream:
                    rejection = Rejection(error_code=ae.error_code, text=str(ae))
                    return AgentResponse(type = AgentResponse.Type.Reject,
                                         rejection = rejection)
                else:
                    return _generate_rejection(ae)

    # the implementation method of `agent.run` to be overriden by subclass agent
    ## return a series of chunks of executed step
    @abstractmethod
    async def _run_impl(self, task:Prompt, context:Optional[AgentContext]) -> AgentStream:
        pass

    def add_message(self, message:MessageType)->None:
        self.memory.add(message)

    def add_message_list(self, messages:List[MessageType])->None:
        for message in messages:
            self.memory.add(message)

    def get_messages(self)->List[MessageType]:
        return self.memory.get_messages()
        
    # add user query
    def add_user_message(self, content:str|Prompt)->None:
        user_prompt = content or ''
        if isinstance(user_prompt,str):
            user_prompt = Prompt(text = user_prompt)
        message = prompt_to_message(user_prompt)
        self.add_message(message)
    # add agent response
    def add_agent_message(self, content:str)->None:
        self.memory.add({"role":"assistant", "content":content})

    # add context before executing task, add to user message
    def add_context(self, content:str|Prompt)->None:
        self.add_user_message(content)
    
    # convert exception to AgentException and log the exception
    def _preprocess_exception(self, e:Exception)->AgentException:
        # convert exception to AgentException
        if isinstance(e, APIConnectionError):
            e = AgentException(details=str(e), error_code=ErrorCode.LlmAccessError, module=self.name)
        elif isinstance(e, RateLimitError):
            e = AgentException(details=str(e), error_code=ErrorCode.LlmAccessError, module=self.name)
        elif isinstance(e, BadRequestError):
            e = AgentException(details=str(e), error_code=ErrorCode.InvalidInput, module=self.name)
        elif isinstance(e, OpenAIError):
            e = AgentException(details=str(e), error_code=ErrorCode.LlmAccessError, module=self.name)
        elif isinstance(e, AgentException):
            pass
        else:
            e = AgentException(details=str(e), error_code=ErrorCode.Unknown, module=self.name)
        # log the exception
        self.log(f'!!!Caught agent exception. module:{e.module}, error_code:{e.error_code}, details:{str(e)}', level=logging.ERROR)
        if self.verbose: # log trace back in verbose mode for debugging
            tb_str = traceback.format_exc()
            self.log(f"  - Traceback: {tb_str}", level=logging.ERROR)
        # return AgentException
        return e
    
    async def call_llm(self,
                       messages:Optional[List[MessageType]], # the message list including user prompt and conversation history
                       system:Optional[str]=None,
                       context:Optional[AgentContext] = AgentContext(),
                       options:Optional[Options] = None,
                       stream:Optional[bool] = False) -> str|AsyncGenerator[RawChunk]:
        
        # replace place holders "{{...}}" with context state variables
        if system is not None \
            and context is not None \
            and context.context:
            system = formatter.format_template_with_json(system, context.context)
            
        response = await self.llm_client.send(
            prompt=messages,
            system=system,
            context=context,
            options=options,
            stream=stream)
        if not stream:
            return response.text or ""
        else:
            return response
    
    # raise exception if found duplicate tool names
    def register_tools(self, tools: list[FunctionTool]):
        for tool in tools:
            if tool.name not in self.tools_map.keys():
                self.tools_map[tool.name] = tool
                self.tools.append(tool)
            else:
                raise AgentException(
                    f"Duplicate tool names found. tool:{tool.name}, agent:{self.name}",
                    error_code = ErrorCode.ModelBehaviorError,
                    module=self.name
                    )

    def get_tools(self):
        return self.tools
    
    # only load mcp tools once
    async def _register_all_mcp_tools_async(self)->list[FunctionTool]:
        if self.mcps and not self.mcp_tools: # if there are mcps
            self.mcp_tools = await MCPAccessUtil.get_all_function_tools(self.mcps)
            self.register_tools(self.mcp_tools)
    
    def get_tool_list_str(self):
        tool_list = []
        tool_names = []
        for function in self.tools:
            tool_names.append(function.name)
            tool_str = f'— **{function.name}**:\n  - Description: {function.description}\n  - Input Arguments and Types: {function.input_arguments}\n'
            tool_list.append(tool_str)
        return '\n'.join(tool_list)

    async def _invoke_function(self, function:FunctionTool, **arguments):
        if not function.is_async: # sync function
            result = function(**arguments)
        else:
            result = await function(**arguments) # async function
        return result
    
    async def call_function(self, name:str, arguments:dict, context:AgentContext)->str|object:
        name = name or ""
        arguments = arguments or {}
        self.log(f"-> call function `{name}` with {arguments}", logging.DEBUG)
        result = None
        if name in self.tools_map.keys():
            # Run the tool in a thread to avoid blocking the event loop.
            function = self.tools_map[name]
            # inject context:ContextType as the message history before the function call as context
            if function.has_context_argument():
                # inject last message to context, but remove the function call messsage
                context.messages = self.memory.get_messages()
                arguments[__CTX_NAME__] = context
            result = await self._invoke_function(function, **arguments)
        else:
            result = f'**Error**: Unknown tool name "{name}"'
            self.log(result, logging.DEBUG)

        self.log(f"<- call result: {result}", logging.DEBUG)
        return result
    
    def as_tool(
        self,
        tool_name: str | None = None,
        tool_description: str | None = None,
        stateful:bool | None = False,
        # - True: the agent tool keep memory for each run
        # - False (default): the agent tool forget memory for each run
        with_parent_context: bool | None = False
        # - True: The agent tool use conversations history of parent agent
        # - False (default): The agent tool doesn't realy on any conversation history of parent agent.
    ) -> FunctionTool:
        @function_tool(
            name=tool_name or transform_string_function_style(self.name),
            description=tool_description or self.description or ""
        )
        async def run_agent(input:str, context:AgentContext) -> str: # the actual agent as tool
            """Call the tool with string input return string output.
            Parameters:
                input (str): input to the tool
            Returns (str): the call result
            """
            self.log(f'run agent_tool with input:"{input}", context:{context.context}', logging.DEBUG)
            if not stateful: # **empty** memory for the run if not stateful
                self.memory = Memory()

            if with_parent_context: # erase memory and **inject** conversations of parent agent to the memory
                self.memory = Memory(messages=context.messages[:-1]) # **remove** the last message which is tool call message
            result = await self.run(input, context=context, stream=False)

            # clear the memory for the run if not stateful or if the agent use conversations of parent agent
            if not stateful or with_parent_context:
                self.memory = Memory()
            return str(result) # this will return text of final answer or rejection
        return run_agent
    
    # Helper method to log information in log file (if self.logger is provided) and on screen (if self.verbose==True).
    def log(self, message:str, level:int = logging.INFO):
        self.get_logger().log(level, f"[{self.name}] {message}")
        if self.verbose:
            print(f"[{self.name}] {message}")

    def get_logger(self)->logging.Logger:
        return self.logger

    # context management
    async def __aenter__(self):
        await self.startup()

    async def __aexit__(self, exc_type, exc, tb):
        """cleanup all resources"""
        pass # for future implemenation
    # clean up all resources by user or context manager of agent
    async def startup(self)->None:
        """Startup all resources."""
        self.log("startup", logging.INFO)
        await self._register_all_mcp_tools_async()

    async def cleanup(self)->None:
        """Cleanup all resources."""
        self.log("cleanup", logging.INFO)
        async with self._cleanup_lock:
            pass
        """
            try:
                if self.mcps:
                    tasks = [mcp.cleanup() for mcp in self.mcps]
                    await asyncio.gather(*tasks)
                    self.mcps = []
                    self.mcp_tools = []
                self.tools = []
            except Exception as e:
                self.get_logger().error(f"Error cleaning up server: {e}")
        """
        