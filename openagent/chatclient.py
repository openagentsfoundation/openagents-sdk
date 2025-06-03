from .utils import env
import json, asyncio
from urllib.parse import urlparse
import re
from openai import AsyncAzureOpenAI, AsyncOpenAI, NOT_GIVEN
from openai import APIConnectionError, BadRequestError, RateLimitError, OpenAIError
from collections.abc import AsyncGenerator
from typing import Literal, Optional, override, Dict, List, Any, cast
from pydantic import BaseModel
from abc import ABC, abstractmethod
from .utils.formatter import AtomWordReader
#Azure OpenAI GPT-4o by default
from .default import (
    OPENAGENT_LLM_ENDPOINT,
    OPENAGENT_LLM_API_KEY,
    OPENAGENT_LLM_DEPLOYMENT,
    OPENAGENT_LLM_API_VERSION,
    OPENAGENT_LLM_TEMPERATURE,
    OPENAGENT_LLM_TOP_P,
    OPENAGENT_LLM_IS_REASONING,
    OPENAGENT_LLM_REASONING_EFFORT
)

from .tools import FunctionTool, function_tool, make_function_tool, __CTX_NAME__, ContextType

SYSTEM_PROMPT = "You are helpful assistant that helps user find information"

class Prompt(BaseModel):
    text: Optional[str] = ""
    images: Optional[list[str]] = None # input images
    audio: Optional[str] = None # input audio, base64 encoded

MAX_ITERATIONS = 5
class Options(BaseModel):
    voice:Optional[str] = None  # voice name, #alloy
    voice_mode:Optional[bool] = False
    session:Optional[dict] = None # the session of the context
    max_iterations:Optional[int] = MAX_ITERATIONS # call tools within steps < max_iterations-1
    response_format: Optional[dict] = None

class Function(BaseModel):
    name: str
    arguments: str # arguments in json. not, it's maybe corrupted. need handle exeception case

class Response(BaseModel):
    text: str
    audio: Optional[str] = None # input audio, base64 encoded

class Chunk(BaseModel):
    text:str = None
    audio: Optional[str] = None # input audio, base64 encoded
    done:bool = False
    function_call: Optional[Function] = None # {"name":..., "arguments":...}

ResponseStream = AsyncGenerator[Chunk]

class ClientResponse(BaseModel):
    text: Optional[str] = None
    audio: Optional[str] = None # output audio, base64 encoded
    tool_calls: Optional[list[object]] = None
    raw_response: Optional[object] = None

MessageType = Dict[str, Any]

class ChatContext(ContextType):
    messages: Optional[List[MessageType]] = None # the conversation history so far in the call

# helper function to convert prompt to user message
def prompt_to_message(prompt:Prompt)->object:
    if not prompt.images and not prompt.audio: # simple text prompt
        return {"role":"user", "content":str(prompt.text)}
        
    message = {
        "role":"user",
        "content":[]
    }
    if prompt.text:
        message["content"].append({"type":"text", "text":str(prompt.text)})
    if prompt.images:
        for image in prompt.images:
            message["content"].append({"type":"image_url", "image_url":{"url":image}})
    if prompt.audio:
        message["content"].append({"type":"input_audio", "input_audio":{"data":prompt.audio, "format":"wav"}})
    return message 

"""
Base class to for LLM model.
"""
class BaseChatClient(ABC):
    def __init__(self,
                 system_prompt: Optional[str] = "You are helpful assistant that helps user find information.",
                 end_point: Optional[str] = None,
                 model: Optional[str] = None,
                 api_key: Optional[str] = None,
                 api_version: Optional[str] = None,
                 tools: Optional[List[FunctionTool]] = [],
                 tool_choice: Optional[Literal["auto", "required"]] = "auto",
                 modalities:Optional[List[str]] = None,
                 temperature:Optional[float] = None,
                 top_p:Optional[float] = None,
                 frequency_penalty: Optional[float]  = None,
                 presence_penalty: Optional[float] = None,
                 verbose:Optional[bool] = False
                 ):
        super().__init__()
        self.system_prompt = system_prompt
        self.end_point = end_point
        self.model = model
        self.api_key = api_key
        self.api_version = api_version
        self.tool_choice = tool_choice
        self.modalities = modalities
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.verbose = verbose
        self.tool_callbacks: Optional[Dict[str, FunctionTool]] = None
        for tool in tools:
            self.register_tool(tool)

    @abstractmethod
    def clone(self, with_tools:bool|None=False)->ABC:
        pass
    # override for different LLM models
    
    def register_tool(self, tool:FunctionTool) -> None:
        """
        Register a tool by using its function name as the key.
        """
        if self.tool_callbacks is None:
            self.tool_callbacks = {}
        if self.tool_callbacks.get(tool.name) is None:
            self.tool_callbacks[tool.name] = tool

    def get_tools(self)->list[FunctionTool]:
        return [self.tool_callbacks[key] for key in self.tool_callbacks]

    @abstractmethod
    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def call_llm_model(self,
                             messages:list,
                             options:Options | None = None,
                             call_tool:bool | None = True,
                             stream:bool=False)->ClientResponse:
        pass

    async def call_function(self,
                            tool:FunctionTool, 
                            arguments:str,
                            context:Optional[ChatContext] = None):
        # validate
        if not tool:
            if self.verbose:
                print("**Warning**: empty tool")
            return '[ERROR]: tool not exist'
        try:
            js_arguments = json.loads(arguments)
        except Exception as e:
            if self.verbose:
                print(f"**Warning**: failed to parse function arguments. function:{tool.name}, arguments:{str(arguments)}")
            return f'[ERROR]: failed to parse tool arguments text as json. arguments:`{str(arguments)}`, fail_reason:{str(e)}'
        # call function
        if self.verbose:
            print(f"-> call function `{tool.name}` with {arguments}")

        # inject context to function call
        if context is not None and tool.has_context_argument():
            js_arguments[__CTX_NAME__] = context

        if not tool.is_async: # sync function
            result = tool(**js_arguments)
        else:
            result = await tool(**js_arguments) # async function

        if self.verbose:
            print(f"<- call result: {result}")
        return result

    """Send prompt and system prompt, returns one-shot response or stream of chunks.
    - Inputs:
      @prompt:
        - type:str: text message
        - type:prompt: multi-modal message
        - type:ContextType: full message list including conversation history and the current message
      @options:
        

    """
    # Send prompt and system prompt, returns one-shot response or stream of chunks:
    ## 
    async def send(self,
                   prompt:Optional[str|Prompt|List[MessageType]] = None,
                   system:Optional[str] = None,
                   options:Optional[Options] = None,
                   context:Optional[ChatContext] = ChatContext(),
                   stream:bool| None = False)->Response|ResponseStream:
        messages = [{"role":"system", "content":system or self.system_prompt}]
        if isinstance(prompt, Prompt):
            messages.append(prompt_to_message(prompt)) # add to message list sent to openai
        elif isinstance(prompt, list): # a list of chat conversation
            messages.extend(prompt)
        else:
            messages.append({"role":"user", "content":str(prompt)})

        if not stream:
            return await self.do_send(messages, options=options, context=context)
        else:
            return self.do_send_stream(messages, options=options, context=context)

    # send with completion response, to be overriden
    @abstractmethod
    async def do_send(self,
                      messages:list,
                      options:Optional[Options] = None,
                      context:Optional[ChatContext]=None)->Response: # deployment model
        pass

    # send with stream response, to be overriden
    @abstractmethod
    async def do_send_stream(self,
                             messages:list,
                             options:Optional[Options] = None,
                             context:Optional[ChatContext]=None)->AsyncGenerator[Chunk]:
        pass

class OpenAIChatClient(BaseChatClient):
    def __init__(
            self,
            system_prompt: Optional[str] = "You are helpful assistant that helps user find information.",
            end_point: Optional[str] = OPENAGENT_LLM_ENDPOINT,
            model: Optional[str] = OPENAGENT_LLM_DEPLOYMENT,
            api_key: Optional[str] = OPENAGENT_LLM_API_KEY,
            api_version: Optional[str] = OPENAGENT_LLM_API_VERSION,
            tools: Optional[List[FunctionTool]] = [],
            tool_choice: Optional[Literal["auto", "required"]] = "auto",
            modalities:Optional[List[str]] = None,
            temperature:Optional[float] = OPENAGENT_LLM_TEMPERATURE,
            top_p:Optional[float] = OPENAGENT_LLM_TOP_P,
            frequency_penalty: Optional[float]  = None,
            presence_penalty: Optional[float] = None,
            is_reasoning:Optional[bool] = OPENAGENT_LLM_IS_REASONING,
            reasoning_effort:Optional[Literal["low", "medium", "high"]] = OPENAGENT_LLM_REASONING_EFFORT,
            verbose:Optional[bool] = False):
        super().__init__(
            system_prompt = system_prompt,
            end_point = end_point,
            model = model,
            api_key = api_key,
            api_version = api_version,
            tools = tools,
            tool_choice = tool_choice,
            modalities = modalities,
            temperature = temperature,
            top_p = top_p,
            frequency_penalty = frequency_penalty,
            presence_penalty = presence_penalty,
            verbose = verbose
        )
        self.is_reasoning = is_reasoning
        self.reasoning_effort = reasoning_effort
        if self.is_reasoning:
            self.temperature = None
            self.top_p = None
        
    def get_client(self)->AsyncOpenAI:
        client = getattr(self, '_client', None)
        # support OpenAI/Azure OpenAI and Ollama/vllm API compatible
        if not client:
            parsed_url = urlparse(self.end_point)
            pattern = re.compile(r"^/v\d+(?:/|$)") # to match whehter it's "/v1...n"
            match = pattern.match(parsed_url.path or '')
            if not match: # Azure OpenAI endpoint
                client = self._client = AsyncAzureOpenAI(azure_endpoint = self.end_point,
                                                         api_key= self.api_key,
                                                         api_version= self.api_version)
            else: # OpenAI or Ollama/vllm openai api comptabile endpoint
                client = self._client = AsyncOpenAI(base_url = self.end_point,
                                                    api_key= self.api_key)
        return client
    
    # override from base class
    def clone(self, with_tools:bool|None=False)->BaseChatClient:
        return OpenAIChatClient(
            system_prompt = self.system_prompt,
            end_point = self.end_point,
            model = self.model,
            api_key = self.api_key,
            api_version = self.api_version,
            tools = [v for _,v in self.tool_callbacks.items()] if with_tools and self.tool_callbacks else [],
            tool_choice = self.tool_choice,
            modalities = self.modalities,
            temperature = self.temperature,
            top_p = self.top_p,
            frequency_penalty = self.frequency_penalty,
            presence_penalty = self.presence_penalty,
            is_reasoning = self.is_reasoning,
            reasoning_effort = self.reasoning_effort,
            verbose = self.verbose
        )

    # override method BaseChatClient
    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Return the list of tool definitions in the format expected by the OpenAI API.
        """
        tool_defs = [self.tool_callbacks[key].json_schema for key in self.tool_callbacks]
        return tool_defs

    # override method
    async def call_llm_model(self,
                             messages:list,
                             options:Options | None = None,
                             call_tool:bool | None = True,
                             stream:bool=False)->ClientResponse:
        client = self.get_client()

        def _get_value(obj, attribute:str):
            value = getattr(obj, attribute, NOT_GIVEN) if obj else NOT_GIVEN
            return value if value is not None else NOT_GIVEN
        
        use_tools = call_tool and not (not self.tool_callbacks)
        completion = await client.chat.completions.create(
            model = self.model,
            messages = messages,
            modalities = _get_value(self, "modalities"),
            audio = {"voice": options.voice, "format": "wav"} if (self.modalities and "audio" in self.modalities and options.voice) else NOT_GIVEN,
            tools = self.get_tool_definitions() if use_tools else NOT_GIVEN,
            tool_choice = self.tool_choice if use_tools else NOT_GIVEN,
            stream = stream,
            reasoning_effort = _get_value(self, "reasoning_effort") if self.is_reasoning else NOT_GIVEN,
            temperature = _get_value(self, "temperature") if not self.is_reasoning else NOT_GIVEN,
            top_p = _get_value(self, "top_p") if not self.is_reasoning else NOT_GIVEN,
            frequency_penalty = _get_value(self, "frequency_penalty") if not self.is_reasoning else NOT_GIVEN,
            presence_penalty = _get_value(self, "presence_penalty") if not self.is_reasoning else NOT_GIVEN,
            response_format = _get_value(options, "response_format")
            )

        oai_response = ClientResponse(raw_response=completion)
        if not stream:
            message = completion.choices[0].message
            if message.tool_calls:
                oai_response.tool_calls = message.tool_calls
            elif message.audio:
                oai_response.text = message.audio.transcript
                oai_response.audio = message.audio.data
            else:
                oai_response.text = message.content
        else:
            async for chunk in completion:
                if (len(chunk.choices)==0): #skip the header
                    continue
                delta = chunk.choices[0].delta
                # the first chunk may have content for some OpenAI API compatible endpoint implementation
                oai_response.text = delta.content or '' 
                tool_calls = delta.tool_calls
                if tool_calls:
                    tool_call = tool_calls[0]
                    if tool_call.function.name: # append
                        oai_response.tool_calls = oai_response.tool_calls or []
                        oai_response.tool_calls.append(tool_call.model_copy(deep=True)) # must use deep copy
                        oai_response.tool_calls[-1].function.arguments = ''
                    if tool_call.function.arguments:
                        # the arguments string returns character by character
                        oai_response.tool_calls[-1].function.arguments += tool_call.function.arguments
                elif tool_calls is None and delta.content is None: #skip empty chunk
                    continue
                elif chunk.choices[0].finish_reason != 'tool_calls':
                    #not the end chunk of tool calls, should be start chunk of message content
                    break #jump out of the loop to leave the outsider methon to handle stream content
        return oai_response
    
    # helper method to convert function call to LLM message
    def call_result_to_message(self, call:object, call_result:str)->dict:
        role = "tool" if not self.is_reasoning else "user"
        return {"tool_call_id":call.id, "role":role, "content":call_result}
    
    # send with completion response 
    # override from base class
    async def do_send(self,
                      messages:list,
                      options:Optional[Options] = None,
                      context:Optional[ChatContext]=None)->Response: # deployment model
        # call functions for steps < max_reason_steps-1 
        max_iterations = options.max_iterations if options else MAX_ITERATIONS
        if context: # context store all newly generated message from LLM, including function calls and text response
            context.messages = messages[1:] # store the conversation to context, skip system prompt
        for step in range(max_iterations):
            call_tool = step < max_iterations-1 # last step should not call tool
            client_response = await self.call_llm_model(messages,options,call_tool=call_tool,stream=False)
            if client_response.tool_calls:
                # handle tool calls
                ## step-1: add responded message to message list
                call_message = ({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [t.model_dump() for t in client_response.tool_calls]
                })
                messages.append(call_message)
                if context:
                    context.messages.append(call_message)
                ## step-2: add call function and add response to message list
                calls = [self.call_function(self.tool_callbacks.get(toolCall.function.name), 
                                            toolCall.function.arguments,
                                            context=context)
                        for toolCall in client_response.tool_calls]
                callResults = await asyncio.gather(*calls) # make function call
                for (toolCall, callResult) in zip(client_response.tool_calls, callResults):
                    result_message = self.call_result_to_message(toolCall, callResult)
                    messages.append(result_message)
                    if context:
                        context.messages.append(result_message)
            else:
                break # return the result
        
        if context:
            context.messages.append({"role":"assistant", "content":client_response.text})
        return Response(text = client_response.text, audio = client_response.audio)
    # send with stream response
    # override from base class
    async def do_send_stream(self,
                             messages:list,
                             options:Optional[Options] = None,
                             context:Optional[ChatContext]=None)->AsyncGenerator[Chunk]:
        # call functions for steps < max_reason_steps-1
        max_iterations = options.max_iterations if options else MAX_ITERATIONS
        if context: # context store all newly generated message from LLM, including function calls and text response
            context.messages = messages[1:] # store the conversation to context, skip system prompt
        for step in range(max_iterations):
            call_tool = step < max_iterations-1 # last step should not call tool
            client_response = await self.call_llm_model(messages, options,call_tool=call_tool,stream=True)
            if client_response.tool_calls:
                # handle tool calls
                ## step-1: add responded message to message list
                call_message = ({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [t.model_dump() for t in client_response.tool_calls]
                })
                messages.append(call_message)
                if context:
                    context.messages.append(call_message)

                for toolCall in client_response.tool_calls:
                    yield Chunk(function_call=Function(name=toolCall.function.name, arguments=toolCall.function.arguments))
                ## step-2: add call function and add response to message list
                if context:
                    context.messages = messages[1:] # store the conversation to context
                calls = [self.call_function(self.tool_callbacks.get(toolCall.function.name), 
                                            toolCall.function.arguments,
                                            context=context)
                        for toolCall in client_response.tool_calls]
                callResults = await asyncio.gather(*calls) # make function call
                for (toolCall, callResult) in zip(client_response.tool_calls, callResults):
                    result_message = self.call_result_to_message(toolCall, callResult)
                    messages.append(result_message)
                    if context:
                        context.messages.append(result_message)
            else:
                break # return the result

        client_response.text = client_response.text or ''
        # yield the remaining text word by word (rather than token by token)
        atom_w_reader = AtomWordReader()
        if client_response.text:
            # yield word by word, rather than token by token. one token may have multiple words.
            #yield Chunk(text=client_response.text, done = False)
            for word in atom_w_reader.emit_atoms(client_response.text):
                yield Chunk(text=word, done = False)

        async for chunk in client_response.raw_response:
            if len(chunk.choices) == 0: #skip the head
                continue
            if chunk.choices[0].finish_reason == 'content_filter':
                yield Chunk(text='', done=True) # return empty string if filtered.
                return
            chunk_message = chunk.choices[0].delta.content
            if chunk_message == '': #skip the first empty character ''
                continue
            if chunk_message is None: # for the convience to handle message in the last chunk
                chunk_message = ''

            client_response.text += chunk_message
            # yield word by word, rather than token by token
            # this will remove the dependancy on tokenizer and make downstream text parser easier.
            if chunk_message:
                #yield Chunk(text=chunk_message, done = False)
                for word in atom_w_reader.emit_atoms(chunk_message):
                    yield Chunk(text=word, done = False)
        if context:
            context.messages.append({"role":"assistant", "content":client_response.text})
        yield Chunk(text=client_response.text, done=True)

