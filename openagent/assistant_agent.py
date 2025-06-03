"""
A simple assistant agent based on OpenAIChatClient.
Supporting tool calls enabled by llm endpoint, such as openai and OpenAI API compatibily endpoint of ollama, vllm
"""
import json
from typing import Optional, override, Any, Dict, List, Literal
from pydantic import BaseModel, ValidationError
from datetime import datetime
# OpenAI chat model
from .base_agent import (
    BaseAgent,
    Prompt,
    FunctionTool,
    AgentContext,
    MessageType,
    MAX_STEPS,
    BaseChatClient,
    OpenAIChatClient,
    Options,
    Chunk,
    AgentStream,
    ResponseStream,
    logging,
)
from .utils import format_template_with_json
from .exceptions import *
from .mcp import *

SYSTEM_PROMPT_TEMPLATE = """# Instructions
<instructions>
{{instructions}}
</instructions>

# Agent name: {{agent_name}}.
# The time now: {{time_now}}.

# Language Rule
- Always reply in the same language as the user's original problem statement or task, as the **working language** 
- Natural language arguments in tool calls must be in the **working language**

# Tools:
You can use following tools to take action or query information for solving problems:
<tools>
{{tools}}
</tools>

{{handoff_section}}
"""

# the system prompts for actor mode (implement tools in agent)
SYSTEM_PROMPT_TEMPLATE_ACTOR = """# Instructions:
<instructions>
{{instructions}}
</instructions>

# Agent name: {{agent_name}}.
# The time now: {{time_now}}.

# Working Language: English (default)
- Use the language of user task as the working language
- Natural language arguments in tool calls must be in the working language

# Tools:
You can use following tools to take action or query information for solving problems:
<tools>
{{tools}}
</tools>

{{handoff_section}}

# Reasoning and Solving Chain-of-Thoughts (CoT):
<chain_of_thought>
Before answering, if user's query is a complex task, think in your mind to breakdown the complex task into actionable sub-tasks.
When answering, strictly follow below steps (and data json data schema below section) to resolve tasks:
- action: If you decide to use a tool, output tool call as `{"action": {"name":"...", "arguments":{...}}}` in json object.
- action_result: After a tool is executed, call result is returned as `{"action_result": "..."}` in json object.
... (repeat action/action_result more times, if you need more tool calls to conclude an answer.)
- answer: finally when you find enough proof from action_result to answer user's query, summarize an answer and output as `{"answer": "..."}`in json object.
</chain_of_thought>

## Guidlines and Constraints of CoT:
- Only invoke **one tool at once** if you need tool calls.
- action, action_result, answer should each in seperate step.
- Typical flow of reasoning steps:
  1. no action: you can conclude an answer yourself without need of tools.
  2. single action: action->action_result->answer
  3. multiple actions or retry with reflection: action->action_result->action->action_result->...answer
- You should only respond in **json** object.

## Json Schema for response format of action, action_result, answer:
- action:
{
  "action": {"name":"...", "arguments":{...}}  #(value in json object). name: The name of tool to call, e.g., 'calculator'. arguments: A JSON object of parameters for the tool call (e.g., {"expression": "2*4"})
}
- action_result:
{
  "action_result": "..." # (vaue in string) The text of the action result from the tool call.
}
- answer:
{
  "answer": "..." # (value in string) The summary in string text as answer to user's query. 
}
"""

HANDOFF_SECTION = """# Handoffs:
When you encounter requests that should be handled by appropriate agent, invoke the `handoff_to_agent` tool. use: `handoff_to_agent(agent_name="...", message="...")`.
The message should be clear enough consists of the reason for handing off, which agent handed off to which agent.
You must choose the **agent_name** from **Available Handoff Agents* below:
{{handoffs}}

## Notes of Handoffs:
- **agent_name** in `handoff_to_agent` must be from available agents list.
- You must avoid **duplicated** handoffs if the handed-off agent rejected to handle the same user problem for multiple times.
- If you cannot handle the user problem, explain the reason to user or to upstream agent.
- If there's rejection caused by non-existed agent name, you should call `handoff_to_agent` again with valid **agent_name** from **available handoff agents** list above.
"""

HANDOFF_SECTION_TO_TRIAGE_AGENT = """
If you cannot answer the question, transfer back to the **triage agent** with agent_name = "TRIAGE_AGENT". 
(Use "TRIAGE_AGENT" as triage agent name, which will be specially handled. Don't change the text "TRIAGE_AGENT")
"""
# -------------------------------------------
# Assistant agent supporting function calling
# -------------------------------------------
class AgentResponse(BaseModel):
    action: Optional[dict[str,Any]] = None
    action_result: Optional[str] = None
    answer: Optional[str] = None
    def __str__(self)->str:
        output_str = ''
        if self.action:
            output_str += f'**action**:\n{str(self.action)}\n'
        if self.action_result:
            output_str += f'**action_result**:\n{str(self.action_result)}\n'
        if self.answer:
            output_str += f'**answer**:\n{str(self.answer)}\n'
        return output_str
    
def parse_agent_response(response_text: str) -> Optional[AgentResponse]:
    """
    Parses the agent response text (expected to be valid JSON) into an AgentResponse.
    """
    try:
        response_text = response_text.strip()
        data = json.loads(response_text)
        return AgentResponse(**data)
    except (json.JSONDecodeError, ValidationError) as e:
        print("Parsing error:", e)
        print(response_text)
        return None

class AssistantAgent(BaseAgent):
    """
    A basic chat agent that enables tool calling.
    """
    def __init__(self,
                 llm_client:Optional[BaseChatClient] = None,
                 name:Optional[str] = "AssistantAgent",
                 description:Optional[str] = "Assist user to solve problems", # describe what the agent do
                 instructions:Optional[str] = "You are a helpful agent that help user resolve problems.",
                 tools:Optional[list[FunctionTool]] = [],
                 tool_choice: Optional[Literal["auto", "required"]] = "auto",
                 mcps: Optional[list[MCPClient]] = [],
                 handoffs:Optional[List["AssistantAgent"]] = [],
                 response_format:Optional[Dict[str, Any]] = None,
                 use_actor_tools:Optional[bool] = False, # use actor tools (implemented by AssistantAgent) or use llm tools
                 max_steps:Optional[int] = MAX_STEPS,
                 logger:Optional[logging.Logger] = None,
                 verbose:Optional[bool] = False) -> None:
        # don't clone tools
        llm_client = llm_client.clone(with_tools=False) if llm_client else OpenAIChatClient(verbose=verbose)
        llm_client.tool_choice = tool_choice # set tool choice for llm_client
        self.response_format:Optional[Dict[str, Any]] = response_format
        self.use_actor_tools:Optional[bool] = use_actor_tools
        self.handoffs:Optional[List[AssistantAgent]] = handoffs
        self._is_triage_agent = False

        super().__init__(llm_client = llm_client,
                         name = name,
                         description = description,
                         instructions = instructions,
                         tools = tools,
                         mcps = mcps,
                         max_steps = max_steps,
                         logger = logger,
                         verbose = verbose)
    # override abstract method from BaseAgent
    @property
    @override
    def system_prompt(self)->str:
        PROMPT_TEMPLATE = SYSTEM_PROMPT_TEMPLATE if not self.use_actor_tools else SYSTEM_PROMPT_TEMPLATE_ACTOR
        prompt = str(PROMPT_TEMPLATE.replace("{{instructions}}", self.instructions or ''))
        variables = {
            "agent_name" : self.name or '',
            "time_now": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "tools": self.get_tool_list_str(),
            "handoff_section":""
        }
        # handoffs
        if self.handoffs and len(self.handoffs):
            variables["handoff_section"] = HANDOFF_SECTION.replace("{{handoffs}}", self.get_handoffs_str())
            if not self._is_triage_agent:
                variables["handoff_section"] += HANDOFF_SECTION_TO_TRIAGE_AGENT
            else:
                variables["handoff_section"] += "\nYou are the triage agent."

        return format_template_with_json(prompt, variables)

    def get_handoffs_str(self)->str:
        self.handoffs
        handoff_list = []
        for handoff in self.handoffs:
            handoff_str = f"- **{handoff.name}** : {handoff.description}"
            handoff_list.append(handoff_str)
        return '\n'.join(handoff_list)
    
    # Set this agent as triage_agent, it's called by handoff agent
    def _set_triage_agent(self)->None:
        self._is_triage_agent = True
    
    @override
    def register_tools(self, tools: list[FunctionTool]):
        super().register_tools(tools)
        if not self.use_actor_tools:
            for tool in tools:
                self.llm_client.register_tool(tool)
        
            
    async def _execute_with_llm_tools(self, input:Prompt, context:Optional[AgentContext]) -> AgentStream:
        if input is not None:
            self.add_user_message(input)
        
        response_stream:ResponseStream = await self.call_llm(
            messages = self.get_messages(), # message list
            system = self.system_prompt,
            context = context,
            options = Options(response_format = self.response_format, max_iterations=self.max_steps),
            stream = True)
        
        async for chunk in response_stream:
            if chunk.function_call: # step
                yield Chunk(type=Chunk.Type.Step, done=False, text = f'**take action**: {chunk.function_call.name}\n')
            elif not chunk.done: # it's the tokens of answer
                yield Chunk(type=Chunk.Type.Answer, done=False, text = chunk.text)
            else: # done with final answer. chunk.text is the full answer.
                yield Chunk(type=Chunk.Type.Answer, done=True, text = chunk.text)
                self.add_agent_message(chunk.text)

    async def _execute_with_actor_tools(self, input:Prompt, context:Optional[AgentContext]) -> AgentStream:
        if input is not None:
            self.add_user_message(input)
        
        for _ in range(self.max_steps):
            # send message with LLM
            response_text = await self.call_llm(
                messages = self.get_messages(), # send all messages list
                system = self.system_prompt,
                context = context,
                options = Options(response_format={ "type": "json_object"}, max_iterations=self.max_steps),
                stream = False # respond in stream mode
                )
            
            self.add_agent_message(response_text)
            actor_response = parse_agent_response(response_text)
            if not actor_response:
                raise AgentException(details=f"Failed to parse agent response. text:{response_text}",
                                     error_code=ErrorCode.ModelBehaviorError,
                                     module=self.name)

            if actor_response.action:
                self.add_agent_message(response_text) # add action to memory
                tool_name = actor_response.action.get("name") or ''
                tool_args = actor_response.action.get("arguments") or {}
                yield(Chunk(type=Chunk.Type.Step, done=False, text = f'**take_action**. name:{tool_name}, arguments:{tool_args}\n'))
                call_result  = await self.call_function(tool_name, tool_args, context=context)
                actor_response.action_result = str(call_result)
                yield(Chunk(type=Chunk.Type.Step, done=False, text = f'**action_result**:\n{actor_response.action_result}\n'))
                action_result_text = json.dumps({"action_result" : actor_response.action_result}, ensure_ascii=False)
                self.add_agent_message(action_result_text)
            if actor_response.answer:
                yield(Chunk(type=Chunk.Type.Answer, done=True, text = actor_response.answer))
                return
        # failed to reach an answer within max_steps
        raise AgentException(details=f"Failed to reach an answer in {self.max_steps} steps.",
                             error_code=ErrorCode.ModelBehaviorError,
                             module=self.name)

    # override abstract method from BaseAgent
    async def _run_impl(self,
                        input:Prompt,
                        context:Optional[AgentContext]) -> AgentStream:
        """
        Processes the user's input by building a prompt from memory,
        sending it to the LLM asynchronously, and yield tool/token/answer to the up stream
        Parameters:
            input (str): The user's task.
        Returns:
            AsyncGenerator[Chunk]: the streamed chunks (tool/token/answer) of the answer
        """
        if not self.use_actor_tools:
            corontine = self._execute_with_llm_tools(input, context=context)
        else:
            corontine = self._execute_with_actor_tools(input, context=context)
        # relay the chunks
        async for chunk in corontine:
            yield chunk
        
