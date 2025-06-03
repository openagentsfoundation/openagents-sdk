from .utils import env
import json
from typing import Optional, override
from pydantic import BaseModel, ValidationError
from datetime import datetime
# OpenAI chat model
from .base_agent import (
    BaseAgent,
    FunctionTool,
    AgentContext,
    MAX_STEPS,
    BaseChatClient,
    Prompt,
    OpenAIChatClient,
    Options,
    AgentStream,
    Chunk,
    logging
)

from .exceptions import *
from .utils import format_template_with_json
from .mcp import *

# -----------------------------------------------------------------------------
# Azure OpenAI Asynchronous Call Function
# -----------------------------------------------------------------------------
SYSTEM_PROMPT_TEMPLATE = """# Instructions:
<instructions>
{{instructions}}
</instructions>

# Agent Name: {{agent_name}}
# The Time Now: {{time_now}}.
# User Profile:
<user_profile>
{{user_profile}}
</user_profile>

# 5. Language Rule
- Always reply in the same language as the user's original problem statement or task, as the **working language** 
- Natural language arguments in tool calls must be in the **working language**

# Tools:
You can use following tools to take action for solving problems:
<tools>
{{tools}}
</tools>

# Reasoning and Solving Steps (ReAct):
When answering, please follow below reasoning_loop step-by-step in ReAct mode strictly:
<reasoning_loop>
- **thought**: Describe your reasoning about how to complete the task in working language (language of user task), then decide the next action to take.
- **action**: If you decide to use a tool, output "action: {name: [tool name], arguments: [arguments in json object of tool call]".
- **observation**: After a tool is executed, tool's result as "observation: …".
- **thought**: Reflect on the Observation to find the gap and decide the next Action (if need find more proof) or conclude with an Answer (enough proof to answer).
... (**repeat** `thought/action/observation` loop **until** you find enough proof and detail to conclude an answer.)
(now, you are ready to conclude an answer as you've found all the proof)
- **final_answer**: "[your final answer in working language]".
</reasoning_loop>

Always start with a **thought** for any user query, and only invoke **one tool at once** if needed.

## Note:
**DON'T** conclude to answer if **observation** is not enough to address user's problem statement or task.

# Output Format:
- Your responses must be valid JSON object, following this schema exactly:
'''
{
    "thought": "..." or null, # (Optional) Your internal reasoning. Use null if not needed.
    "action": {"name":"...", "arguments":{...}} or null, # (Optional) name: The name of tool to call, e.g., 'calculator'. arguments: A JSON object of parameters for the tool call (e.g., {"expression": "2*4"}) 
    "observation": "..." or null, # (Optional) The observation from the tool call. Use null if not applicable.
    "final_answer": "..." or null # (Optional) The final answer when ready. Use null if not finished.
}
'''
- Though and final answer must be in working language.
"""
#"action_input": {...} or null, # (Optional) A JSON object of parameters for the tool (e.g., {"expression": "2*4"}). Use null if no action.
# action_input: Follow with "Action Input: {…}" where the JSON specifies the arguments for the tool call.

# -----------------------------------------------------------------------------
# Structured Agent Response Model using Pydantic
# -----------------------------------------------------------------------------
class AgentResponse(BaseModel):
    thought: Optional[str] = None
    action: Optional[dict] = None
    observation: Optional[str] = None
    final_answer: Optional[str] = None
    def __str__(self)->str:
        output_str = ''
        if self.thought:
            output_str = f'**thought**:\n{self.thought}\n'
        if self.action:
            output_str += f'**action**:\n{self.action}\n'
        if self.observation:
            output_str += f'**observation**:\n{self.observation}\n'
        if self.final_answer:
            output_str += f'**answer**:\n{self.final_answer}\n'
        return output_str

def parse_agent_response(response_text: str) -> Optional[AgentResponse]:
    """
    Parses the agent response text (expected to be valid JSON) into an AgentResponse.
    """
    try:
        data = json.loads(response_text)
        return AgentResponse(**data)
    except (json.JSONDecodeError, ValidationError) as e:
        print("Parsing error:", e)
        print(response_text)
        return None

# -----------------------------------------------------------------------------
# Asynchronous ReAct Agent Implementation
# -----------------------------------------------------------------------------
class ReactAgent(BaseAgent):
    """
    A ReAct-style agent that iteratively constructs prompts from conversation memory,
    calls the Azure OpenAI model asynchronously, parses its output, and executes
    tool actions if requested.
    
    Tools are registered in a dictionary mapping tool names to callable functions.
    """
    def __init__(self,
                 llm_client:Optional[BaseChatClient] = None,
                 name:Optional[str] = "ReactAgent",
                 description:Optional[str] = "Execute user tasks in ReAct agent loop",
                 instructions:str | None = "You are a helpful agent that help user resolve problems.",
                 tools: list[FunctionTool]|None = [],
                 mcps: Optional[list[MCPClient]] = [],
                 max_steps:int|None=MAX_STEPS,
                 logger = None,
                 verbose:bool|None=False) -> None:
        llm_client = llm_client.clone(with_tools=False) if llm_client else OpenAIChatClient(verbose=verbose)
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
        prompt = str(SYSTEM_PROMPT_TEMPLATE.replace("{{instructions}}", self.instructions))
        variables = {
            "agent_name" : self.name or '',
            "time_now": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "tools": self.get_tool_list_str(),
        }
        return format_template_with_json(prompt, variables)
    
    
    # override abstract method from BaseAgent
    async def _run_impl(self, task:Prompt, context:AgentContext) -> AgentStream:
        """Execute the task and yield intermidiate reasoning step and final answer
        Parameters:
            task: the user task
        Returns: return chunks in stream mode
        """
        """ not working
        if self.is_executing:
            yield(Chunk(type=ChunkType.Answer, done=True, text = f'Agent {self.name}: previous task is in running'))
            return
        else:
            self.is_executing = True
        """
        # add to user question to message list
        if task is not None:
            self.add_user_message(task)

        #yield(Chunk(type=ChunkType.Step, done=False, text = "<think>\n"))
        done = False
        for i in range(self.max_steps):
            if self.interrupted:
                self.interrupted = False
                yield(Chunk(type=Chunk.Type.Answer, done=True, text = f'Agent {self.name}: task interrupted'))
                break

            self.log(f"iteration #{i+1}", logging.DEBUG)
            #user_prompt = "Agent:"
            messages = self.memory.get_messages()
            response = await self.call_llm(
                messages = messages, # send all messages as a list
                system =self.system_prompt,
                context = context,
                options = Options(response_format={ "type": "json_object"}),
                stream = False
                )
            self.add_agent_message(response)
            react_response = parse_agent_response(response)
            if not react_response:
                #yield(Chunk(type=ChunkType.Step, done=False, text = "</think>\n"))
                raise AgentException(details=f"Failed to parse agent response. text:{response}",
                                     error_code=ErrorCode.ModelBehaviorError,
                                     module=self.name)
            
            if react_response.thought:
                yield(Chunk(type=Chunk.Type.Step, done=False, text = f'**thought**: {react_response.thought}\n'))

            if react_response.action:
                tool_name = react_response.action["name"] or {}
                tool_args = react_response.action.get("arguments") or {}
                yield(Chunk(type=Chunk.Type.Step, done=False, text = f'**take action**: {tool_name}, arguments: {tool_args}\n'))
                call_result  = await self.call_function(tool_name, tool_args, context=context)
                react_response.observation = str(call_result)
                yield(Chunk(type=Chunk.Type.Step, done=False, text = f'**observation**:\n{react_response.observation}\n'))
                self.add_agent_message(json.dumps({"observation" : react_response.observation}, ensure_ascii=False))
                
            if react_response.final_answer:
                #yield(Chunk(type=ChunkType.Step, done=False, text = "</think>\n"))
                yield(Chunk(type=Chunk.Type.Answer, done=True, text = react_response.final_answer))
                done = True
                break
        #yield(Chunk(type=ChunkType.Step, done=False, text = "</think>\n"))
        if not done:
            raise AgentException(details=f"Failed to reach an answer in {self.max_steps} steps.",
                                 error_code=ErrorCode.ModelBehaviorError,
                                 module=self.name)
            #self.log(f"Failed to reach an answer in {self.max_steps} steps.", logging.DEBUG)
            #yield(Chunk(type=Chunk.Type.Answer, done=True, text = f"Failed to reach an answer in {self.max_steps} steps."))
        else:
            self.log(f"Completed in {i+1} steps", logging.DEBUG)
        
