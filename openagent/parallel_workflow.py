import asyncio
from typing import Optional, override, Any, Dict, List
from .base_agent import (
    BaseAgent, 
    Prompt, 
    AgentContext, 
    AgentStream, 
    AgentResponse, 
    Chunk,
    prompt_to_message
    ) 

# ParallelWorkflow execute sub agents in pararrel and provided aggregated answer
# ParallelWorkflow itself is a agent
class ParallelWorkflow(BaseAgent):
    """
    A basic chat agent that enables tool calling.
    """
    def __init__(self,
                 name:Optional[str] = "ParallelWorkflowAgent",
                 description:Optional[str] = "Execute sub-agent in parallel",
                 agents:Optional[List[BaseAgent]] = None, # a list of sub agents
                 logger = None,
                 verbose:Optional[bool] = False) -> None:
        super().__init__(llm_client = None,
                         name = name,
                         description = description,
                         instructions = None, # no instructions needed for worflow
                         logger = logger,
                         verbose = verbose)
        self.agents = agents or []
    
    # override abstract method from BaseAgent
    @property
    @override
    def system_prompt(self)->str:
        pass
    
    # override abstract method from BaseAgent
    async def _run_impl(self,
                        input:Prompt,
                        context:Optional[AgentContext]) -> AgentStream:
        """
        Processes the user's input by building a prompt from memory,
        sending it to the LLM asynchronously, and yield step/answer to the up stream
        Parameters:
            input (Prompt): The user's task.
        Returns:
            AsyncGenerator[Chunk]: the streamed chunks (tool/token/answer) of the answer
        """
        # add initial user input to memory
        if input is not None:
            self.add_user_message(input)

        async def run_subagent(sub_agent:BaseAgent,
                               input:list,
                               context:Optional[AgentContext])->str:
            # the sub-agent will make a copy of input_messages as it's memory
            if self.verbose:
                print(f"[{self.name}] -> run sub_agent: {sub_agent.name}")
            response = await sub_agent.run(input = input, context=context, stream=False)
            return f"<{sub_agent.name}>\n{response.answer}\n</{sub_agent.name}>"

        input_messages = self.get_messages()
        async_tasks = [run_subagent(sub_agent=sub_agent, input=input_messages, context=context) 
                       for sub_agent in self.agents
                       ]
        results = await asyncio.gather(*async_tasks)
        aggregated_answer = "\n".join(results)
        # add aggregated result to memory
        self.add_agent_message(aggregated_answer)
        # return to upstream
        yield Chunk(type=Chunk.Type.Answer, text = aggregated_answer, done=True)

