from typing import Optional, override, Any, Dict, List
from .base_agent import BaseAgent, Prompt, AgentContext, AgentStream, AgentStream, Chunk

# SequentailWorkflow execute sub agents sequantially.
# SequentailWorkflow itself is a agent
class SequentialWorkflow(BaseAgent):
    """
    A basic chat agent that enables tool calling.
    """
    def __init__(self,
                 name:Optional[str] = "SequentialWorkflowAgent",
                 description:Optional[str] = "Execute sub-agent sequentially",
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
            input (str|Prompt): The user's task.
        Returns:
            AsyncGenerator[Chunk]: the streamed chunks (tool/token/answer) of the answer
        """
        # add inital user prompt (can be none)
        if input is not None:
            self.add_user_message(input)

        for index in range(len(self.agents)):
            # add the initial input or the answer of the last agent to message list
            sub_agent = self.agents[index]
            is_last_agent = index == len(self.agents)-1
            if self.verbose:
                print(f"[{self.name}] -> run sub_agent: {sub_agent.name}")
            # send the whole message list
            response_stream:AgentStream = await sub_agent.run(input=self.get_messages(), context=context, stream=True)
            async for chunk in response_stream:
                if not is_last_agent: # send response of sub agents as step message
                    if not chunk.done: # ignore the final answer for sub agents to avoid duplication
                        yield Chunk(type=Chunk.Type.Step, done = False, text = chunk.text)
                    else:
                        # add sub-agent response answer to memory
                        self.add_agent_message(f"<{sub_agent.name}>\n{chunk.text}\n</{sub_agent.name}>")
                        yield Chunk(type=Chunk.Type.Step, done = False, text = "\n") # add a new line for last agent answer output
                else:
                    yield chunk # pass through the response of last agent
                    if chunk.done: # add final answer to memory
                        self.add_agent_message(f"<{sub_agent.name}>\n{chunk.text}\n</{sub_agent.name}>")
