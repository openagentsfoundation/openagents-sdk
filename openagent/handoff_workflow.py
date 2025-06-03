from typing import Optional, override, Any, Dict, List
from .base_agent import BaseAgent, Prompt, AgentContext, AgentStream, Chunk, function_tool, MAX_STEPS
from .assistant_agent import AssistantAgent

def _normalize_agent_name(agent_name:str):
    return agent_name.lower().replace("*", "").replace(" ","")
# HandoffWorkflow execute user task and handoff to sub agents accordingly
# HandoffWorkflow itself is a agent
class HandoffWorkflow(BaseAgent):
    """
    A basic chat agent that enables tool calling.
    """
    def __init__(self,
                 name:Optional[str] = "HandoffWorkflow",
                 description:Optional[str] = "Handoff to sub-agents according to user query",
                 triage_agent:Optional[AssistantAgent] = None,
                 max_steps:Optional[int] = MAX_STEPS, # max hanoffs
                 logger = None,
                 verbose:Optional[bool] = False) -> None:
        super().__init__(llm_client = None,
                         name = name,
                         description = description,
                         instructions = None, # no instructions needed for worflow
                         max_steps=max_steps,
                         logger = logger,
                         verbose = verbose)
        
        self.triage_agent = triage_agent
        self.current_agent = triage_agent
        self.agent_map:Dict[str, AssistantAgent] = {}
        self._register_handoffs()
    
    # override abstract method from BaseAgent
    @property
    @override
    def system_prompt(self)->str:
        pass

    def _register_handoffs(self):
        if not self.triage_agent:
            return
        
        self.triage_agent._set_triage_agent() # set as triage agent
        # traverse and lookup
        lookups = [self.triage_agent]
        while(len(lookups)>0):
            agent = lookups.pop(0)
            if _normalize_agent_name(agent.name) not in self.agent_map.keys():
                if self.triage_agent not in agent.handoffs and not agent._is_triage_agent:
                    agent.handoffs.append(self.triage_agent) # always append triage agent to handoff list of sub-agents
                # register handoff function ONLY if the agent has handoffs
                if len(agent.handoffs):
                    self.register_handoff(agent)
                    lookups.extend(agent.handoffs) # traverse sub agents
            else:
                pass
    
    def register_handoff(self, agent:AssistantAgent):
        if _normalize_agent_name(agent.name) not in self.agent_map.keys():
            agent.register_tools([self.handoff_to_agent])
            self.agent_map[_normalize_agent_name(agent.name)] = agent # add to agent map

    #TO-DO: use agent_name="TriageAgent" as default triage agent
    @function_tool
    def handoff_to_agent(self, agent_name:str, message:str, context:AgentContext)->str:
        """Handoff the conversation to the agent who will take over the conversation and further handle user request.
        Parameters:
            - agent_name (type:str, required): the name of agent handed off to, who will take over the conversation handle user request further. Use text "TRIAGE_AGENT" for triage agent name.
            - message (type:str, required): the message to the agent, explained the reason of conversation handoff.
        Returns:
            str: the handoff result.
        """
        if _normalize_agent_name(agent_name) == "triage_agent":
            handoff_to = self.triage_agent
        else:
            handoff_to = self.agent_map.get(_normalize_agent_name(agent_name))
        if handoff_to is None:
            result = f"Failed to handed off to agent **{agent_name}**. reason: agent **{agent_name}** not exist"
            self.active_agent = self.triage_agent
        else:
            result = f'Handed off **{agent_name}** to handle user request further. Message to **{agent_name}**:"{message}"'
            self.active_agent = handoff_to
        return result
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
        # **Note**: all agents share one conversation history hosted by HandoffWorkflow memory via self.get_messages()
        ## all tools calls is added back to HandoffWorkflow memory, so each sub-agents knows the whole tracing routes.
        self.active_agent = self.current_agent
        for _ in range(self.max_steps):
            if self.verbose:
                print(f"[{self.name}] -> run sub_agent: {self.current_agent.name}")
            inital_message_count = len(self.get_messages())
            response_stream:AgentStream = await self.current_agent.run(input=self.get_messages(), context=context, stream=True)
            handed_off = False
            async for chunk in response_stream:
                if chunk.type == Chunk.Type.Step: # step
                    yield chunk
                else: # answer
                    # The function has been called in steps. now let whether there's agent change
                    handed_off = self.current_agent!=self.active_agent
                    if not handed_off:
                        yield chunk
                    else:
                        if not chunk.done: # yield answer tokens as step
                            yield Chunk(type=Chunk.Type.Step, done = False, text = chunk.text)
                        else:
                            yield Chunk(type=Chunk.Type.Step, done = False, text = "\n") # add a new line for last agent answer output
            # newly generated message by the sub-agent
            new_messages = context.messages[inital_message_count:]
            #from pprint import pprint
            #pprint(new_messages)
            self.add_message_list(new_messages)
            # add the full answer to memory
            # self.add_agent_message(chunk.text) # add to memory
            if handed_off:
                self.current_agent = self.active_agent # hand off to the new agent and continue the loop
            else:
                return # exit
        yield(Chunk(type=Chunk.Type.Answer, done=True, text = f"Failed to reach an answer in {self.max_steps} steps."))

