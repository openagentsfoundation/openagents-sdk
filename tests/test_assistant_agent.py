import asyncio
from typing import Optional
from openagent import (
    OpenAIChatClient,
    AssistantAgent, 
    AgentStream,
    AgentContext,
    function_tool, 
    get_logger,
    logging
    )

# user defined function call
## recommend to clearly descibe what that function does and what's the expeted input and output with function doc """..."""
## the function doc will be readed by SDK passed to LLM client.
@function_tool
def calculator(expression: str)->str:
    """Securely evaluates an arithmetic expression in python __builtins__ runtime.
    Parameters:
        expression (str): A string containing a single arithmetic expression valid in python.
    Returns (str): The the eval(...) result of the arithmetic expression or an error message.
    """
    try:
        # Evaluate with restricted built-ins for basic arithmetic.
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
         return f"Error: {str(e)}"

# Optionally define application level logger . the log file will be 'assistant_agent_app.log'
app_logger = get_logger("assistant_agent_app", logging.INFO)

# Optionally define agent context.
"""
The variable place holder "{{...}}" in instructions will be replaced by variables defined in sub class AgentContext,
so agent will know the context in system prompt level.

if want this context auto replacement feature, pass context in agent.run:
  `await agent.run("Hi", context=agent_context)`
"""

class MyAgentContext(AgentContext):
    user_name:Optional[str] = None

agent_context = MyAgentContext()
agent_context.user_name = "Sam Altman"

agent = AssistantAgent(
    name="PersonalAssistant",
    instructions="You are helpful personal assistant. Your user's name is {{user_name}}.",
    tools= [calculator],
    logger = app_logger,
    verbose = True)
# You can declear an LLM client and pass to the agent rather than using the default client.
## support OpenAI/Azure OpenAI, Ollama (OpenAI API compatible endpoint) and Vllm (OpenAI API compatible endpoint)
"""
llm_client =  OpenAIChatClient(
    end_point = "...",
    model = "...",
    api_key = ""...",
    api_version = "...")
agent = AssistantAgent(llm_client=llm_client, ...)
"""
# respond with final answer
async def chat_loop() -> None:
    while True:
        user_input = input("You: ").strip() #"What is 20+(2*4)? Calculate step by step."
        # Exit the loop if the user enters 'exit'
        if not user_input:
            print("Please Enter Something")
            continue
        if user_input.lower() == "exit":
            break
        print(agent_context.context)
        result = await agent.run(user_input, context=agent_context)
        print(f"\n\033[1;32;40mAI: {str(result)}\033[0m") 

async def chat_loop_stream() -> None:
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            print("Please Enter Something")
            continue
        if user_input.lower() == "exit":
            break
        # request stream response
        response_stream:AgentStream = await agent.run(user_input, context=agent_context, stream=True)
        async for chunk in response_stream:
            if not chunk.done: # token by token chunks of reasoning steps and answer
                print(chunk.text, end="", flush=True) 
            else:              # the final answer
                print(f"\n\033[1;32;40mAI: {chunk.text}\033[0m")
        
if __name__ == "__main__":
    asyncio.run(chat_loop_stream())
