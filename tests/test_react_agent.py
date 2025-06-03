import asyncio
from openagent import (
    env,
    OpenAIChatClient,
    ReactAgent,
    AgentResponse,
    AgentStream,
    function_tool,
    get_logger,
    logging
)
from openagent.common_tools import search_from_web

# GPT-4o
OPENAI_LLM_ENDPOINT = env.get("OPENAGENT_OPENAI_LLM_ENDPOINT")
OPENAI_LLM_API_KEY = env.get("OPENAGENT_OPENAI_LLM_API_KEY")
OPENAI_LLM_DEPLOYMENT = env.get("OPENAGENT_OPENAI_LLM_DEPLOYMENT")
OPENAI_LLM_API_VERSION = env.get("OPENAGENT_OPENAI_LLM_API_VERSION")

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
# -----------------------------------------------------------------------------
# Main: Asynchronous Testing of the ReAct Agent
# -----------------------------------------------------------------------------
llm_client =  OpenAIChatClient(end_point = OPENAI_LLM_ENDPOINT,
    model = OPENAI_LLM_DEPLOYMENT,
    api_key = OPENAI_LLM_API_KEY,
    api_version = OPENAI_LLM_API_VERSION,
    temperature = 0.2)


# define app level logger
app_logger = get_logger("assistant_agent_app", level=logging.INFO)
# define agent
agent = ReactAgent(
    llm_client = llm_client, 
    tools = [calculator, search_from_web(llm_client)],
    logger = app_logger,
    verbose = True)

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

        result = await agent.run(user_input)
        print(f"\n\033[1;32;40mAI: {result.answer}\033[0m") 

# respond with streamed message
async def chat_loop_stream() -> None:
    while True:
        user_input = input("You: ").strip() #"What is 20+(2*4)? Calculate step by step."
        # Exit the loop if the user enters 'exit'
        if not user_input:
            print("Please Enter Something")
            continue
        if user_input.lower() == "exit":
            break

        response_stream:AgentStream = await agent.run(user_input, stream=True)
        async for chunk in response_stream:
            if not chunk.done: # streamed text including step and answer tokens
                print(chunk.text, end="", flush=True) 
            else: # final answer
                print(f"\n\033[1;32;40mAI: {chunk.text}\033[0m") 

if __name__ == "__main__":
    asyncio.run(chat_loop_stream())
