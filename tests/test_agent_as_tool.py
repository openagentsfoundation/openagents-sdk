
import asyncio
from openagent import (
    AssistantAgent,
    AgentStream
    )

spanish_agent = AssistantAgent(
    name="Spanish agent",
    instructions="You translate the user's message to Spanish",
)
french_agent = AssistantAgent(
    name="French agent",
    instructions="You translate the user's message to French",
)

# the user-facing agent
user_agent = AssistantAgent(
    name="TranslationAgent",
    description="You can translate language in Spanish or French",
    instructions=(
        "You are a translation agent. You use the tools given to you to translate."
        "If asked for multiple translations, you call the relevant tools."
    ),
    tools=[
        spanish_agent.as_tool(
            tool_name="translate_to_spanish",
            tool_description="Translate the user's message to Spanish",
            with_parent_context=True
        ),
        french_agent.as_tool(
            tool_name="translate_to_french",
            tool_description="Translate the user's message to French",
        ),
    ],
    verbose=True,
)

tool = spanish_agent.as_tool(
            tool_name="translate_to_spanish",
            tool_description="Translate the user's message to Spanish",
        )

async def agent_chat_loop() -> None:
    print("You are in agent as a tool test example")
    while True:
        user_input = input("You: ").strip() #"What is 20+(2*4)? Calculate step by step."
        # Exit the loop if the user enters 'exit'
        if not user_input:
            print("Please Enter Something")
            continue
        if user_input.lower() == "exit":
            break

        response_stream:AgentStream = await user_agent.run(user_input, stream=True)
        async for chunk in response_stream:
            if not chunk.done:
                print(chunk.text, end="", flush=True)
            else:
                #append_history(history=history, role="assistant", content=chunk.text)
                print(f"\n\033[1;32;40mAI: {chunk.text}\033[0m")

if __name__ == "__main__":
    asyncio.run(agent_chat_loop())
