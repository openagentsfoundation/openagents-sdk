import asyncio
from openagent import AssistantAgent, AgentStream, SequentialWorkflow
from openagent.common_tools import *

"""
    user -> code_workflow:
         1. call coder agent
         <- response of coder agent
         2. call reviewer agent
         <- response of reviewr agent
         3. call call refector agent
         <- response of refector agent
"""

CODER_INSTRUCTIONS = """You are a Python Code Generator.
Based *only* on the user's request, write Python code that fulfills the requirement.
Output *only* the complete Python code block, enclosed in triple backticks (```python ... ```). 
Do not add any other text before or after the code block.
"""

REVEWER_INSTRUCTIONS = """You are an expert Python Code Reviewer. 
Your task is to provide constructive feedback on the provided code.

**Review Criteria:**
1.  **Correctness:** Does the code work as intended? Are there logic errors?
2.  **Readability:** Is the code clear and easy to understand? Follows PEP 8 style guidelines?
3.  **Efficiency:** Is the code reasonably efficient? Any obvious performance bottlenecks?
4.  **Edge Cases:** Does the code handle potential edge cases or invalid inputs gracefully?
5.  **Best Practices:** Does the code follow common Python best practices?

**Output:**
Provide your feedback as a concise, bulleted list. Focus on the most important points for improvement.
If the code is excellent and requires no changes, simply state: "No major issues found."
Output *only* the review comments or the "No major issues" statement.
"""

REFACTOR_INSTRUCTIONS = """You are a Python Code Refactoring AI.
Your goal is to improve the given Python code based on the provided review comments.

**Task:**
Carefully apply the suggestions from the review comments to refactor the original code.
If the review comments state "No major issues found," return the original code unchanged.
Ensure the final code is complete, functional, and includes necessary imports and docstrings.

**Output:**
Output *only* the final, refactored Python code block, enclosed in triple backticks (```python ... ```). 
Do not add any other text before or after the code block.
"""

async def chat_loop() -> None:
    coder_agent = AssistantAgent(
        name = "CodeWriterAgent",
        instructions=CODER_INSTRUCTIONS,
        verbose=True)
    
    reviwer_agent = AssistantAgent(
        name = "CodeReviewerAgent",
        instructions = REVEWER_INSTRUCTIONS,
        verbose=True)
    
    refactor_agent = AssistantAgent(
        name = "CodeRefactorAgent",
        instructions = REFACTOR_INSTRUCTIONS,
        verbose=True
    )

    code_workflow = SequentialWorkflow(
        name = "CodeWorkflow",
        description ="Executes a sequence of code writing, reviewing, and refactoring.",
        agents = [coder_agent, reviwer_agent, refactor_agent],
        verbose = True
        )
    
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            print("Please Enter Something")
            continue
        if user_input.lower() == "exit":
            break
        # request stream response
        response_stream:AgentStream = await code_workflow.run(user_input, stream=True)
        async for chunk in response_stream:
            if not chunk.done: # token by token chunks of reasoning steps and answer
                print(chunk.text, end="", flush=True) 
            else:              # the final answer
                print(f"\n\033[1;32;40mAI: {chunk.text}\033[0m")

        #from pprint import pprint
        #pprint(code_workflow.get_messages())
if __name__ == "__main__":
    asyncio.run(chat_loop())
