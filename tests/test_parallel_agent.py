import asyncio
from openagent import AssistantAgent, AgentStream, SequentialWorkflow, ParallelWorkflow, function_tool
from openagent.common_tools import *

@function_tool
def search_from_web(query:str):
    """Search from web with query"""
    return "I found nothing"

"""
  user -> orchestrator_workflow:
          1. call research agents in **parrel**:
             - call research1
             - call research2
             - call research3
          2. call aggreate agent to summarize
       <- agent response
"""

RESEARCHER1_INSTRUCTIONS = """You are an AI Research Assistant specializing in energy.
1. Research the latest advancements in 'renewable energy sources'.
2. Use the search tool provided.
3. Summarize your key findings concisely (1-2 sentences).
4. Output *only* the summary.
"""

RESEARCHER2_INSTRUCTIONS = """You are an AI Research Assistant specializing in transportation.
1. Research the latest developments in 'electric vehicle technology'.
2. Use the search tool provided.
3. Summarize your key findings concisely (1-2 sentences).
4. Output *only* the summary.
"""

RESEARCHER3_INSTRUCTIONS = """You are an AI Research Assistant specializing in climate solutions.
1. Research the current state of 'carbon capture methods'.
2. Use the search tool provided.
3. Summarize your key findings concisely (1-2 sentences).
4. Output *only* the summary.
"""

RESEARCH_AGGREGATOR_INSTRUCTIONS = """You are an AI Assistant responsible for combining research findings into a structured report.
Your primary task is to synthesize the following research summaries, clearly attributing findings to their source areas. Structure your response using headings for each topic. Ensure the report is coherent and integrates the key points smoothly.
**Crucially: Your entire response MUST be grounded *exclusively* on the information provided in the **received search summaries**. 
Do NOT add any external knowledge, facts, or details not present in these specific summaries.

**Output Format:**
## Summary of Recent Sustainable Technology Advancements
### Renewable Energy Findings
  (Based on RenewableEnergyResearcher's findings)
  [Synthesize and elaborate *only* on the renewable energy input summary provided above.]

### Electric Vehicle Findings
  (Based on EVResearcher's findings)
  [Synthesize and elaborate *only* on the EV input summary provided above.]

### Carbon Capture Findings
  (Based on CarbonCaptureResearcher's findings)
  [Synthesize and elaborate *only* on the carbon capture input summary provided above.]

### Overall Conclusion
  [Provide a brief (1-2 sentence) concluding statement that connects *only* the findings presented above.]

**Notes:**
- Output *only* the structured report following this format. 
- Do not include introductory or concluding phrases outside this structure, and strictly adhere to using only the provided input summary content.
"""

async def chat_loop() -> None:
    reseacher1 = AssistantAgent(
        name = "RenewableEnergyResearcher",
        description="Researches renewable energy sources.",
        instructions = RESEARCHER1_INSTRUCTIONS,
        tools = [search_from_web],
        verbose = True,
    )
    
    reseacher2 = AssistantAgent(
        name = "EVResearcher",
        instructions = RESEARCHER2_INSTRUCTIONS,
        description="Researches electric vehicle technology.",
        tools = [search_from_web],
        verbose = True,
    )
    
    reseacher3 = AssistantAgent(
        name = "CarbonCaptureResearcher",
        instructions = RESEARCHER3_INSTRUCTIONS,
        description="Researches carbon capture methods.",
        tools = [search_from_web],
        verbose = True
    )

    # a parral workflow executing sub research agents in parallel 
    research_workflow = ParallelWorkflow(
        name = "ParallelWebResearchAgent",
        description="Runs multiple research agents in parallel to gather information.",
        agents = [reseacher1, reseacher2, reseacher3],
        verbose = True
    )

    aggregator = AssistantAgent(
        name = "AggregatorAgent",
        description="Combines research findings from parallel agents into a structured, cited report, strictly grounded on provided inputs.",
        instructions = RESEARCH_AGGREGATOR_INSTRUCTIONS,
        verbose = True
    )

    # a sequatial workflow consists the reseacher parrel workflow and aggregator agent
    orchestrator_workflow = SequentialWorkflow(
        name = "ResearchOrestratorWorkflow",
        description = "Coordinates parallel research and synthesizes the results.",
        agents = [research_workflow, aggregator],
        verbose = True
    )

    print("Welcome to parallel agent test workflow.\nExample input: Give me a report of renewable energy 2024-2025")
    
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            print("Please Enter Something")
            continue
        if user_input.lower() == "exit":
            break
        
        # request stream response
        response_stream:AgentStream = await orchestrator_workflow.run(user_input, stream=True)
        async for chunk in response_stream:
            if not chunk.done: # token by token chunks of reasoning steps and answer
                print(chunk.text, end="", flush=True) 
            else:              # the final answer
                print(f"\n\033[1;32;40mAI: {chunk.text}\033[0m")

if __name__ == "__main__":
    asyncio.run(chat_loop())
