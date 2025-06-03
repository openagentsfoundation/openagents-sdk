# Reference: https://github.com/openai/openai-agents-python/tree/main/examples/customer_service
import asyncio
from typing import Optional
from openagent import (
    AssistantAgent, 
    AgentStream,
    HandoffWorkflow, 
    AgentContext, 
    function_tool
)

### CONTEXT
class AirlineAgentContext(AgentContext):
    confirmation_number:Optional[str] = None
    seat_number:Optional[str] = None

### TOOLS
@function_tool
async def faq_lookup_tool(question:str) -> str:
    """Lookup frequently asked questions.
    Parameters:
        - question (type:str, required): the user question
    Returns(str): the FAQ lookup answer 
    """
    question = question.lower()
    result = ""
    if "bag" in question or "baggage" in question:
        result +=(
            "You are allowed to bring one bag on the plane. "
            "It must be under 50 pounds and 22 inches x 14 inches x 9 inches."
            )
    if "seats" in question or "plane" in question:
        result += (
            "There are 120 seats on the plane. "
            "There are 22 business class seats and 98 economy seats. "
            "Exit rows are rows 4 and 16. "
            "Rows 5-8 are Economy Plus, with extra legroom. "
            )
        
    if "wifi" in question:
        result += "We have free wifi on the plane, join Airline-Wifi"

    return result or "I'm sorry, I don't know the answer to that question."

@function_tool
async def update_seat(confirmation_number: str, new_seat: str, context: AirlineAgentContext)->str:
    """Update the seat for a given confirmation number.
    Parameters:
        - confirmation_number (type:str, required): The confirmation number for the flight.
        - new_seat (type:str, required): The new seat to update to.
    Returns(str): the seat update result
    """
    # Update the context based on the customer's input
    context.confirmation_number = confirmation_number
    context.seat_number = new_seat
    # Ensure that the flight number has been set by the incoming handoff
    #assert context.flight_number is not None, "Flight number is required"
    return f"Updated seat to {new_seat} for confirmation number {confirmation_number}"

### AGENTS
faq_agent = AssistantAgent(
    name="FAQAgent",
    description="A helpful agent that can answer questions about the airline.",
    instructions="""You are an FAQ agent. If you are speaking to a customer, you probably were transferred to from the triage agent.
    Use the following routine to support the customer.
    # Routine
    1. Identify the last question asked by the customer.
    2. Use the faq lookup tool to answer the question. Do not rely on your own knowledge.""",
    tools=[faq_lookup_tool],
    verbose=True,
)

seat_booking_agent = AssistantAgent(
    name="SeatBookingAgent",
    description="A helpful agent that can update a seat on a flight.",
    instructions="""You are a seat booking agent. If you are speaking to a customer, you probably were transferred to from the triage agent.
    Use the following routine to support the customer.
    # Routine
    1. Ask for their confirmation number if confirmation_number is not provided.
    2. Ask the customer what their desired seat number is.
    3. Use the update seat tool to update the seat on the flight.

    # Context:
    - confirmation_number: {{confirmation_number}}
    """,
    tools=[update_seat],
    verbose=True,
)

triage_agent = AssistantAgent(
    name="TriageAgent",
    description="A triage agent that can delegate a customer's request to the appropriate agent.",
    instructions="You are a helpful triaging agent. You can use your tools to delegate questions to other appropriate agents.",
    handoffs=[
        faq_agent,
        seat_booking_agent,
    ],
    verbose=True,
)

### RUN
async def chat_loop():
    cs_workflow = HandoffWorkflow(
        name = "AirlineServiceWorkflow",
        description = "Airline customer service",
        triage_agent = triage_agent,
        verbose = True,
    )
    
    context = AirlineAgentContext()
    context.confirmation_number = "C01" # preset the confirmation number, so the agent will skip asking for it.

    # Normally, each input from the user would be an API request to your app, and you can wrap the request in a trace()
    # Here, we'll just use a random UUID for the conversation ID
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            print("Please Enter Something")
            continue
        if user_input.lower() == "exit":
            break
        # request stream response
        response_stream:AgentStream = await cs_workflow.run(user_input, context = context, stream=True)
        async for chunk in response_stream:
            if not chunk.done: # token by token chunks of reasoning steps and answer
                print(chunk.text, end="", flush=True) 
            else:              # the final answer
                print(f"\n\033[1;32;40m[{cs_workflow.current_agent.name}]: {chunk.text}\033[0m")

if __name__ == "__main__":
    asyncio.run(chat_loop())
