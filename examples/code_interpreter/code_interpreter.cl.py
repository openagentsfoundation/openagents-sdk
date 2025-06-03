# !chainlit run test_data_analysis.cl.py
# install required packages:
"""
pip install pandas sympy numpy matplotlib plotly seaborn scikit-learn statsmodels prophet
"""
import os, shutil, time
from typing import cast, Optional
from pydantic import BaseModel
import chainlit as cl
from openagent import (
    env,
    CodeInterpreterAgent,
    AgentStream,
    AgentResponse,
    PythonRuntime,
    OpenAIChatClient,
    get_logger,
    logging,
    SANDBOX_LOCAL,
    SANDBOX_BLOB,
    identity
)

# LLM endpoint: recommend Azure OpenAI GPT-4.1
OPENAI_CODE_ENDPOINT = env.get("OPENAGENT_OPENAI_CODE_ENDPOINT")
OPENAI_CODE_API_KEY = env.get("OPENAGENT_OPENAI_CODE_API_KEY")
OPENAI_CODE_DEPLOYMENT = env.get("OPENAGENT_OPENAI_CODE_DEPLOYMENT")
OPENAI_CODE_API_VERSION = env.get("OPENAGENT_OPENAI_CODE_API_VERSION")

app_logger = get_logger("CodeInterpreter", level=logging.INFO)

USER_ID = "__USER_ID__"
AGENT = "__AGENT__"

# initialize a new session
@cl.on_chat_start
async def start_chat() -> None:
    print("@cl.on_chat_start")
    user_id = identity.unique_hash(8)
    cl.user_session.set(USER_ID, user_id)
    # create sandbox data dirctory for the user, if not created.
    sandbox_data_dir = SANDBOX_LOCAL + f"/{user_id}/data"
    if not os.path.isdir(sandbox_data_dir):
        os.makedirs(sandbox_data_dir, exist_ok=True)
    
    code_client =  OpenAIChatClient(
        end_point = OPENAI_CODE_ENDPOINT,
        model = OPENAI_CODE_DEPLOYMENT,
        api_key = OPENAI_CODE_API_KEY,
        api_version = OPENAI_CODE_API_VERSION,
        temperature = 0,
        top_p = 0
        )
    
    SANDBOX_USER = user_id
    SANDBOX_PATH = SANDBOX_LOCAL + f"/{SANDBOX_USER}"
    SANDBOX_BLOB_PATH = SANDBOX_BLOB + f"/{SANDBOX_USER}"

    agent = CodeInterpreterAgent(
        code_client=code_client,
        runtime=PythonRuntime.NativeUnsafe,
        sandbox_local=SANDBOX_PATH, # local data directory
        sandbox_blob=SANDBOX_BLOB_PATH, # web blob directory
        logger=app_logger,
        verbose=True)
    
    cl.user_session.set(AGENT, agent)

    welcome = """Welcome to code interpreter agent.
    - Please upload CSV file and ask for any data analysis task.
    - Note: the CSV should contain header clearly imply the meaning of each column.    
    """
    await cl.Message(content=welcome).send()

class File(BaseModel):
    path:str # full local file path or url
    name:str # file display name
    size:Optional[int] = None # None means not set

def copy_files_to_sandbox(files:list[File], user_id:str)->str:
    """Copy files to sandbox directory
    Parameters:
        files (list of str, required): names of files to copy
        user_id (str, required): the current user id
    Returns (str): the output context of the function
    """
    sandbox_data_dir = SANDBOX_LOCAL + f"/{user_id}/data"
    if not os.path.isdir(sandbox_data_dir):
        os.makedirs(sandbox_data_dir, exist_ok=True)
    file_names = [file.name for file in files]
    try:
        for file in files:
            shutil.copy2(file.path, os.path.join(sandbox_data_dir, file.name))
        return f"The below files are copied to container direcotry {sandbox_data_dir}:\n{'\n'.join(file_names)}"
    except Exception as e:
        return f"Error: failed to copy file. reason: {str(e)}"

@cl.on_message  # type: ignore
async def chat(message: cl.Message) -> None:
    print("@cl.on_message")
    user_id = cl.user_session.get(USER_ID)
    print(f"User ID is {user_id}")
    agent = cast(CodeInterpreterAgent, cl.user_session.get(AGENT))
    # handle attachments
    csv_files = []
    if message.elements:
        for file in message.elements:
            print(file)
            _, ext = os.path.splitext(file.name.lower())
            if ext == ".csv":
                csv_files.append(File(path=file.path, name=file.name, size=file.size))
            else:
                await cl.Message(content=f"Warning!!! only csv file is allowed\n.The file {file.name} is ignored").send()
    
    if len(csv_files):
        returned_context = copy_files_to_sandbox(csv_files, user_id=user_id)
        print(returned_context)
        agent.add_context(returned_context)

    # run agent to execute data analysis task
    task = str(message.content).strip()
    if not task:
        await cl.Message(content = "Please enter your analysis task").send()
        return
    
    final_answer = cl.Message(content="")
    async with cl.Step(name="Thinking") as thinking_step:
        start = time.time()
        response_stream:AgentStream = await agent.run(task, stream=True)
        async for chunk in response_stream:
            await thinking_step.stream_token("\n")
            if not chunk.done: # including chunk.Step or (chunk.Answer and chunk.done==False)
                thinking_seconds = round(time.time() - start)
                thinking_step.name = f"thinking for {thinking_seconds} seconds"
                await thinking_step.update()
                await thinking_step.stream_token(chunk.text)
                print(chunk.text, end="", flush=True)
            else:
                await thinking_step.send() # end of thinking
                # send/stream final answer
                await final_answer.stream_token(chunk.text)
                print(f"\n\033[1;32;40mAI: {chunk.text}\033[0m")

    await final_answer.send()

# sample questions:
"""
- "Predict the sales revenue for next 7 days." (Time series forecast)
- "What factors contribute the most to drop of yield?" (Feature importance analysis)
- "What factors contribute the most to accident frequency?" (Feature importance analysis)
- "Which areas are at the highest risk of accidents?" (Classification/Clustering)
- "How does traffic fine amount influence the number of accidents?" (Regression/Causal inference)
- "Can we determine the optimal fine amounts to reduce accident rates?" (Optimization models)
- "Do higher fines correlate with lower average speeds or reduced accidents?" (Correlation/Regression)
"""