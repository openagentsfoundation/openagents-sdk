# OpenAgents SDK

OpenAgents SDK is an open-source framework for building intelligent agents powered by large language models (LLMs). It provides a modular, extensible core library for defining agent behaviors, workflows, and memory, along with reference implementations and hands-on examples. Whether you need a data analysis “Code Interpreter” agent, a Researcher agent, or a Coder agent, OpenAgents SDK offers the building blocks to develop, test, and deploy LLM-driven agents in real-world scenarios.

---

## 1. Brief Introduction

OpenAgents SDK (short for OpenAI Agents SDK) is designed to help developers and enterprises quickly create intelligent agents that leverage tools, memory, and structured workflows. Key features include:

- **Agent Core**: Memory management, tool invocation, ReAct reasoning, MCP, and Agent-to-Agent (A2A) communication.
- **Agentic Workflows**: Implemenation of sequential, parallel, looped, and handoff workflows.
- **Pre-built Agents & Functions**: Ready-to-use templates like a Data Analyst (Code Interpreter) agent, Researcher agent, Coder agent, Search and Retrieval-Augmented Generation (RAG) pipelines.
- **Reference Implementations & Examples**: Sample projects in Python (and soon JavaScript/TypeScript) that demonstrate how to integrate and extend core SDK components.
- **Testing Suite**: A collection of unit tests and sample scenarios to validate SDK functionality.

---

## 2. Breakdown of `openagents-sdk`

```

openagents-sdk/
├── openagent/ # Core SDK modules
│
├── examples/ # Reference implementations & templates
│ ├── code_interpreter/ # Example project for Data Analyst agent
│ └── ...
│
├── tests/ # Sample codes of using OpenAgents SDK
│ ├── test\test_assistant_agent.py
│ └── ...
│
├── README.md # This file: Introduction + setup + examples
├── requirements.txt # Python dependencies for openagents-sdk
└── setup.py # Package metadata (used for `pip install -e .`)

```

- **OpenAgent Core SDK (`openagent/`)**:
  - **Agent Core**: Provides abstractions for memory storage, tool usage (tool invocation interface), ReAct-style reasoning, Multi-Chain-of-Thought Planning (MCP), and Agent-to-Agent (A2A) communication patterns.
  - **Agentic Workflows**: Modules to orchestrate agent tasks in sequential, parallel, or looped workflows. Includes “handoff” workflows where one agent can seamlessly pass control to another.
  - **Pre-built Agents & Functions**: A set of ready-made agents:
    1. **Data Analyst (Code Interpreter)**: Executes Python code, processes data files (CSV/Excel), and returns plots or analysis results.
    2. **Researcher Agent**: Gathers information, performs literature review tasks, and synthesizes findings.
    3. **Coder Agent**: Writes or reviews code based on user prompts.
    4. **Search & RAG (Retrieval-Augmented Generation)**: Performs document retrieval (e.g., from a vector store) and uses LLMs to generate answers with citing.
- **Reference Implementations & Examples (`examples/`)**:
  - **`code_interpreter/`**: A fully functional “Code Interpreter” project showcasing how to integrate the Data Analyst agent into a web UI (e.g., Chainlit).
  - Additional folders (e.g., `researcher_agent/`) demonstrating step-by-step tutorials for other agent types.
- **Tests Codes (`tests/`)**:
  - Unit tests and sample scripts that exercise core modules of the SDK. For instance, verifying that the AssistantAgent can load tools, maintain memory, and produce streaming responses.

---

## 3. Install

You have two options:

### 3.1. Install from python pip

```bash
pip install openagents-sdk
```

This will install the latest stable release of OpenAgents SDK and all dependencies listed in `requirements.txt`.

### 3.2. Install from Source

```bash
git clone https://github.com/openagentsfoundation/openagents-sdk.git
cd openagents-sdk
pip install -e .
```

- `pip install -e .` installs the SDK itself in “editable” mode so you can make changes locally.

---

## 4. Quick Start

### 4.1. Initialize a Basic Agent (Python)

Below is a minimal example demonstrating how to create an AssistantAgent that uses a simple `calculator` tool. Save this as `quickstart_agent.py` (or run directly from `tests/test_assistant_agent.py`).

```python
# quickstart_agent.py

import asyncio
from openagent import AssistantAgent, AgentStream
from openagent.common_tools import calculator

async def chat_loop() -> None:
    agent = AssistantAgent(
        name="Calculator Agent",
        instructions="You are a calculator assistant that helps users solve mathematics problems.",
        tools=[calculator()]
    )

    print("Type a math expression (or 'exit' to quit):")
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            print("Please enter something.")
            continue
        if user_input.lower() == "exit":
            break

        # Request a streaming response
        response_stream: AgentStream = await agent.run(user_input, stream=True)
        print("AI:", end=" ", flush=True)
        async for chunk in response_stream:
            if not chunk.done: # token by token chunks of reasoning steps and answer
                print(chunk.text, end="", flush=True)
            else:              # the final answer
                print(f"\nAI: {chunk.text}")

if __name__ == "__main__":
    asyncio.run(chat_loop())
```

### 4.2. Run the Example

```bash
python quickstart_agent.py
```

You should see a prompt like `You: `. Type a math expression (e.g., `23 * 47 + 5`) and watch the agent compute the result using the embedded `calculator` tool.

### 4.3. Explore More Examples

Browse the `examples/` directory for additional tutorials:

- **Data Analyst agent** (Code Interpreter):

  - Located at `examples/code_interpreter/`.
  - Walks through setting up a Chainlit UI, uploading CSV/Excel files, and examining plots.

- **Researcher agent**:

  - Sample code illustrating how to query academic sources and summarize results.

- **Coder agent**:

  - Demonstrates code generation and review workflows.

---

## 5. Project Mission

Our mission is to foster an open ecosystem where features, improvements, and new open-source agents for industry can be contributed by anyone. By providing robust, reusable components (agent memory, tool interfaces, workflows) and clear examples, we aim to:

- **Bolster Innovation**: Allow researchers and developers to prototype new agent types quickly.
- **Support Real-World Adoption**: Offer production-ready templates that can be extended for custom enterprise scenarios (e.g., data analysis, research assistance, automated coding).
- **Encourage Community Contributions**: Welcome pull requests, issue reports, and new example submissions that demonstrate novel use cases.

---

## 6. Requirements

All required Python packages are listed in [`requirements.txt`](./requirements.txt). At a minimum, you’ll find:

```
openai
mcp
pandas
numpy
matplotlib
chainlit
# ...other dependencies used by built-in tools and agents
```

Install them with:

```bash
pip install -r requirements.txt
```

---

## 7. Contributing

1. Fork the repository on GitHub.
2. Create a new branch for your feature or bug fix:

   ```bash
   git checkout -b feature/my-new-agent
   ```

3. Install dependencies and set up your development environment:

   ```bash
   pip install -e .
   ```

4. Make your changes, add tests under `tests/`, and ensure all tests pass:

   ```bash
   pytest
   ```

5. Commit your changes, push to your fork, and submit a pull request.

Please follow standard Python style (PEP 8) and write clear documentation for any new modules or examples.

---

## 8. License

This project is licensed under the **MIT License**. See the [`LICENSE`](./LICENSE) file for details.
Feel free to fork, modify, and redistribute under the same terms.
