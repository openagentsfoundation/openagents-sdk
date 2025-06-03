# References projects:
from .utils import env as env
import os, subprocess, json, csv
import pandas as pd
from typing import Optional, override, Dict, Any, Tuple
from pydantic import BaseModel, ValidationError
from datetime import datetime
from enum import Enum
# OpenAI chat model
from .base_agent import (
    BaseAgent,
    FunctionTool,
    function_tool,
    MAX_STEPS,
    BaseChatClient,
    Prompt,
    Options,
    logging
)
from .react import ReactAgent, MAX_STEPS
from .utils.multitask import run_in_thread_pool
from .utils import format_template_with_json
from .mcp import *

SANDBOX_LOCAL = env.get("SANDBOX_PATH", "c:/dev/sandbox")
SANDBOX_BLOB = env.get("SANDBOX_BLOB", "http://localhost:8080/blob")
# converted string to list [,,,]
SANDBOX_LIBS = json.loads(env.get("SANDBOX_LIBS", '["pandas", "sympy", "numpy", "matplotlib", "plotly", "seaborn", "scikit-learn", "statsmodels"]'))

AGENT_INSTRUCTION = """{{user_instructions}}

You are good at leveraging coding skills to fulfill data analysis tasks.
## Guidlines
Your tasks include analyzing CSV formatted data and generating Python code to address user queries.
Follow these guidelines:
0. Rephrase user query in professional data analytics language, clarify with user if task is not clear.
1. The user may provide mathematics problem to solve, use python coding tools to solve the problem.
2. The user may provide CSV data in one of below manners, follow below rules to pre-process the data:
  - CSV formatted data in user query, proceed the data analysis task
  - The name of a CSV file located in the directory `{{sandbox_local}}/data` or a web url to the CSV file.
    -- Call `locate_and_desc_csv_file` to locate and describe the CSV content samples before writing codes to analyze.
  - The API to fetch data from business system and populate data as CSV file
3. The user will also supply context, including:
  - Column names and their descriptions. (optional)
  - Sample data from the CSV (headers and a few rows) to help understand data types.
4. The user will provide data analysis task
5. Call `write_and_run_python_code` to analyze the data via coding (use English for arguments) with detailed data analysis task description:
   - Start with basic data analysis models unless user explicitly ask for deep or advanced analysis. e.g.,:
      - start with `ExponentialSmoothing` for seasonality data forecast.
      - start with Random Forrest for feature importance analysis.
6. Summarization: interpret the output results of the code execution and provide analysis report to the user:
  - The **final_answer** content should be written in **markdown** and in **language** of user's task.
  - For any files included in the final answer, change the local directory `{{sandbox_local}}/data` to web-based directory `{{sandbox_blob}}/data`
  - Files should be in **markdown** for example: "![Daily Sales Revenue Over Time](.../daily_sales_timeseries.png)"
  - Include detailed **data tables** in the report for better interpretation if available.
  - Include **visualization** diagram in the report for better interpretation and readability.
  - Provide **insights** and **recommendations** with support data in business language.

## Output language: in the same language as the user's task.
"""

CODER_SYSTEM_PROMPT = """You are a Python code assistant specializing in data analysis, forecasting, and visualization.
You will receive a user task and context.
Your job is to generate Python code to fulfill the user's task.
**Only** output Python code enclosed in triple-backticks with "python" as the language tag:
```python
# Your code here

and any necessary code comments. Do not output anything else.

## Pre-installed Libraries
You may use any of these:
{{use_libs}}

## Language: English
Always use **English** for text content, comments, filename, plot labels in your generated code.

## Sandbox & Runtime Rules
1. **No GUI pop-ups** — e.g. do **not** call plt.show().
2. **Data paths**: All input files live under {{sandbox_local}}/data. Any output files or saved figures must also go there.
3. **Filename sanitization**
    - When you build filenames from variable names, **replace all non-alphanumeric characters** (spaces, `/`, `(`, `)`, `-`, etc.) with underscores.  
    - Example:  
      ```python
      safe_var = re.sub(r'[^0-9A-Za-z]+', '_', var)
      plot_path = f"{BASE_DIR}/trend_{safe_var}_vs_yield.png"
      ```
4. **Quoting in f-strings**
    - **Do not** escape inner quotes with backslashes.
    - Alternate quote styles instead.
5. **Stdout only**: Code outputs via print() to console.
6. **No outside access**: Do not read or write outside {{sandbox_local}}/data.
7. **Security**: Refuse to generate code that compromises security.

## Data Loading & Preprocessing
- Read CSVs, parse date/time columns correctly, and set an index if needed.
- Handle duplicates and missing values.
- Convert data types as required.
- For forcast task, **sort** the dataset first.

## Exploratory Data Analysis (EDA)
- Print summary statistics (.describe()) for relevant columns.
- If visuals are requested, save line plots, histograms, box plots to files (not plt.show()).

## Advanced Analytics (any task)
- **Correlation analysis**: compute correlation matrices, heatmaps, and interpret key drivers.
- **Feature importance**: use tree-based models or permutation methods to rank predictors.
- **Clustering / Dimensionality reduction**: apply K-means, PCA, t-SNE, etc., when asked.
- **Forecasting (trend & seasonality)**
  - If user asks for forecast, prefer `ExponentialSmoothing` from `statsmodels.tsa.holtwinters` over ARIMA or Prophet for seasonality data analysis (apply seasonal),
    or another method that fits the data's characteristics.
  - If user asks for deep analysis, follow below rules:
    - **Detect seasonality** in the series (e.g. via autocorrelation plots or seasonal decomposition).  
      - If seasonality is present, use `ExponentialSmoothing` from `statsmodels.tsa.holtwinters`, choosing additive or multiplicative seasonality as appropriate.  
    - **No or weak seasonality**: select a more suitable model — e.g.:
      - **ARIMA** or **SARIMA** for autoregressive and integrated components,  
      - **Simple Exponential Smoothing** for baseline smoothing,  
      - **Prophet** for flexible holiday/trend modeling,
      - or another method that fits the data's characteristics.
  - **Model selection & tuning**:  
    - Compare candidate models on hold-out or cross-validation metrics (e.g. AIC, RMSE).  
    - Print or save the chosen model's forecast vs. actual plot and its performance scores.
- **Any other statistical or ML technique** that the user requests.

## Data Visualization & Output
- All plots must have titles, axis labels, and legends.
- Save figures to {{sandbox_local}}/data and print() their full paths.
- Print any summary numbers or tables to stdout.
- For Chinese labels for plot, use the proper font settings:
  ```python
  import matplotlib.pyplot as plt
  plt.rcParams['font.sans-serif'] = ['SimHei']          # for Chinese characters
  plt.rcParams['axes.unicode_minus'] = False            # to correctly show minus sign
  ```

Your final code must:
- Load and preprocess the CSV.
- Do detailed EDA and visualizations.
- Perform the **requested analytics** (correlation, feature importance, clustering, forecasting, etc.).
- Output clear **visual** and **statistical** results.
- Print full paths of any created files.

## Context
'''
{{context}}
'''
"""

def detect_delimiter(filename):
    with open(filename, 'r', newline='') as f:
        # Read a sample of the file to analyze the delimiter
        sample = f.read(2048)
        sniffer = csv.Sniffer()
        # Sniff and check against common delimiters, e.g., comma and tab
        dialect = sniffer.sniff(sample, delimiters=[',', '\t'])
        return dialect.delimiter

class PythonRuntime(Enum):
    NativeUnsafe  = 1 # run codes directly, unsafe; import is allowed. recommend for test-only.
    DockerSafe  = 2 # reserved. run codes in docker process, safe
    JupyterSafe = 3 # reserved. run codes in jupyer notebook, safe
    RemoteSafe  = 4 # reserved. run codes in remote container, safe

class CodeInterpreterAgent(ReactAgent):
    def __init__(
        self,
        llm_client:Optional[BaseChatClient] = None, # assistant LLM
        code_client:Optional[BaseChatClient] = None , # coding LLM
        name:Optional[str] = "Code Interpreter",
        description:Optional[str] = "Solve data analysis tasks",
        instructions:Optional[str] = "You are code interpreter agent designated as professional data scientist.",
        tools:list[FunctionTool]|None = [], # extra tools
        mcps: Optional[list[MCPClient]] = [],
        runtime: PythonRuntime | None = PythonRuntime.NativeUnsafe,
        python_libs:list[str] | None = SANDBOX_LIBS,
        sandbox_local:str |None = SANDBOX_LOCAL,
        sandbox_blob:str |None = SANDBOX_BLOB,
        max_steps:int | None = MAX_STEPS,
        logger = None,
        verbose:bool|None=False):
        
        self.sandbox_local = sandbox_local
        self.sandbox_blob = sandbox_blob
        agent_instruct = str(AGENT_INSTRUCTION).replace("{{user_instructions}}", instructions)
        agent_instruct = format_template_with_json(agent_instruct, {
                "sandbox_local" : self.sandbox_local,
                "sandbox_blob" : self.sandbox_blob
                })
        self.runtime = runtime
        self.python_libs = python_libs or []
        tools = tools or []
        tools.extend([self.locate_and_desc_csv_file, self.write_and_run_python_code])

        super().__init__(
            llm_client = llm_client,
            name = name,
            description = description,
            instructions=agent_instruct,
            tools = tools,
            mcps = mcps,
            max_steps = max_steps,
            logger = logger,
            verbose = verbose)
        
        self.code_client = code_client or self.llm_client # set code agent LLM client
    
    #To-do: copy file to docker
    @function_tool
    async def locate_and_desc_csv_file(self, filepath:str)->str:
        """Locate and describe the contents of a CSV file: CSV data header and sample data rows
        Parameters:
            filepath (required): Full path or URL of the file to load and describe
        Returns:
            str: The location of file, readed CSV header and sample data data rows
        """
        if not filepath.endswith('.csv'):
            error_msg = "Error: The file is not a CSV file."
            self.log(f"{error_msg} - Invalid CSV file provided: {filepath}", level=logging.WARNING)
            return error_msg
        data_path = f'{self.sandbox_local}/data'
        # Ensure the path is correct
        if not os.path.dirname(filepath):
            filepath = os.path.join(data_path, filepath)
        self.log(f"Attempting to read file at path: {filepath}", level=logging.DEBUG)
        try:
            # Read the CSV file using the detected delimiter
            delimiter = detect_delimiter(filepath)
            df = pd.read_csv(filepath, delimiter=delimiter)
            self.log(f"File '{filepath}' loaded successfully.", level=logging.DEBUG)
            # If you need a CSV-format representation of the first 15 rows:
            # copy_output = self.copy_file_to_container(filename)
            head_str = df.head(15).to_csv(sep=delimiter, index=False)
            return f"The file ({filepath}) content for the first 15 rows is:\n{head_str}"
            #return f"{copy_output}\nThe file content for the first 15 rows is:\n{head_str}" # commented
        except FileNotFoundError:
            error_msg = f"Error: The file '{filepath}' was not found."
            self.log(error_msg, level=logging.WARNING)
            return error_msg
        except Exception as e:
            error_msg = f"Error while reading the CSV file: {str(e)}"
            self.log(error_msg, level=logging.WARNING)
            return error_msg        

    @function_tool
    async def write_and_run_python_code(self, task:str)->str:
        """Fulfill the task or requirement via coding, returns the written codes and python runtime output.
        Parameters:
            task (required): The **detailed** description of user task/requirement
        Returns:
            str: The python source codes in standalone program and the execution output of codes from docker.
        """
        python_code = await self.write_python_code(task)
        output = await self.execute_python_code(python_code)

        return f"## Python code:\n{python_code}\n\n## Python runtime output:\n{output}"
        
    async def write_python_code(self, task:str)->str:
        """Write codes in python and run the codes to fulfill the task or requirement.
        Parameters:
            task (required): A string containing the coding task/requirement
        Returns:
            str: The python source codes in standalone program.
        """
        uselibs = f"You can use Python libs: {', '.join(self.python_libs)}"
        coder_system_prompt = format_template_with_json(
            CODER_SYSTEM_PROMPT, 
            {
                "sandbox_local":self.sandbox_local,
                "use_libs":uselibs,
                "context":self.memory.get_messages_str()
            }
        )
        response = await self.code_client.send(system=coder_system_prompt,
                                               prompt=Prompt(text=task),
                                               stream=False)
        return response.text

    # execute python code, throw no exception. 
    ## any exceptions caught will be sent back to LLM for reflection
    async def execute_python_code(self, python_code: str) -> str:
        """Helper function that actually runs Python code and return the executed result in str.
        Parameters:
            python_code (required): A string of python code block.
        Returns:
            str: The result of the executed codes or an error message.
        """
        self.log(f"Execute Python Code:\n{python_code}\n", logging.DEBUG)
        python_code_stripped = python_code.strip()
        if python_code_stripped.startswith("```python"):
            python_code_stripped = python_code_stripped[len("```python"):]
        if python_code_stripped.endswith("```"):
            python_code_stripped = python_code_stripped[:-3]

        try:
            match self.runtime:
                case PythonRuntime.DockerSafe:
                    output, errors = await run_in_thread_pool(self._run_code_in_container, python_code_stripped)
                case PythonRuntime.NativeUnsafe:
                    output, errors = await run_in_thread_pool(self._run_code_native, python_code_stripped)
                case _:
                    return "Error: Invalid python runtime"
        
            result = ''
            if output:
                result += f"[STDOUT]\n{output}"
            if errors:
                result += f"[ALARM]\n{errors}"
            return result
        except Exception as e:
            result = f"[THROWN EXCEPTION]\n{str(e)}"
            self.log(result, logging.ERROR)
            return result

    def _run_code_in_container(self, python_code:str, container_name:str = "sandbox") -> Tuple[str, str]:
        """
        Helper function that actually runs Python code inside a Docker container named `sandbox` (by default).
        """
        cmd = [
            "docker", "exec", "-i",
            container_name,
            "python", "-c", "import sys; exec(sys.stdin.read())"
        ]
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        out, err = process.communicate(python_code)
        return out, err

    # only use it for testonly, it's unsafe.
    def _run_code_native(self, python_code: str) -> Tuple[str, str]:
        """
        Helper function that actually runs Python code in native runtime.
        Caution: it's unsafe method, which may result in illegal access to files outside of sandbox.
        """
        cmd = ["python", "-c", "import sys; exec(sys.stdin.read())"]
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True)
        out, err = process.communicate(python_code)
        return out, err
        
# Advanced Data Analytics
"""
You are a Python code generation expert specialized in data analysis, visualization, and time series forecasting. Your task is to write a complete, well-structured, and well-commented Python script that performs an end-to-end analysis on a provided CSV file. The CSV file contains historical time series data such as sales revenue and inventory, which typically exhibit seasonal patterns. Your generated Python code should include the following sections:

1. Data Loading & Preprocessing:
   - Import necessary libraries such as pandas, numpy, matplotlib, seaborn, and datetime.
   - Read the CSV file and correctly parse any date/time columns. Set an appropriate index if needed.
   - Handle missing values and duplicates, and convert data types as necessary to ensure the data is ready for analysis.

2. Exploratory Data Analysis (EDA):
   - Compute and print summary statistics for the relevant columns (e.g., sales revenue and inventory).
   - Visualize data distributions, trends, and seasonal patterns using line plots, histograms, and box plots.
   - Optionally perform a seasonal decomposition using statsmodels' `seasonal_decompose` to clearly display the trend, seasonal, and residual components of the data.

3. Forecasting with Seasonality (Including Holt-Winters):
   - Provide guidance on how to handle seasonality and trends in the data.
   - **Include a forecasting section with multiple approaches:**
       - **Holt-Winters Method:**
         Utilize the `ExponentialSmoothing` class from `statsmodels.tsa.holtwinters` to implement the Holt-Winters method. Clearly specify if an additive or multiplicative model should be used based on the data characteristics, and include steps for parameter tuning (e.g., smoothing level, trend, and seasonal components).
       - **Alternative Approaches:**  
         Optionally include guidance or code snippets for other forecasting methods like SARIMA (or auto_arima from pmdarima) and Prophet, but ensure that the Holt-Winters method is prominently featured.
   - Split the data into training and test sets. Fit the forecasting model on the training data and generate predictions for the test period.
   - Create plots comparing the actual data with forecasted values to visually assess model performance.

4. Data Visualization and Output:
   - Generate detailed plots for:
       - Time series trends with the forecast overlay.
       - Seasonal decomposition components (if applicable).
       - Residual plots to check for forecasting errors.
   - Save key plots to files and print summary statistics along with forecast error metrics such as MAE, MSE, or RMSE. These outputs should be clear and structured for further summarization by a data_insight_agent.

5. Code Structure & Documentation:
   - Organize the script by defining functions for data loading, preprocessing, EDA, forecasting, and visualization.
   - Add clear and thorough comments explaining each major step and the rationale behind model choice, especially for the Holt-Winters method.
   - Include error handling to manage possible issues such as file not found, parsing errors, or model convergence issues.

Your final script should be a standalone Python program that:
   - Loads and pre-processes a CSV file containing seasonal time series data.
   - Performs detailed exploratory analysis and visualizations.
   - Implements forecasting using the Holt-Winters method (ExponentialSmoothing) to capture trend and seasonal patterns, along with alternative methods if desired.
   - Outputs comprehensive visual and statistical results that can be later summarized by a data_insight_agent.

Please write the Python code accordingly.
"""