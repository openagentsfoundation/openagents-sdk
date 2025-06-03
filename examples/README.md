# Code Interpreter

This project is a fully open-sourced implementation of OpenAI’s Code Interpreter, built on top of the [OpenAgents SDK](https://github.com/your-repo/openagents-sdk) and Chainlit. It provides a web-based UI where users can upload, execute, and analyze code (e.g., Python scripts, data analysis notebooks) in a sandboxed environment. The goal is to replicate the functionality of OpenAI’s Code Interpreter but with a transparent, self-hosted setup.

---

## Table of Contents

1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Directory Structure](#directory-structure)
4. [Installation & Setup](#installation--setup)

   * [1. Download and Install NGINX (Windows)](#1-download-and-install-nginx-windows)
   * [2. Clone this Repository](#2-clone-this-repository)
   * [3. Create and Configure `.env`](#3-create-and-configure-env)
   * [4. Configure NGINX](#4-configure-nginx)
   * [5. Install Python Packages](#5-install-python-packages)
5. [Usage](#usage)

   * [1. Start NGINX](#1-start-nginx)
   * [2. Start Code Interpreter UI](#2-start-code-interpreter-ui)
6. [Environment Variables (`.env`)](#environment-variables-env)
7. [NGINX Configuration](#nginx-configuration)
8. [Notes & Tips](#notes--tips)
9. [License](#license)

---

## Features

* **Full Open-Source**: Implements the core logic of Code Interpreter using the OpenAgents SDK and Chainlit, with no proprietary code.
* **Web-Based UI**: A Chainlit-powered frontend where users can:

  * Upload Python scripts, data files (CSV, Excel), or Jupyter notebooks.
  * Run code in a sandboxed “sandbox” directory.
  * View results (plots, tables, logs) directly in the browser.
* **Sandboxed Execution**: All code executes inside a controlled local directory (configurable via `.env`) to prevent unauthorized file system access.
* **NGINX Reverse Proxy**: Uses NGINX (Windows build) to serve static assets (for example, sandboxed data) via HTTP/HTTPS.
* **Extensible**: Built on top of OpenAgents’s agent framework—easily swap out or extend to other LLM endpoints (e.g., Azure OpenAI, OpenAI, local LLM).

---

## Prerequisites

1. **Windows 10+ / Windows Server 2016+**
2. **Python 3.10+**
3. **NGINX (Windows version)**
4. **Git (for cloning this repository)**
5. An LLM endpoint (e.g., Azure OpenAI or OpenAI) and corresponding API key/deployment settings.

---

## Directory Structure

```
code_interpreter/
├── nginx-windows/                      # Downloaded NGINX Windows build
│   ├── conf/
│   │   └── nginx.conf                  # NGINX configuration (modify as needed)
│   └── nginx.exe                       # NGINX Windows binary
├── .env                                # Environment variables (not checked into Git)
├── code_interpreter.cl.py              # Chainlit script to run the Code Interpreter UI
├── requirements.txt                    # Python dependencies (Chainlit + data analysis libs)
├── run_code_interpreter.bat            # Windows batch file: runs Chainlit UI
└── start_nginx.bat                     # Windows batch file: starts NGINX process
```

---

## Installation & Setup

### 1. Download and Install NGINX (Windows)

1. Visit the [official NGINX download page](http://nginx.org/en/download.html) and download the **“nginx/x.x.x (zip)”** for Windows.
2. Unzip the downloaded archive.
3. Rename or move the entire `nginx-x.x.x` folder into this project’s root, and rename it `nginx-windows`.

   * After moving, you should see:

     ```
     code_interpreter/
     └── nginx-windows/
         ├── nginx.exe
         ├── conf/
         └── ...
     ```
4. Verify you can run NGINX by double-clicking `code_interpreter/nginx-windows/nginx.exe` or via the batch file in the next section.

---

### 2. Clone this Repository

```bash
git https://github.com/openagentsfoundation/openagents-sdk
cd examples/code_interpreter
```

---

### 3. Create and Configure `.env`

In the project root, create a file named `.env`. Copy the example below and replace with your own LLM endpoint, API key, and deployment names. Update the `SANDBOX_LOCAL` path if you want a different sandbox folder.

```dotenv
# ──────────
# LLM Endpoint Settings
# ──────────

# Primary text-model (e.g., GPT-4/4o)
OPENAGENT_OPENAI_LLM_ENDPOINT="https://your-azure-openai-endpoint.openai.azure.com/"
OPENAGENT_OPENAI_LLM_API_KEY="YOUR_LLM_API_KEY"
OPENAGENT_OPENAI_LLM_DEPLOYMENT="gpt-4o"
OPENAGENT_OPENAI_LLM_API_VERSION="2024-12-01-preview"

# Code-model (e.g., GPT-4.1 or code-specialized model)
OPENAGENT_OPENAI_CODE_ENDPOINT="https://your-azure-openai-endpoint.openai.azure.com/"
OPENAGENT_OPENAI_CODE_API_KEY="YOUR_CODE_MODEL_API_KEY"
OPENAGENT_OPENAI_CODE_DEPLOYMENT="gpt-4.1"
OPENAGENT_OPENAI_CODE_API_VERSION="2024-12-01-preview"

# ──────────
# Sandbox Settings
# ──────────

# Local directory where user-uploaded code will run
SANDBOX_LOCAL="C:/dev/sandbox"
# Public URL exposed by NGINX to serve sandbox outputs
SANDBOX_BLOB="http://localhost:8080/blob"
```

> **Tip:** On Windows, use forward slashes (`/`) in the `SANDBOX_LOCAL` path or escape backslashes (`C:/dev/sandbox`).

---

### 4. Configure NGINX

1. Open `./nginx-windows/conf/nginx.conf` in a text editor.

2. Update the following sections to match your `.env` settings (especially the `SANDBOX_BLOB` URL and the local directory path you want to expose).

   ```nginx
   worker_processes  1;

   events {
       worker_connections  1024;
   }

   http {
       include       mime.types;
       default_type  application/octet-stream;
       sendfile        on;
       keepalive_timeout  65;

       server {
           listen       8080;
           server_name  localhost;

           # Serve static files from the sandbox directory:
           location /blob/ {
               alias  C:/dev/code_interpreter/sandbox/;   # <-- adjust to your SANDBOX_LOCAL
               autoindex on;
           }
       }
   }
   ```

3. Save and close `nginx.conf`.

---

### 5. Install Python Packages

1. Make sure you have Python 3.10+ installed.

2. (Optional but recommended) Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install the core OpenAgents SDK:

   ```bash
   pip install openagents-sdk
   ```

4. Install all other dependencies via `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` should include at least:

   ```
   chainlit
   pandas
   matplotlib
   numpy
   # ...any other data analysis packages you need
   ```

---

## Usage

### 1. Start NGINX

Open a Command Prompt (or PowerShell) in the project root (`examples/code_interpreter/`) and run:

```batch
start_nginx.bat
```

* This will launch `nginx.exe` using the configuration file at `./nginx-windows/conf/nginx.conf`.
* You should see a new Command Prompt window open for the NGINX process. Leave it running.

> **Tip:** If you need to stop NGINX, go to the NGINX console window and press `CTRL+C`, or run `nginx -s stop` from the NGINX directory.

---

### 2. Start Code Interpreter UI

#### Option A: Using the Batch File

From the same project root (`code_interpreter/`), run:

```batch
run_code_interpreter.bat
```

The contents of `run_code_interpreter.bat` should be:

```batch
@echo off
REM Starts the Chainlit app for Code Interpreter
chainlit run code_interpreter.cl.py
```

* This spins up a local web server (default port: 8000).
* Once started, open your browser and navigate to `http://localhost:8000/`.

#### Option B: Manually via Chainlit

If you prefer to run it manually or customize the port:

1. Activate your Python virtual environment (if using).

2. Run:

   ```bash
   chainlit run code_interpreter.cl.py --host 0.0.0.0 --port 8000
   ```

3. Visit `http://localhost:8000/` in your browser.

---

## Environment Variables (`.env`)

Below is an example `.env` (for reference). **You must replace** the placeholders (`YOUR_LLM_API_KEY`, etc.) with your actual LLM endpoint and API key. Adjust `SANDBOX_LOCAL` to a valid folder on your machine.

```dotenv
# ──────────
# LLM Endpoint Settings
# ──────────

# Primary Chat LLM (e.g., GPT-4 / GPT-4o)
OPENAGENT_OPENAI_LLM_ENDPOINT="https://kslabopenai5.openai.azure.com/"
OPENAGENT_OPENAI_LLM_API_KEY="b963c8b33ab747b79d3031ac7baf3c21"
OPENAGENT_OPENAI_LLM_DEPLOYMENT="gpt-4o"
OPENAGENT_OPENAI_LLM_API_VERSION="2024-12-01-preview"

# Code-specialized LLM (e.g., GPT-4.1)
OPENAGENT_OPENAI_CODE_ENDPOINT="https://kslabopenai5.openai.azure.com/"
OPENAGENT_OPENAI_CODE_API_KEY="b963c8b33ab747b79d3031ac7baf3c21"
OPENAGENT_OPENAI_CODE_DEPLOYMENT="gpt-4.1"
OPENAGENT_OPENAI_CODE_API_VERSION="2024-12-01-preview"

# ──────────
# Sandbox Settings
# ──────────

# Local path where user code is executed
SANDBOX_LOCAL="C:/dev/code_interpreter/sandbox"
# Public URL via NGINX to serve sandbox outputs
SANDBOX_BLOB="http://localhost:8080/blob"
```

> **Important:**
>
> * Do **not** commit your `.env` to version control—keep it secret.
> * Ensure the SANDBOX\_LOCAL directory exists. Create it if necessary:
>
>   ```bash
>   mkdir C:\dev\code_interpreter\sandbox
>   ```

---

## NGINX Configuration

Open `./nginx-windows/conf/nginx.conf` and verify the following:

```nginx
worker_processes  1;

events {
    worker_connections  1024;
}

http {
    include       mime.types;
    default_type  application/octet-stream;
    sendfile        on;
    keepalive_timeout  65;

    server {
        listen       8080;
        server_name  localhost;

        # Serve static files (sandbox results) here:
        location ~ ^/blob/([^/]+)/data/(.*)$ {
            alias c:/dev/sandbox/$1/data/$2;   # <-- match SANDBOX_LOCAL
            autoindex off; # disable directory listing
        }
    }
}
```

* If you change `SANDBOX_LOCAL` in `.env`, be sure to update the `alias` path accordingly.
* If you want to serve over HTTPS, you’ll need to configure SSL certificates in this file (beyond the scope of this README).

---

## Notes & Tips

* **Sandbox Safety**: All code runs inside the `SANDBOX_LOCAL` directory. Avoid pointing it to sensitive system folders.
* **Data Analysis Examples**: Place sample CSV or Excel files in `sandbox/input/` (you may need to create `sandbox/input` yourself). The UI will show an upload button—uploaded files go to `sandbox/input/` by default.
* **Logs & Debugging**:

  * Chainlit outputs logs to your console.
  * For deeper debugging, set `LOG_LEVEL=DEBUG` in your environment before running Chainlit:

    ```bash
    set LOG_LEVEL=DEBUG
    run_code_interpreter.bat
    ```
* **Customizing the UI**:

  * The main app logic lives in `code_interpreter.cl.py`. Feel free to modify the Chainlit flows, add new commands, or tweak the front end.
* **Port Conflicts**:

  * By default, Chainlit uses port 8000, and NGINX uses 8080. If those ports are in use, specify alternative ports:

    ```bash
    chainlit run code_interpreter.cl.py --port 9000
    ```

    and update `SANDBOX_BLOB="http://localhost:8081/blob"` (and `nginx.conf → listen 8081;`) accordingly.

---

## License

This project is released under the [MIT License](./LICENSE). Feel free to fork, modify, and redistribute under the same terms.
