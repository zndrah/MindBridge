# 🌿 MindBridge 2.0 — Native Windows Execution Guide

This guide describes how to run and stop **MindBridge 2.0** on **Windows 10/11** using the native Windows launcher scripts without WSL2 or Linux dependencies.

---

## 🏗️ Architecture & Runtime Overview

MindBridge runs natively on Microsoft Windows:
- **Execution Environment**: Native Windows 10/11 x86-64.
- **Python Environment**: Isolated local virtual environment at `.venv\Scripts\python.exe`.
- **Node.js Environment**: Native Windows Node.js and npm.
- **User Interface**: Rendered in your host Windows browser (`http://localhost:5173`).
- **Backend Service**: FastAPI ASGI application running on port `8000`.
- **Frontend Service**: React + Vite development server running on port `5173` with reverse proxy to port `8000`.
- **Process Orchestration**: Command Prompt (`cmd.exe`) opens independent, live console windows for both the backend and frontend.

```
                      +------------------------------------------+
                      |         Windows Host Environment         |
                      |                                          |
                      |   Default Web Browser                    |
                      |   http://localhost:5173                  |
                      +--------------------+---------------------+
                                           |
                        +------------------+------------------+
                        |                                     |
                        v                                     v
             +-----------------------+             +-----------------------+
             |   Console Window      |             |   Console Window      |
             |  MindBridge Backend   |             |  MindBridge Frontend  |
             +-----------+-----------+             +-----------+-----------+
                         |                                     |
+------------------------v-------------------------------------v------------------------+
| Windows 10 / 11 Native Host (.venv\Scripts\python.exe & node.exe)                      |
|                                                                                       |
|   +------------------------------------+   +------------------------------------+     |
|   | FastAPI Backend (:8000)            |   | Vite Frontend (:5173)              |     |
|   | - RoBERTa Crisis Classifier (CPU)  |   | - React 18 Workspace               |     |
|   | - BGE-small Embeddings + FAISS     |   | - Reverse proxy /ready -> :8000    |     |
|   | - Deterministic Safety Graph       |   | - Hot Module Reloading (HMR)       |     |
|   +------------------------------------+   +------------------------------------+     |
+---------------------------------------------------------------------------------------+
```

---

## 🛠️ First-Time Setup (`FIRST_TIME_SETUP.bat`)

Before running for the first time on a new computer:
1. Double-click `FIRST_TIME_SETUP.bat`.
2. This automated script checks:
   - Windows host OS version.
   - Python 3.11 availability.
   - Node.js and npm availability.
   - Disk space.
   - Creates a native Windows virtual environment (`.venv`).
   - Installs backend dependencies from `requirements-windows.txt` (PyTorch CPU, FAISS CPU, Transformers, FastAPI).
   - Installs frontend packages via `npm ci`.
   - Validates that RoBERTa models, BGE embeddings, and FAISS vector indexes are present.
   - Initializes a fresh `.env` file from `.env.example`.

---

## 🚀 Running MindBridge (`run.bat`)

To launch the full system:
1. Double-click `run.bat`.
2. The script:
   - Validates setup and model files.
   - Checks that ports `8000` and `5173` are not in use.
   - Launches the FastAPI backend in a separate window titled `MindBridge Backend (FastAPI)`.
   - Launches the React frontend in a separate window titled `MindBridge Frontend (Vite)`.
   - Pre-warms the RoBERTa model and BGE FAISS indexes in memory.
   - Continuously queries `http://127.0.0.1:8000/ready` until all components are ready.
   - Automatically opens `http://localhost:5173` in your default browser.

---

## 🛑 Stopping MindBridge (`stop.bat`)

To cleanly shut down:
1. Double-click `stop.bat`.
2. The script:
   - Reads the recorded process IDs from `.run\backend.pid` and `.run\frontend.pid`.
   - Verifies the process status and terminates only the MindBridge process trees.
   - Queries `netstat -ano` to confirm ports `8000` and `5173` are freed.
   - Cleans up runtime PID tracking files.
   - **Never kills unrelated python or node instances.**
