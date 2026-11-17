# 🌸 MindBridge 2.0 — Native Windows Teammate Setup & Quickstart Guide

Welcome to **MindBridge 2.0**! This guide will walk you through setting up and running the project natively on your Windows laptop in just a few minutes.

> [!NOTE]
> **No Linux, WSL2, or Docker required!**  
> MindBridge 2.0 now runs 100% natively on Microsoft Windows 10 and 11 using native Windows Python and Node.js.

---

## 🌿 1. What is MindBridge?

MindBridge is an agentic, AI-powered mental health first-response companion and safe conversational workspace:
- **Crisis Detection**: A fine-tuned RoBERTa model that immediately detects crisis or self-harm disclosures and triggers automated emergency bypass with certified crisis hotlines.
- **Grounded Support**: A Retrieval-Augmented Generation (RAG) system using dense vector embeddings (BGE-small + FAISS) grounded in clinical rubrics and validated psychoeducation materials.
- **Calm, Sanctuary UI**: A soothing React + Vite frontend workspace designed with empathetic non-clinical styling.
- **Privacy-by-Design**: Ephemeral sessions, zero user message logging, and strict data sanitization.

---

## 💻 2. System Requirements

To run MindBridge on your Windows laptop, you need:
- **Operating System**: Windows 10 (64-bit) or Windows 11.
- **Python**: Python 3.11 (e.g. Python 3.11.9 from [python.org](https://www.python.org/downloads/)).
  *(Important: Check the box "Add python.exe to PATH" during installation!)*
- **Node.js**: Node.js 18+ or 20+ LTS from [nodejs.org](https://nodejs.org/).
- **Disk Space**: At least 4 GB free space for models and dependencies.
- **RAM**: 8 GB minimum (16 GB recommended).

---

## 📂 3. Extracting the Project

1. Download or copy `MindBridge_Portable.zip` to your computer.
2. Right-click `MindBridge_Portable.zip` and select **Extract All...**.
3. You can extract it to **any folder you want**, for example:
   - `C:\Users\YourName\Downloads\MindBridge`
   - `C:\Users\YourName\Desktop\MindBridge`
   - `D:\Projects\MindBridge`  
   *(Paths with spaces like `My Projects` are fully supported!)*

---

## 🛠️ 4. First-Time Setup (`FIRST_TIME_SETUP.bat`)

Before running MindBridge for the first time, run the automated setup script once to prepare your Python virtual environment and install dependencies.

1. Open the extracted folder in Windows File Explorer.
2. Double-click:
   ```cmd
   FIRST_TIME_SETUP.bat
   ```
3. The setup script will automatically:
   - Verify your Windows environment.
   - Detect your Python 3.11 installation.
   - Detect Node.js and npm.
   - Create a clean local virtual environment at `.venv`.
   - Install CPU-optimized PyTorch, FAISS CPU, Hugging Face Transformers, and FastAPI via `requirements-windows.txt`.
   - Install React frontend dependencies via `npm ci`.
   - Verify that all AI/ML model weights and FAISS vector indexes are intact.
   - Create your initial `.env` file from `.env.example`.
4. When finished, you will see:
   ```
   ==============================================================================
     M I N D B R I D G E   2 . 0  --  S E T U P   C O M P L E T E D !
   ==============================================================================
   ```

---

## 🔑 5. Adding Your API Keys (Optional)

MindBridge supports cascading LLM generation:
1. **Primary Provider**: Google Gemini Developer API (`gemini-3.5-flash-lite`).
2. **Failover Provider**: Groq Cloud API (`llama-3.3-70b-versatile`).
3. **Offline Fallback**: Deterministic grounded psychoeducation response.

### How to configure:
1. Open the file `.env` in the project root using **Notepad** (or any text editor).
2. Find the lines:
   ```ini
   GEMINI_API_KEY=your_gemini_api_key_here
   GROQ_API_KEY=your_groq_api_key_here
   ```
3. Replace the placeholder with your personal API key (free keys are available at [Google AI Studio](https://aistudio.google.com/) and [Groq Console](https://console.groq.com/)).
4. Save and close the file.

> [!TIP]
> **No API keys? No problem!**  
> If you don't add API keys, MindBridge automatically operates in **deterministic grounded fallback mode**. In this mode, the local RoBERTa classifier, BGE dense embeddings, and clinical FAISS vector database still perform full semantic search and safe response composition offline.

---

## 🚀 6. Launching MindBridge (`run.bat`)

Whenever you want to use MindBridge:

1. Double-click:
   ```cmd
   run.bat
   ```
2. The launcher will:
   - Check that ports `8000` and `5173` are free.
   - Start the FastAPI backend in a dedicated console window (`MindBridge Backend`).
   - Start the React/Vite frontend in a dedicated console window (`MindBridge Frontend`).
   - Pre-warm the local RoBERTa crisis classifier and BGE dense vector index.
   - Continuously poll the `/ready` endpoint until models are loaded into memory.
   - Automatically open your default web browser to:
     **`http://localhost:5173`**
3. You can now chat with MindBridge, test grounded responses, explore grounding exercises, or test crisis safety bypasses!

---

## 🛑 7. Stopping MindBridge (`stop.bat`)

When you are done testing:

1. Double-click:
   ```cmd
   stop.bat
   ```
2. The stopper will:
   - Identify only the backend and frontend processes spawned by MindBridge.
   - Safely terminate their process trees without affecting any other running Python or Node applications.
   - Clean up runtime PID files and verify that ports `8000` and `5173` are released.

---

## ❓ 8. Troubleshooting

### Python Not Recognized
- **Symptom**: `[ERROR] Python was not found on your system!`
- **Solution**: Install Python 3.11 from [python.org](https://www.python.org/downloads/). During installation, make sure to check the box **"Add python.exe to PATH"**.

### Node / npm Not Recognized
- **Symptom**: `[ERROR] Node.js was not found on your system!`
- **Solution**: Install Node.js LTS from [nodejs.org](https://nodejs.org/) and restart your terminal / File Explorer.

### Port 8000 or 5173 Already in Use
- **Symptom**: `[ERROR] Port 8000 is already in use by another process!`
- **Solution**: Double-click `stop.bat` to terminate any previous MindBridge instances, or check `netstat -ano | findstr :8000` in Command Prompt to see what application is using that port.
