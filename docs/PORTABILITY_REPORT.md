# 🔍 MindBridge 2.0 — Portability & Diagnostic Audit Report

**Date**: November 11, 2026  
**Auditor**: Senior DevOps & Full-Stack Reliability Engineering Team  
**Scope**: Complete codebase inspection for portable Windows 11 + WSL2 teammate deployment.

---

## 1. Executive Summary

This diagnostic audit evaluates the portability of the **MindBridge 2.0** application across Windows 11 host machines running Ubuntu on WSL2. The goal is to produce a self-contained ZIP package that can be extracted to any folder on a teammate's computer and run without manual Linux configuration or dependencies on the author's machine (`/home/adarsh/`).

---

## 2. Comprehensive Portability Findings

### 2.1 Machine-Specific Absolute Paths & `/home/adarsh/` Dependencies
During code inspection, the following hardcoded references to `/home/adarsh/` and machine-specific paths were identified:

| File Location | Hardcoded Path Reference | Portability Risk | Remediation |
| :--- | :--- | :--- | :--- |
| `run.bat` | `PROJECT_DIR=/home/adarsh/projects/MindBridge` | High (Fails on teammate laptop) | Dynamically resolve via `%PROJECT_ROOT%` and `wslpath -u` |
| `stop.bat` | `PROJECT_DIR=/home/adarsh/projects/MindBridge` | High (Fails on teammate laptop) | Dynamically resolve via `%PROJECT_ROOT%` and `wslpath -u` |
| `scripts/start_backend.sh` | `PROJECT_DIR="/home/adarsh/projects/MindBridge"` | High (Fails on teammate laptop) | Dynamically resolve relative to script directory (`${BASH_SOURCE[0]}`) |
| `scripts/start_backend.sh` | `PATH="/home/adarsh/.local/bin:..."` | Medium (Node/uv lookup failure) | Dynamically resolve `uv` and `node` in PATH and user `~/.local/bin` |
| `scripts/start_frontend.sh` | `PROJECT_DIR="/home/adarsh/projects/MindBridge"` | High (Fails on teammate laptop) | Dynamically resolve relative to script directory (`${BASH_SOURCE[0]}`) |
| `scripts/start_frontend.sh` | `PATH="/home/adarsh/.local/bin:..."` | Medium (Node/npm lookup failure) | Dynamically resolve `node` and `npm` in PATH and user `~/.local/bin` |
| `scripts/stop_services.sh` | `PROJECT_DIR="/home/adarsh/projects/MindBridge"` | High (Fails on teammate laptop) | Dynamically resolve relative to script directory (`${BASH_SOURCE[0]}`) |
| `docs/ARCHITECTURAL_ALIGNMENT_PLAN.md` | `file:///home/adarsh/projects/...` | Low (Documentation link only) | Keep as historical doc or use relative paths |

*Note*: In `backend/` Python files, paths are dynamically derived relative to `Path(__file__).resolve().parent.parent` (e.g. in `backend/config.py`, `backend/retrieval/embeddings.py`, `backend/retrieval/ingestion.py`, and `backend/safety/crisis_detector.py`). The Python codebase itself is already clean of hardcoded absolute paths.

### 2.2 Dependencies on Windows Username
- Launchers previously assumed Windows username `adars` or default paths.
- The portable launchers must use `%USERPROFILE%`, `%SYSTEMROOT%`, and `%~dp0` to remain independent of any Windows username.

### 2.3 Hardcoded Extraction Directory
- No extraction directory is hardcoded in the application logic.
- Launchers are refactored to use `cd /d "%~dp0"` and `for %%I in ("%~dp0.") do set "PROJECT_ROOT=%%~fI"`, allowing the project to reside in any drive (`C:`, `D:`, etc.) or folder (including folders with spaces such as `C:\Users\Jane Doe\Downloads\MindBridge`).

### 2.4 Relative Path & Working Directory Dependencies
- Backend relies on `PROJECT_ROOT` being in `sys.path` (`PYTHONPATH=.`). Launchers must explicitly set `PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"` and `cd "${PROJECT_DIR}"`.
- Frontend relies on working directory being `frontend/` when running `npm run dev`. Launchers must explicitly `cd "${PROJECT_DIR}/frontend"`.

### 2.5 Required Model & Checkpoint Files
The application requires the following local model weights:

1. **RoBERTa Crisis Detection Classifier** (`models/crisis_detector/roberta/`):
   - `model.safetensors` (476 MB, fine-tuned classification head)
   - `config.json` (764 B, architecture config)
   - `threshold_config.json` (250 B, calibrated decision threshold = 0.509765625)
   - `tokenizer.json` (3.4 MB)
   - `vocab.json` (780 KB)
   - `merges.txt` (446 KB)
   - `special_tokens_map.json` (280 B)
   - `tokenizer_config.json` (1.3 KB)

2. **BAAI/bge-small-en-v1.5 Embedding Model** (`models/embeddings/bge-small-en-v1.5/`):
   - `model.safetensors` (128 MB, dense embedding weights)
   - `config.json` (743 B)
   - `tokenizer.json` (695 KB)
   - `vocab.txt` (227 KB)
   - `tokenizer_config.json` (366 B)
   - `special_tokens_map.json` (125 B)
   - `modules.json` (349 B)
   - `1_Pooling/config.json` (190 B)

*Integrity Check*: All above weights are verified present and intact on disk.

### 2.6 Required Knowledge Base & FAISS Files
The retrieval pipeline depends on:

1. **Canonical Clinical Knowledge Base** (`data/knowledge_base/`):
   - 22 Markdown files in `clinical_rubrics/`, `cognitive_reframing/`, `legal_and_privacy/`, `psychoeducation/`, and `somatic_regulation/`.
   - Registered in `backend/retrieval/governance.py` (`DOCUMENT_GOVERNANCE_REGISTRY`).
2. **Clinical FAISS Index**:
   - `models/embeddings/faiss_index/index.faiss` (47 KB, 31 vectors)
   - `models/embeddings/faiss_index/metadata.json` (90 KB, chunk metadata)
   - Mirror in `data/knowledge_base/faiss_index/`
3. **CounselChat Consolidated FAISS Index**:
   - `data/knowledge_base/counselchat_faiss_index/index.faiss` (2.9 MB)
   - `data/knowledge_base/counselchat_faiss_index/metadata.json` (8.2 MB)
4. **Safety Hotline Directory**:
   - `backend/safety/hotlines.json` (authoritative emergency contacts for IN, US, UK, CA, AU, GL).

### 2.7 Required Environment Variables & Secrets
- `APP_ENV`: Defaults to `development`.
- `HOST`: Defaults to `0.0.0.0`.
- `PORT`: Defaults to `8000`.
- `ALLOWED_ORIGINS`: Defaults to `http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173`.
- `DEFAULT_JURISDICTION`: Defaults to `IN`.
- `CRISIS_MODEL_DIR`: Defaults to `models/crisis_detector/roberta`.
- `EMBEDDING_MODEL_DIR`: Defaults to `models/embeddings/bge-small-en-v1.5`.
- `GEMINI_API_KEY`: **Optional**. If omitted, system engages local deterministic grounded fallback.
- `GROQ_API_KEY`: **Optional**. Secondary fallback.
- `ENABLE_GEMINI`: Defaults to `true`.

*Secret Protection*: No personal API keys or `.env` files will be packaged into the portable ZIP.

### 2.8 Startup Dependencies & Toolchain
- **Windows**: Windows 11 with WSL2 enabled, `wsl.exe`, `curl.exe`, Windows Terminal (`wt.exe` preferred, fallback to `cmd.exe`).
- **WSL Linux**: Ubuntu 22.04+ (tested on Ubuntu 26.04 LTS).
- **Python**: Python 3.11 (`>=3.11,<3.12`), managed via `uv`.
- **Node.js**: Node.js >= 18 and npm, for React + Vite frontend.

---

## 3. Package Size & Archiving Analysis

| Component | Uncompressed Size | Notes |
| :--- | :--- | :--- |
| `models/crisis_detector/roberta` | ~481 MB | Essential crisis classifier weights |
| `models/embeddings/bge-small-en-v1.5` | ~383 MB | Dense embedding model weights |
| `data/knowledge_base` | ~12 MB | Markdown rubrics + FAISS vector indexes |
| `backend/` source | ~2 MB | FastAPI application & LangGraph orchestrator |
| `frontend/` source | ~2 MB | React application source (excluding node_modules) |
| Configuration, scripts, docs | ~1 MB | Setup, launchers, manifests |
| **Total Included Runtime Assets** | **~881 MB** | **Compresses to ~650–750 MB in ZIP** |

### Excluded from Portable ZIP:
- `.venv/` (~1.4 GB) — Recreated cleanly via `uv sync` on teammate machine.
- `frontend/node_modules/` (~101 MB) — Recreated cleanly via `npm ci` on teammate machine.
- `.git/` (~15 MB) — Excluded to reduce size and isolate from dev branches.
- `.env` — Developer credentials excluded for security.
- `logs/` and `.run/` — Runtime ephemeral data.
- `data/raw/` (~27 MB) and `data/processed/*.csv` (~83 MB) — Raw training datasets not required for inference.

---

## 4. Remediation Plan

1. **Create `scripts/setup_linux_env.sh`**: Robust Linux environment preparation script with automated checks for `uv`, Python 3.11, Node, npm, model assets, and `.env` initialization.
2. **Create `FIRST_TIME_SETUP.bat`**: Windows entrypoint running pre-flight checks and delegating to `setup_linux_env.sh`.
3. **Refactor `run.bat` & `stop.bat`**: Universal path detection (`%~dp0`), WSL path resolution via `wslpath -u`, dedicated logging (`logs/`), and PID tracking (`.run/`).
4. **Refactor `scripts/start_backend.sh` & `scripts/start_frontend.sh`**: Fully remove hardcoded `/home/adarsh` references.
5. **Create `PACKAGE_FOR_TEAMMATE.bat`**: Automatic packaging script with asset validation, checksum generation, and ZIP creation.
6. **Create `docs/TEAMMATE_SETUP_GUIDE.md`**: Step-by-step documentation for the teammate.
7. **Create `docs/PACKAGE_MANIFEST.md`**: Complete file inventory and SHA-256 verification manifest.
