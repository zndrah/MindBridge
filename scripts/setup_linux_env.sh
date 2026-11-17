#!/usr/bin/env bash
# ==============================================================================
# MindBridge 2.0 — First-Time Linux Environment Setup (Portable)
# Location: scripts/setup_linux_env.sh
# Purpose: Prepares Python virtualenv, installs dependencies, validates models,
#          FAISS indexes, knowledge base, and configures .env.
# ==============================================================================

set -uo pipefail

# 1. Dynamically resolve project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORIGINAL_PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_DIR="${ORIGINAL_PROJECT_DIR}"

# If an explicit path argument was provided from Windows
if [ -n "${1:-}" ]; then
    RAW_PATH="$1"
    if [[ "${RAW_PATH}" =~ ^\\\\(wsl\.localhost|wsl\$)(\\|/)[^\\/]+(.*)$ ]]; then
        RESOLVED_PATH="${BASH_REMATCH[3]//\\//}"
    else
        RESOLVED_PATH="$(wslpath -u "${RAW_PATH}" 2>/dev/null || echo "${RAW_PATH}")"
    fi
    if [ -d "${RESOLVED_PATH}" ]; then
        ORIGINAL_PROJECT_DIR="${RESOLVED_PATH}"
        PROJECT_DIR="${ORIGINAL_PROJECT_DIR}"
    fi
fi

# ------------------------------------------------------------------------------
# High-Performance WSL2 Workspace Optimization (Section 5)
# If extracted on a Windows drive (/mnt/c/...), deploy a fast native ext4 copy.
# ------------------------------------------------------------------------------
IS_DRVFS=0
if [[ "${ORIGINAL_PROJECT_DIR}" =~ ^/mnt/[a-z]/ ]]; then
    IS_DRVFS=1
    RUNTIME_DIR="${HOME}/.mindbridge/runtime"
    mkdir -p "${RUNTIME_DIR}"
    echo "=================================================================="
    echo "  [WSL Optimization] Detected project extracted on Windows drive (/mnt/...)."
    echo "  Deploying fast native WSL2 ext4 workspace to avoid 9P filesystem lag..."
    echo "  Destination: ${RUNTIME_DIR}"
    echo "=================================================================="
    rsync -a --delete \
        --exclude='.venv' \
        --exclude='frontend/node_modules' \
        --exclude='.git' \
        --exclude='logs' \
        --exclude='.run' \
        "${ORIGINAL_PROJECT_DIR}/" "${RUNTIME_DIR}/"

    echo "${RUNTIME_DIR}" > "${ORIGINAL_PROJECT_DIR}/.wsl_runtime_path"
    PROJECT_DIR="${RUNTIME_DIR}"
else
    rm -f "${ORIGINAL_PROJECT_DIR}/.wsl_runtime_path" 2>/dev/null || true
fi

cd "${PROJECT_DIR}"

LOGS_DIR="${PROJECT_DIR}/logs"
RUN_DIR="${PROJECT_DIR}/.run"
mkdir -p "${LOGS_DIR}" "${RUN_DIR}"

SETUP_LOG="${LOGS_DIR}/setup.log"

# Log to both stdout and setup.log
exec > >(tee -a "${SETUP_LOG}") 2>&1

echo "=================================================================="
echo "  MindBridge 2.0 — First-Time Environment Setup"
echo "=================================================================="
echo " Active Project Directory : ${PROJECT_DIR}"
if [ "${IS_DRVFS}" -eq 1 ]; then
    echo " Windows Extracted Source : ${ORIGINAL_PROJECT_DIR}"
fi
echo " Setup Timestamp          : $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo " Host Architecture        : $(uname -m)"
echo " Linux Kernel             : $(uname -r)"
echo " Log Destination          : ${SETUP_LOG}"
echo "=================================================================="
echo ""

# Ensure user local bin is in PATH
export PATH="${HOME}/.local/bin:${PATH}"

# ------------------------------------------------------------------------------
# Step 1: Detect / Install `uv` Package Manager
# ------------------------------------------------------------------------------
echo "[1/7] Checking for Python package manager (uv)..."

if ! command -v uv >/dev/null 2>&1; then
    if [ -x "${HOME}/.local/bin/uv" ]; then
        export PATH="${HOME}/.local/bin:${PATH}"
    else
        echo "  -> uv was not found in PATH. Attempting automated installation..."
        if command -v curl >/dev/null 2>&1; then
            curl -LsSf https://astral.sh/uv/install.sh | sh
            export PATH="${HOME}/.local/bin:${PATH}"
        else
            echo "  [ERROR] Neither uv nor curl was found in this Linux environment."
            echo "  Please install uv manually inside WSL by running:"
            echo "      curl -LsSf https://astral.sh/uv/install.sh | sh"
            exit 1
        fi
    fi
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "  [ERROR] Failed to locate or install 'uv'."
    exit 1
fi

UV_VERSION=$(uv --version)
echo "  [OK] ${UV_VERSION} available."

# ------------------------------------------------------------------------------
# Step 2: Ensure Compatible Python (3.11)
# ------------------------------------------------------------------------------
echo "[2/7] Verifying Python 3.11 environment..."

# Instruct uv to ensure CPython 3.11 is installed hermetically
uv python install 3.11 >/dev/null 2>&1 || true

PYTHON_BIN=$(uv python find 3.11 2>/dev/null || command -v python3.11 || command -v python3)
echo "  [OK] Using Python: ${PYTHON_BIN}"

# ------------------------------------------------------------------------------
# Step 3: Install Backend Virtualenv & Dependencies
# ------------------------------------------------------------------------------
echo "[3/7] Setting up Python virtual environment and dependencies via uv sync..."

cd "${PROJECT_DIR}"
export UV_LINK_MODE=copy
uv sync --all-groups

if [ ! -f "${PROJECT_DIR}/.venv/bin/python" ]; then
    echo "  [ERROR] Virtual environment was not created at ${PROJECT_DIR}/.venv"
    exit 1
fi

echo "  [OK] Python virtual environment ready."

# ------------------------------------------------------------------------------
# Step 4: Detect Node.js & Install Frontend Dependencies
# ------------------------------------------------------------------------------
echo "[4/7] Verifying Node.js, npm, and frontend dependencies..."

if ! command -v node >/dev/null 2>&1; then
    if [ -x "${HOME}/.local/bin/node" ]; then
        export PATH="${HOME}/.local/bin:${PATH}"
    else
        echo "  [ERROR] Node.js is not installed in your WSL environment."
        echo "  Please install Node.js (v18 or newer) inside Ubuntu:"
        echo "      sudo apt-get update && sudo apt-get install -y nodejs npm"
        exit 1
    fi
fi

if ! command -v npm >/dev/null 2>&1; then
    if [ -x "${HOME}/.local/bin/npm" ]; then
        export PATH="${HOME}/.local/bin:${PATH}"
    else
        echo "  [ERROR] npm is not installed in your WSL environment."
        echo "  Please install npm inside Ubuntu: sudo apt-get install -y npm"
        exit 1
    fi
fi

NODE_VERSION=$(node -v)
NPM_VERSION=$(npm -v)
echo "  [OK] Node.js ${NODE_VERSION} & npm ${NPM_VERSION} detected."

cd "${PROJECT_DIR}/frontend"
if [ ! -d "node_modules" ] || [ ! -f "node_modules/.bin/vite" ]; then
    echo "  -> Installing frontend packages via npm ci..."
    if [ -f "package-lock.json" ]; then
        npm ci || npm install
    else
        npm install
    fi
fi

echo "  [OK] Frontend dependencies ready."

# ------------------------------------------------------------------------------
# Step 5: Validate Models & Knowledge Base Assets
# ------------------------------------------------------------------------------
echo "[5/7] Validating machine learning models and knowledge base assets..."

ERRORS_FOUND=0

# A. RoBERTa Crisis Classifier
ROBERTA_WEIGHTS="${PROJECT_DIR}/models/crisis_detector/roberta/model.safetensors"
ROBERTA_CONFIG="${PROJECT_DIR}/models/crisis_detector/roberta/config.json"
ROBERTA_THRESH="${PROJECT_DIR}/models/crisis_detector/roberta/threshold_config.json"

if [ -f "${ROBERTA_WEIGHTS}" ] && [ -s "${ROBERTA_WEIGHTS}" ]; then
    echo "  [OK] RoBERTa model weights present ($(du -h "${ROBERTA_WEIGHTS}" | cut -f1))."
else
    echo "  [ERROR] Missing or empty RoBERTa weights: ${ROBERTA_WEIGHTS}"
    ERRORS_FOUND=1
fi

if [ -f "${ROBERTA_CONFIG}" ] && [ -f "${ROBERTA_THRESH}" ]; then
    echo "  [OK] RoBERTa configuration and threshold files present."
else
    echo "  [ERROR] Missing RoBERTa config or threshold_config.json"
    ERRORS_FOUND=1
fi

# B. BGE Embedding Model
BGE_WEIGHTS="${PROJECT_DIR}/models/embeddings/bge-small-en-v1.5/model.safetensors"
BGE_CONFIG="${PROJECT_DIR}/models/embeddings/bge-small-en-v1.5/config.json"

if [ -f "${BGE_WEIGHTS}" ] && [ -s "${BGE_WEIGHTS}" ]; then
    echo "  [OK] BGE embedding model weights present ($(du -h "${BGE_WEIGHTS}" | cut -f1))."
else
    echo "  [ERROR] Missing or empty BGE embedding weights: ${BGE_WEIGHTS}"
    ERRORS_FOUND=1
fi

# C. FAISS Indexes & Metadata
CLINICAL_FAISS="${PROJECT_DIR}/models/embeddings/faiss_index/index.faiss"
CLINICAL_META="${PROJECT_DIR}/models/embeddings/faiss_index/metadata.json"
COUNSEL_FAISS="${PROJECT_DIR}/data/knowledge_base/counselchat_faiss_index/index.faiss"
COUNSEL_META="${PROJECT_DIR}/data/knowledge_base/counselchat_faiss_index/metadata.json"

if [ -f "${CLINICAL_FAISS}" ] && [ -f "${CLINICAL_META}" ]; then
    echo "  [OK] Clinical knowledge base FAISS index and metadata present."
else
    echo "  [ERROR] Missing clinical FAISS index or metadata at ${CLINICAL_FAISS}"
    ERRORS_FOUND=1
fi

if [ -f "${COUNSEL_FAISS}" ] && [ -f "${COUNSEL_META}" ]; then
    echo "  [OK] CounselChat consolidated FAISS index and metadata present."
else
    echo "  [WARN] CounselChat FAISS index missing. System will default to canonical KB."
fi

# D. Knowledge Base Markdown Documents
KB_COUNT=$(find "${PROJECT_DIR}/data/knowledge_base" -maxdepth 3 -name "*.md" 2>/dev/null | wc -l)
if [ "${KB_COUNT}" -ge 20 ]; then
    echo "  [OK] Canonical knowledge base markdown rubrics verified (${KB_COUNT} documents)."
else
    echo "  [ERROR] Knowledge base documents incomplete (found ${KB_COUNT}, expected >= 20)."
    ERRORS_FOUND=1
fi

# E. Emergency Hotline Directory
HOTLINES_FILE="${PROJECT_DIR}/backend/safety/hotlines.json"
if [ -f "${HOTLINES_FILE}" ] && [ -s "${HOTLINES_FILE}" ]; then
    echo "  [OK] Safety emergency directory verified (backend/safety/hotlines.json)."
else
    echo "  [ERROR] Missing emergency directory: ${HOTLINES_FILE}"
    ERRORS_FOUND=1
fi

if [ "${ERRORS_FOUND}" -ne 0 ]; then
    echo "  [ERROR] Model asset verification failed. Please inspect the log above."
    exit 1
fi

# ------------------------------------------------------------------------------
# Step 6: Environment Configuration (.env initialization)
# ------------------------------------------------------------------------------
echo "[6/7] Configuring environment settings..."

cd "${PROJECT_DIR}"
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp ".env.example" ".env"
        echo "  [OK] Created .env from template (.env.example)."
    fi
else
    echo "  [NOTE] Existing .env file found; preserving user settings."
fi

if [ ! -f "${PROJECT_DIR}/frontend/.env" ] && [ -f "${PROJECT_DIR}/frontend/.env.example" ]; then
    cp "${PROJECT_DIR}/frontend/.env.example" "${PROJECT_DIR}/frontend/.env"
    echo "  [OK] Created frontend/.env from frontend/.env.example."
fi

# If using separate runtime directory, mirror .env back to Windows folder
if [ "${IS_DRVFS}" -eq 1 ] && [ -f "${PROJECT_DIR}/.env" ] && [ ! -f "${ORIGINAL_PROJECT_DIR}/.env" ]; then
    cp "${PROJECT_DIR}/.env" "${ORIGINAL_PROJECT_DIR}/.env" 2>/dev/null || true
fi

# ------------------------------------------------------------------------------
# Step 7: Sanity Test Imports & Complete Setup
# ------------------------------------------------------------------------------
echo "[7/7] Executing backend import sanity check..."

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

"${PROJECT_DIR}/.venv/bin/python" -c "
import fastapi
import uvicorn
import torch
import transformers
import faiss
print('  [OK] Successfully imported core runtime packages: fastapi, uvicorn, torch, transformers, faiss.')
"

# Write completion markers in both runtime and original Windows directory
PY_VER="$("${PROJECT_DIR}/.venv/bin/python" --version 2>/dev/null || echo "Python 3.11")"
cat << EOF > "${PROJECT_DIR}/.setup_completed"
setup_timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
python_version=${PY_VER}
node_version=$(node -v)
npm_version=$(npm -v)
status=completed
EOF

if [ "${IS_DRVFS}" -eq 1 ]; then
    cp "${PROJECT_DIR}/.setup_completed" "${ORIGINAL_PROJECT_DIR}/.setup_completed" 2>/dev/null || true
    mkdir -p "${ORIGINAL_PROJECT_DIR}/logs"
    cp "${SETUP_LOG}" "${ORIGINAL_PROJECT_DIR}/logs/setup.log" 2>/dev/null || true
fi

echo ""
echo "=================================================================="
echo "  First-Time Setup Completed Successfully!"
echo "=================================================================="
if [ "${IS_DRVFS}" -eq 1 ]; then
    echo " Native WSL2 Runtime : ${PROJECT_DIR}"
    echo " Windows Project Dir : ${ORIGINAL_PROJECT_DIR}"
fi
echo " Next Steps:"
echo "   1. (Optional) To use external LLMs, add your API key in .env:"
echo "        GEMINI_API_KEY=your_key_here"
echo "        GROQ_API_KEY=your_key_here"
echo "      (If left blank, MindBridge operates in deterministic local fallback mode)."
echo ""
echo "   2. Double-click 'run.bat' on Windows to start MindBridge!"
echo "=================================================================="
