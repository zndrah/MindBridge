#!/usr/bin/env bash
# ==============================================================================
# MindBridge 2.0 — FastAPI Backend Launcher (Portable)
# Location: scripts/start_backend.sh
# ==============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -n "${1:-}" ]; then
    RAW_PATH="$1"
    if [[ "${RAW_PATH}" =~ ^\\\\(wsl\.localhost|wsl\$)(\\|/)[^\\/]+(.*)$ ]]; then
        RESOLVED_PATH="${BASH_REMATCH[3]//\\//}"
    else
        RESOLVED_PATH="$(wslpath -u "${RAW_PATH}" 2>/dev/null || echo "${RAW_PATH}")"
    fi
    if [ -d "${RESOLVED_PATH}" ]; then
        PROJECT_DIR="${RESOLVED_PATH}"
    fi
fi

# High-performance native WSL workspace check
if [ -f "${PROJECT_DIR}/.wsl_runtime_path" ]; then
    RUNTIME_CANDIDATE="$(cat "${PROJECT_DIR}/.wsl_runtime_path" 2>/dev/null | tr -d '[:space:]')"
    if [ -d "${RUNTIME_CANDIDATE}" ]; then
        # Sync any user modifications from Windows before launch
        rsync -u -r \
            --exclude='.venv' \
            --exclude='frontend/node_modules' \
            --exclude='logs' \
            --exclude='.run' \
            "${PROJECT_DIR}/" "${RUNTIME_CANDIDATE}/" 2>/dev/null || true
        PROJECT_DIR="${RUNTIME_CANDIDATE}"
    fi
fi

RUN_DIR="${PROJECT_DIR}/.run"
LOGS_DIR="${PROJECT_DIR}/logs"
PID_FILE="${RUN_DIR}/backend.pid"
LOG_FILE="${LOGS_DIR}/backend.log"

mkdir -p "${RUN_DIR}" "${LOGS_DIR}"

# Ensure user binaries and virtualenv are in PATH
export PATH="${HOME}/.local/bin:${PROJECT_DIR}/.venv/bin:/usr/local/bin:/usr/bin:/bin:${PATH}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

SERVER_PID=""

cleanup() {
    local exit_code=$?
    if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo ""
        echo "[MindBridge Backend] Stopping server process ${SERVER_PID}..."
        pkill -TERM -P "${SERVER_PID}" 2>/dev/null || true
        kill -TERM "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    rm -f "${PID_FILE}"
    exit ${exit_code}
}
trap cleanup EXIT INT TERM

cd "${PROJECT_DIR}"

echo "=================================================================="
echo "  MindBridge 2.0 — FastAPI Backend Service"
echo "=================================================================="
echo " Project Directory : ${PROJECT_DIR}"
echo " Host / Port       : 0.0.0.0:8000"
echo " Liveness Probe    : http://127.0.0.1:8000/health"
echo " Readiness Probe   : http://127.0.0.1:8000/ready"
echo " API Documentation : http://127.0.0.1:8000/docs"
echo " Log Destination   : ${LOG_FILE}"
echo "=================================================================="
echo ""
echo "Pre-warming RoBERTa crisis classifier and BGE dense vector index..."
echo ""

# Run uvicorn with live stdout display while logging to backend.log
if [ -x "${PROJECT_DIR}/.venv/bin/uvicorn" ]; then
    "${PROJECT_DIR}/.venv/bin/uvicorn" backend.main:app --host 0.0.0.0 --port 8000 2>&1 | tee "${LOG_FILE}" &
elif command -v uv >/dev/null 2>&1; then
    uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000 2>&1 | tee "${LOG_FILE}" &
else
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 2>&1 | tee "${LOG_FILE}" &
fi
SERVER_PID=$!
echo "${SERVER_PID}" > "${PID_FILE}"

wait "${SERVER_PID}"
STATUS=$?

rm -f "${PID_FILE}"

if [ ${STATUS} -ne 0 ] && [ ${STATUS} -ne 130 ] && [ ${STATUS} -ne 143 ]; then
    echo ""
    echo "=================================================================="
    echo " [ERROR] Backend service exited with failure code ${STATUS}."
    echo " Logs available at: ${LOG_FILE}"
    echo "=================================================================="
    echo "Press Enter to close this window..."
    read -r
fi
