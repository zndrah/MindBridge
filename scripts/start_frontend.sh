#!/usr/bin/env bash
# ==============================================================================
# MindBridge 2.0 — React + Vite Frontend Launcher (Portable)
# Location: scripts/start_frontend.sh
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
        PROJECT_DIR="${RUNTIME_CANDIDATE}"
    fi
fi

FRONTEND_DIR="${PROJECT_DIR}/frontend"
RUN_DIR="${PROJECT_DIR}/.run"
LOGS_DIR="${PROJECT_DIR}/logs"
PID_FILE="${RUN_DIR}/frontend.pid"
LOG_FILE="${LOGS_DIR}/frontend.log"

mkdir -p "${RUN_DIR}" "${LOGS_DIR}"

# Ensure user node/npm binaries are in PATH
export PATH="${HOME}/.local/bin:/usr/local/bin:/usr/bin:/bin:${PATH}"

SERVER_PID=""

cleanup() {
    local exit_code=$?
    if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo ""
        echo "[MindBridge Frontend] Stopping development server process ${SERVER_PID}..."
        pkill -TERM -P "${SERVER_PID}" 2>/dev/null || true
        kill -TERM "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    rm -f "${PID_FILE}"
    exit ${exit_code}
}
trap cleanup EXIT INT TERM

cd "${FRONTEND_DIR}"

echo "=================================================================="
echo "  MindBridge 2.0 — React + Vite Frontend Workspace"
echo "=================================================================="
echo " Working Directory : ${FRONTEND_DIR}"
echo " Host / Port       : 0.0.0.0:5173"
echo " Browser URL       : http://localhost:5173"
echo " Backend Proxy     : http://127.0.0.1:8000"
echo " Log Destination   : ${LOG_FILE}"
echo "=================================================================="
echo ""

npm run dev -- --host 0.0.0.0 --port 5173 2>&1 | tee "${LOG_FILE}" &
SERVER_PID=$!
echo "${SERVER_PID}" > "${PID_FILE}"

wait "${SERVER_PID}"
STATUS=$?

rm -f "${PID_FILE}"

if [ ${STATUS} -ne 0 ] && [ ${STATUS} -ne 130 ] && [ ${STATUS} -ne 143 ]; then
    echo ""
    echo "=================================================================="
    echo " [ERROR] Frontend service exited with failure code ${STATUS}."
    echo " Logs available at: ${LOG_FILE}"
    echo "=================================================================="
    echo "Press Enter to close this window..."
    read -r
fi
