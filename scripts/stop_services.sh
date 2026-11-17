#!/usr/bin/env bash
# ==============================================================================
# MindBridge 2.0 — Safe Process Stopper & Port Cleaner (Portable)
# Location: scripts/stop_services.sh
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

# Also check for separate runtime workspace
ACTIVE_DIR="${PROJECT_DIR}"
if [ -f "${PROJECT_DIR}/.wsl_runtime_path" ]; then
    RUNTIME_CANDIDATE="$(cat "${PROJECT_DIR}/.wsl_runtime_path" 2>/dev/null | tr -d '[:space:]')"
    if [ -d "${RUNTIME_CANDIDATE}" ]; then
        ACTIVE_DIR="${RUNTIME_CANDIDATE}"
    fi
fi

RUN_DIRS=("${PROJECT_DIR}/.run" "${ACTIVE_DIR}/.run")

echo "=================================================================="
echo "  MindBridge 2.0 — Stopping Services"
echo "=================================================================="
echo " Project Directory : ${PROJECT_DIR}"
if [ "${ACTIVE_DIR}" != "${PROJECT_DIR}" ]; then
    echo " Runtime Directory : ${ACTIVE_DIR}"
fi
echo "=================================================================="

# Helper function to inspect /proc/$PID/cmdline safely
get_cmdline() {
    local pid="$1"
    if [ -f "/proc/${pid}/cmdline" ]; then
        tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true
    fi
}

# Stop process recorded by PID file
stop_pid_service() {
    local service_label="$1"
    local pid_file="$2"
    local match_pattern="$3"

    if [ ! -f "${pid_file}" ]; then
        return 0
    fi

    local target_pid
    target_pid=$(cat "${pid_file}" 2>/dev/null | tr -d '[:space:]' || true)

    if [ -z "${target_pid}" ]; then
        rm -f "${pid_file}"
        return 0
    fi

    if kill -0 "${target_pid}" 2>/dev/null; then
        local cmd
        cmd=$(get_cmdline "${target_pid}")
        if echo "${cmd}" | grep -Eq "${match_pattern}"; then
            echo "[OK] Terminating ${service_label} (PID ${target_pid})..."
            # Terminate descendant processes first
            pkill -TERM -P "${target_pid}" 2>/dev/null || true
            kill -TERM "${target_pid}" 2>/dev/null || true

            # Wait up to 5 seconds for graceful exit
            local stopped=0
            for _ in {1..10}; do
                if ! kill -0 "${target_pid}" 2>/dev/null; then
                    stopped=1
                    break
                fi
                sleep 0.5
            done

            if [ ${stopped} -eq 0 ] && kill -0 "${target_pid}" 2>/dev/null; then
                echo "[WARN] ${service_label} did not exit gracefully; issuing SIGKILL..."
                pkill -KILL -P "${target_pid}" 2>/dev/null || true
                kill -KILL "${target_pid}" 2>/dev/null || true
            fi
            echo "[OK] ${service_label} stopped."
        else
            echo "[WARN] PID ${target_pid} does not match expected signature (${match_pattern})."
            echo "       Skipping termination to protect unrelated processes."
        fi
    else
        echo "[INFO] ${service_label} (PID ${target_pid}) is already terminated."
    fi

    rm -f "${pid_file}"
}

# Secondary cleanup for port listeners belonging strictly to MindBridge
cleanup_port_listener() {
    local service_label="$1"
    local port="$2"
    local match_pattern="$3"

    local pids
    pids=$(lsof -t -i :"${port}" 2>/dev/null || true)
    if [ -z "${pids}" ]; then
        return 0
    fi

    for p in ${pids}; do
        if [ -n "${p}" ] && kill -0 "${p}" 2>/dev/null; then
            local cmd
            cmd=$(get_cmdline "${p}")
            if echo "${cmd}" | grep -Eq "${match_pattern}"; then
                echo "[OK] Releasing port ${port} from lingering ${service_label} (PID ${p})..."
                pkill -TERM -P "${p}" 2>/dev/null || true
                kill -TERM "${p}" 2>/dev/null || true
                sleep 0.5
                if kill -0 "${p}" 2>/dev/null; then
                    kill -KILL "${p}" 2>/dev/null || true
                fi
            else
                echo "[SKIP] Port ${port} is held by PID ${p}, but cmdline does not match MindBridge."
                echo "       Process preserved: ${cmd:0:80}"
            fi
        fi
    done
}

# 1. Stop Backend Service across known run directories
for rd in "${RUN_DIRS[@]}"; do
    stop_pid_service "Backend" "${rd}/backend.pid" "(uvicorn|backend\.main|fastapi)"
done
cleanup_port_listener "Backend" "8000" "(uvicorn|backend\.main|mindbridge)"

# 2. Stop Frontend Service across known run directories
for rd in "${RUN_DIRS[@]}"; do
    stop_pid_service "Frontend" "${rd}/frontend.pid" "(vite|npm run dev|node.*vite)"
done
cleanup_port_listener "Frontend" "5173" "(vite|node.*vite|mindbridge.*frontend)"

# 3. Clean up any remaining .run files
for rd in "${RUN_DIRS[@]}"; do
    rm -f "${rd}/backend.pid" "${rd}/frontend.pid" 2>/dev/null || true
done

echo ""
echo "Verifying port availability:"
if ss -tln | grep -q ':8000 '; then
    echo " [!] Port 8000 is still occupied by an external/unrelated process."
else
    echo " [OK] Port 8000 is free."
fi

if ss -tln | grep -q ':5173 '; then
    echo " [!] Port 5173 is still occupied by an external/unrelated process."
else
    echo " [OK] Port 5173 is free."
fi

echo "=================================================================="
echo " MindBridge shutdown process complete."
echo "=================================================================="
