"""
MindBridge 2.0 - Native Windows Output Mirroring Runner (`tee_runner.py`)
=========================================================================

Executes a command while mirroring stdout/stderr to both the active console
terminal and a target log file in real time.
"""

import sys
import subprocess
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 3:
        sys.stderr.write("Usage: python tee_runner.py <log_file> <cmd> [args...]\n")
        sys.exit(1)

    log_path = Path(sys.argv[1]).resolve()
    cmd = sys.argv[2:]

    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", encoding="utf-8", buffering=1) as log_file:
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except Exception as exc:
            err_msg = f"[ERROR] Failed to start process: {exc}\n"
            sys.stderr.write(err_msg)
            log_file.write(err_msg)
            sys.exit(1)

        try:
            if proc.stdout is not None:
                for line in proc.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log_file.write(line)
                    log_file.flush()
            proc.wait()
            sys.exit(proc.returncode if proc.returncode is not None else 0)
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            sys.exit(0)


if __name__ == "__main__":
    main()

