#!/usr/bin/env python3
"""
MindBridge 2.0 - Hardware Profiling & System Inspection Script
Phase 2: Environment Scaffolding & Hardware Profiling

Reads hardware metrics directly using standard Python libraries and Linux /proc pseudo-filesystem.
Zero external package dependencies required.
"""

import os
import sys
import json
import platform
from datetime import datetime, timezone


def get_cpu_info():
    info = {
        "logical_cpus": os.cpu_count() or 0,
        "model_name": "Unknown",
        "cpu_family": "Unknown",
        "cores_detected": 0
    }
    
    cpuinfo_path = "/proc/cpuinfo"
    if os.path.exists(cpuinfo_path):
        try:
            with open(cpuinfo_path, "r", encoding="utf-8") as f:
                cores = 0
                for line in f:
                    line = line.strip()
                    if line.startswith("model name"):
                        if info["model_name"] == "Unknown":
                            info["model_name"] = line.split(":", 1)[1].strip()
                    elif line.startswith("processor"):
                        cores += 1
                info["cores_detected"] = cores
        except Exception as e:
            info["proc_error"] = str(e)
            
    try:
        load1, load5, load15 = os.getloadavg()
        info["load_avg"] = {"1min": round(load1, 2), "5min": round(load5, 2), "15min": round(load15, 2)}
    except Exception:
        info["load_avg"] = "Not available"

    return info


def get_memory_info():
    mem = {}
    meminfo_path = "/proc/meminfo"
    if os.path.exists(meminfo_path):
        try:
            with open(meminfo_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        key = parts[0].strip()
                        val_str = parts[1].strip().split()[0]
                        if val_str.isdigit():
                            mem[key] = int(val_str)
        except Exception as e:
            return {"error": str(e)}

    total_kb = mem.get("MemTotal", 0)
    avail_kb = mem.get("MemAvailable", 0)
    free_kb = mem.get("MemFree", 0)
    swap_total_kb = mem.get("SwapTotal", 0)
    swap_free_kb = mem.get("SwapFree", 0)

    used_kb = total_kb - avail_kb if total_kb >= avail_kb else 0

    return {
        "mem_total_mb": round(total_kb / 1024, 2),
        "mem_total_gb": round(total_kb / (1024 * 1024), 2),
        "mem_available_mb": round(avail_kb / 1024, 2),
        "mem_available_gb": round(avail_kb / (1024 * 1024), 2),
        "mem_used_mb": round(used_kb / 1024, 2),
        "mem_used_gb": round(used_kb / (1024 * 1024), 2),
        "mem_free_mb": round(free_kb / 1024, 2),
        "mem_free_gb": round(free_kb / (1024 * 1024), 2),
        "mem_used_percent": round((used_kb / total_kb * 100), 2) if total_kb > 0 else 0,
        "swap_total_mb": round(swap_total_kb / 1024, 2),
        "swap_total_gb": round(swap_total_kb / (1024 * 1024), 2),
        "swap_free_mb": round(swap_free_kb / 1024, 2),
        "swap_free_gb": round(swap_free_kb / (1024 * 1024), 2),
    }


def get_disk_info(path="/"):
    try:
        st = os.statvfs(path)
        total = (st.f_blocks * st.f_frsize) / (1024 * 1024 * 1024)
        free = (st.f_bfree * st.f_frsize) / (1024 * 1024 * 1024)
        avail = (st.f_bavail * st.f_frsize) / (1024 * 1024 * 1024)
        used = total - free
        return {
            "total_gb": round(total, 2),
            "used_gb": round(used, 2),
            "available_gb": round(avail, 2),
            "used_percent": round((used / total * 100), 2) if total > 0 else 0
        }
    except Exception as e:
        return {"error": str(e)}


def assess_headroom(mem_available_gb):
    """
    Evaluates whether available RAM meets the MindBridge 2.0 runtime targets:
    - Base OS & Daemons: ~1.2 GB
    - Server & LangGraph Orchestrator: ~0.3 GB
    - BGE-small Embeddings: ~0.2 GB
    - FAISS Vector Index (15k docs): ~0.1 GB
    - DistilBERT ONNX Safety Classifier: ~0.3 GB
    - Quantized Qwen2.5-1.5B (GGUF Q4_K_M): ~1.6 GB
    Total Projected Peak: ~3.7 GB
    """
    projected_peak_gb = 3.70
    headroom_gb = mem_available_gb - projected_peak_gb
    is_sufficient = headroom_gb >= 0.5  # At least 500 MB surplus headroom

    return {
        "projected_peak_ram_gb": projected_peak_gb,
        "surplus_headroom_gb": round(headroom_gb, 2),
        "assessment": "SUFFICIENT" if is_sufficient else "CONSTRAINED",
        "recommendation": (
            "System possesses adequate memory for local CPU execution of ONNX safety classifier, "
            "BGE-small embeddings, FAISS vector store, and 4-bit quantized local model."
            if is_sufficient else
            "Memory headroom is tight. Recommend offloading generation to free-tier API adapter "
            "(Groq / HuggingFace) while running safety triage and FAISS locally on CPU."
        )
    }


def main():
    timestamp = datetime.now(timezone.utc).isoformat()
    cpu = get_cpu_info()
    memory = get_memory_info()
    disk = get_disk_info("/")
    headroom = assess_headroom(memory.get("mem_available_gb", 0))

    report = {
        "project": "MindBridge 2.0",
        "phase": "Phase 2: Environment Scaffolding & Hardware Profiling",
        "timestamp": timestamp,
        "python_environment": {
            "version": sys.version.split()[0],
            "full_version": sys.version,
            "executable": sys.executable,
            "prefix": sys.prefix,
            "is_virtualenv": sys.prefix != getattr(sys, "base_prefix", sys.prefix)
        },
        "operating_system": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "architecture": platform.machine(),
            "is_wsl2": "microsoft" in platform.release().lower() or "wsl" in platform.release().lower()
        },
        "cpu": cpu,
        "memory": memory,
        "disk": disk,
        "headroom_assessment": headroom
    }

    # Print Formatted Report
    print("=" * 72)
    print(" MindBridge 2.0 - Phase 2 Hardware & System Profile Report")
    print("=" * 72)
    print(f" Timestamp (UTC)       : {report['timestamp']}")
    print(f" Python Version        : {report['python_environment']['version']}")
    print(f" Python Executable     : {report['python_environment']['executable']}")
    print(f" Virtual Environment   : {'Active' if report['python_environment']['is_virtualenv'] else 'Inactive'}")
    print(f" Operating System      : {report['operating_system']['system']} {report['operating_system']['release']}")
    print(f" WSL2 Environment      : {'Detected (Microsoft WSL2 Kernel)' if report['operating_system']['is_wsl2'] else 'Native Linux'}")
    print(f" Architecture          : {report['operating_system']['architecture']}")
    print("-" * 72)
    print(f" CPU Model             : {cpu['model_name']}")
    print(f" Logical Cores / Threads: {cpu['logical_cpus']}")
    print(f" Load Average (1,5,15m): {cpu['load_avg']}")
    print("-" * 72)
    print(f" Total RAM (Allocated) : {memory['mem_total_gb']} GB ({memory['mem_total_mb']} MB)")
    print(f" Available RAM         : {memory['mem_available_gb']} GB ({memory['mem_available_mb']} MB)")
    print(f" Used RAM              : {memory['mem_used_gb']} GB ({memory['mem_used_percent']}%)")
    print(f" Total Swap            : {memory['swap_total_gb']} GB (Free: {memory['swap_free_gb']} GB)")
    print("-" * 72)
    print(f" Root Disk ('/') Total : {disk['total_gb']} GB")
    print(f" Root Disk Available   : {disk['available_gb']} GB (Used: {disk['used_percent']}%)")
    print("-" * 72)
    print(" Hardware Headroom Assessment for MindBridge 2.0:")
    print(f" Projected Peak RAM    : {headroom['projected_peak_ram_gb']} GB")
    print(f" Estimated Surplus     : {headroom['surplus_headroom_gb']} GB")
    print(f" Status                : [{headroom['assessment']}]")
    print(f" Guidance              : {headroom['recommendation']}")
    print("=" * 72)

    return report


if __name__ == "__main__":
    main()
