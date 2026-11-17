#!/usr/bin/env bash
# -*- mode: python -*-
""":"
# Bash wrapper allowing direct execution or execution via python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="python3"
fi
exec "${PYTHON_BIN}" "$0" "$@"
"""

import hashlib
import os
from pathlib import Path
import sys
import zipfile
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXCLUDE_PATTERNS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "logs",
    ".run",
    "release",
}

EXCLUDE_FILENAMES = {
    ".env",
    ".setup_completed",
    "test_pid.txt",
}

EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".log",
    ".pid",
    ".bak",
    ".swp",
    ".swo",
    ".DS_Store",
}


def should_exclude(rel_path: Path) -> bool:
    parts = set(rel_path.parts)
    if parts & EXCLUDE_PATTERNS:
        return True
    if rel_path.name in EXCLUDE_FILENAMES:
        return True
    if rel_path.name.startswith(".env.") and rel_path.name != ".env.example":
        return True
    if rel_path.suffix in EXCLUDE_EXTENSIONS:
        return True
    # Exclude temporary .cache inside models
    if ".cache" in parts:
        return True
    return False


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def human_size(n_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(n_bytes) < 1024.0:
            return f"{n_bytes:3.1f} {unit}"
        n_bytes /= 1024.0
    return f"{n_bytes:.1f} TB"


def main():
    print("=" * 70)
    print("  MindBridge 2.0 — Portable Release Packaging Engine")
    print(f"  Project Root: {PROJECT_ROOT}")
    print("=" * 70)

    release_dir = PROJECT_ROOT / "release"
    release_dir.mkdir(exist_ok=True)
    zip_path = release_dir / "MindBridge_Portable.zip"
    manifest_path = PROJECT_ROOT / "docs" / "PACKAGE_MANIFEST.md"

    # 1. Asset Pre-Flight Checks
    print("\n[1/5] Verifying required runtime assets before packaging...")
    required_assets = [
        PROJECT_ROOT / "models" / "crisis_detector" / "roberta" / "model.safetensors",
        PROJECT_ROOT / "models" / "crisis_detector" / "roberta" / "threshold_config.json",
        PROJECT_ROOT / "models" / "embeddings" / "bge-small-en-v1.5" / "model.safetensors",
        PROJECT_ROOT / "models" / "embeddings" / "faiss_index" / "index.faiss",
        PROJECT_ROOT / "data" / "knowledge_base" / "faiss_index" / "index.faiss",
        PROJECT_ROOT / "data" / "knowledge_base" / "counselchat_faiss_index" / "index.faiss",
        PROJECT_ROOT / "backend" / "main.py",
        PROJECT_ROOT / "backend" / "safety" / "hotlines.json",
        PROJECT_ROOT / "frontend" / "package.json",
        PROJECT_ROOT / "pyproject.toml",
        PROJECT_ROOT / "uv.lock",
        PROJECT_ROOT / "FIRST_TIME_SETUP.bat",
        PROJECT_ROOT / "run.bat",
        PROJECT_ROOT / "stop.bat",
    ]

    missing = []
    for asset in required_assets:
        if not asset.exists():
            missing.append(str(asset.relative_to(PROJECT_ROOT)))

    if missing:
        print("\n[ERROR] Packaging aborted: Required assets are missing:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    print("  [OK] All core runtime assets and model weights confirmed present.")

    # 2. Collect files to package
    print("\n[2/5] Scanning directory tree and cataloging files...")
    files_to_pack = []
    total_uncompressed = 0

    for root, dirs, files in os.walk(PROJECT_ROOT):
        # In-place directory filtering to skip walking into heavy excluded folders
        dirs[:] = [d for d in dirs if d not in EXCLUDE_PATTERNS and d != ".cache"]
        for f in files:
            full_path = Path(root) / f
            rel_path = full_path.relative_to(PROJECT_ROOT)
            if not should_exclude(rel_path):
                size = full_path.stat().st_size
                files_to_pack.append((rel_path, full_path, size))
                total_uncompressed += size

    files_to_pack.sort(key=lambda x: x[0].as_posix())
    print(f"  [OK] Found {len(files_to_pack)} files ({human_size(total_uncompressed)} uncompressed).")

    # 3. Generate Manifest with SHA-256 Checksums
    print("\n[3/5] Calculating SHA-256 checksums and building manifest...")
    manifest_lines = [
        "# 📦 MindBridge 2.0 — Portable Package Manifest & Checksums",
        "",
        f"**Generated**: {datetime.now(timezone.utc).isoformat()}",
        f"**Total Files**: {len(files_to_pack)}",
        f"**Uncompressed Size**: {human_size(total_uncompressed)} ({total_uncompressed:,} bytes)",
        "",
        "This manifest documents every file included in `release/MindBridge_Portable.zip` along with its size and cryptographic SHA-256 hash.",
        "",
        "## Key Runtime Assets",
        "",
        "| File | Size | SHA-256 Hash |",
        "| :--- | :--- | :--- |",
    ]

    file_hashes = []
    print("  Calculating file hashes (this may take a few moments for model weights)...")

    for rel_path, full_path, size in files_to_pack:
        posix_rel = rel_path.as_posix()
        # Compute SHA256
        sha = sha256_file(full_path)
        file_hashes.append((posix_rel, size, sha))

        # Include major assets in top table
        if any(key in posix_rel for key in [
            "model.safetensors",
            "threshold_config.json",
            "index.faiss",
            "hotlines.json",
            "FIRST_TIME_SETUP.bat",
            "run.bat",
            "stop.bat",
            "PACKAGE_FOR_TEAMMATE.bat",
            ".env.example",
            "uv.lock",
        ]):
            manifest_lines.append(f"| `{posix_rel}` | {human_size(size)} | `{sha}` |")

    manifest_lines.extend([
        "",
        "---",
        "",
        "## Complete File Inventory",
        "",
        "| File Path | Size (Bytes) | SHA-256 |",
        "| :--- | :--- | :--- |",
    ])

    for posix_rel, size, sha in file_hashes:
        manifest_lines.append(f"| `{posix_rel}` | {size:,} | `{sha}` |")

    manifest_path.parent.mkdir(exist_ok=True)
    manifest_path.write_text("\n".join(manifest_lines), encoding="utf-8")
    print(f"  [OK] Manifest written to {manifest_path.relative_to(PROJECT_ROOT)}.")

    # Also add the freshly generated manifest to the files to pack!
    manifest_full = manifest_path
    manifest_size = manifest_full.stat().st_size
    files_to_pack = [item for item in files_to_pack if item[0].as_posix() != "docs/PACKAGE_MANIFEST.md"]
    files_to_pack.append((Path("docs/PACKAGE_MANIFEST.md"), manifest_full, manifest_size))

    # 4. Create Portable ZIP Archive
    print(f"\n[4/5] Creating portable ZIP archive: {zip_path.name}...")
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        count = 0
        for rel_path, full_path, size in files_to_pack:
            # Preserve Linux permissions
            st = full_path.stat()
            zinfo = zipfile.ZipInfo.from_file(full_path, arcname=rel_path.as_posix())
            zinfo.external_attr = (st.st_mode & 0xFFFF) << 16
            with open(full_path, "rb") as f:
                zf.writestr(zinfo, f.read())
            count += 1
            if count % 50 == 0 or count == len(files_to_pack):
                print(f"  Archived {count}/{len(files_to_pack)} files...", end="\r", flush=True)

    print()
    zip_size = zip_path.stat().st_size
    print(f"  [OK] Successfully created {zip_path.name} ({human_size(zip_size)}).")

    # 5. Integrity Verification
    print("\n[5/5] Verifying archive integrity and secret exclusion...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = set(zf.namelist())

        # Assert no secrets included
        for name in namelist:
            if name.endswith("/.env") or name == ".env":
                raise AssertionError(f"SECURITY VIOLATION: {name} found in ZIP archive!")
            if (name.startswith(".env.") or "/.env." in name) and not name.endswith(".example"):
                raise AssertionError(f"SECURITY VIOLATION: {name} found in ZIP archive!")
            assert "node_modules" not in name, f"DIR VIOLATION: node_modules found in {name}"
            assert ".venv" not in name, f"DIR VIOLATION: .venv found in {name}"

        # Assert critical assets present
        assert "models/crisis_detector/roberta/model.safetensors" in namelist, "Missing RoBERTa weights in ZIP"
        assert "models/embeddings/bge-small-en-v1.5/model.safetensors" in namelist, "Missing BGE weights in ZIP"
        assert "models/embeddings/faiss_index/index.faiss" in namelist, "Missing clinical FAISS in ZIP"
        assert "data/knowledge_base/counselchat_faiss_index/index.faiss" in namelist, "Missing CounselChat FAISS in ZIP"
        assert "FIRST_TIME_SETUP.bat" in namelist, "Missing FIRST_TIME_SETUP.bat in ZIP"
        assert "run.bat" in namelist, "Missing run.bat in ZIP"
        assert "stop.bat" in namelist, "Missing stop.bat in ZIP"
        assert ".env.example" in namelist, "Missing .env.example in ZIP"

    print("  [OK] All integrity and security checks PASSED.")

    print("\n" + "=" * 70)
    print("  PACKAGE SUMMARY:")
    print(f"  - Target Archive     : {zip_path}")
    print(f"  - Compressed Size    : {human_size(zip_size)} ({zip_size:,} bytes)")
    print(f"  - Uncompressed Size  : {human_size(total_uncompressed)}")
    print(f"  - Packaged Files     : {len(namelist)}")
    print(f"  - Manifest Location  : docs/PACKAGE_MANIFEST.md")
    print(f"  - Secrets Excluded   : .env, credentials, and personal logs verified excluded")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
