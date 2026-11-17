"""
MindBridge 2.0 - ASGI Main Entrypoint (`main.py`)
=================================================

Executable entrypoint for local execution:
    python -m backend.main
Or via uvicorn directly:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""

import uvicorn
from backend.api.app import app
from backend.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=settings.web_concurrency if not settings.debug else 1,
        log_level="info",
    )
