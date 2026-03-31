"""
server/app.py — OpenEnv-compatible entry point.
This is the canonical server module referenced by pyproject.toml [project.scripts].
"""
import sys
import os

# Ensure root is on path so models/environment are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # re-export the FastAPI app from root app.py


def main():
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    host = os.getenv("HOST", "0.0.0.0")
    workers = int(os.getenv("WORKERS", 1))
    uvicorn.run("server.app:app", host=host, port=port, workers=workers)


if __name__ == "__main__":
    main()
