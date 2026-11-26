"""Start the Advanced Impactor RAG web app."""

import os
from pathlib import Path

import uvicorn


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    port = int(os.getenv("PORT", "8000"))
    print(f"Starting Advanced Impactor at http://127.0.0.1:{port}")
    uvicorn.run("app.api:app", host="127.0.0.1", port=port, reload=False)
