"""Start the Advanced Impactor RAG web app."""

import os
from pathlib import Path

import uvicorn


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    print("Starting Advanced Impactor at http://127.0.0.1:8000")
    uvicorn.run("app.api:app", host="127.0.0.1", port=8000, reload=False)
