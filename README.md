# Chatbot Advanced Impactor

## Run on Windows

From the project directory, create a Python 3.11 environment and start the web app:

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install -r requirements.txt
.\.venv311\Scripts\python.exe run.py
```

Open <http://127.0.0.1:8000>. Keep the terminal open while using the app; press
Ctrl+C to stop the server. The `.env` file must contain a valid `GOOGLE_API_KEY`
for AI answers.

`app/rag.py` contains the retrieval and generation pipeline. Run `run.py` to
start the web application; running `app/rag.py` directly only loads its function
definitions and does not start a server.
