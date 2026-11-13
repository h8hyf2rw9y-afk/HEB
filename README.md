# HEB AI Assistant

Technical assessment project: Spanish-language assistant for retail operations questions across structured data and internal policy documents.

## Proposed stack
- **UI:** Streamlit
- **API:** FastAPI + Pydantic
- **Structured data:** DuckDB + Python
- **Document retrieval:** PyMuPDF, local embeddings through Ollama, lightweight indexed retrieval
- **LLM:** Ollama (model selected after local performance tests)
- **Tests:** pytest

The local prototype is implemented on this branch. Llama 3.2 can call verified DuckDB and RAG tools; outputs remain subject to factual checks. This is not a production service.

## Architecture and agent coordination
- [Requirements and verification status](docs/REQUIREMENTS.md)
- [Architecture](docs/ARCHITECTURE.md)
- [API/tool contracts](docs/CONTRACTS.md)
- [Decision log](docs/DECISIONS.md)
- [Handoff guide](docs/HANDOFF.md)
- [Local Git worktrees setup](docs/WORKTREES.md)
- [Codex instructions](AGENTS.md)
- [Claude Code instructions](CLAUDE.md)

Codex owns structured data/backend work; Claude Code owns PDF/RAG work. Both coordinate through GitHub Issues and Pull Requests, not through automatic chat-to-chat messaging. The user approves merges and deployments.

## Local setup — Windows PowerShell (PR #17)

Run these commands from the repository root. Never merge the PR just to test it.

**1. Update the PR branch.**
```powershell
git fetch origin
git switch feature/ollama-grounded-answering
git pull --ff-only origin feature/ollama-grounded-answering
git rev-parse --short HEAD
```

**2. Install Python requirements (only if .venv is missing).**
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**3. Start Ollama in a separate terminal.** Do not start a second server if the Windows Ollama application is already listening on port 11434.
```powershell
ollama serve
```
From another terminal: 
```powershell
ollama list
ollama pull llama3.2
Invoke-RestMethod http://127.0.0.1:11434/api/tags
```

**4. Build the local policy index.**
```powershell
.\.venv\Scripts\python.exe -m app.rag build --pdf-dir candidatos/politicas --index-dir data/rag_index --embeddings off
```

**5. Start the FastAPI server in a new terminal; set these variables BEFORE starting Python.**
```powershell
$env:HEB_LLM_ENABLED = "true"
$env:HEB_LLM_MODE = "agente"
$env:HEB_LLM_MODEL = "llama3.2"
$env:HEB_LLM_TIMEOUT_S = "120"
$env:OLLAMA_BASE_URL = "http://127.0.0.1:11434"
.\.venv\Scripts\python.exe -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

Verify from another terminal:
```powershell
Invoke-RestMethod http://127.0.0.1:8000/health/llm
Invoke-RestMethod http://127.0.0.1:8000/health/rag
```
Expected: enabled=true, mode=agente, and rag ready=true. /health/llm reports the startup configuration, not an actual network probe of Ollama.

**6. Start Streamlit in another terminal.**
```powershell
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```
Open the local URL printed by Streamlit (usually http://localhost:8501). Follow up on a return, receipt and original-card question and inspect tool_calls, evidence and fallback_reason.

**7. Evaluate actual Llama 3.2 in a separate terminal.**
```powershell
$env:HEB_LLM_REAL_TESTS = "1"
.\.venv\Scripts\python.exe -m pytest tests\llm\test_real_ollama.py -q -s
.\.venv\Scripts\python.exe scripts\eval_agent_conversations.py --out data\eval_agent_conversations.json
.\.venv\Scripts\python.exe scripts\eval_llm_modes.py --out data\eval_llm_modes.json
Remove-Item Env:HEB_LLM_REAL_TESTS
```

These real-model results require human checks of factual accuracy and citations. Use Ctrl+C to stop processes; to disable generation change HEB_LLM_ENABLED=false and restart FastAPI. See [the agent README](app/llm/README.md) for behavior and limitations.

**Do not commit** .venv, secrets or unpublished assessment inputs.
