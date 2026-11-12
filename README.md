# HEB AI Assistant

Technical assessment project: Spanish-language assistant for retail operations questions across structured data and internal policy documents.

## Proposed stack
- **UI:** Streamlit
- **API:** FastAPI + Pydantic
- **Structured data:** DuckDB + Python
- **Document retrieval:** PyMuPDF, local embeddings through Ollama, lightweight indexed retrieval
- **LLM:** Ollama (model selected after local performance tests)
- **Tests:** pytest

The stack is a proposal pending validation against the full assignment brief. The application is not yet implemented.

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

## Local setup (after application files are implemented)
Create a Python virtual environment, install the project dependencies, and run Ollama locally. Exact install/run commands will be added when the dependency manifest and entrypoints are committed.

**Do not commit** `.venv`, credentials, or original assessment data/documents without distribution permission.
