# Architecture (proposed)

```text
Streamlit UI
    | POST /chat (JSON)
    v
FastAPI + Pydantic validation
    |
    v
Ollama orchestration adapter
    |-- structured query tools --> DuckDB (CSV-derived tables)
    |-- policy retrieval tool ---> PDF extraction + indexed search + citations
    |
    v
Grounded response + evidence + per-question metrics
```

## Boundaries
- HTTP GET/POST transports requests; neither stores the source data.
- CSV files are loaded once or queried via DuckDB; do not pass raw tables to the LLM.
- PDF files are extracted and indexed offline; retrieval returns only relevant passages with metadata.
- The model interprets intent, chooses permitted tools, and summarizes results. It must not generate unrestricted executable SQL.
- Backend enforces source provenance, dates, validation, error handling, and abstention.
- Every Ollama call contributes to aggregate per-question token and latency metrics; embedding work is tracked separately when measurable.

## Deployment
Develop and validate locally first. A Render-hosted API cannot directly reach a laptop's `localhost` Ollama service. Select and document a reachable inference deployment before cloud deployment.

## Ownership
- Codex: `app/api/`, `app/data/`, `app/tools/`.
- Claude Code: `app/rag/`.
- Shared after approved Issue: `app/llm/`, `frontend/`, dependencies, integration wiring and CI.

See `docs/CONTRACTS.md` for provisional interface definitions.
