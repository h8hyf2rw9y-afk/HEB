# Codex instructions — HEB AI Assistant

Read `docs/REQUIREMENTS.md`, `docs/ARCHITECTURE.md`, `docs/CONTRACTS.md`, and `docs/DECISIONS.md` before proposing or changing code.

## Role and ownership
- Codex owns `app/api/`, `app/data/`, `app/tools/`, and backend integration tests unless a task explicitly changes ownership.
- Claude Code owns `app/rag/` and document-retrieval tests.
- Shared modules (`app/llm/`, `frontend/`, contracts, dependency manifests, CI) require an agreed GitHub Issue before edits.
- Do not modify another agent's owned files without an explicit handoff and user approval.

## Workflow
1. Read the linked GitHub Issue and acceptance criteria.
2. Work on a feature branch; never push directly to `main`.
3. Record cross-agent dependencies in the Issue or PR; do not rely on local chat history or a stale handoff file.
4. Run relevant tests and include commands/results in the PR.
5. Request cross-review and wait for the user's approval before merge or deployment.
6. Never upload secrets, `.venv`, raw recruitment datasets, or internal PDFs without explicit authorization.

## Engineering rules
- Structured numbers come from validated DuckDB/Python queries, not LLM arithmetic.
- PDF answers must include document/page evidence and handle conflicting policies.
- Keep Ollama calls isolated behind an adapter; log per-call prompt/output tokens and latency, aggregating all calls per user question.
- Validate tool arguments with Pydantic; never execute untrusted model-generated SQL.
- Prefer small changes, typed interfaces, deterministic tests, and clear error handling.
