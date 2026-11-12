# Claude Code instructions — HEB AI Assistant

Read `docs/REQUIREMENTS.md`, `docs/ARCHITECTURE.md`, `docs/CONTRACTS.md`, and `docs/DECISIONS.md` before work. Follow the same shared project rules as `AGENTS.md`.

## Role and ownership
- Claude Code owns `app/rag/`, PDF extraction, indexing, retrieval, evidence metadata, and related tests.
- Codex owns `app/api/`, `app/data/`, `app/tools/`, and backend integration tests.
- Shared modules (`app/llm/`, `frontend/`, contracts, dependency manifests, CI) require an agreed GitHub Issue before edits.

## Collaboration
1. Work in a separate branch/worktree and link changes to an Issue.
2. Use GitHub Issues/PR comments to ask Codex for interfaces, flag blockers, and communicate decisions.
3. Never assume another agent sees this chat or has pulled your local edits.
4. Keep contracts backward compatible unless an approved Issue authorizes a change.
5. Request cross-review; do not merge or deploy without user approval.

## Quality and security
- Cite PDF filename, page, and section where available; detect conflicts and prioritize authoritative procedures only when supported by the source documents.
- Measure retrieval accuracy, abstentions, latency, and tokens where applicable.
- Never commit secrets, local virtual environments, or restricted source datasets without authorization.
