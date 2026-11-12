# Git worktrees — Codex and Claude Code

Git worktrees allow separate working directories backed by one local Git repository. They are **local folders**, not a GitHub feature, and GitHub cannot create them remotely.

## Prerequisites
- A local clone of this repository with Git installed.
- Commit or stash local changes before creating worktrees.
- Fetch current remote branches with `git fetch origin`.
- Merge the approved documentation PR into `main` before using these instructions as the project's source of truth.

## Recommended Windows layout
Keep repositories outside OneDrive when possible to reduce sync conflicts:

```text
C:\Dev\HEB\
  main\       # original clone, main branch
  codex\      # separate worktree, feature/codex-bootstrap
  claude\     # separate worktree, feature/claude-rag
```

Do not move an existing repository by copying its `.git` directory blindly. If the current project is under OneDrive, a fresh clone into `C:\Dev\HEB\main` is safer after pushing all intended changes.

## PowerShell commands: fresh clone
```powershell
New-Item -ItemType Directory -Force C:\Dev\HEB | Out-Null
git clone https://github.com/h8hyf2rw9y-afk/HEB.git C:\Dev\HEB\main
Set-Location C:\Dev\HEB\main
git fetch origin
git switch main
git pull --ff-only origin main

git worktree add -b feature/codex-bootstrap ..\codex main
git worktree add -b feature/claude-rag ..\claude main
git worktree list
```

The branch names are examples. Use a distinct branch per Issue; avoid reusing branches that already exist.

## Agent workflow
- Codex opens `C:\Dev\HEB\codex`; Claude Code opens `C:\Dev\HEB\claude`.
- Each agent works only in its assigned worktree and owned directories, with explicit Issue-scoped exceptions.
- Each worktree has its own working files and branch, but all worktrees share Git object storage and branch refs.
- Each worktree needs its own ignored local `.venv` and secrets/configuration; never commit either.
- Agents commit and push their own branches, then open separate PRs targeting `main`.
- User reviews and approves merges. Merge one PR at a time when contracts overlap; refresh the other branch afterward.
- For remote/cloud agents, configure their own isolated checkout or worktree in the remote environment. Local worktrees do not grant access to local files or Ollama automatically.

## Inspect and maintain
```powershell
git worktree list
git -C C:\Dev\HEB\codex status
git -C C:\Dev\HEB\claude status
```

Remove a worktree only after committing/pushing or explicitly discarding its work. For example, from `main`:

```powershell
git worktree remove ..\codex
```

This removes the linked working directory but not necessarily its Git branch. Do not use `--force` without reviewing uncommitted changes. Do not use `git worktree prune` as a substitute for safely removing an active worktree.

## Integration checklist
1. GitHub Issue defines scope, ownership and acceptance criteria.
2. Each agent works in its own worktree/branch.
3. Shared interface changes are proposed in the Issue and reflected in `docs/CONTRACTS.md`.
4. Each agent runs tests, pushes, and opens a PR.
5. Cross-review occurs; user approves before merge/deployment.
6. After merge, update the other worktree's branch from current `origin/main` using a deliberate merge/rebase strategy, resolving conflicts before new work.
