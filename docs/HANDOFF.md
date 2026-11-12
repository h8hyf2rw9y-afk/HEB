# Cross-agent handoff protocol

This file describes the process; **live communication happens in GitHub Issues and Pull Requests**. Branch-local edits to this file are not instantly visible to the other agent.

For local folder isolation and commands, see [WORKTREES.md](WORKTREES.md).

## Before starting
- Read requirements, architecture, contracts, and decisions.
- Confirm the assigned Issue, ownership, dependencies, and acceptance tests.
- Use a dedicated branch/worktree; synchronize with the latest approved `main`.

## When blocked
Open or comment on a GitHub Issue with:
1. What you need from the other agent.
2. The affected function/schema/file.
3. A proposed contract and test case.
4. Whether your work can continue independently.

## When delivering
Create a PR stating:
- Scope and files changed.
- How to run tests and actual results.
- New dependencies, assumptions, and unresolved risks.
- Token/latency impact if relevant.
- Specific questions for cross-review.

The other agent reviews the PR when explicitly assigned or triggered; GitHub does not automatically wake either agent. User approval is required before merge or deployment.
