# Architecture Decision Records

Durable decisions for this package (`puzzlegrid-study`). Read these alongside the
[guided walkthrough](../README.md) and the [docs index](../docs/README.md).

| ADR | Title |
|-----|-------|
| [0001](0001-split-source-layout.md) | Split source layout (`common`, `engine`, `solvers`, `replay`, …) |
| [0002](0002-authoritative-engine-bfs-placement.md) | Authoritative engine and BFS placement |
| [0003](0003-replay-clients.md) | Replay clients as the play-side interface |
| [0004](0004-solver-progression-ranking.md) | Solvers as a progressive study; ranking over generation |
| [0005](0005-python-tooling-and-style.md) | Python tooling and style |
| [0006](0006-agent-openai-env.md) | Agent OpenAI credentials via `.env` |
| [0007](0007-mit-human-ai-friendly.md) | MIT license and human/AI-friendly project |

Status of all records above: **Accepted** (2026-07-26).

## ADRs 0001 through 0007 - Foundaational ADRs
- 0001 through 0007 ADRs are initially retroactive documentation, but should keep the current patterns on track: and set expectations for AI during future interactions
- The current game engine is on it's 3rd rewrite, which implies a trend.  It's been stable for the current use-cases. 
