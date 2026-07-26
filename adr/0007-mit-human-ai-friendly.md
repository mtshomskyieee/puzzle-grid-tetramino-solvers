# ADR 0007: MIT license and human/AI-friendly project

- **Status:** Accepted
- **Date:** 2026-07-26

## Context

This package is meant to be studied, forked, and extended by both people and
coding agents. Opaque licensing or docs that only humans (or only agents) can
navigate would fight that goal. Secrets and durable architecture choices need
clear homes.  This entire project started as a human-only endeavor but didn't
cross the finish line until AI reviewed and fixed some of the game engine issues.
Needless to say, human and AI have touched all parts of this repo at this point.

## Decision

1. **License:** The package is released under the **MIT License**. The license
   text lives at the package root in `LICENSE`.

2. **Human-friendly surfaces:**
   - `README.md` — guided walkthrough and command reference
   - `docs/` — step write-ups and the results arc
   - `adr/` — durable architecture decisions (this set)

3. **AI-friendly surfaces:** The same documents are the source of truth for
   agents. Preferred reading order: `README.md` → `docs/` → `adr/` → `src/`.
   Agents must treat `.env` as secret (use `.env.example` as the template) and
   must not invent rules outside `puzzlegrid` (ADR 0002).

4. **No parallel “agent-only” policy tree** is required beyond these ADRs and
   the existing README/docs. Keep contribution and architecture guidance in
   places both audiences already open.

## Consequences

- Downstream use, modification, and redistribution follow MIT terms.
- New architectural choices should land as ADRs here, not only in chat or
  ephemeral notes.
- README stays the on-ramp; ADRs stay short and decision-focused so agents can
  retrieve them without dumping the full design write-ups.
