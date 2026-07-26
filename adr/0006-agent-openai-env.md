# ADR 0006: Agent OpenAI credentials via `.env`

- **Status:** Accepted
- **Date:** 2026-07-26

## Context

Step 5 (`uv run play-agents`) uses a LangGraph flow whose primary placement
chooser is the transformer scorer. An LLM is only a recovery fallback when a
placement is missed or navigation fails. That fallback needs credentials
without baking secrets into the repository or requiring a key for the happy
path.

We wire recovery through **OpenAI’s API / `ChatOpenAI` out of convenience**, and
because that interface is a familiar fit when pointing at **local large language
models** (OpenAI-compatible servers). This is not a claim that OpenAI is the
only or best provider for this project.

## Decision

1. **Secrets live only in a local `.env` file**, loaded with `python-dotenv`
   (`load_dotenv()` in `src/replay/replay_using_agents.py`). Today the only
   variable for LLM recovery is `OPENAI_API_KEY`.

2. **Ship `.env.example`**, not real keys. Document that the key is optional and
   only enables the recovery path. `.env` is gitignored and must never be
   committed.

3. **Missing key is non-fatal:** log a warning and continue with the transformer
   scorer alone; skip or fail soft on LLM assist.

4. **LLM wiring (current default):** lazy `langchain_openai.ChatOpenAI` (default
   model `gpt-4o-mini`, overridable). The scorer remains the primary placement
   decision maker on every move.

5. **Provider is swappable.** Replacing OpenAI with another hosted LLM (or an
   OpenAI-compatible local endpoint) is welcome — go do it. Keep secrets in
   `.env`, keep the LLM on the recovery path only, and update `.env.example` /
   deps when the wiring changes.

## Consequences

- Agents play works without any LLM provider for the main path; demos do not
  require a paid key.
- Docs and prompts must reference `.env.example` only — never paste live keys.
- New providers should extend `.env.example` and this ADR’s variable list rather
  than hard-coding secrets in source; OpenAI remains a convenience default, not
  a locked-in choice.
