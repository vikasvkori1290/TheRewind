# Rewind: Implementation Plan

2026-09-18

Rewind is built in eight phases. Each phase ends with passing tests and a working demo of that phase's feature before the next starts.

## Principles

- **Testable without the network.** Every component takes its LLM client as a parameter, so unit tests use a fake client and cost nothing. Live API runs are opt-in.
- **One job per module.** Storage, compaction, recall and the chat session are separate modules behind small interfaces, so any one can be replaced (for example, JSONL storage to SQLite) without touching the others.
- **Two compaction strategies share one interface.** `PlainCompactor` (the baseline) and `RewindCompactor` (archive + pointers) are interchangeable, which makes the side-by-side demo and the benchmark a configuration change.
- **Measure from day one.** Every model call records input, output and cache tokens.

## Phases

| Phase | Deliverable | Key files | Done when |
| --- | --- | --- | --- |
| 0 | Project scaffold | `pyproject.toml`, `src/rewind/config.py`, `tests/` | `uv run pytest` passes on an empty suite |
| 1 | Chat loop, token counting, plain compaction | `session.py`, `messages.py`, `compaction.py`, `cli.py` | Tests pass with a fake client; CLI chat compacts at the limit |
| 2 | Archive store (JSONL + hash files + BM25) | `store.py` | Dedupe, reload and session-isolated search tests pass |
| 3 | Rewind compaction with pointers | `compaction.py` (`RewindCompactor`) | Every pointer in a summary resolves to an archived record |
| 4 | Recall tool, threshold, failed-search memory | `recall.py`, `session.py` tool loop | Model recalls a planted fact by ID and by search; unknown facts return NOT_FOUND |
| 5 | Metrics and benchmark | `metrics.py`, `eval/` | Benchmark runs all four setups and writes a results file |
| 6 | API server and split-screen demo UI | `server.py`, `web/` | Two panes and a live token/cost chart update each turn |
| 7 | Stretch: MCP server, cache-aware compaction, SQLite | `mcp_server.py`, `store_sqlite.py` | Chosen stretch goal works end to end |

## Phase details

### Phase 0: Scaffold
- `uv` project with `src/` layout, `pytest`, `.gitignore` (ignores `data/`, `.venv/`, `.env`).
- `config.py` reads settings from environment variables with safe defaults (model, context limit, turns kept, recall threshold, data directory).

### Phase 1: Chat loop and plain compaction
- `messages.py`: turn any message (string or content blocks) into plain text; find a safe cut point that never separates a `tool_use` from its `tool_result`.
- `compaction.py`: `Compactor` protocol plus `PlainCompactor`, which summarizes old turns and prepends the summary to the first kept user message.
- `session.py`: `Session` sends messages, counts tokens before each call, compacts when over the limit, records usage and compaction events.
- `cli.py`: `uv run rewind-chat` for a manual chat in the terminal.
- Tests: to-text conversion, cut-point safety, compaction output shape, usage accounting, compaction trigger.

### Phase 2: Archive store
- `ArchiveStore` protocol (`put`, `get_by_id`, `search`) and `JsonlArchive` implementation.
- Content ID = first 12 hex characters of SHA-256 over session + text; identical text in one session is stored once, and IDs are validated before use as file names.
- Search ranks with BM25 and gates on query coverage (share of meaningful query terms found), which does not depend on archive size.
- BM25 index built per session and rebuilt only after that session changes.

### Phase 3: Rewind compaction
- `RewindCompactor` archives each old user+assistant pair verbatim before summarizing and writes a pointer table (`§id → gist`) into the summary header.
- Gist: first meaningful line in v1; a model-written gist as an option.

### Phase 4: Recall
- `recall` tool definition, handler with ID lookup, BM25 fallback, score threshold, gist/full levels.
- Remember failed queries per session and return NOT_FOUND without searching again.
- Recalled text wrapped as quoted archive content (it is data, not instructions).

### Phase 5: Metrics and benchmark
- Cost estimate from per-model prices.
- Scripted conversations with planted facts; runner for four setups; exact-match grading for code and numbers; results saved as JSON and a Markdown table.

### Phase 6: Server and UI
- FastAPI: create session (plain or Rewind), send message, get metrics, list archive.
- Single-page UI: two chat panes on the same script, live chart of tokens and cost.

### Phase 7: Stretch
- Pick one: MCP server exposing `recall`; cache-aware compaction with `cache_read_input_tokens` on the dashboard; SQLite storage.

## Cost control
- Unit tests never call the API.
- Live runs use a small context limit (8k tokens) so each demo conversation stays cheap.
- The model is set in one place (`REWIND_MODEL`); switching to a cheaper model is a one-line change.

## Status

- [x] Phase 0: Scaffold
- [x] Phase 1: Chat loop, token counting, plain compaction
- [x] Phase 2: Archive store
- [ ] Phase 3: Rewind compaction
- [ ] Phase 4: Recall
- [ ] Phase 5: Metrics and benchmark
- [ ] Phase 6: Server and UI
- [ ] Phase 7: Stretch
