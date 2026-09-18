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
| 5 | Metrics and benchmark | `pricing.py`, `bench/` | Benchmark runs all three setups and writes a results file |
| 6 | API server and split-screen demo UI | `server.py`, `web/index.html` | Two panes and a live token/cost chart update each turn |
| 7 | Prompt caching, SQLite store, MCP server | `llm.py`, `store_sqlite.py`, `mcp_server.py` | Same store tests pass on SQLite; MCP tools answer over stdio |

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
- `RewindCompactor` archives each old turn (a user message plus everything up to the next user message) verbatim, then summarizes and writes a pointer table (`§id → gist`) into the summary header.
- Pointers from earlier compactions are carried forward; the table shows the latest 40 and says how many older turns are reachable by search. Old headers are stripped before archiving.
- Gist: the user's first line plus any function or class names defined in the turn.

### Phase 4: Recall
- `recall` tool definition, handler with ID lookup, BM25 fallback, a query-coverage threshold (default 0.5), gist/full levels.
- The session runs a bounded tool loop (5 rounds), keeps each tool_use paired with its tool_result, and rolls back the whole turn on a refusal.
- `factory.py` builds `none`, `plain` and `rewind` sessions, so the CLI, server and benchmark agree.
- Remember failed queries per session and return NOT_FOUND without searching again.
- Recalled text wrapped as quoted archive content (it is data, not instructions).

### Phase 5: Metrics and benchmark
- Cost estimate from per-model prices.
- Two scripted scenarios (billing service, mobile app), each with 4 planted facts (code, numbers, a person, a decision), 14 filler turns and 4 graded questions plus 1 control question about something never discussed.
- Three setups: `none` (no compaction, upper bound), `plain`, `rewind`. The "recent window + running summary" setup is what `plain` already does.
- `uv run rewind-bench` asks for confirmation (live API), then writes `results/<timestamp>/results.json` and `report.md` with accuracy, tokens, cost and tokens per correct answer.

### Phase 6: Server and UI
- FastAPI (`uv run rewind-serve`, localhost only): create a plain + Rewind pair, send one message to both in parallel, get state, list the archive. Messages are length-checked, a pair accepts one message at a time, and at most 20 pairs are kept in memory.
- Single page (`src/rewind/web/index.html`): two chat panes, per-pane stats and context meter, compaction and recall markers, a "Play demo script" button for the scenarios, and live charts of cumulative cost and context size. Model output is rendered as text, never HTML.

### Phase 7: Stretch
- All three built: automatic prompt caching (cache reads counted in usage and cost); `SqliteArchive` (FTS5, same tests as JSONL, `REWIND_STORE=sqlite`); MCP server (`rewind-mcp`) with `remember`, `recall` and `list_memories`.

## Not yet verified
- No live API run yet: all tests use fake models. Run `uv run rewind-bench` to get real accuracy and cost numbers.
- The demo page was checked over HTTP with a fake model, not visually in a browser.

## Cost control
- Unit tests never call the API.
- Live runs use a small context limit (8k tokens) so each demo conversation stays cheap.
- The model is set in one place (`REWIND_MODEL`); switching to a cheaper model is a one-line change.

## Status

- [x] Phase 0: Scaffold
- [x] Phase 1: Chat loop, token counting, plain compaction
- [x] Phase 2: Archive store
- [x] Phase 3: Rewind compaction
- [x] Phase 4: Recall
- [x] Phase 5: Metrics and benchmark
- [x] Phase 6: Server and UI
- [x] Phase 7: Prompt caching, SQLite store, MCP server
