# Rewind

Context compaction that archives what it drops and recalls it on demand, so a model doesn't have to regenerate (or guess) what it already produced.

When the conversation passes the context limit, Rewind archives the oldest turns word for word, replaces them with a summary plus a pointer table (`§id → gist`), and gives the model a `recall` tool to fetch exact text by ID or keyword search. Anything not found is reported as NOT_FOUND so the model generates it fresh.

## Docs

- [Hackathon plan](docs/HACKATHON_PLAN.md)
- [Learning and implementation guide](docs/LEARNING_AND_IMPLEMENTATION_GUIDE.md)
- [Implementation plan and status](docs/IMPLEMENTATION_PLAN.md)

## Setup

```bash
uv sync --all-extras
uv run pytest                 # offline tests with a fake model; no API calls
export ANTHROPIC_API_KEY=...
```

Settings are environment variables; see `.env.example`.

## Commands

All of these except `pytest` call the live API and cost money.

| Command | What it does |
| --- | --- |
| `uv run rewind-serve` | Demo at http://127.0.0.1:8000: plain vs Rewind side by side, live cost and context charts, scripted demo player |
| `uv run rewind-chat` | Terminal chat with Rewind (`--plain` for the baseline); `/stats`, `/archive`, `/quit` |
| `uv run rewind-bench` | Benchmark on planted facts; writes `results/<time>/report.md` and `results.json` (asks before spending) |
| `uv run --extra mcp rewind-mcp` | MCP server with `remember`, `recall` and `list_memories` tools |

Useful benchmark flags: `--strategies none,plain,rewind`, `--scenarios billing-service`, `--runs 3`, `--context-limit 8000`, `--yes`.

### Use the MCP server in Claude Code

```bash
claude mcp add rewind -e REWIND_DATA_DIR="$HOME/.rewind" -e REWIND_MCP_SESSION=my-project \
  -- uv run --directory "/Users/apple/Documents/App Development/rewind" --extra mcp rewind-mcp
```

## Settings

| Variable | Default | Meaning |
| --- | --- | --- |
| `REWIND_MODEL` | `claude-opus-5` | Model for chat and summaries |
| `REWIND_CONTEXT_LIMIT` | `8000` | Tokens before compaction (small for demos) |
| `REWIND_KEEP_RECENT_MESSAGES` | `4` | Latest messages never compacted |
| `REWIND_RECALL_MIN_COVERAGE` | `0.5` | Share of query terms a search hit must contain |
| `REWIND_PROMPT_CACHING` | `1` | Automatic prompt caching |
| `REWIND_REFUSAL_FALLBACKS` | `1` | Server-side refusal fallback (turn off on Bedrock, Vertex, Foundry) |
| `REWIND_STORE` | `jsonl` | `jsonl` or `sqlite` |
| `REWIND_DATA_DIR` | `data` | Where the archive lives |

## Layout

```
src/rewind/
  config.py         settings from environment variables
  llm.py            gateway around the Anthropic client (fallbacks, caching)
  messages.py       history helpers: to_text, safe cut points, turns, gists
  compaction.py     NullCompactor, PlainCompactor, RewindCompactor
  search.py         tokenizer, BM25, query coverage
  store.py          ArchiveStore protocol + JsonlArchive
  store_sqlite.py   SqliteArchive (FTS5)
  recall.py         recall tool: by id or search, gist or full, NOT_FOUND memory
  session.py        chat session: token counting, compaction, tool loop
  factory.py        builds none / plain / rewind sessions
  pricing.py        cost estimates per model
  bench/            scenarios, grading, runner, rewind-bench CLI
  server.py         FastAPI demo server; web/index.html is the UI
  mcp_server.py     MCP server
  cli.py            terminal chat
tests/              offline tests with fake models
docs/               plans and guides
```
