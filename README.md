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
cp .env.example .env          # then fill in the provider and key
```

Settings are environment variables or lines in `.env`; see `.env.example`.

### Providers and keys

The easiest way: run `uv run rewind-serve`, open **http://127.0.0.1:8000/settings**, and for each provider paste a key, click **Save**, then **Test** (which lists the models the key can use). Pick the active provider and model at the top of the page. Keys are saved in `data/rewind_config.json` (readable only by your user, git-ignored) and are only ever shown masked. A key already in your shell environment (e.g. `OPENAI_API_KEY`, `NVIDIA_API_KEY`) is used when none is saved.

Supported: Claude (Anthropic API), Amazon Bedrock (API key or AWS login), OpenAI, Google Gemini, NVIDIA NIM, OpenRouter, Groq, Mistral, DeepSeek, Together AI, and any custom OpenAI-compatible endpoint.

**Cost tracking.** Every model call (chat, compaction summaries, recalls) is recorded in `data/usage.jsonl`, and the settings page shows calls, tokens and cost per provider and model. Claude models have built-in prices and are tracked in dollars. Any other model is tracked in tokens only, unless you add its price under **Model prices**; from then on its calls are tracked in dollars too. Prices are estimates; your provider's bill is authoritative.

**Security.** The server listens on 127.0.0.1 only, answers only requests addressed to localhost, and refuses cross-site requests, so other websites can't read or change your keys or spend your credits.

The same saved keys, active model and ledger are used by `rewind-chat` and `rewind-bench`. Environment variables still work without the settings page:

| Provider | Settings | Credentials |
| --- | --- | --- |
| Claude API | `REWIND_PROVIDER=anthropic` | `REWIND_API_KEY` = Console key (`sk-ant-api...`). Subscription tokens (`sk-ant-oat...`) are rejected. |
| Amazon Bedrock, API key | `REWIND_PROVIDER=bedrock` | `REWIND_API_KEY` or `AWS_BEARER_TOKEN_BEDROCK` = Bedrock API key. Short-term keys expire after at most 12 hours. |
| Amazon Bedrock, AWS login | `REWIND_PROVIDER=bedrock`, `REWIND_BEDROCK_AUTH=aws` | `aws login` (or a profile via `AWS_PROFILE`) |
| NVIDIA NIM (or any OpenAI-compatible API) | `REWIND_PROVIDER=nim`, `REWIND_MODEL=openai/gpt-oss-20b`, optional `REWIND_BASE_URL` | `REWIND_API_KEY` or `NVIDIA_API_KEY` (`nvapi-...`) |

On NIM, choose a model that supports tool calling: `openai/gpt-oss-20b` is fast; `deepseek-ai/deepseek-v4-flash-0731` works but is slow. Token counts are estimated (about 4 characters per token) and cost shows as tokens because NIM models have no price in `pricing.py`. Set `REWIND_MAX_OUTPUT_TOKENS=4096` for NIM models with smaller output limits.

A Claude Pro or Max subscription can't be used: its login token (`sk-ant-oat...`) only works in Claude apps and Claude Code.

On Bedrock, model IDs get the `anthropic.` prefix automatically, the server-side refusal fallback is off (Bedrock doesn't offer it), and the region comes from `REWIND_AWS_REGION`, `AWS_REGION`, or `~/.aws/config`. Claude Opus 5 needs model access granted in the Bedrock console; `REWIND_MODEL=claude-sonnet-5` or `claude-opus-4-8` are open to all accounts.

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
| `REWIND_PROVIDER` | `anthropic` | `anthropic`, `bedrock` or `nim` |
| `REWIND_BASE_URL` | NIM endpoint | nim: any OpenAI-compatible base URL |
| `REWIND_API_KEY` | – | Key for the provider (overrides the SDK's own variables) |
| `REWIND_BEDROCK_AUTH` | `key` | Bedrock: `key` or `aws` |
| `REWIND_AWS_REGION` | – | Bedrock region override |
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
