# Rewind

Context compaction that archives what it drops and recalls it on demand, so a model doesn't have to regenerate what it already produced.

## Docs

- [Hackathon plan](docs/HACKATHON_PLAN.md)
- [Learning and implementation guide](docs/LEARNING_AND_IMPLEMENTATION_GUIDE.md)
- [Implementation plan and status](docs/IMPLEMENTATION_PLAN.md)

## Quick start

```bash
uv sync
uv run pytest                 # offline tests, no API calls
export ANTHROPIC_API_KEY=...
uv run rewind-chat            # live terminal chat; compacts at 8,000 tokens
```

Settings are environment variables; see `.env.example`.

## Layout

```
src/rewind/
  config.py       settings from environment variables
  llm.py          gateway around the Anthropic client (swappable, fakeable)
  messages.py     history helpers: to_text, safe cut points
  compaction.py   Compactor protocol + PlainCompactor (baseline)
  metrics.py      token usage and compaction events
  session.py      chat session: count tokens, compact, call the model
  cli.py          terminal chat
tests/            offline tests with a fake LLM
docs/             plans and guides
```
