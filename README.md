# llm-cost

Token and spend reports over the [llm](https://llm.datasette.io) logs database.

```sh
llm install llm-cost

llm cost                        # daily sparkline + today/week/month/all-time
llm cost today                  # per-model table for today (local time)
llm cost --since 2026-04-01     # month-to-date
llm cost --days 7               # last 7 days
llm cost --model 'gemini/%'     # filter by model (SQL LIKE pattern)
llm cost all                    # full all-time breakdown (escape hatch)
llm cost --json                 # machine-readable

llm cost top                    # most expensive individual responses
llm cost top -n 20 --days 7
llm cost top --by input         # sort by input tokens

llm cost dupes                  # requests that could have been replayed
llm cost dupes --days 30

llm cost models                 # list models in the active price table
```

## The default landing

Bare `llm cost` gives you a 14-day sparkline plus headline totals:

```
Spend — last 14 days

date        resps      cost  bar
----------  -----  --------  --------------------
2026-04-09    817  $26.4998  ▓▓▓▓▓▓▓▓▓
2026-04-10   1019  $16.8762  ▓▓▓▓▓▓
...
2026-04-22     15   $0.0317

Today       $0.0317
This week   $8.0463      (last 7 days)
This month  $741.2975    (month-to-date)
All time    $1,311.7484

Top models this month:
  gemini/gemini-3-flash-preview  $  433.8507  (58.5%)
  gemini/gemini-3.1-pro-preview  $  223.8888  (30.2%)
  ...
```

## How costs are computed

For each canonical model group:

1. If llm itself logged a `cost_usd` (typically written by a provider
   plugin), that wins at the subgroup level.
2. Otherwise the bundled price table (`llm_cost/prices.yaml`) is used. Prices
   are USD per **one million** tokens. Input and output tokens are priced
   separately.
3. When a canonical group spans subgroups with both logged and priced costs,
   the row shows `source=mixed` and the total is the per-subgroup sum so
   nothing drops out.
4. Models that aren't in the price table and had no `cost_usd` logged are
   labelled `unpriced` and called out in a footnote.

Model names are collapsed via `llm.get_model_aliases()` so historical variants
of the same model (e.g. `gemini-3-flash-preview`, `gemini/gemini-3-flash-preview`,
`gemini-flash-latest`) roll up into a single row.

Override the price table with your own YAML:

```sh
llm cost today --prices ~/models.yaml
# or
export LLM_COST_PRICES=~/models.yaml
```

Two YAML shapes are accepted:

```yaml
# flat (the bundled shape)
claude-opus-4-6:
  input: 5.0
  output: 25.0
```

```yaml
# models.json-style wrapper
models:
  anthropic/claude-opus-4-6:
    input_cost_per_1m: 5.0
    output_cost_per_1m: 25.0
```

## Paired plugins

- [`llm-confirm-tokens`](https://github.com/olivergeorge/llm-confirm-tokens)
  gates big prompts before they send. Pair with `llm cost top` to spot
  accidents after the fact, then tune the token threshold to head off
  similar ones in future.
- [`llm-replay`](https://github.com/olivergeorge/llm-replay) short-circuits
  identical requests by returning the previously-logged response instead
  of re-hitting the API. `llm cost dupes` joins against its `replay_index`
  to show how much you'd save if you turned replay on.

## Dev

```sh
cd llm-cost
uv sync
uv run pytest
```

## Backlog

Unprioritised laundry list, roughly sorted by bang-for-buck at the top:

- `llm cost --by day` / `--by week` / `--by month` inside the per-model
  table (the default landing already does daily — this is for window reports).
- CSV export (`--csv` or `--format csv`) for dropping into a spreadsheet.
- Cached-input pricing for Gemini (the `cached_input_cost_per_1m` field in
  `models.json` — currently we use the full input price).
- Group by conversation (`llm cost --by conversation`) so you can see which
  piece of work cost the most.
- Group by schema tag (the `schema_id` column) to report cost per workflow.
- Break out reasoning / thinking tokens using the `token_details` JSON
  (Gemini's `thoughtsTokenCount`, Anthropic's thinking tokens, OpenAI's
  reasoning tokens) — useful when thinking-heavy runs dominate spend.
- Break out input modalities from `token_details.promptTokensDetails`
  (TEXT vs IMAGE vs AUDIO) so image-heavy prompts are visible.
- Budget alerts: `llm cost --budget 5` exits non-zero when today's spend
  exceeds the threshold, so it can gate CI or a shell prompt.
- `llm cost watch` that streams new responses as they're logged and keeps a
  running total for the current session.
- Publish the snapshot of `prices.yaml` somewhere shared (a GitHub gist
  or a tiny companion repo) so it can drift forward between plugin
  releases without a code change.
- `llm cost diff --since A --until B` to compare two windows side-by-side
  (this month vs last month, etc.).
- Currency conversion for non-USD reporting (hit an FX rate once per run).
- Export to the `responses` table itself: backfill missing `cost_usd`
  values using the price table, so downstream tools see consistent data.
- `llm cost top --conversation ID` — drill into the spend of one
  specific chat session.
- Dupe report `--since-install` flag that respects when llm-replay
  started indexing, so the "X of Y indexed" line is less misleading.
