# llm-cost

Token and spend reports over the [`llm`](https://llm.datasette.io) CLI's
logs database.

```sh
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

## Requirements

The reporting commands (`llm cost …`) work against a vanilla `llm`
install — they read straight from the logs SQLite, no hookspecs
required. The optional [inline `--cost` flag](#inline-per-response-cost---cost)
depends on **one hookspec that is not yet in upstream `llm`**, on a
branch of [olivergeorge/llm](https://github.com/olivergeorge/llm)
pending upstream merge.

| Hook | Purpose in this plugin | Branch |
| ---- | ---------------------- | ------ |
| [`after_log_to_db`](https://github.com/olivergeorge/llm/blob/llm-after-log-to-db/docs/plugins/plugin-hooks.md#after_log_to_dbresponse-db) | Fires after the response is persisted, so the plugin can compute cost from the in-memory token counts and print the inline cost line at the same point `llm -u` prints `Token usage:`. | [`llm-after-log-to-db`](https://github.com/olivergeorge/llm/tree/llm-after-log-to-db) |
| The above + the prompt-gates and replay-stores hookspecs | Single-branch superset, shared with `llm-confirm-tokens` and `llm-replay`. | [`combined-prs`](https://github.com/olivergeorge/llm/tree/combined-prs) |

Without `after_log_to_db` there is no clean place in `llm`'s surface
to attach a per-response side effect — the alternative is
monkey-patching `_BaseResponse.log_to_db`, which is fragile across
versions and hostile to other plugins doing the same. The reporting
commands don't touch this hook; skip the fork if you don't want the
inline flag.

## Install

For the reporting commands only (works against stock `llm`):

```bash
llm install llm-cost
```

For the inline `--cost` flag, also install the fork that carries
`after_log_to_db`. `combined-prs` is the single-branch superset:

```bash
# Clone and install the fork on the combined branch
git clone -b combined-prs https://github.com/olivergeorge/llm.git
llm install -e ./llm

# Install the plugin
llm install llm-cost
```

If you only need the inline flag (no other forked plugins),
[`llm-after-log-to-db`](https://github.com/olivergeorge/llm/tree/llm-after-log-to-db)
on its own is sufficient.

If you already have a local checkout of `../llm`, `pyproject.toml`
points at it as an editable dependency — check out `combined-prs`
(or `llm-after-log-to-db`) there and run `uv sync` / `pip install -e .`.

## Inline per-response cost (`--cost`)

Analog of llm's `-u` usage flag: prints the cost of a single call to
stderr after it finishes.

```sh
llm 'hello' --cost
# Hi, how are?
# Cost: $0.0000 (priced)

llm 'write a 200 word essay about sqlite' --cost -u
# ...essay...
# Token usage: 11 input, 253 output, {...}
# Cost: $0.0004 (priced)
```

Enable globally with `LLM_COST=1`, override per-call with `--no-cost`.
When the model isn't in the price table the line reads
`Cost: $0.0000 (unpriced)` so you can see something's off.

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
