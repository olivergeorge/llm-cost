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

date        resps    cost  bar
----------  -----  ------  --------------------
2026-04-09    817  $26.50  ▓▓▓▓▓▓▓▓▓
2026-04-10   1019  $16.88  ▓▓▓▓▓▓
...
2026-04-22     56   $0.27

Today       $0.27
This week   $7.39  (week-to-date)
This month  $741.55  (month-to-date)
All time    $1,515.06

Top models this month:
  gemini/gemini-3-flash-preview                 $433.96  (58.5%)
  gemini/gemini-3.1-pro-preview                 $223.89  (30.2%)
  gemini/gemini-3.1-pro-preview-customtools      $36.39  ( 4.9%)

Drill down: `llm cost today` · `llm cost --since YYYY-MM-DD` · `llm cost all`
```

## How costs are computed

For each canonical model group:

1. If llm itself logged a `cost_usd` (typically written by a provider
   plugin), that wins at the subgroup level.
2. Otherwise the active price table is consulted. Rates are **per
   token** (matching LiteLLM's `model_prices_and_context_window.json`).
   Input and output tokens are priced separately.
3. When a canonical group spans subgroups with both logged and priced
   costs, the row shows `source=mixed` and the total is the per-subgroup
   sum so nothing drops out.
4. Models that aren't in the price table and had no `cost_usd` logged
   are labelled `unpriced` and called out in a footnote.

Model names are collapsed via `llm.get_model_aliases()` so historical
variants of the same model (e.g. `gemini-3-flash-preview`,
`gemini/gemini-3-flash-preview`, `gemini-flash-latest`) roll up into a
single row.

### Fetching prices

**No price data ships with the plugin** — a stale bundled snapshot is
worse than an explicit refresh. Run this once after install:

```sh
llm cost refresh-prices
```

It downloads
[LiteLLM's `model_prices_and_context_window.json`](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json)
(≈2,600 models) to your user cache (`~/.config/llm-cost/prices.json`
on Linux, `~/Library/Application Support/llm-cost/prices.json` on
macOS, `%APPDATA%\llm-cost\prices.json` on Windows). Subsequent
`llm cost` invocations pick it up automatically. Re-run whenever you
want a fresh snapshot — the command overwrites the cache atomically
and only replaces it if the download parses as valid JSON, so a
failed refresh can't wipe good data.

### Custom price tables

Override the active table with your own YAML (or JSON — the loader
accepts both, since YAML is a superset):

```sh
llm cost today --prices ~/models.yaml
# or
export LLM_COST_PRICES=~/models.yaml
```

The schema matches LiteLLM's, so you can paste an entry straight from
their catalog:

```yaml
claude-opus-4-6:
  input_cost_per_token: 5e-6
  output_cost_per_token: 2.5e-5
my-local-model:
  input_cost_per_token: 0
  output_cost_per_token: 0
```

Resolution order: `--prices PATH` beats `LLM_COST_PRICES` beats the
user cache populated by `refresh-prices`.

### Caveat: one price per model, forever

The price table is a flat snapshot — there's no notion of an effective
date, so a rate change retroactively rewrites the cost of every
historical row for that model. Rows with a logged `cost_usd`
(computed at call time) are unaffected; only the priced fallback
drifts. If you care about historical accuracy, keep `cost_usd`
populated at log time (via the fork's `after_log_to_db` hook or a
provider plugin that writes it), or pin `--prices` to a dated file
when reporting on old windows.

## How dupes are detected

`llm cost dupes` groups responses by a fingerprint over everything the
model sees — calls with identical fingerprints are dupes of each other.
The fingerprint combines:

- Canonical model name (so `gemini/gemini-3-flash-preview` folds into
  `gemini-3-flash-preview`).
- `system` and `prompt` text.
- `options_json` (temperature, max tokens, etc.) and `schema_id`.
- The ordered prior conversation turns — same final prompt in a
  different chat history is a different request.
- Attachments, joined via `prompt_attachments.attachment_id`.
- Prompt + system fragments, joined via `fragments.hash`.

Attachments and fragments are already content-addressed by `llm`
(`attachments.id` is a SHA-256 of the file contents; `fragments.hash`
is a SHA-256 of the fragment text), so the same file uploaded twice
produces the same id and folds into one dupe group automatically.

Savings per group assume you'd keep the first call and skip the rest:
`sum(cost) − first_call_cost`. Costs use the logged `cost_usd` when llm
has one, otherwise fall back to the price-table estimate.

**Uses only the core llm schema** — no plugin dependency. Runs straight
against `responses` + `prompt_attachments` + `prompt_fragments` +
`system_fragments` + `fragments`. Scales to ~50k responses in seconds;
if your history is huge, narrow the window with `--days` or `--since`.

## Paired plugins

- [`llm-confirm-tokens`](https://github.com/olivergeorge/llm-confirm-tokens)
  gates big prompts before they send. Pair with `llm cost top` to spot
  accidents after the fact, then tune the token threshold to head off
  similar ones in future.
- [`llm-replay`](https://github.com/olivergeorge/llm-replay) short-circuits
  identical requests by returning the previously-logged response instead
  of re-hitting the API. Pair with `llm cost dupes` to see how much that
  would save you.

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
- Time-varying prices: keep dated snapshots from each `refresh-prices`
  run and pick the one effective at each row's `datetime_utc`, so a
  rate change doesn't retroactively rewrite historical costs.
- CSV export (`--csv` or `--format csv`) for dropping into a spreadsheet.
- Cached-input pricing for Gemini (LiteLLM's `cache_read_input_token_cost`
  field — currently we use the full input price).
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
- `llm cost diff --since A --until B` to compare two windows side-by-side
  (this month vs last month, etc.).
- Currency conversion for non-USD reporting (hit an FX rate once per run).
- Export to the `responses` table itself: backfill missing `cost_usd`
  values using the price table, so downstream tools see consistent data.
- `llm cost top --conversation ID` — drill into the spend of one
  specific chat session.
