# llm-cost

Token and spend reports over the [llm](https://llm.datasette.io) logs database.

```sh
llm install llm-cost

llm cost today                  # spend today (local time)
llm cost --since 2026-04-01     # month-to-date
llm cost --days 7               # last 7 days
llm cost --model 'gemini/%'     # filter by model (SQL LIKE pattern)
llm cost --json                 # machine-readable
```

## How costs are computed

For each response row in `responses`:

1. If llm itself logged a `cost_usd` (because another plugin wrote one), that
   wins.
2. Otherwise the bundled price table (`llm_cost/prices.yaml`) is used. Prices
   are USD per **one million** tokens. Input and output tokens are priced
   separately.
3. Models that aren't in the table are surfaced in an "unpriced models"
   footnote with their token totals so nothing silently disappears.

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

## Dev

```sh
cd llm-cost
uv sync
uv run pytest
```

## Backlog

Unprioritised laundry list, roughly sorted by bang-for-buck at the top:

- Collapse duplicate rows where the same model appears once with
  `resolved_model` set and once without (currently shown as separate rows —
  informative, but noisy).
- `llm cost --by day` / `--by week` / `--by month` for trend reports.
- `--top N` and `--sort cost|tokens|responses` for the table renderer.
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
- Sparkline of daily spend next to the totals line, or a small ASCII chart
  via `--chart`.
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

