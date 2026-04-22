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
