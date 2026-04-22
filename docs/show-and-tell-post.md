# llm-cost: a post-hoc "where did my LLM spend go?" report over `llm`'s logs

I wanted to know where my LLM spend was actually going without scraping dashboards across multiple providers, and `llm` already logs every request/response to SQLite — so I wrote **llm-cost**: a plugin that turns that log into token and spend reports.

https://github.com/olivergeorge/llm-cost

**Upfront: this is a proof-of-concept.** It's had minimal testing beyond my own logs database and is tagged `0.1a0` for a reason. I'm posting it to get feedback on the shape — particularly around model-name canonicalisation and the inline `--cost` hookspec — rather than because it's ready to depend on.

## The default landing

Bare `llm cost` gives a 14-day sparkline plus headline totals — the intent is that a single command tells you whether anything's on fire:

```
$ llm cost

Spend — last 14 days

date        resps    cost  bar
----------  -----  ------  --------------------
2026-04-09    817  $26.50  ▓▓▓▓▓▓▓▓▓
2026-04-10   1019  $16.88  ▓▓▓▓▓▓
2026-04-11    788   $7.13  ▓▓
2026-04-12    650   $6.70  ▓▓
2026-04-13   4286  $60.39  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
2026-04-14     18   $1.05  
2026-04-15     26   $0.50  
2026-04-16      9   $0.24  
2026-04-17    115   $0.65  
2026-04-18      0   $0.00  
2026-04-19      0   $0.00  
2026-04-20     58   $3.46  ▓
2026-04-21    849   $3.66  ▓
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

## Other commands

```sh
llm cost today              # per-model table for today
llm cost --days 7           # last 7 days
llm cost --since 2026-04-01 # month-to-date
llm cost --model 'gemini/%' # filter by model
llm cost all                # full all-time breakdown
llm cost --json             # machine-readable

llm cost top                # most expensive individual responses
llm cost top --by input     # sort by input tokens

llm cost dupes              # requests you could have replayed
llm cost models             # list models in the price table
```

## How costs are computed

1. If `llm` logged a `cost_usd` (some provider plugins do), that wins.
2. Otherwise the active price table is consulted — per-token rates matching [LiteLLM's `model_prices_and_context_window.json`](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json) schema, input/output priced separately.
3. Models missing from both are flagged `unpriced` with a footnote, so a missing price table entry is visible rather than silently dropping spend.
4. When a canonical model group spans subgroups with both logged and priced costs, the row shows `source=mixed` and the total is the per-subgroup sum, so nothing drops out.

Model names are collapsed via `llm.get_model_aliases()` so historical variants (`gemini-3-flash-preview`, `gemini/gemini-3-flash-preview`, `gemini-flash-latest`) roll up into one row. This is the part most likely to misbehave on price tables or provider prefixes I haven't seen.

**No price data ships with the plugin** — a stale bundled snapshot is worse than an explicit refresh. Run `llm cost refresh-prices` once after install; it downloads LiteLLM's catalogue (~2,600 models) to a user cache and subsequent `llm cost` invocations pick it up automatically.

Override with your own table via `--prices ~/models.yaml` or `LLM_COST_PRICES=…`. The schema is LiteLLM's, so you can paste an entry straight from their catalogue:

```yaml
claude-opus-4-6:
  input_cost_per_token: 5e-6
  output_cost_per_token: 2.5e-5
```

## Optional: inline `--cost` per call

Analog of `llm -u`, printed to stderr after the response:

```sh
$ llm 'write a 200 word essay about sqlite' --cost -u
...essay...
Token usage: 11 input, 253 output, {...}
Cost: $0.0004 (priced)
```

Enable globally with `LLM_COST=1`, override per-call with `--no-cost`. Unpriced models read `Cost: $0.0000 (unpriced)` so the gap is visible.

**Caveat:** this one flag depends on the `after_log_to_db` hookspec — same one [llm-replay](https://github.com/olivergeorge/llm-replay) needs, not yet in upstream `llm`. There's a PR branch on a fork, and a `combined-prs` branch that carries both. **The reporting commands above don't need the fork** — they read the logs database directly against stock `llm`.

## Install

```sh
llm install llm-cost
```

For the reporting commands only. For the inline flag too, the README has the fork one-liner.

## The honest caveat

Single-author PoC, only exercised against my own workflow. The LiteLLM snapshot is only as current as your last `llm cost refresh-prices`, and override it if you need exactness beyond that. If your provider names or model aliases look different from mine, the canonicalisation step is the first thing likely to trip, and I'd value bug reports with a snippet of `llm cost models` output.
