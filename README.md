# quanttrade

A from-scratch quantitative trading research system, built to learn the full stack
of a data-driven trading strategy — and engineered like production software.

> **Status:** Phase 0 (scaffolding). See the roadmap below.

## What this is

A modular pipeline that ingests market data (and, later, news), learns a predictive
signal, turns predictions into trading decisions, and evaluates them with a
**leak-free backtester built from scratch**. A simple web UI lets you explore results.

The design deliberately separates *prediction* (a model that outputs a number) from
*decision* (a policy that turns that number into buy/sell/hold), so each half can be
tested independently.

## Roadmap

| Phase | Focus | Key concept |
|-------|-------|-------------|
| 0 | Scaffolding, tooling, CI | Reproducible research setup |
| 1 | Data layer + features | Point-in-time correctness (no lookahead) |
| 2 | Supervised model | Walk-forward / time-series validation |
| 3 | Backtest engine + policy | Honest performance (costs, slippage, drawdown) |
| 4 | News signal (NLP) | Alternative data, multimodal features |
| 5 | Web app (Streamlit → FastAPI) | Model serving, API design |
| 6 | RL agent (stretch) | MDPs, reward shaping |

## Project layout

```
src/quanttrade/
  data/       # fetching, caching, cleaning market data
  features/   # technical indicators, (later) news features
  models/     # baselines + PyTorch models
  backtest/   # engine, transaction costs, metrics
  policy/     # prediction -> position sizing
  api/        # web backend
  config.py   # typed, YAML-driven configuration
tests/        # pytest (incl. leakage & backtest-correctness checks)
notebooks/    # exploration (kept out of src)
configs/      # YAML configs (no hard-coded params)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Design principles

- **No lookahead / leakage** — the cardinal sin of backtesting; guarded by tests.
- **Realistic costs** — every backtest applies commission and slippage.
- **Config-driven** — parameters live in `configs/`, never hard-coded.
- **Tested & typed** — pytest + mypy + ruff, enforced in CI.
