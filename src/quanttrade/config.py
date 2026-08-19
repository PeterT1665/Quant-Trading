"""Typed, YAML-driven configuration for quanttrade.

Everything tunable (which tickers, date range, transaction costs, ...) lives here
and in ``configs/*.yaml`` rather than being scattered as magic numbers across the
codebase. Config is validated by pydantic, so a typo or wrong type fails loudly
and early instead of silently producing garbage.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Which market data to pull and where to cache it."""

    tickers: list[str] = Field(default_factory=lambda: ["AAPL", "MSFT", "SPY"])
    start: str = "2015-01-01"
    end: str | None = None  # None => up to today
    interval: str = "1d"
    cache_dir: Path = Path("data/cache")


class BacktestConfig(BaseModel):
    """Assumptions that make a backtest realistic rather than fantasy."""

    initial_cash: float = 100_000.0
    # Commission as a fraction of traded notional (0.0005 = 5 basis points).
    commission: float = 0.0005
    # Slippage as a fraction of price (execution is worse than the quoted price).
    slippage: float = 0.0005


class Config(BaseModel):
    """Top-level config aggregating every sub-config."""

    data: DataConfig = Field(default_factory=DataConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    """Load and validate a YAML config file.

    Falls back to built-in defaults if the file does not exist, so the project
    is runnable out of the box.
    """
    path = Path(path)
    if not path.exists():
        return Config()
    raw = yaml.safe_load(path.read_text()) or {}
    return Config.model_validate(raw)
