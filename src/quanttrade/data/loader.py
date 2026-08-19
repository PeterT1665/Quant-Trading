"""Fetch and cache daily OHLCV price data from Yahoo Finance.

Point-in-time note: a backtest run "as of" day D must never see information
that wasn't actually available on day D. This loader just gets clean OHLCV
bars onto disk -- it doesn't try to enforce that on its own. The rule to keep
in mind downstream (features/backtest) is that day D's close is only known
*after* D's session ends, so a signal computed from D's close can only be
acted on starting D+1.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from quanttrade.config import DataConfig

_COLUMNS = ["open", "high", "low", "close", "volume"]


def _cache_path(cache_dir: Path, ticker: str, interval: str) -> Path:
    return cache_dir / f"{ticker}_{interval}.parquet"


def fetch_ticker(ticker: str, start: str, end: str | None, interval: str) -> pd.DataFrame:
    """Download raw OHLCV bars for one ticker from Yahoo Finance.

    ``auto_adjust=True`` folds splits/dividends into OHLC so prices are
    continuous across corporate actions -- the right default for a
    return-prediction model, since a 2:1 split shouldn't look like a 50% drop.
    """
    raw = yf.Ticker(ticker).history(start=start, end=end, interval=interval, auto_adjust=True)
    if raw.empty:
        raise ValueError(f"yfinance returned no data for {ticker!r} ({start}..{end}, {interval})")
    df = raw.rename(columns=str.lower)[_COLUMNS].copy()
    df.index.name = "date"
    return df


def load_ticker(
    ticker: str, start: str, end: str | None, interval: str, cache_dir: Path
) -> pd.DataFrame:
    """Load one ticker's OHLCV bars, using the on-disk parquet cache when possible."""
    path = _cache_path(cache_dir, ticker, interval)
    if path.exists():
        return pd.read_parquet(path)

    df = fetch_ticker(ticker, start, end, interval)
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def load_prices(cfg: DataConfig) -> dict[str, pd.DataFrame]:
    """Load OHLCV bars for every ticker in ``cfg``, returning ``{ticker: DataFrame}``."""
    return {
        ticker: load_ticker(ticker, cfg.start, cfg.end, cfg.interval, cfg.cache_dir)
        for ticker in cfg.tickers
    }
