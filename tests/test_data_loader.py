"""Tests for the price-data loader.

These stub out the network call (fetch_ticker) entirely, so they exercise
just the caching logic: fetch-then-write on a miss, read-without-fetching on
a hit. No real yfinance calls happen in this test module.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quanttrade.config import DataConfig
from quanttrade.data import loader


def _fake_bars() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=3, freq="D", name="date")
    return pd.DataFrame(
        {"open": [1.0, 2.0, 3.0], "high": [1.0, 2.0, 3.0], "low": [1.0, 2.0, 3.0],
         "close": [1.0, 2.0, 3.0], "volume": [100, 200, 300]},
        index=idx,
    )


def test_load_ticker_fetches_and_caches_on_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def fake_fetch(ticker: str, start: str, end: str | None, interval: str) -> pd.DataFrame:
        calls.append(ticker)
        return _fake_bars()

    monkeypatch.setattr(loader, "fetch_ticker", fake_fetch)

    df = loader.load_ticker("AAPL", "2024-01-01", None, "1d", tmp_path)

    assert calls == ["AAPL"]
    assert list(df["close"]) == [1.0, 2.0, 3.0]
    assert (tmp_path / "AAPL_1d.parquet").exists()


def test_load_ticker_reads_from_cache_on_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def fake_fetch(ticker: str, start: str, end: str | None, interval: str) -> pd.DataFrame:
        calls.append(ticker)
        return _fake_bars()

    monkeypatch.setattr(loader, "fetch_ticker", fake_fetch)

    loader.load_ticker("AAPL", "2024-01-01", None, "1d", tmp_path)  # miss: populates cache
    loader.load_ticker("AAPL", "2024-01-01", None, "1d", tmp_path)  # hit: should not fetch again

    assert calls == ["AAPL"]  # only the first call actually fetched


def test_load_prices_loads_every_configured_ticker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loader, "fetch_ticker", lambda *a, **kw: _fake_bars())

    cfg = DataConfig(tickers=["AAPL", "MSFT"], cache_dir=tmp_path)
    result = loader.load_prices(cfg)

    assert set(result) == {"AAPL", "MSFT"}
    assert all(isinstance(df, pd.DataFrame) for df in result.values())
