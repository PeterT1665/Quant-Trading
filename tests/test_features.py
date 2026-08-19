"""Tests for feature engineering and the prediction target.

These pin down exact formulas and, more importantly, encode the no-lookahead
invariants: a feature at row t must not depend on prices after t, and the
target must be a genuinely forward-looking value with correct row alignment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quanttrade.features.dataset import build_dataset
from quanttrade.features.target import forward_return
from quanttrade.features.technical import (
    add_momentum,
    add_returns,
    add_rolling_volatility,
    add_sma_ratio,
    build_features,
)


def _price_df(closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="B", name="date")
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1000] * len(closes),
        },
        index=idx,
    )


class TestReturns:
    def test_first_row_is_nan(self) -> None:
        df = _price_df([100, 101, 102, 103])
        r = add_returns(df, periods=1)
        assert pd.isna(r.iloc[0])

    def test_values_match_formula(self) -> None:
        df = _price_df([100, 110, 121])
        r = add_returns(df, periods=1)
        assert r.iloc[1] == pytest.approx(0.10)
        assert r.iloc[2] == pytest.approx(0.10)

    def test_multi_period(self) -> None:
        df = _price_df([100, 110, 121])
        r = add_returns(df, periods=2)
        assert pd.isna(r.iloc[0])
        assert pd.isna(r.iloc[1])
        assert r.iloc[2] == pytest.approx(0.21)


class TestRollingVolatility:
    def test_warmup_is_nan(self) -> None:
        df = _price_df([100] * 10)
        vol = add_rolling_volatility(df, window=5)
        assert vol.iloc[:5].isna().all()

    def test_constant_price_has_zero_vol(self) -> None:
        df = _price_df([100] * 10)
        vol = add_rolling_volatility(df, window=5)
        assert vol.iloc[5:].fillna(0).abs().max() < 1e-9

    def test_does_not_look_ahead(self) -> None:
        # Two series identical up to (and including) index 5, diverging from
        # index 6 onward -- vol at index 5 must be identical between them,
        # since a point-in-time rolling window can only see data up to index 5.
        a = _price_df([100, 101, 99, 102, 98, 101, 200, 5, 300])
        b = a.copy()
        b.iloc[6:, b.columns.get_loc("close")] = [50, 50, 50]
        vol_a = add_rolling_volatility(a, window=5)
        vol_b = add_rolling_volatility(b, window=5)
        assert vol_a.iloc[5] == pytest.approx(vol_b.iloc[5])


class TestMomentum:
    def test_matches_n_day_return(self) -> None:
        df = _price_df([100, 105, 110, 121])
        mom = add_momentum(df, window=3)
        assert pd.isna(mom.iloc[2])
        assert mom.iloc[3] == pytest.approx(0.21)


class TestSmaRatio:
    def test_equal_averages_gives_zero(self) -> None:
        df = _price_df([100] * 60)
        ratio = add_sma_ratio(df, short=10, long=50)
        assert ratio.iloc[-1] == pytest.approx(0.0)

    def test_uptrend_gives_positive_ratio(self) -> None:
        closes = list(np.linspace(100, 200, 60))
        df = _price_df(closes)
        ratio = add_sma_ratio(df, short=10, long=50)
        assert ratio.iloc[-1] > 0


class TestBuildFeatures:
    def test_returns_expected_columns(self) -> None:
        closes = list(np.linspace(100, 150, 80))
        df = _price_df(closes)
        feats = build_features(df)
        assert isinstance(feats, pd.DataFrame)
        assert feats.index.equals(df.index)
        assert set(feats.columns) >= {
            "returns_1d",
            "volatility_20d",
            "momentum_10d",
            "sma_ratio",
        }


class TestForwardReturn:
    def test_matches_formula(self) -> None:
        df = _price_df([100, 110, 121, 133.1])
        target = forward_return(df, horizon=1)
        assert target.iloc[0] == pytest.approx(0.10)
        assert target.iloc[1] == pytest.approx(0.10)
        assert pd.isna(target.iloc[-1])

    def test_horizon_two(self) -> None:
        df = _price_df([100, 110, 121, 133.1])
        target = forward_return(df, horizon=2)
        assert target.iloc[0] == pytest.approx(0.21)
        assert pd.isna(target.iloc[-1])
        assert pd.isna(target.iloc[-2])

    def test_is_forward_not_backward(self) -> None:
        # A classic bug: shift(horizon) instead of shift(-horizon). Catch it directly --
        # target[0] must reflect close[1]/close[0]-1 = 1.0, using the FUTURE bar.
        df = _price_df([100, 200, 100, 400])
        target = forward_return(df, horizon=1)
        assert target.iloc[0] == pytest.approx(1.0)


class TestBuildDataset:
    def test_no_nans_survive(self) -> None:
        closes = list(np.linspace(100, 150, 80))
        df = _price_df(closes)
        dataset = build_dataset(df, horizon=1)
        assert not dataset.isna().any().any()

    def test_has_target_column(self) -> None:
        closes = list(np.linspace(100, 150, 80))
        df = _price_df(closes)
        dataset = build_dataset(df, horizon=1)
        assert "target" in dataset.columns

    def test_row_alignment_is_correct(self) -> None:
        # Spot-check one surviving row: its target must equal the forward
        # return independently computed for that same date.
        closes = list(np.linspace(100, 150, 80))
        df = _price_df(closes)
        dataset = build_dataset(df, horizon=1)
        full_target = forward_return(df, horizon=1)
        sample_date = dataset.index[10]
        assert dataset.loc[sample_date, "target"] == pytest.approx(full_target.loc[sample_date])
