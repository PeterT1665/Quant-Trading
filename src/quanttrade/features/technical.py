"""Technical indicator features computed from OHLCV bars.

TODO(you): implement each function below. All of them must be point-in-time
safe -- a value at row t may only use price data from row t and earlier.
`.rolling()` and `.pct_change()` are trailing by default (they never look
ahead) as long as you don't pass `center=True`, so stick to the defaults.
The tests in tests/test_features.py pin down the exact formulas expected --
run them (`pytest tests/test_features.py -v`) to check your work as you go.
"""

from __future__ import annotations

import pandas as pd


def add_returns(df: pd.DataFrame, periods: int = 1) -> pd.Series:
    """Simple percentage return over `periods` bars, using the `close` column.

        return[t] = close[t] / close[t - periods] - 1

    The first `periods` rows are NaN (no prior bar to compare against).
    """
    return df["close"].pct_change(periods)


def add_rolling_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Rolling standard deviation of 1-bar returns over a trailing `window`.

    Compute 1-bar returns (via `add_returns(df, periods=1)`), then take a
    trailing (non-centered) rolling standard deviation over `window` bars.
    The first `window` rows are NaN.
    """
    return add_returns(df, periods=1).rolling(window).std()


def add_momentum(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """Momentum: the `window`-bar return, i.e. `add_returns(df, periods=window)`."""
    return add_returns(df, periods=window)


def add_sma_ratio(df: pd.DataFrame, short: int = 10, long: int = 50) -> pd.Series:
    """Trend feature: (SMA(short) / SMA(long)) - 1, using trailing simple moving
    averages of `close`. Positive means the short-term average is above the
    long-term average (an uptrend signal); negative means the opposite.
    """
    short_sma = df["close"].rolling(short).mean()
    long_sma = df["close"].rolling(long).mean()
    return short_sma / long_sma - 1


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble the feature matrix, indexed the same as `df`, with these columns:

        "returns_1d"     -- add_returns(df, periods=1)
        "volatility_20d" -- add_rolling_volatility(df, window=20)
        "momentum_10d"   -- add_momentum(df, window=10)
        "sma_ratio"      -- add_sma_ratio(df, short=10, long=50)

    Rows in the warmup period (before enough history exists for the longest
    window) will contain NaNs -- that's expected. Dropping them is the
    caller's job (see `quanttrade.features.dataset.build_dataset`), not this
    function's -- keep this function a pure "compute what you can" step.
    """

    return pd.concat(
        {
            "returns_1d": add_returns(df, periods=1),
            "volatility_20d": add_rolling_volatility(df, window=20),
            "momentum_10d": add_momentum(df, window=10),
            "sma_ratio": add_sma_ratio(df, short=10, long=50),
        },
        axis=1,
    )
