"""The prediction target: forward returns.

This is the one place in the pipeline where looking past row t is *correct* --
the target IS the future outcome we're trying to predict. The invariant to
never break: the target for row t must never be usable as an input FEATURE
for row t or any earlier row. Keeping prediction (features/technical.py) and
target (this file) in separate modules makes that mistake structurally
harder to make by accident.
"""

from __future__ import annotations

import pandas as pd


def forward_return(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """The return an investor earns holding from day t's close to day (t+horizon)'s close.

        target[t] = close[t + horizon] / close[t] - 1

    Implementation note: this needs a *negative* shift (`.shift(-horizon)`),
    the opposite direction from a normal lagging feature -- get this sign
    backwards and every "prediction" downstream is silently cheating by
    looking at data from the past instead of the future. The last `horizon`
    rows will be NaN (no future close exists yet to compute a return into);
    dropping them is handled by `quanttrade.features.dataset.build_dataset`.
    """
    future_close = df["close"].shift(-horizon)
    return future_close / df["close"] - 1
