"""Assemble a clean, aligned (features, target) training matrix.

This is the point-in-time assembly step: for every row that survives, the
features are computable using only data up to and including that row's date,
and the target is a known, non-NaN future outcome. Rows where either side is
NaN (feature warmup at the start, or the target's forward window at the end)
are dropped.
"""

from __future__ import annotations

import pandas as pd

from quanttrade.features.target import forward_return  # noqa: F401 -- used once implemented
from quanttrade.features.technical import build_features  # noqa: F401 -- used once implemented


def build_dataset(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Build a model-ready DataFrame: feature columns plus a `target` column, NaN-free.

    Steps:
      1. Compute features via `build_features(df)`.
      2. Compute the target via `forward_return(df, horizon)`, as a column named "target".
      3. Concatenate them (they share `df`'s index) into one DataFrame.
      4. Drop any row containing a NaN in *either* the features or the target.
    """
    features = build_features(df)
    target = forward_return(df, horizon).rename("target")
    dataset = pd.concat([features, target], axis=1)
    return dataset.dropna()
