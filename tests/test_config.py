"""Smoke tests for the config layer.

These do double duty: they prove the package imports and the tooling works
(Phase 0), and they document the expected shape of the config.
"""

from quanttrade.config import Config, load_config


def test_default_config_loads() -> None:
    cfg = load_config("configs/default.yaml")
    assert isinstance(cfg, Config)
    assert cfg.data.tickers, "expected at least one ticker"
    assert cfg.backtest.initial_cash > 0


def test_config_defaults_without_file() -> None:
    # A missing file should fall back to built-in defaults, not crash.
    cfg = load_config("configs/does_not_exist.yaml")
    assert isinstance(cfg, Config)
    assert cfg.backtest.commission >= 0
