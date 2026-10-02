"""Datos sintéticos compartidos por las pruebas (deterministas, semilla fija)."""
import numpy as np
import pandas as pd
import pytest

from src import config

BASE_PARAMS = {"ema_fast": 10, "ema_gap": 30, "rsi_window": 14, "rsi_lower": 30,
               "rsi_upper": 70, "bb_window": 20, "bb_k": 2.0, "m_sl": 2.0,
               "m_tp": 3.0, "max_hold": 20}


def synthetic_ohlcv(n: int = 600, seed: int = config.SEED, start_price: float = 100.0) -> pd.DataFrame:
    """Caminata aleatoria geométrica con OHLC consistente (high ≥ max(o,c), low ≤ min(o,c))."""
    rng = np.random.default_rng(seed)
    close = start_price * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    open_ = np.r_[start_price, close[:-1]] * np.exp(rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.015, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.015, n))
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": rng.integers(1_000_000, 5_000_000, n)}, index=idx)


@pytest.fixture
def ohlcv():
    return synthetic_ohlcv()


@pytest.fixture
def params():
    return dict(BASE_PARAMS)


def make_market(ohlc: list[tuple[float, float, float, float]], start: str = "2024-01-01"):
    """MarketData de un activo a partir de tuplas (open, high, low, close)."""
    from src.backtest import MarketData
    arr = np.array(ohlc, dtype=float)
    col = lambda j: arr[:, j].reshape(-1, 1)
    return MarketData(pd.bdate_range(start, periods=len(arr)), ["X"], col(0), col(1), col(2), col(3),
                      np.zeros((len(arr), 1)))


def market_from_ohlcv(df: pd.DataFrame, ticker: str = "X"):
    """MarketData de un activo a partir de un DataFrame OHLCV."""
    from src.backtest import MarketData
    panel = pd.concat({ticker: df}, axis=1).swaplevel(axis=1)
    return MarketData.from_panel(panel, [ticker])


def make_inputs(signal, atr=1e6, m_sl=2.0, m_tp=3.0, max_hold=100, strength=None):
    """StrategyInputs de un activo; escalares se expanden a todo el periodo."""
    from src.backtest import StrategyInputs
    sig = np.asarray(signal, dtype=int).reshape(-1, 1)
    T = len(sig)
    full = lambda v, dt=float: np.broadcast_to(np.asarray(v, dtype=dt).reshape(-1, 1) if np.ndim(v) else v, (T, 1)).astype(dt)
    stren = np.abs(sig) * 2 if strength is None else np.asarray(strength).reshape(-1, 1)
    return StrategyInputs(sig, stren, full(atr), full(m_sl), full(m_tp), full(max_hold, int),
                          np.ones((T, 1), dtype=bool))


def synthetic_panel(n: int = 700, seed: int = config.SEED) -> pd.DataFrame:
    """Panel (campo, ticker) de 6 activos correlacionados con tramos de volatilidad alta."""
    rng = np.random.default_rng(seed)
    vol = np.where((np.arange(n) // 120) % 3 == 2, 0.04, 0.012)
    common = rng.normal(0.0004, 1, n) * vol
    frames = {}
    for j, t in enumerate(config.TICKERS):
        r = 0.7 * common + 0.3 * rng.normal(0, 1, n) * vol
        close = 100 * np.exp(np.cumsum(r))
        open_ = np.r_[100, close[:-1]]
        frames[t] = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.005,
                                  "low": np.minimum(open_, close) * 0.995, "close": close,
                                  "volume": np.full(n, 1e6)})
    panel = pd.concat(frames, axis=1).swaplevel(axis=1)
    panel.index = pd.bdate_range("2018-01-01", periods=n)
    return panel[[(f, t) for f in ["open", "high", "low", "close", "volume"] for t in config.TICKERS]]
