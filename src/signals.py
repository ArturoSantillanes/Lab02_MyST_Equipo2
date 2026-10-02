"""Indicadores técnicos (implementados a mano) y regla de confirmación 2 de 3.

Tres familias distintas:
    1. Tendencia   — cruce de EMAs.
    2. Momento     — RSI con suavizado de Wilder.
    3. Volatilidad — Bandas de Bollinger.
El ATR (Wilder) no vota; se usa para colocar SL/TP en el motor.

Causalidad: todo valor en t usa solo datos hasta el cierre de t (EMAs recursivas
y ventanas móviles hacia atrás). No hay shift(-k), center=True ni normalizaciones
sobre la serie completa. La ejecución ocurre en el open de t+1 (backtest.py).

Los indicadores se calculan una vez sobre toda la historia disponible y se
recortan después: al ser causales, el valor en t no depende de datos futuros.
"""
from functools import cache

import numpy as np
import pandas as pd

from src import config

INDICATORS = ("ema", "rsi", "bb")


def ema(price: pd.Series, span: int) -> pd.Series:
    """Media móvil exponencial.

    EMA_t = α·P_t + (1−α)·EMA_{t−1},  α = 2/(h+1),  EMA_0 = P_0.
    `ewm(adjust=False)` implementa exactamente esta recursión.
    """
    return price.ewm(span=span, adjust=False).mean()


def wilder_smooth(x: pd.Series, window: int) -> pd.Series:
    """Suavizado de Wilder: W_t = W_{t−1} + (x_t − W_{t−1})/n  (EMA con α = 1/n).

    Los primeros `window` valores quedan en NaN (calentamiento).
    """
    out = x.ewm(alpha=1.0 / window, adjust=False).mean()
    out.iloc[:window] = np.nan
    return out


def rsi(close: pd.Series, window: int) -> pd.Series:
    """Relative Strength Index de Wilder.

    G_t = max(ΔC_t, 0),  L_t = max(−ΔC_t, 0)
    RS_t = Wilder(G)_t / Wilder(L)_t,   RSI_t = 100 − 100/(1 + RS_t)
    """
    delta = close.diff()
    gain = wilder_smooth(delta.clip(lower=0), window)
    loss = wilder_smooth((-delta).clip(lower=0), window)
    rs = gain / loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.where(loss != 0, 100.0).where(gain.notna())


def rolling_mean_std(close: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    """SMA_N y σ_N poblacional (ddof=0) en ventana móvil hacia atrás."""
    return close.rolling(window).mean(), close.rolling(window).std(ddof=0)


def bollinger(close: pd.Series, window: int, k: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bandas de Bollinger con desviación estándar poblacional.

    MB = SMA_N(C),  UB = MB + k·σ_N,  LB = MB − k·σ_N
    Regresa (LB, MB, UB).
    """
    mid, std = rolling_mean_std(close, window)
    return mid - k * std, mid, mid + k * std


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = config.ATR_WINDOW) -> pd.Series:
    """Average True Range de Wilder.

    TR_t = max(H_t − L_t, |H_t − C_{t−1}|, |L_t − C_{t−1}|),  ATR_t = Wilder_n(TR)_t
    """
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    tr.iloc[0] = high.iloc[0] - low.iloc[0]
    return wilder_smooth(tr, window)


class IndicatorCache:
    """Memoiza indicadores por (indicador, ventana) para un OHLCV fijo.

    El optimizador evalúa cientos de θ sobre la misma serie; las ventanas se
    repiten mucho entre trials, así que cada indicador se calcula una sola vez.
    """

    def __init__(self, ohlcv: pd.DataFrame):
        self.ohlcv = ohlcv
        self.close = ohlcv["close"]
        self.ema = cache(lambda span: ema(self.close, span).to_numpy())
        self.rsi = cache(lambda window: rsi(self.close, window).to_numpy())
        self.bands = cache(lambda window: tuple(x.to_numpy() for x in rolling_mean_std(self.close, window)))
        self.atr = atr(ohlcv["high"], ohlcv["low"], self.close).to_numpy()
        self.close_np = self.close.to_numpy()


def indicator_votes(cache_: IndicatorCache, params: dict) -> dict[str, np.ndarray]:
    """Voto individual de cada indicador en {−1, 0, +1} (0 durante el calentamiento).

    s_ema = +1 si EMA_rápida > EMA_lenta, −1 si es menor.
    s_rsi = +1 si RSI < umbral_inf (sobreventa), −1 si RSI > umbral_sup (sobrecompra).
    s_bb  = +1 si C < LB, −1 si C > UB, 0 dentro de las bandas.
    """
    fast = cache_.ema(params["ema_fast"])
    slow = cache_.ema(params["ema_fast"] + params["ema_gap"])
    s_ema = np.sign(fast - slow)
    s_ema[: params["ema_fast"] + params["ema_gap"]] = 0  # calentamiento de la EMA lenta

    r = cache_.rsi(params["rsi_window"])
    s_rsi = np.where(r < params["rsi_lower"], 1, np.where(r > params["rsi_upper"], -1, 0))

    mid, std = cache_.bands(params["bb_window"])
    c = cache_.close_np
    s_bb = np.where(c < mid - params["bb_k"] * std, 1, np.where(c > mid + params["bb_k"] * std, -1, 0))

    # Comparaciones con NaN dan False -> 0 en RSI y BB; en EMA np.sign(NaN) = NaN.
    return {"ema": np.nan_to_num(s_ema).astype(np.int8),
            "rsi": s_rsi.astype(np.int8),
            "bb": s_bb.astype(np.int8)}


def confirm(votes: np.ndarray, min_agree: int = config.MIN_AGREE) -> tuple[np.ndarray, np.ndarray]:
    """Regla de confirmación k de m sobre una matriz de votos (T × m).

    S_t = +1 si Σ_i 1[s_i,t = +1] ≥ k
    S_t = −1 si Σ_i 1[s_i,t = −1] ≥ k
    S_t =  0 en otro caso (sin nueva apertura)
    Fuerza_t = max(Σ 1[s_i,t = +1], Σ 1[s_i,t = −1]) cuando S_t ≠ 0, si no 0.

    Con k = 2 y m = 3 no pueden cumplirse ambas condiciones a la vez.
    """
    votes = np.atleast_2d(votes)
    n_long = (votes == 1).sum(axis=1)
    n_short = (votes == -1).sum(axis=1)
    signal = np.where(n_long >= min_agree, 1, np.where(n_short >= min_agree, -1, 0)).astype(np.int8)
    strength = np.where(signal != 0, np.maximum(n_long, n_short), 0).astype(np.int8)
    return signal, strength


def generate_signals(ohlcv: pd.DataFrame, params: dict, min_agree: int = config.MIN_AGREE,
                     indicators: tuple[str, ...] = INDICATORS,
                     cache_: IndicatorCache | None = None) -> pd.DataFrame:
    """Señal confirmada S_t, fuerza, votos netos, votos individuales y ATR para un activo.

    `strength` = indicadores que coinciden con S_t (2 o 3); `net_votes` = Σ_j x_j cuando
    S_t ≠ 0 (puede ser ±1 si un indicador vota en contra), que da la fuerza continua
    s = Σ_j x_j / 3 del portafolio.

    `min_agree` e `indicators` solo se cambian en el experimento de un solo
    indicador (indicators=("rsi",), min_agree=1). La estrategia oficial usa los
    tres indicadores con min_agree = 2.
    """
    cache_ = cache_ or IndicatorCache(ohlcv)
    votes = indicator_votes(cache_, params)
    matrix = np.column_stack([votes[i] for i in indicators])
    signal, strength = confirm(matrix, min_agree)
    net = np.where(signal != 0, matrix.sum(axis=1), 0)
    return pd.DataFrame({"s_ema": votes["ema"], "s_rsi": votes["rsi"], "s_bb": votes["bb"],
                         "signal": signal, "strength": strength, "net_votes": net, "atr": cache_.atr},
                        index=ohlcv.index)


def signal_correlation(signals: pd.DataFrame) -> pd.DataFrame:
    """Correlación de Pearson entre los votos de los 3 indicadores (redundancia)."""
    return signals[["s_ema", "s_rsi", "s_bb"]].corr()
