"""Causalidad: la señal en t calculada con df.iloc[:t+1] es igual a la de la serie completa."""
import numpy as np
import pandas as pd
import pytest

from src.signals import generate_signals

COLUMNS = ["s_ema", "s_rsi", "s_bb", "signal", "strength", "atr"]


@pytest.mark.parametrize("t", [120, 300, 450, 598, 599])
def test_senal_en_t_no_depende_del_futuro(ohlcv, params, t):
    full = generate_signals(ohlcv, params)
    truncated = generate_signals(ohlcv.iloc[: t + 1], params)
    pd.testing.assert_series_equal(full[COLUMNS].iloc[t], truncated[COLUMNS].iloc[t], check_names=False)


def test_alterar_el_futuro_no_cambia_el_pasado(ohlcv, params):
    """Si se modifican las barras posteriores a t, ninguna señal ≤ t cambia."""
    t = 400
    shocked = ohlcv.copy()
    shocked.iloc[t + 1:, :4] *= 1.5
    a = generate_signals(ohlcv, params).iloc[: t + 1]
    b = generate_signals(shocked, params).iloc[: t + 1]
    pd.testing.assert_frame_equal(a, b)


def test_hay_senales_en_ambas_direcciones(ohlcv, params):
    s = generate_signals(ohlcv, params)["signal"]
    assert (s == 1).any() and (s == -1).any()
    assert np.isin(s.unique(), [-1, 0, 1]).all()
