"""Regla 2 de 3: con un solo indicador a favor no se abre; con 2 o 3 sí."""
import numpy as np
import pytest

from src.signals import bollinger, confirm, generate_signals, rsi


@pytest.mark.parametrize("votes, expected_signal, expected_strength", [
    ((1, 0, 0), 0, 0),      # un solo indicador largo
    ((0, -1, 0), 0, 0),     # un solo indicador corto
    ((0, 0, 0), 0, 0),
    ((1, 1, 0), 1, 2),      # largo confirmado por 2
    ((1, 1, 1), 1, 3),      # largo confirmado por 3
    ((-1, 0, -1), -1, 2),   # corto confirmado por 2
    ((-1, -1, -1), -1, 3),
    ((1, -1, 0), 0, 0),     # mixto sin mayoría
    ((1, -1, -1), -1, 2),   # mixto con mayoría corta
    ((1, 1, -1), 1, 2),     # mixto con mayoría larga
])
def test_regla_dos_de_tres(votes, expected_signal, expected_strength):
    signal, strength = confirm(np.array([votes]))
    assert signal[0] == expected_signal
    assert strength[0] == expected_strength


def test_crisis_exige_tres_de_tres():
    signal, _ = confirm(np.array([(1, 1, 0), (1, 1, 1)]), min_agree=3)
    assert list(signal) == [0, 1]


def test_indicador_unico_abre_con_un_voto():
    signal, strength = confirm(np.array([[1], [0], [-1]]), min_agree=1)
    assert list(signal) == [1, 0, -1] and list(strength) == [1, 0, 1]


def test_votos_coinciden_con_las_formulas(ohlcv, params):
    """Los votos de RSI y Bollinger corresponden a sus umbrales y bandas."""
    sig = generate_signals(ohlcv, params)
    lb, _, ub = bollinger(ohlcv["close"], params["bb_window"], params["bb_k"])
    expected_bb = np.where(ohlcv["close"] < lb, 1, np.where(ohlcv["close"] > ub, -1, 0))
    np.testing.assert_array_equal(sig["s_bb"], expected_bb)
    r = rsi(ohlcv["close"], params["rsi_window"])
    expected_rsi = np.where(r < params["rsi_lower"], 1, np.where(r > params["rsi_upper"], -1, 0))
    np.testing.assert_array_equal(sig["s_rsi"], expected_rsi)


def test_votos_netos_definen_la_fuerza_continua(ohlcv, params):
    """net_votes = Σ votos cuando hay señal (±1, ±2 o ±3) y 0 si no; mismo signo que la señal."""
    sig = generate_signals(ohlcv, params)
    opened = sig["signal"] != 0
    assert (sig.loc[~opened, "net_votes"] == 0).all()
    assert (np.sign(sig.loc[opened, "net_votes"]) == sig.loc[opened, "signal"]).all()
    assert sig.loc[opened, "net_votes"].abs().isin([1, 2, 3]).all()
