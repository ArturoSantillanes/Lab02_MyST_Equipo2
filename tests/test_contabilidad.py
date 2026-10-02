"""Contabilidad del motor: identidad de equity, golden file al centavo y suma de costos."""
import numpy as np
import pytest

from conftest import BASE_PARAMS, make_inputs, make_market, market_from_ohlcv, synthetic_ohlcv
from src import config
from src.backtest import run_backtest, single_asset_inputs
from src.signals import generate_signals

C = config.COMMISSION


def golden_case():
    """Largo del día 1 al 4, reversa a corto el día 4 y time-stop del corto el día 6.

    ATR enorme: SL/TP nunca se tocan, solo actúan la señal y el time-stop.
    """
    ohlc = [(100, 101, 99, 100), (100, 111, 99, 110), (110, 121, 109, 120), (120, 126, 119, 125),
            (125, 126, 119, 120), (120, 121, 114, 115), (110, 111, 109, 110), (110, 111, 109, 110)]
    signal = [1, 0, 0, -1, 0, 0, 0, 0]
    max_hold = [10, 10, 10, 2, 2, 2, 2, 2]   # se lee el día de la señal
    return make_market(ohlc), make_inputs(signal, max_hold=max_hold)


def test_golden_file_al_centavo():
    market, inputs = golden_case()
    res = run_backtest(market, inputs)

    # --- Cálculo a mano ---
    e0 = 1_000_000.0
    q_long = e0 / (100 * (1 + C))                 # día 1: compra al open 100
    cash = e0 - q_long * 100 * (1 + C)            # = 0
    long_exit = q_long * 125                      # día 4: vende al open 125
    cash += long_exit * (1 - C)
    long_gross = q_long * (125 - 100)
    long_costs = C * q_long * (100 + 125)
    e_star = cash                                 # equity al open neto del costo de salida
    q_short = e_star / (125 * (1 + C))            # día 4: abre corto al open 125
    cash += q_short * 125 * (1 - C)
    cash -= q_short * 110 * (1 + C)               # día 6: time-stop, recompra al open 110
    short_gross = q_short * (125 - 110)
    short_costs = C * q_short * (125 + 110)
    final_equity = cash

    t = res.trades
    assert list(t["direccion"]) == ["largo", "corto"]
    assert list(t["motivo_salida"]) == ["señal", "time-stop"]
    assert t["pnl_bruto"].tolist() == pytest.approx([long_gross, short_gross], abs=0.01)
    assert t["costos"].tolist() == pytest.approx([long_costs, short_costs], abs=0.01)
    assert t["pnl_neto"].tolist() == pytest.approx([long_gross - long_costs, short_gross - short_costs], abs=0.01)
    assert res.equity.iloc[-1] == pytest.approx(final_equity, abs=0.01)
    assert res.cash.iloc[-1] == pytest.approx(final_equity, abs=0.01)
    assert res.total_costs == pytest.approx(long_costs + short_costs, abs=0.01)
    assert res.equity.iloc[-1] - e0 == pytest.approx(t["pnl_neto"].sum(), abs=0.01)


def test_equity_igual_efectivo_mas_posiciones():
    ohlcv = synthetic_ohlcv(500)
    sig = generate_signals(ohlcv, BASE_PARAMS)
    res = run_backtest(market_from_ohlcv(ohlcv), single_asset_inputs(sig, BASE_PARAMS))
    marked = res.cash + res.positions["X"] * ohlcv["close"]
    np.testing.assert_allclose(res.equity, marked, rtol=0, atol=1e-6)
    assert res.n_trades > 5


def test_costos_igual_suma_de_nocional_por_comision():
    ohlcv = synthetic_ohlcv(500)
    res = run_backtest(market_from_ohlcv(ohlcv), single_asset_inputs(generate_signals(ohlcv, BASE_PARAMS), BASE_PARAMS))
    f = res.fills
    still_open = int(res.positions["X"].iloc[-1] != 0)
    entries_exits = f[f["motivo"] != "límite de apalancamiento"]
    assert len(entries_exits) == 2 * res.n_trades + still_open   # cada operación = entrada + salida
    assert res.total_costs == pytest.approx(C * f["nocional"].sum(), abs=1e-6)
    np.testing.assert_allclose(f["costo"], C * f["nocional"], atol=1e-9)


def test_nocional_fijo_costos_igual_operaciones_por_comision():
    """Precio constante: la entrada y la salida tienen el mismo nocional N; costos = #ejecuciones × c × N."""
    market = make_market([(100, 100, 100, 100)] * 6)
    res = run_backtest(market, make_inputs([1, 0, 0, 0, 0, 0], max_hold=2))
    notional = 1_000_000 / (1 + C)
    assert len(res.fills) == 2 and res.n_trades == 1
    np.testing.assert_allclose(res.fills["nocional"], notional)
    assert res.total_costs == pytest.approx(2 * C * notional, abs=0.01)
