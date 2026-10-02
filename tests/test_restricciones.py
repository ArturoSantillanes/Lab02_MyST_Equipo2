"""Restricciones del motor: sin apalancamiento, SL primero, gaps al open y ejecución en t+1."""
import numpy as np
import pytest

from conftest import BASE_PARAMS, make_inputs, make_market, market_from_ohlcv, synthetic_ohlcv
from src.backtest import run_backtest, single_asset_inputs
from src.signals import generate_signals


def test_sin_apalancamiento_en_toda_la_serie():
    ohlcv = synthetic_ohlcv(800, seed=7)
    res = run_backtest(market_from_ohlcv(ohlcv), single_asset_inputs(generate_signals(ohlcv, BASE_PARAMS), BASE_PARAMS))
    assert res.exec_gross.max() <= 1 + 1e-9
    assert (res.cash + res.positions["X"].clip(upper=0) * ohlcv["close"] >= -1e-6).all()  # nunca se pide prestado efectivo


def test_sl_primero_si_sl_y_tp_en_la_misma_barra():
    # Largo al open 100 del día 1; ATR = 5 -> SL = 90, TP = 115. La barra 2 toca ambos.
    ohlc = [(100, 100, 100, 100), (100, 101, 99, 100), (100, 120, 85, 100), (100, 100, 100, 100)]
    res = run_backtest(make_market(ohlc), make_inputs([1, 0, 0, 0], atr=5.0))
    t = res.trades.iloc[0]
    assert t["motivo_salida"] == "SL" and t["precio_salida"] == pytest.approx(90.0)


def test_gap_se_ejecuta_al_open():
    # SL = 90; el día 2 abre en 80 (debajo del SL): la salida es a 80, no a 90.
    ohlc = [(100, 100, 100, 100), (100, 101, 99, 100), (80, 82, 78, 81), (81, 81, 81, 81)]
    res = run_backtest(make_market(ohlc), make_inputs([1, 0, 0, 0], atr=5.0))
    t = res.trades.iloc[0]
    assert t["motivo_salida"] == "SL" and t["precio_salida"] == pytest.approx(80.0)
    assert t["fecha_salida"] == res.equity.index[2]


def test_gap_favorable_toma_tp_al_open():
    ohlc = [(100, 100, 100, 100), (100, 101, 99, 100), (130, 131, 129, 130), (130, 130, 130, 130)]
    res = run_backtest(make_market(ohlc), make_inputs([1, 0, 0, 0], atr=5.0))
    t = res.trades.iloc[0]
    assert t["motivo_salida"] == "TP" and t["precio_salida"] == pytest.approx(130.0)


def test_corto_sl_primero():
    # Corto al open 100: SL = 110, TP = 85. La barra 2 toca ambos.
    ohlc = [(100, 100, 100, 100), (100, 101, 99, 100), (100, 115, 80, 100), (100, 100, 100, 100)]
    res = run_backtest(make_market(ohlc), make_inputs([-1, 0, 0, 0], atr=5.0))
    t = res.trades.iloc[0]
    assert t["direccion"] == "corto" and t["motivo_salida"] == "SL" and t["precio_salida"] == pytest.approx(110.0)


def test_ejecucion_en_t_mas_1_al_open():
    ohlc = [(100, 100, 100, 100), (100, 100, 100, 100), (104, 106, 103, 105), (105, 105, 105, 105)]
    res = run_backtest(make_market(ohlc), make_inputs([0, 1, 0, 0]))
    entry = res.fills.iloc[0]
    assert entry["fecha"] == res.equity.index[2]      # señal al cierre del día 1 -> open del día 2
    assert entry["precio"] == pytest.approx(104.0)
    assert res.positions["X"].iloc[1] == 0             # el día de la señal aún no hay posición


def test_un_solo_indicador_no_abre_posicion_en_el_motor(ohlcv, params):
    """Si en ningún día coinciden 2 indicadores, el motor no opera."""
    from src.signals import confirm
    votes = np.zeros((len(ohlcv), 3), dtype=int)
    votes[::7, 0] = 1                                   # solo la EMA vota largo
    votes[3::11, 1] = -1                                # solo el RSI vota corto (otros días)
    signal, strength = confirm(votes)
    assert not signal.any()
    res = run_backtest(market_from_ohlcv(ohlcv), make_inputs(signal, atr=2.0))
    assert res.n_trades == 0 and res.equity.iloc[-1] == pytest.approx(1_000_000)


def test_portafolio_rp_sin_apalancamiento_y_con_ambas_direcciones():
    """Portafolio de 6 activos con RP: Σ|w| ≤ 1 tras cada ejecución en toda la serie."""
    import pandas as pd
    from conftest import synthetic_panel
    from src import config
    from src.backtest import MarketData
    from src.portfolio import PortfolioSizer, compose_strategy_inputs

    panel = synthetic_panel(n=600)
    dates = panel.index[150:]
    regimes = pd.Series(np.tile([0] * 40 + [1] * 30 + [2] * 20, 10)[: len(panel)], index=panel.index)
    inputs = compose_strategy_inputs(panel, config.TICKERS, dates, lambda d, tk, r: BASE_PARAMS, regimes).inputs
    returns = panel["close"][config.TICKERS].pct_change()
    sizer = PortfolioSizer(returns.loc[dates].to_numpy(), regimes.loc[dates].to_numpy())
    res = run_backtest(MarketData.from_panel(panel, config.TICKERS).slice(150, 600), inputs, sizer)
    assert res.exec_gross.max() <= 1 + 1e-9
    trades = res.trades
    assert (trades["direccion"] == "largo").any() and (trades["direccion"] == "corto").any()
