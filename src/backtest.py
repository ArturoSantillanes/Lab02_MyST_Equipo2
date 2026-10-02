"""Motor de backtesting event-driven, multi-activo, con costos y sin apalancamiento.

Un solo motor: el backtest de un activo (optimizador, benchmark individual) es este
mismo motor con n = 1 y `FullCapitalSizer`; el portafolio usa un sizer de
portfolio.py. SL/TP, costos y contabilidad existen en un único lugar.

Orden de eventos en el día t
    1. Open de t:  (a) stops por gap: si el open ya rebasó SL/TP, se sale al open;
                   (b) se ejecutan las órdenes pendientes (entradas, salidas,
                       rebalanceos) al open y se cobran costos;
                   (c) límite de apalancamiento: si Σ|q_i·P_i| > equity, se escala.
    2. Durante t:  SL/TP con high/low, incluida la barra de entrada. Si SL y TP caen
                   en la misma barra se ejecuta primero el SL (convención conservadora).
    3. Cierre de t: marca a mercado, registro de equity y decisiones con datos ≤ t;
                   las órdenes quedan pendientes para el open de t+1.

Flujos de efectivo (c = tasa de costo de la operación)
    Compra de q:  Cash −= q·P + c·q·P      Venta de q:  Cash += q·P − c·q·P
    Corto: la venta abre la posición (q < 0) y registra el pasivo q·P.
    Equity_t = Cash_t + Σ_i q_i·P_i,t   (q_i < 0 en cortos)

Sin apalancamiento: después de cada ejecución Σ|q_i·P_i| ≤ equity. Con exposición
bruta ≤ 100% el margen de cortos de Reg-T (50% inicial, 25–30% de mantenimiento)
siempre se cumple, por eso no se modela una cuenta de margen aparte (supuesto).
"""
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from src import config

EXIT_REASONS = ("señal", "SL", "TP", "time-stop", "cambio de régimen", "rebalanceo", "fin de ventana")
ADV_WINDOW = 20
LEVERAGE_TRIM = "límite de apalancamiento"   # recorte de una posición existente, no es salida


@dataclass
class CostModel:
    """Componentes del costo por operación, como fracción del nocional.

    tasa = comisión + spread_bps/2 + slippage_bps + η·(|q|/ADV)^(2/3)
    `spread_bps` es el spread bid-ask completo: cada operación cruza medio spread.
    `borrow_fee_annual` se cobra diario sobre el nocional corto: fee/252·|q·C_t|.
    Oficial: solo la comisión; el resto vale 0 salvo en el escenario realista.
    """
    commission: float = config.COMMISSION
    spread_bps: float = 0.0
    slippage_bps: float = 0.0
    impact_eta: float = 0.0
    borrow_fee_annual: float = 0.0

    def rate(self, qty_abs: float, adv: float) -> float:
        r = self.commission + self.spread_bps / 2e4 + self.slippage_bps / 1e4
        if self.impact_eta > 0 and adv > 0:
            r += self.impact_eta * (qty_abs / adv) ** (2 / 3)
        return r


@dataclass
class MarketData:
    """Precios alineados (T × n) de los activos simulados."""
    dates: pd.DatetimeIndex
    tickers: list[str]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    adv: np.ndarray  # volumen promedio de 20 días hasta t−1 (para impacto)

    @classmethod
    def from_panel(cls, panel: pd.DataFrame, tickers: list[str]) -> "MarketData":
        get = lambda f: panel[f][tickers].to_numpy(dtype=float)
        adv = panel["volume"][tickers].rolling(ADV_WINDOW).mean().shift(1).to_numpy(dtype=float)
        return cls(panel.index, list(tickers), get("open"), get("high"), get("low"), get("close"),
                   np.nan_to_num(adv))

    def slice(self, start: int, stop: int) -> "MarketData":
        s = slice(start, stop)
        return MarketData(self.dates[s], self.tickers, self.open[s], self.high[s], self.low[s],
                          self.close[s], self.adv[s])


@dataclass
class StrategyInputs:
    """Señales y parámetros de salida vigentes en cada día (T × n).

    `signal` ya es la señal efectiva de apertura (confirmada con la regla del
    régimen vigente). SL/TP/max_hold se leen en el día de la señal y quedan fijos.
    """
    signal: np.ndarray
    strength: np.ndarray
    atr: np.ndarray
    m_sl: np.ndarray
    m_tp: np.ndarray
    max_hold: np.ndarray
    entry_mask: np.ndarray

    def slice(self, start: int, stop: int) -> "StrategyInputs":
        s = slice(start, stop)
        return StrategyInputs(*(getattr(self, f)[s] for f in
                                ("signal", "strength", "atr", "m_sl", "m_tp", "max_hold", "entry_mask")))


class Sizer(Protocol):
    """Traduce direcciones deseadas en pesos objetivo al cierre de t.

    Regresa (pesos con signo, motivo). NaN = conservar la cantidad actual.
    Un peso 0 para un activo con dirección deseada ≠ 0 lo cierra o bloquea.
    `daily = False` indica que solo hay que consultarlo en días con eventos.
    """
    daily: bool

    def target_weights(self, t: int, desired: np.ndarray, strength: np.ndarray,
                       weights: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, str]: ...


class FullCapitalSizer:
    """Un activo con 100% del capital: w = dirección al entrar; entre eventos no se toca."""
    daily = False

    def target_weights(self, t, desired, strength, weights, events):
        return np.where(events, desired.astype(float), np.nan), "rebalanceo"


@dataclass
class BacktestResult:
    """Salida del motor. Los arreglos se guardan crudos y se exponen como pandas."""
    dates: pd.DatetimeIndex
    tickers: list[str]
    equity_array: np.ndarray
    cash_array: np.ndarray
    positions_array: np.ndarray
    weights_array: np.ndarray
    exec_gross_array: np.ndarray       # Σ|w| inmediatamente después de ejecutar en el open
    trade_records: list = field(repr=False)
    fill_records: list = field(repr=False)
    total_costs: float = 0.0
    n_leverage_scalings: int = 0

    @property
    def equity(self) -> pd.Series:
        return pd.Series(self.equity_array, index=self.dates, name="equity")

    @property
    def cash(self) -> pd.Series:
        return pd.Series(self.cash_array, index=self.dates, name="cash")

    @property
    def positions(self) -> pd.DataFrame:
        return pd.DataFrame(self.positions_array, index=self.dates, columns=self.tickers)

    @property
    def weights(self) -> pd.DataFrame:
        return pd.DataFrame(self.weights_array, index=self.dates, columns=self.tickers)

    @property
    def exec_gross(self) -> pd.Series:
        return pd.Series(self.exec_gross_array, index=self.dates, name="exec_gross")

    @property
    def gross_exposure(self) -> pd.Series:
        return self.weights.abs().sum(axis=1)

    @property
    def trades(self) -> pd.DataFrame:
        cols = ["activo", "direccion", "fecha_entrada", "precio_entrada", "fecha_salida",
                "precio_salida", "dias", "pnl_bruto", "costos", "pnl_neto", "motivo_salida"]
        return pd.DataFrame(self.trade_records, columns=cols)

    @property
    def fills(self) -> pd.DataFrame:
        cols = ["fecha", "activo", "cantidad", "precio", "nocional", "costo", "motivo"]
        return pd.DataFrame(self.fill_records, columns=cols)

    @property
    def n_trades(self) -> int:
        return len(self.trade_records)

    @property
    def daily_turnover(self) -> pd.Series:
        """T_t = ½·Σ_i |Δq_i·P_i| / Equity_t en cada día con operaciones (½ cuenta un viaje redondo)."""
        if not self.fill_records:
            return pd.Series(dtype=float)
        f = self.fills
        f = f.assign(frac=f["nocional"].to_numpy() / self.equity.reindex(f["fecha"]).to_numpy())
        return 0.5 * f.groupby("fecha")["frac"].sum()

    @property
    def turnover(self) -> float:
        """Rotación anualizada: Σ_t T_t por año. Costo anual ≈ rotación · 2c."""
        years = max(len(self.dates) / config.TRADING_DAYS, 1 / config.TRADING_DAYS)
        return float(self.daily_turnover.sum() / years)


def run_backtest(market: MarketData, strat: StrategyInputs, sizer: Sizer | None = None,
                 costs: CostModel = CostModel(), initial_capital: float = config.INITIAL_CAPITAL,
                 force_close_last: bool = False) -> BacktestResult:
    """Simula día por día y regresa equity, posiciones, pesos, operaciones y costos.

    El loop usa escalares de Python (listas) porque n ≤ 6: las operaciones numpy
    sobre arreglos diminutos cuestan más que la aritmética que hacen.
    `force_close_last`: purga del walk-forward; cierra todo al cierre del último día.
    """
    sizer = sizer or FullCapitalSizer()
    T, n = market.close.shape
    dates, tickers = market.dates, market.tickers
    O, H, L, C = (a.tolist() for a in (market.open, market.high, market.low, market.close))
    ADV = market.adv.tolist()
    finite_atr = np.isfinite(strat.atr)
    SIG = np.where(finite_atr, strat.signal, 0).tolist()   # sin ATR no hay SL/TP: no se abre
    STR, ATR = strat.strength.tolist(), strat.atr.tolist()
    MSL, MTP, MH, MASK = strat.m_sl.tolist(), strat.m_tp.tolist(), strat.max_hold.tolist(), strat.entry_mask.tolist()
    rate = costs.rate
    assets = range(n)

    cash = float(initial_capital)
    q = [0.0] * n
    direction = [0] * n
    entry_idx = [-1] * n
    sl = [0.0] * n
    tp = [0.0] * n
    pos_hold = [0] * n
    pos_strength = [0] * n
    # Acumuladores de la operación abierta por activo.
    avg_px = [0.0] * n
    entry_px = [0.0] * n
    realized = [0.0] * n
    trade_cost = [0.0] * n

    pending_w = [None] * n          # None = sin orden (conservar)
    pending_reason = [None] * n
    pending_setup = [None] * n      # (atr, m_sl, m_tp, max_hold, strength) del día de la señal
    pending_rebal_reason = "rebalanceo"

    eq_arr, cash_arr, exec_gross = np.empty(T), np.empty(T), np.zeros(T)
    pos_arr, w_arr = np.empty((T, n)), np.empty((T, n))
    trades, fills = [], []
    total_costs = 0.0
    n_scalings = 0

    def fill(i, t, dq, px, reason):
        """Ejecuta Δq al precio px, cobra costo y actualiza la operación abierta."""
        nonlocal cash, total_costs
        notional = abs(dq) * px
        cost = notional * rate(abs(dq), ADV[t][i])
        cash -= dq * px + cost
        total_costs += cost
        trade_cost[i] += cost
        if q[i] == 0 or (dq > 0) == (q[i] > 0):
            avg_px[i] = (abs(q[i]) * avg_px[i] + abs(dq) * px) / (abs(q[i]) + abs(dq))
        else:
            realized[i] += -dq * (px - avg_px[i])
        q[i] += dq
        if abs(q[i]) < 1e-9:
            q[i] = 0.0
        fills.append((dates[t], tickers[i], dq, px, notional, cost, reason))

    def open_position(i, t, qty, px, setup):
        a, m_sl, m_tp, max_hold, strength = setup
        d = 1 if qty > 0 else -1
        realized[i] = trade_cost[i] = avg_px[i] = 0.0
        fill(i, t, qty, px, "entrada")
        direction[i], entry_idx[i], entry_px[i] = d, t, px
        sl[i] = px - d * m_sl * a
        tp[i] = px + d * m_tp * a
        pos_hold[i], pos_strength[i] = int(max_hold), int(strength)

    def close_position(i, t, px, reason):
        fill(i, t, -q[i], px, reason)
        trades.append((tickers[i], "largo" if direction[i] > 0 else "corto", dates[entry_idx[i]],
                       entry_px[i], dates[t], px, t - entry_idx[i] + 1, realized[i], trade_cost[i],
                       realized[i] - trade_cost[i], reason))
        direction[i], entry_idx[i] = 0, -1

    for t in range(T):
        o, h, l, c = O[t], H[t], L[t], C[t]

        # 1a. Stops por gap en posiciones de días anteriores.
        for i in assets:
            d = direction[i]
            if d == 0:
                continue
            hit = "SL" if d * (o[i] - sl[i]) <= 0 else "TP" if d * (o[i] - tp[i]) >= 0 else None
            if hit:
                close_position(i, t, o[i], hit)
                if pending_setup[i] is None:        # la orden pendiente no era una entrada
                    pending_w[i] = pending_reason[i] = None

        # 1b. Órdenes pendientes -> cantidades objetivo.
        #     q* = w·E* / (P·(1 + c)),  E* = equity al open − costo de las salidas.
        #     Dividir entre (1 + c) hace que nocional + costo de entrada = w·E*.
        equity_open = cash + sum(q[i] * o[i] for i in assets)
        ordered = [pending_w[i] is not None for i in assets]
        q_target = list(q)
        if any(ordered):
            exit_cost = 0.0
            for i in assets:
                w = pending_w[i]
                if ordered[i] and direction[i] != 0 and (w == 0 or (w > 0) != (direction[i] > 0)):
                    exit_cost += abs(q[i]) * o[i] * rate(abs(q[i]), ADV[t][i])
            sizing_equity = equity_open - exit_cost
            for i in assets:
                if ordered[i]:
                    q_target[i] = pending_w[i] * sizing_equity / (o[i] * (1 + rate(0.0, ADV[t][i])))

        # 1c. Sin apalancamiento: Σ|q*·P| ≤ equity después de costos (recorta, nunca agrega).
        for _ in range(3):
            gross = sum(abs(q_target[i]) * o[i] for i in assets)
            if gross == 0:
                break
            cost_est = sum(abs(q_target[i] - q[i]) * o[i] * rate(abs(q_target[i] - q[i]), ADV[t][i])
                           for i in assets if q_target[i] != q[i])
            limit = equity_open - cost_est
            if gross <= limit * (1 + 1e-12):
                break
            scale = limit / gross * (1 - 1e-9)
            q_target = [x * scale for x in q_target]
            n_scalings += 1

        for i in assets:
            target = q_target[i]
            if target == q[i]:
                continue
            if direction[i] != 0 and (target == 0 or (target > 0) != (direction[i] > 0)):
                close_position(i, t, o[i], pending_reason[i] or pending_rebal_reason)
            if target != 0 and direction[i] == 0:
                if pending_setup[i] is not None:
                    open_position(i, t, target, o[i], pending_setup[i])
            elif direction[i] != 0 and abs(target - q[i]) > 1e-9:
                fill(i, t, target - q[i], o[i], pending_rebal_reason if ordered[i] else LEVERAGE_TRIM)
        eq_o = cash + sum(q[i] * o[i] for i in assets)
        exec_gross[t] = sum(abs(q[i]) * o[i] for i in assets) / eq_o if eq_o > 0 else np.inf
        pending_w = [None] * n
        pending_reason = [None] * n
        pending_setup = [None] * n

        # 2. SL/TP intradía (SL primero si ambos caen en la barra).
        for i in assets:
            d = direction[i]
            if d == 0:
                continue
            if (d > 0 and l[i] <= sl[i]) or (d < 0 and h[i] >= sl[i]):
                close_position(i, t, sl[i], "SL")
            elif (d > 0 and h[i] >= tp[i]) or (d < 0 and l[i] <= tp[i]):
                close_position(i, t, tp[i], "TP")

        # 3. Cierre: costo de préstamo de cortos, purga y marca a mercado.
        if costs.borrow_fee_annual > 0:
            for i in assets:
                if q[i] < 0:
                    fee = costs.borrow_fee_annual / config.TRADING_DAYS * abs(q[i]) * c[i]
                    cash -= fee
                    total_costs += fee
                    trade_cost[i] += fee
        if force_close_last and t == T - 1:
            for i in assets:
                if direction[i] != 0:
                    close_position(i, t, c[i], "fin de ventana")
        equity = cash + sum(q[i] * c[i] for i in assets)
        eq_arr[t], cash_arr[t] = equity, cash
        pos_arr[t] = q
        w_now = [q[i] * c[i] / equity for i in assets]
        w_arr[t] = w_now
        if t == T - 1:
            break

        # Decisiones para el open de t+1 con información ≤ cierre de t.
        desired = list(direction)
        events = [False] * n
        exit_reason = [None] * n
        for i in assets:
            s = SIG[t][i]
            if direction[i] != 0:
                if s == -direction[i]:
                    desired[i] = s if MASK[t][i] else 0
                    events[i], exit_reason[i] = True, "señal"
                elif t - entry_idx[i] + 1 >= pos_hold[i]:
                    desired[i], events[i], exit_reason[i] = 0, True, "time-stop"
            elif s != 0 and MASK[t][i]:
                desired[i], events[i] = s, True
        if not (sizer.daily or any(events)):
            continue

        strength_now = np.array([pos_strength[i] if direction[i] != 0 else STR[t][i] for i in assets])
        weights, pending_rebal_reason = sizer.target_weights(t, np.array(desired), strength_now,
                                                             np.array(w_now), np.array(events))
        for i in assets:
            w = 0.0 if events[i] and desired[i] == 0 else float(weights[i])
            if w != w:      # NaN: conservar
                continue
            pending_w[i] = w
            pending_reason[i] = exit_reason[i]
            if desired[i] != 0 and w != 0 and desired[i] != direction[i]:
                pending_setup[i] = (ATR[t][i], MSL[t][i], MTP[t][i], MH[t][i], STR[t][i])

    return BacktestResult(dates, list(tickers), eq_arr, cash_arr, pos_arr, w_arr, exec_gross,
                          trades, fills, total_costs, n_scalings)


def single_asset_inputs(signals: pd.DataFrame, params: dict,
                        entry_mask: np.ndarray | None = None) -> StrategyInputs:
    """Entradas del motor para un activo con θ constante (columna única)."""
    T = len(signals)
    col = lambda x: np.asarray(x, dtype=float).reshape(T, 1)
    full = lambda v: np.full((T, 1), v, dtype=float)
    mask = np.ones((T, 1), dtype=bool) if entry_mask is None else np.asarray(entry_mask, bool).reshape(T, 1)
    return StrategyInputs(signal=col(signals["signal"]).astype(int), strength=col(signals["strength"]).astype(int),
                          atr=col(signals["atr"]), m_sl=full(params["m_sl"]), m_tp=full(params["m_tp"]),
                          max_hold=full(params["max_hold"]).astype(int), entry_mask=mask)
