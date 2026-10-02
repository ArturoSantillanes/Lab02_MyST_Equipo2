"""Métricas de desempeño (Rf = 0, 252 días hábiles por año).

R_t = V_t / V_{t−1} − 1 son los rendimientos diarios de la curva de equity V.
"""
import numpy as np
import pandas as pd

from src import config

A = config.TRADING_DAYS


def daily_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def cagr(equity: pd.Series) -> float:
    """CAGR = (V_N / V_0)^(252/N) − 1, con N = número de rendimientos diarios."""
    n = len(equity) - 1
    if n <= 0 or equity.iloc[0] <= 0:
        return 0.0
    growth = equity.iloc[-1] / equity.iloc[0]
    return float(growth ** (A / n) - 1) if growth > 0 else -1.0


def volatility(returns: pd.Series) -> float:
    """σ_anual = std(R)·√252."""
    return float(returns.std(ddof=1) * np.sqrt(A)) if len(returns) > 1 else 0.0


def sharpe(returns: pd.Series) -> float:
    """Sharpe = E[R] / σ(R) · √252   (Rf = 0)."""
    sd = returns.std(ddof=1) if len(returns) > 1 else 0.0
    return float(returns.mean() / sd * np.sqrt(A)) if sd > 0 else 0.0


def sortino(returns: pd.Series) -> float:
    """Sortino = E[R] / σ_d · √252,  σ_d = √(1/N · Σ min(0, R_t)²)."""
    sd = np.sqrt(np.mean(np.minimum(returns.to_numpy(), 0.0) ** 2)) if len(returns) else 0.0
    return float(returns.mean() / sd * np.sqrt(A)) if sd > 0 else 0.0


def drawdown(equity: pd.Series) -> pd.Series:
    """DD_t = (Peak_t − V_t) / Peak_t,  Peak_t = max_{s≤t} V_s."""
    peak = equity.cummax()
    return (peak - equity) / peak


def max_drawdown(equity: pd.Series) -> float:
    """MDD = max_t DD_t (positivo)."""
    return float(drawdown(equity).max()) if len(equity) else 0.0


def max_drawdown_duration(equity: pd.Series) -> int:
    """Días hábiles más largos consecutivos por debajo del máximo previo."""
    under = (drawdown(equity) > 0).to_numpy()
    longest = run = 0
    for u in under:
        run = run + 1 if u else 0
        longest = max(longest, run)
    return longest


def calmar(equity: pd.Series, mdd_floor: float = config.CALMAR_MDD_FLOOR) -> float:
    """Calmar = CAGR / max(|MDD|, piso).

    El piso de 1% evita que una ventana casi sin drawdown produzca un Calmar
    que explote numéricamente y domine la optimización.
    """
    return cagr(equity) / max(abs(max_drawdown(equity)), mdd_floor)


def trade_stats(trades: pd.DataFrame) -> dict:
    """Win rate, payoff, profit factor y conteos sobre operaciones cerradas (PnL neto).

    Win rate = #{PnL > 0}/#ops;  payoff = media(ganancias)/|media(pérdidas)|;
    profit factor = Σ ganancias / |Σ pérdidas|.
    """
    if trades.empty:
        return {"n_operaciones": 0, "n_largas": 0, "n_cortas": 0, "win_rate": np.nan,
                "payoff": np.nan, "profit_factor": np.nan}
    pnl = trades["pnl_neto"]
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    return {
        "n_operaciones": len(trades),
        "n_largas": int((trades["direccion"] == "largo").sum()),
        "n_cortas": int((trades["direccion"] == "corto").sum()),
        "win_rate": float(len(wins) / len(pnl)),
        "payoff": float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) and losses.mean() != 0 else np.nan,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else np.nan,
    }


def returns_table(equity: pd.Series) -> dict[str, pd.Series]:
    """Rendimientos compuestos mensuales, trimestrales y anuales de la curva de equity.

    El primer periodo se mide contra el valor inicial de la curva.
    """
    out = {}
    for name, freq in (("mensual", "ME"), ("trimestral", "QE"), ("anual", "YE")):
        last = equity.resample(freq).last().dropna()
        prev = last.shift(1)
        prev.iloc[0] = equity.iloc[0]
        out[name] = last / prev - 1
    return out


def summary(equity: pd.Series, trades: pd.DataFrame | None = None, total_costs: float = 0.0,
            turnover: float = np.nan, name: str = "") -> dict:
    """Una fila con todas las métricas de la sección 7.

    Recovery factor = retorno total / MDD. Costos vs. retorno bruto =
    costos totales / PnL bruto, con PnL bruto = PnL neto + costos.
    """
    r = daily_returns(equity)
    mdd = max_drawdown(equity)
    net_pnl = float(equity.iloc[-1] - equity.iloc[0])
    gross_pnl = net_pnl + total_costs
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    row = {
        "conjunto": name,
        "inicio": equity.index[0].date(), "fin": equity.index[-1].date(), "dias": len(equity),
        "retorno_total": total_return, "cagr": cagr(equity), "volatilidad": volatility(r),
        "sharpe": sharpe(r), "sortino": sortino(r), "mdd": mdd, "calmar": calmar(equity),
        "recovery_factor": total_return / mdd if mdd > 0 else np.nan,
        "duracion_max_dd": max_drawdown_duration(equity),
    }
    row.update(trade_stats(trades if trades is not None else pd.DataFrame()))
    row.update({
        "turnover_anual": turnover, "costos_totales": total_costs,
        "pnl_bruto": gross_pnl, "pnl_neto": net_pnl,
        "costos_vs_pnl_bruto": total_costs / gross_pnl if gross_pnl > 0 else np.nan,
    })
    return row


def result_summary(result, name: str = "") -> dict:
    """`summary` a partir de un BacktestResult."""
    return summary(result.equity, result.trades, result.total_costs, result.turnover, name)


def metrics_by_regime(equity: pd.Series, labels: pd.Series, names: dict) -> pd.DataFrame:
    """Métricas de los rendimientos diarios agrupados por el régimen vigente al cierre previo.

    El rendimiento del día t se atribuye al régimen de t−1, que es el que decidió
    la posición. MDD por régimen = MDD de la curva compuesta solo con esos días.
    """
    r = daily_returns(equity)
    reg = labels.shift(1).reindex(r.index)
    rows = []
    for code, name in names.items():
        x = r[reg == code]
        if len(x) < 2:
            rows.append({"regimen": name, "n_dias": len(x)})
            continue
        curve = (1 + x).cumprod()
        rows.append({"regimen": name, "n_dias": len(x), "proporcion": len(x) / len(r),
                     "media_anualizada": x.mean() * A, "volatilidad": volatility(x), "sharpe": sharpe(x),
                     "sortino": sortino(x), "mdd": max_drawdown(curve), "dias_positivos": float((x > 0).mean())})
    return pd.DataFrame(rows)
