"""Portafolio multi-activo: Risk Parity, agregación de señales y rebalanceo.

Notación de la clase "Fundamentos Matemáticos de Risk Parity" (pasos 1–8):

Risk Parity (pasos 2–5)
    σ_p = √(wᵀΣw),   ∂σ_p/∂w_k = (Σw)_k/σ_p,   RC_i = w_i·(Σw)_i/σ_p,   Σ_i RC_i = σ_p (Euler)
    Condición: RC_i = σ_p/n para todo i, con Σw_i = 1 y w_i > 0.
    Se resuelve con la formulación convexa de Spinu (2013):
        min_{y>0} ½·yᵀΣy − (1/n)·Σ_i ln y_i,    w = y / Σ_j y_j
    (convexa: forma cuadrática PSD + barrera logarítmica => solución única). Minimizar
    Σ(RC_i − RC_j)² directamente no es convexo y depende del punto de partida.
    Versiones comparadas: pesos iguales (1/n), RP naïve (1/σ_i) y RP optimizado.

Agregación (paso 7)
    s_i = (1/k)·Σ_j x_ij ∈ [−1, 1] si la compuerta 2 de 3 se cumple, 0 si no;
    w̃_i = w_i^RP · s_i,   w^target = m(régimen) · w̃ / max(1, Σ_i |w̃_i|).

Rebalanceo (paso 8)
    T_t = ½·Σ_i |w_{i,t} − w_{i,t⁻}|  (contra los pesos después del drift),
    costo anual ≈ T̄ · f · 2c.  Disparador híbrido: en fechas de calendario (cada f días)
    se rebalancea solo si ‖w_t⁻ − w^target‖₁ > δ.

Σ: muestral de rendimientos diarios en ventana de 126 días con datos ≤ t (oficial);
se comparan EWMA y Ledoit-Wolf (paso 6).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src import config
from src.backtest import StrategyInputs
from src.regimes import CRISIS
from src.signals import INDICATORS, IndicatorCache, generate_signals

REGIME_BY_CODE = config.REGIME_NAMES
WEIGHTING_METHODS = {"rp": "Risk Parity (Spinu)", "iv": "RP naïve (1/σ)", "ew": "Pesos iguales"}


class RiskParityError(RuntimeError):
    """Las contribuciones al riesgo no quedaron iguales dentro de la tolerancia."""


def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Contribución fraccional al riesgo: RC_i/σ_p = w_i·(Σw)_i / wᵀΣw  (suma 1, Euler)."""
    w = np.asarray(w, dtype=float)
    marginal = cov @ w
    return w * marginal / (w @ marginal)


def risk_parity_weights(cov: np.ndarray, budgets: np.ndarray | None = None,
                        tol: float = config.RP_TOLERANCE) -> np.ndarray:
    """Risk Parity con la formulación convexa de Spinu, resuelta con SLSQP.

    min_{y>0} ½·yᵀΣy − Σ_i b_i·ln y_i  (b_i = 1/n en RP estándar),   w = y/Σy.
    En el óptimo y_i·(Σy)_i = b_i, por lo que RC_i/σ_p = b_i al normalizar.
    Verificación obligatoria: max_i |RC_i/σ_p − b_i| < tol; si falla, RiskParityError.
    Σ se normaliza por su traza (los pesos no cambian al escalar Σ).
    """
    cov = np.asarray(cov, dtype=float)
    n = len(cov)
    b = np.full(n, 1.0 / n) if budgets is None else np.asarray(budgets, float) / np.sum(budgets)
    if n == 1:
        return np.ones(1)
    s = cov / np.trace(cov) * n
    y0 = b / np.sqrt(np.diag(s))
    res = minimize(lambda y: 0.5 * y @ s @ y - b @ np.log(y), y0, jac=lambda y: s @ y - b / y,
                   method="SLSQP", bounds=[(1e-12, None)] * n, options={"ftol": 1e-16, "maxiter": 1000})
    w = res.x / res.x.sum()
    err = np.max(np.abs(risk_contributions(w, s) - b))
    if err >= tol:
        raise RiskParityError(f"max|RC_i/σ_p − b_i| = {err:.2e} ≥ {tol:.0e}")
    return w


def inverse_vol_weights(cov: np.ndarray) -> np.ndarray:
    """RP naïve: w_i = (1/σ_i) / Σ_j (1/σ_j). Exacta solo si todas las ρ_ij son iguales."""
    inv = 1.0 / np.sqrt(np.diag(cov))
    return inv / inv.sum()


def base_weights(cov: np.ndarray, method: str) -> np.ndarray:
    """w^base de los n activos: 'rp' (Spinu), 'iv' (1/σ) o 'ew' (1/n)."""
    if method == "rp":
        return risk_parity_weights(cov)
    if method == "iv":
        return inverse_vol_weights(cov)
    return np.full(len(cov), 1.0 / len(cov))


def estimate_covariance(window: np.ndarray, estimator: str = config.COV_ESTIMATOR) -> np.ndarray:
    """Σ a partir de rendimientos (nunca precios) de la ventana ≤ t.

    muestral:    S = cov(r) sobre la ventana (126 días).
    ewma:        Σ_t = λ·Σ_{t−1} + (1−λ)·r_t r_tᵀ, λ = 0.94 (T_eff = 1/(1−λ) ≈ 17 días).
    ledoit_wolf: Σ = δ*·F + (1−δ*)·S con δ* analítico (encoge los eigenvalores extremos).
    """
    if estimator == "ewma":
        centered = window - window.mean(axis=0)
        lam = config.EWMA_LAMBDA
        weights = (1 - lam) * lam ** np.arange(len(window) - 1, -1, -1)
        return (centered * weights[:, None]).T @ centered / weights.sum()
    if estimator == "ledoit_wolf":
        from sklearn.covariance import LedoitWolf
        return LedoitWolf().fit(window).covariance_
    return np.cov(window, rowvar=False)


def signal_strength(direction: np.ndarray, net_votes: np.ndarray, k: int = len(INDICATORS)) -> np.ndarray:
    """s_i = (1/k)·Σ_j x_ij con el signo de la dirección confirmada; 0 si no hay posición.

    Con la compuerta 2 de 3: (1,1,1) → 1, (1,1,0) → 2/3, (1,1,−1) → 1/3.
    """
    return np.sign(direction) * np.abs(net_votes) / k


def compose_target(w_base: np.ndarray, s: np.ndarray, m: float) -> np.ndarray:
    """w̃ = w^base · s;   w^target = m · w̃ / max(1, Σ|w̃|)  (nunca apalancado)."""
    tilde = w_base * s
    return m * tilde / max(1.0, np.abs(tilde).sum())


def turnover(w_new: np.ndarray, w_before: np.ndarray) -> float:
    """T_t = ½·Σ_i |w_{i,t} − w_{i,t⁻}| contra los pesos después del drift."""
    return 0.5 * float(np.abs(np.asarray(w_new) - np.asarray(w_before)).sum())


def condition_number(cov: np.ndarray) -> float:
    """κ(Σ) = λ_max / λ_min."""
    eig = np.linalg.eigvalsh(cov)
    return float(eig[-1] / eig[0]) if eig[0] > 0 else np.inf


@dataclass
class PortfolioSizer:
    """Agregación de señales -> pesos objetivo del portafolio (Sizer del motor).

    Al cierre de t, con datos ≤ t:
    1. w^base de los 6 activos con Σ de 126 días ('rp', 'iv' o 'ew').
    2. s_i = dirección · |Σ votos|/3 (fuerza continua de la clase).
    3. Conflictos: si corr_126(i, j) > 0.7 y direcciones opuestas, gana la de mayor
       |s| y la otra queda en 0; si empatan, ambas reducen s a la mitad.
    4. w^target = m(régimen) · w̃ / max(1, Σ|w̃|).
    Rebalanceo híbrido: en fechas de calendario (cada `rebalance_every` días) solo si
    ‖w − w^target‖₁ > `band`; inmediato al entrar a Crisis. Las entradas y salidas por
    señal, SL, TP o time-stop mueven solo a ese activo al open siguiente.
    """
    returns: np.ndarray                       # T × n, rendimientos diarios (fila t = cierre t)
    regimes: np.ndarray                       # T, código de régimen vigente al cierre de t
    method: str = "rp"
    rebalance_every: int = config.REBALANCE_EVERY
    band: float = config.REBALANCE_BAND
    estimator: str = config.COV_ESTIMATOR
    multipliers: dict = field(default_factory=lambda: dict(config.REGIME_MULTIPLIER))
    cov_window: int = config.COV_WINDOW
    conflict_corr: float = config.CONFLICT_CORR
    daily: bool = True
    log: list = field(default_factory=list)  # (t, tipo, κ(Σ), n_activos, T_t)
    n_conflicts: int = 0
    n_conflict_ties: int = 0
    n_band_skips: int = 0

    def _window(self, t: int) -> np.ndarray:
        r = self.returns[max(0, t - self.cov_window + 1): t + 1]
        return r[~np.isnan(r).any(axis=1)]

    def _resolve_conflicts(self, s: np.ndarray, corr: np.ndarray) -> np.ndarray:
        s = s.copy()
        n = len(s)
        pairs = sorted(((corr[i, j], i, j) for i in range(n) for j in range(i + 1, n)), reverse=True)
        for c, i, j in pairs:
            if c <= self.conflict_corr or s[i] * s[j] >= 0:
                continue
            self.n_conflicts += 1
            if abs(s[i]) > abs(s[j]):
                s[j] = 0.0
            elif abs(s[j]) > abs(s[i]):
                s[i] = 0.0
            else:
                self.n_conflict_ties += 1
                s[i] /= 2
                s[j] /= 2
        return s

    def full_targets(self, t, desired, strength) -> tuple[np.ndarray, np.ndarray, float]:
        """w^target de todos los activos y la fuerza s después de conflictos."""
        window = self._window(t)
        n = len(desired)
        if len(window) > n:
            cov = estimate_covariance(window, self.estimator)
            corr = np.corrcoef(window, rowvar=False)
        else:
            cov, corr = np.eye(n), np.eye(n)
        s = self._resolve_conflicts(signal_strength(desired, strength), corr)
        m = self.multipliers[REGIME_BY_CODE[int(self.regimes[t])]]
        return compose_target(base_weights(cov, self.method), s, m), s, condition_number(cov)

    def target_weights(self, t, desired, strength, weights, events):
        regime_now = int(self.regimes[t])
        crisis_entry = regime_now == CRISIS and t > 0 and int(self.regimes[t - 1]) != CRISIS
        calendar = t % self.rebalance_every == 0
        if not (calendar or crisis_entry or events.any()):
            return np.full(len(desired), np.nan), "rebalanceo"

        targets, s, kappa = self.full_targets(t, desired, strength)
        if crisis_entry or (calendar and np.abs(weights - targets).sum() > self.band):
            self.log.append((t, "crisis" if crisis_entry else "calendario + banda", kappa,
                             int((s != 0).sum()), turnover(targets, weights)))
            return targets, "cambio de régimen" if crisis_entry else "rebalanceo"
        if calendar:
            self.n_band_skips += 1
        # Solo eventos: se mueven los activos que entran/salen o pierden un conflicto.
        out = np.full(len(desired), np.nan)
        changed = events | ((desired != 0) & (s == 0))
        out[changed] = targets[changed]
        return out, "rebalanceo"


class BuyHoldSizer:
    """Benchmark Buy & Hold equiponderado: 1/n por activo al inicio y no se toca."""
    daily = False

    def target_weights(self, t, desired, strength, weights, events):
        n = len(desired)
        return np.where(events, desired / n, np.nan), "rebalanceo"


def buy_hold_inputs(T: int, n: int) -> StrategyInputs:
    """Señal larga solo el primer día, sin SL/TP efectivos ni time-stop (ATR enorme)."""
    signal = np.zeros((T, n), dtype=int)
    signal[0] = 1
    big = np.full((T, n), 1e12)
    return StrategyInputs(signal, np.full((T, n), 3), big, np.ones((T, n)), np.ones((T, n)),
                          np.full((T, n), 10 ** 9, dtype=int), np.ones((T, n), dtype=bool))


@dataclass
class ComposedSignals:
    """Entradas del motor y la señal confirmada antes del filtro de régimen (para análisis).

    `inputs.strength` guarda |Σ votos| (1, 2 o 3), que define s_i = ±|Σ votos|/3.
    """
    inputs: StrategyInputs
    raw_signal: np.ndarray      # T × n, S_t con la regla k de m, sin la regla del régimen


def compose_strategy_inputs(panel: pd.DataFrame, tickers: list[str], dates: pd.DatetimeIndex,
                            theta_for_day, regimes: pd.Series,
                            min_agree_by_regime: dict = config.REGIME_MIN_AGREE,
                            indicators: tuple[str, ...] = INDICATORS, min_agree: int = config.MIN_AGREE,
                            caches: dict | None = None) -> ComposedSignals:
    """Señales efectivas T × n cuando θ cambia por mes y por régimen.

    `theta_for_day(date, ticker, regime_code) -> params | None`. La señal de cada día
    sale del θ vigente ese día (régimen al cierre de t). Encima se aplica la regla de
    entrada del régimen: en Crisis solo cuentan señales con fuerza 3 (3 de 3).
    θ = None (ningún estudio válido en esa ventana) => el activo no abre ese día.
    `indicators`/`min_agree` solo cambian en el experimento de un solo indicador; ahí
    la exigencia del régimen se limita al número de indicadores usados.
    """
    caches = caches if caches is not None else {}
    T, n = len(dates), len(tickers)
    pos = panel.index.get_indexer(dates)
    reg = regimes.reindex(dates).to_numpy().astype(int)
    need = np.array([min(min_agree_by_regime[REGIME_BY_CODE[r]], len(indicators)) for r in reg])
    out = {k: np.zeros((T, n)) for k in ("signal", "raw", "strength", "atr", "m_sl", "m_tp", "max_hold")}
    for j, tk in enumerate(tickers):
        cache_ = caches.setdefault(tk, IndicatorCache(panel.xs(tk, axis=1, level=1)))
        out["atr"][:, j] = cache_.atr[pos]
        computed = {}
        for t, d in enumerate(dates):
            params = theta_for_day(d, tk, int(reg[t]))
            if params is None:
                continue
            key = tuple(sorted(params.items()))
            if key not in computed:
                sig = generate_signals(cache_.ohlcv, params, min_agree, indicators, cache_)
                computed[key] = (sig["signal"].to_numpy(), sig["strength"].to_numpy(), sig["net_votes"].to_numpy())
            signal, agree, net = computed[key]
            s = signal[pos[t]]
            out["raw"][t, j], out["strength"][t, j] = s, abs(net[pos[t]])
            out["signal"][t, j] = s if agree[pos[t]] >= need[t] else 0
            out["m_sl"][t, j], out["m_tp"][t, j], out["max_hold"][t, j] = params["m_sl"], params["m_tp"], params["max_hold"]
    inputs = StrategyInputs(out["signal"].astype(int), out["strength"].astype(int), out["atr"], out["m_sl"],
                            out["m_tp"], out["max_hold"].astype(int), np.ones((T, n), dtype=bool))
    return ComposedSignals(inputs, out["raw"].astype(int))


def realized_risk_contributions(weights: pd.DataFrame, returns: pd.DataFrame,
                                window: int = config.COV_WINDOW) -> pd.DataFrame:
    """RC fraccional ex-ante por activo en cada día con posiciones (pesos con signo).

    `returns` es la historia completa: la covarianza de cada día usa los `window`
    rendimientos previos (≤ t), aunque caigan antes del tramo simulado.
    """
    r = returns[weights.columns].fillna(0.0)
    pos = r.index.get_indexer(weights.index)
    values, w_all, rows = r.to_numpy(), weights.to_numpy(), {}
    for k, t in enumerate(pos):
        w = w_all[k]
        if np.abs(w).sum() < 1e-9 or t + 1 < window:
            continue
        cov = np.cov(values[t - window + 1: t + 1], rowvar=False)
        var = w @ cov @ w
        if var > 0:
            rows[weights.index[k]] = w * (cov @ w) / var
    return pd.DataFrame.from_dict(rows, orient="index", columns=weights.columns)


def risk_contribution_summary(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Contribución al riesgo promedio por activo (|RC| fraccional) y concentración.

    Concentración = promedio diario de max_i |RC_i| y de Σ RC_i² (Herfindahl).
    """
    rc = realized_risk_contributions(weights, returns)
    out = rc.abs().mean().to_frame("rc_promedio_abs")
    out["peso_promedio_abs"] = weights.loc[rc.index].abs().mean()
    out.loc["concentracion_max_rc"] = [rc.abs().max(axis=1).mean(), np.nan]
    out.loc["herfindahl_rc"] = [(rc ** 2).sum(axis=1).mean(), np.nan]
    return out


def correlation_by_regime(returns: pd.DataFrame, labels: pd.Series) -> dict[str, pd.DataFrame]:
    """Matriz de correlación de rendimientos diarios usando solo los días de cada régimen."""
    aligned = returns.loc[labels.index]
    return {name: aligned[labels == code].corr() for code, name in config.REGIME_NAMES.items()
            if (labels == code).sum() > 2}


def rolling_mean_correlation(returns: pd.DataFrame, window: int = config.COV_WINDOW) -> pd.Series:
    """Correlación promedio entre pares (ventana móvil de 126 días, solo datos ≤ t)."""
    n = returns.shape[1]
    corr = returns.rolling(window).corr()
    mean = corr.groupby(level=0).apply(lambda m: (m.to_numpy().sum() - n) / (n * (n - 1)))
    return mean.rename("correlacion_promedio")


def transition_impact(labels: pd.Series, weights: pd.DataFrame) -> pd.DataFrame:
    """Cada cambio de régimen: de/a, posiciones abiertas en ese cierre y regla aplicada."""
    rows = []
    prev = labels.shift()
    for d in labels.index[(labels != prev) & prev.notna()]:
        frm, to = config.REGIME_NAMES[int(prev.loc[d])], config.REGIME_NAMES[int(labels.loc[d])]
        open_pos = int((weights.loc[d].abs() > 1e-9).sum())
        rule = ("rebalanceo inmediato al open siguiente con M = 0.3; entradas nuevas solo 3 de 3"
                if to == "Crisis" else "posiciones conservan SL/TP; θ nuevo para entradas; M nuevo en el siguiente rebalanceo")
        rows.append({"fecha": d, "de": frm, "a": to, "posiciones_abiertas": open_pos, "regla": rule})
    return pd.DataFrame(rows)
