"""Optimización bayesiana (Optuna, TPE) maximizando Calmar y walk-forward en TRAIN.

Por ventana k (train 6 meses -> test 1 mes, paso mensual):
    1. Régimen: scaler + K-means ajustados con features ≤ fin del train_k.
    2. θ por activo i y régimen j:
           θ*_{i,j} = argmax_θ Calmar(backtest(train_k | S_t = j, activo i, θ))
       "| S_t = j" = solo se permiten entradas en días del régimen j (entry_mask).
       Además un estudio global (sin máscara) que sirve de respaldo (fallback).
    3. Purga y embargo: las posiciones se cierran el último día del train y los
       últimos 5 días se excluyen del objetivo.
    4. θ robusto = mediana de cada parámetro en el 10% de mejores trials válidos.
La simulación OOS es UNA corrida continua donde el mes k+1 usa lo optimizado en k.
"""
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed

from src import config
from src import regimes as R
from src.backtest import MarketData, StrategyInputs, run_backtest
from src.metrics import calmar
from src.signals import IndicatorCache, confirm, indicator_votes

log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

INT_PARAMS = ("ema_fast", "ema_gap", "rsi_window", "rsi_lower", "rsi_upper", "bb_window", "max_hold")
GLOBAL = "global"
REGIME_KEYS = {R.TREND: "Tendencia", R.REVERSION: "Reversión", R.CRISIS: "Crisis"}


def suggest_params(trial: optuna.Trial) -> dict:
    """Espacio de búsqueda θ (10 parámetros). El ATR usa ventana fija de 14."""
    return {
        # EMA rápida de 1 a 6 semanas: horizonte de las operaciones (5–40 días).
        "ema_fast": trial.suggest_int("ema_fast", 5, 30),
        # La lenta = rápida + gap garantiza lenta > rápida; gap hasta ~5 meses.
        "ema_gap": trial.suggest_int("ema_gap", 10, 100),
        # RSI de 1.5 a 6 semanas alrededor del estándar de Wilder (14).
        "rsi_window": trial.suggest_int("rsi_window", 7, 28),
        # Umbrales alrededor del 30/70 clásico, sin cruzarse.
        "rsi_lower": trial.suggest_int("rsi_lower", 20, 40),
        "rsi_upper": trial.suggest_int("rsi_upper", 60, 80),
        # Bollinger de 2 a 8 semanas alrededor del estándar de 20 días.
        "bb_window": trial.suggest_int("bb_window", 10, 40),
        # k de 1.5 a 3σ: de bandas sensibles a solo extremos.
        "bb_k": trial.suggest_float("bb_k", 1.5, 3.0),
        # SL de 1 a 4 ATR: fuera del ruido diario sin arriesgar demasiado.
        "m_sl": trial.suggest_float("m_sl", 1.0, 4.0),
        # TP de 1 a 6 ATR: permite payoffs mayores que 1 frente al SL.
        "m_tp": trial.suggest_float("m_tp", 1.0, 6.0),
        # Time-stop de 1 a 8 semanas.
        "max_hold": trial.suggest_int("max_hold", 5, 40),
    }


@dataclass
class Window:
    k: int
    train_start: str
    train_end: str
    test_start: str | None     # None en la ventana final (su θ se congela para TEST)
    test_end: str | None


def build_windows(index: pd.DatetimeIndex, first_test_month: str, last_train_day: str,
                  anchored: bool = False, anchor_start: str | None = None) -> list[Window]:
    """Ventanas mensuales dentro de TRAIN.

    Rolling: train = 6 meses previos al mes de prueba. Anchored: train desde
    `anchor_start` (crece cada mes). La última ventana termina en el último día de
    TRAIN y no tiene mes de prueba: su θ es el que se congela para el TEST.
    """
    last = pd.Timestamp(last_train_day)
    months = pd.date_range(first_test_month, last, freq="MS")
    windows = []
    for k, m in enumerate(months):
        start = pd.Timestamp(anchor_start) if anchored else m - pd.DateOffset(months=config.WF_TRAIN_MONTHS)
        train = index[(index >= start) & (index < m)]
        test = index[(index >= m) & (index < m + pd.DateOffset(months=config.WF_TEST_MONTHS)) & (index <= last)]
        windows.append(Window(k, str(train[0].date()), str(train[-1].date()),
                              str(test[0].date()), str(test[-1].date())))
    final_start = pd.Timestamp(anchor_start) if anchored else last - pd.DateOffset(months=config.WF_TRAIN_MONTHS) + pd.Timedelta(days=1)
    train = index[(index >= final_start) & (index <= last)]
    windows.append(Window(len(months), str(train[0].date()), str(train[-1].date()), None, None))
    return windows


def study_seed(k: int, asset_idx: int, slot: int) -> int:
    """Semilla determinista por (ventana, activo, régimen) derivada de SEED."""
    return (config.SEED * 1_000_003 + k * 7_919 + asset_idx * 104_729 + slot * 15_485_863) % (2 ** 31 - 1)


class AssetWindow:
    """Datos de un activo recortados a un train, con indicadores precalculados (causales)."""

    def __init__(self, ohlcv: pd.DataFrame, cache_: IndicatorCache, a: int, b: int):
        self.cache, self.a, self.b = cache_, a, b
        self.market = MarketData(ohlcv.index[a:b], ["X"], *(ohlcv[f].to_numpy(float)[a:b, None]
                                                           for f in ("open", "high", "low", "close")),
                                 np.zeros((b - a, 1)))

    def backtest(self, params: dict, entry_mask: np.ndarray | None = None, min_agree: int = config.MIN_AGREE):
        """Backtest de train con purga (cierre forzoso al final). Regresa el BacktestResult."""
        votes = indicator_votes(self.cache, params)
        sl = slice(self.a, self.b)
        signal, strength = confirm(np.column_stack([votes["ema"][sl], votes["rsi"][sl], votes["bb"][sl]]), min_agree)
        L = self.b - self.a
        mask = np.ones((L, 1), bool) if entry_mask is None else entry_mask.reshape(L, 1)
        inputs = StrategyInputs(signal[:, None].astype(int), strength[:, None].astype(int),
                                self.cache.atr[sl, None], np.full((L, 1), params["m_sl"]),
                                np.full((L, 1), params["m_tp"]), np.full((L, 1), params["max_hold"], dtype=int), mask)
        return run_backtest(self.market, inputs, force_close_last=True)


def embargoed_calmar(equity: np.ndarray, index: pd.DatetimeIndex) -> float:
    """Calmar de la curva sin los últimos EMBARGO_DAYS días."""
    cut = len(equity) - config.EMBARGO_DAYS
    return calmar(pd.Series(equity[:cut], index=index[:cut]))


def robust_params(study: optuna.Study) -> dict | None:
    """Mediana de cada parámetro en el 10% de mejores trials válidos (centro de la meseta)."""
    valid = [t for t in study.trials if t.value is not None and t.value > config.INVALID_OBJECTIVE / 2]
    if not valid:
        return None
    valid.sort(key=lambda t: t.value, reverse=True)
    top = valid[: max(1, math.ceil(config.ROBUST_TOP_FRACTION * len(valid)))]
    out = {}
    for name in top[0].params:
        med = float(np.median([t.params[name] for t in top]))
        out[name] = int(round(med)) if name in INT_PARAMS else med
    return out


def run_study(objective, seed: int, n_trials: int) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=min(config.N_STARTUP_TRIALS, n_trials))
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return study


def _study_summary(study: optuna.Study, evaluate) -> dict:
    """Argmax literal, θ robusto y su Calmar en train; None si ningún trial fue válido."""
    robust = robust_params(study)
    if robust is None:
        return {"valido": False}
    return {"valido": True, "argmax": study.best_params, "calmar_argmax": study.best_value,
            "robusto": robust, "calmar_robusto": evaluate(robust)}


def optimize_asset_window(ohlcv: pd.DataFrame, asset_idx: int, window: Window, labels: np.ndarray,
                          a: int, b: int, n_trials: int, keep_trials: bool = False) -> dict:
    """Estudio global + 3 por régimen para un activo en un train. Corre en un proceso hijo."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    aw = AssetWindow(ohlcv, IndicatorCache(ohlcv), a, b)
    index = aw.market.dates
    out = {"k": window.k, "activo_idx": asset_idx, "estudios": {}, "fallbacks": [], "n_evaluadas": 0}

    def make_objective(mask, n_min, min_agree):
        def objective(trial):
            res = aw.backtest(suggest_params(trial), mask, min_agree)
            if res.n_trades < n_min:
                return config.INVALID_OBJECTIVE
            return embargoed_calmar(res.equity_array, index)
        return objective

    def evaluate(mask, min_agree):
        return lambda p: embargoed_calmar(aw.backtest(p, mask, min_agree).equity_array, index)

    study = run_study(make_objective(None, config.N_MIN_GLOBAL, config.MIN_AGREE),
                      study_seed(window.k, asset_idx, 3), n_trials)
    out["n_evaluadas"] += len(study.trials)
    out["estudios"][GLOBAL] = _study_summary(study, evaluate(None, config.MIN_AGREE))
    if keep_trials:
        out["trials_global"] = study.trials_dataframe(attrs=("number", "value", "params")).to_dict("list")

    for code, name in REGIME_KEYS.items():
        mask = labels == code
        min_agree = config.REGIME_MIN_AGREE[name]
        if mask.sum() < config.MIN_REGIME_DAYS:
            out["estudios"][name] = {"valido": False, "motivo": f"{int(mask.sum())} días < {config.MIN_REGIME_DAYS}"}
            out["fallbacks"].append(name)
            continue
        study = run_study(make_objective(mask, config.N_MIN_REGIME, min_agree),
                          study_seed(window.k, asset_idx, code), n_trials)
        out["n_evaluadas"] += len(study.trials)
        summary = _study_summary(study, evaluate(mask, min_agree))
        if not summary["valido"]:
            summary["motivo"] = f"ningún trial con ≥ {config.N_MIN_REGIME} operaciones"
            out["fallbacks"].append(name)
        out["estudios"][name] = summary
    return out


def optimize_shared_window(ohlcvs: list[pd.DataFrame], window: Window, labels: np.ndarray,
                           a: int, b: int, n_trials: int) -> dict:
    """Variante compartida: un θ por régimen para los 6 activos.

    Objetivo = Calmar de la curva promedio de los 6 backtests individuales;
    N_MIN se aplica a la suma de operaciones (6 × N_MIN).
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    aws = [AssetWindow(o, IndicatorCache(o), a, b) for o in ohlcvs]
    index = aws[0].market.dates
    n = len(aws)
    out = {"k": window.k, "estudios": {}, "fallbacks": [], "n_evaluadas": 0}

    def run_all(p, mask, min_agree):
        results = [aw.backtest(p, mask, min_agree) for aw in aws]
        curve = np.mean([r.equity_array for r in results], axis=0)
        return curve, sum(r.n_trades for r in results)

    def make_objective(mask, n_min, min_agree):
        def objective(trial):
            curve, trades = run_all(suggest_params(trial), mask, min_agree)
            return config.INVALID_OBJECTIVE if trades < n * n_min else embargoed_calmar(curve, index)
        return objective

    def evaluate(mask, min_agree):
        return lambda p: embargoed_calmar(run_all(p, mask, min_agree)[0], index)

    study = run_study(make_objective(None, config.N_MIN_GLOBAL, config.MIN_AGREE), study_seed(window.k, 99, 3), n_trials)
    out["n_evaluadas"] += len(study.trials)
    out["estudios"][GLOBAL] = _study_summary(study, evaluate(None, config.MIN_AGREE))
    for code, name in REGIME_KEYS.items():
        mask = labels == code
        min_agree = config.REGIME_MIN_AGREE[name]
        if mask.sum() < config.MIN_REGIME_DAYS:
            out["estudios"][name] = {"valido": False, "motivo": f"{int(mask.sum())} días < {config.MIN_REGIME_DAYS}"}
            out["fallbacks"].append(name)
            continue
        study = run_study(make_objective(mask, config.N_MIN_REGIME, min_agree), study_seed(window.k, 99, code), n_trials)
        out["n_evaluadas"] += len(study.trials)
        summary = _study_summary(study, evaluate(mask, min_agree))
        if not summary["valido"]:
            summary["motivo"] = f"ningún trial con ≥ {n * config.N_MIN_REGIME} operaciones"
            out["fallbacks"].append(name)
        out["estudios"][name] = summary
    return out


def theta_by_regime(window_result: dict) -> dict:
    """θ robusto por régimen con fallback al global. Si el global tampoco es válido, None."""
    g = window_result["estudios"][GLOBAL]
    base = g["robusto"] if g.get("valido") else None
    return {name: (window_result["estudios"][name]["robusto"] if window_result["estudios"][name].get("valido") else base)
            for name in REGIME_KEYS.values()}


def fit_window_regime(feats: pd.DataFrame, window: Window, feature_names: list[str],
                      full_index: pd.DatetimeIndex) -> tuple[R.RegimeModel, pd.Series]:
    """Modelo de régimen de la ventana (fit con features ≤ fin del train) y etiquetas del train."""
    fit_end = pd.Timestamp(window.train_end)
    model = R.fit_regime_model(feats, fit_end, feature_names)
    train_feats = feats.loc[window.train_start: window.train_end]
    labels = R.classify(train_feats, [(train_feats.index[0], model)], full_index)
    return model, labels


def _checkpoint_path(cache_dir, variant: str, k: int):
    return cache_dir / variant / f"ventana_{k:03d}.json"


def walk_forward(panel: pd.DataFrame, windows: list[Window], feats: pd.DataFrame, feature_names: list[str],
                 variant: str, n_trials: int, cache_dir, n_jobs: int | None = None,
                 keep_trials_for: tuple[int, int] | None = None) -> dict:
    """Optimiza todas las ventanas (por activo o compartida) con checkpoints por ventana.

    variant ∈ {"por_activo", "compartido", "anclado_por_activo", "anclado_compartido"}.
    Regresa {"ventanas": [...], "modelos": {k: RegimeModel}, "segundos": t, "n_evaluadas": N}.
    """
    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 1)
    shared = variant.endswith("compartido")
    ohlcvs = [panel.xs(t, axis=1, level=1)[["open", "high", "low", "close", "volume"]] for t in config.TICKERS]
    models, labels, bounds = {}, {}, {}
    for w in windows:
        models[w.k], lab = fit_window_regime(feats, w, feature_names, panel.index)
        a = panel.index.get_loc(pd.Timestamp(w.train_start))
        b = panel.index.get_loc(pd.Timestamp(w.train_end)) + 1
        labels[w.k] = lab.reindex(panel.index[a:b]).to_numpy()
        bounds[w.k] = (a, b)

    t0 = time.perf_counter()
    pending = [w for w in windows if not _checkpoint_path(cache_dir, variant, w.k).exists()]
    if shared:
        tasks = [delayed(optimize_shared_window)(ohlcvs, w, labels[w.k], *bounds[w.k], n_trials) for w in pending]
    else:
        tasks = [delayed(optimize_asset_window)(ohlcvs[i], i, w, labels[w.k], *bounds[w.k], n_trials,
                                                keep_trials_for == (w.k, i))
                 for w in pending for i in range(len(ohlcvs))]
    log.info("Walk-forward %s: %d ventanas pendientes, %d tareas, %d procesos", variant, len(pending), len(tasks), n_jobs)
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(tasks) if tasks else []
    elapsed_new = time.perf_counter() - t0

    by_window = {}
    for r in results:
        by_window.setdefault(r["k"], []).append(r)
    for w in pending:
        rs = by_window[w.k]
        record = {"ventana": asdict(w), "segundos_lote": elapsed_new / max(len(pending), 1)}
        if shared:
            record["compartido"] = rs[0]
        else:
            record["activos"] = {config.TICKERS[r["activo_idx"]]: r for r in sorted(rs, key=lambda r: r["activo_idx"])}
        path = _checkpoint_path(cache_dir, variant, w.k)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, default=str), encoding="utf-8")

    records = [json.loads(_checkpoint_path(cache_dir, variant, w.k).read_text(encoding="utf-8")) for w in windows]
    n_eval = sum(rec["compartido"]["n_evaluadas"] if shared else sum(a["n_evaluadas"] for a in rec["activos"].values())
                 for rec in records)
    seconds = sum(rec["segundos_lote"] for rec in records)
    return {"variante": variant, "ventanas": records, "modelos": models, "segundos": seconds, "n_evaluadas": n_eval}


def theta_table(wf: dict) -> dict[int, dict[str, dict]]:
    """{k: {ticker: {régimen: θ}}} con fallback aplicado; la compartida repite θ para todos."""
    out = {}
    for rec in wf["ventanas"]:
        k = rec["ventana"]["k"]
        if "compartido" in rec:
            th = theta_by_regime(rec["compartido"])
            out[k] = {t: th for t in config.TICKERS}
        else:
            out[k] = {t: theta_by_regime(rec["activos"][t]) for t in config.TICKERS}
    return out


def fallback_counts(wf: dict) -> dict:
    """Cuántas veces cada régimen usó el θ global (por días insuficientes o sin trials válidos)."""
    counts = {name: 0 for name in REGIME_KEYS.values()}
    total = 0
    for rec in wf["ventanas"]:
        units = [rec["compartido"]] if "compartido" in rec else list(rec["activos"].values())
        for u in units:
            total += 1
            for name in u["fallbacks"]:
                counts[name] += 1
    return {"por_regimen": counts, "estudios_por_regimen": total}


# --------------------------------------------------------------------------
# Sistema completo (régimen + θ por mes/régimen + portafolio) y su evaluación
# --------------------------------------------------------------------------

@dataclass
class SystemSpec:
    """Todo lo que define una corrida del sistema de portafolio.

    `theta_for_day(date, ticker, regime_code) -> θ | None`;
    `regime_schedule`: [(fecha desde la que rige, modelo de régimen)].
    """
    theta_for_day: object
    regime_schedule: list
    method: str = "rp"
    multipliers: dict | None = None
    rebalance_every: int = config.REBALANCE_EVERY
    band: float = config.REBALANCE_BAND
    estimator: str = config.COV_ESTIMATOR
    costs: object = None
    indicators: tuple = ("ema", "rsi", "bb")
    min_agree: int = config.MIN_AGREE


@dataclass
class SystemRun:
    result: object              # BacktestResult del portafolio
    regimes: pd.Series          # etiqueta diaria usada
    sizer: object               # PortfolioSizer (logs de rebalanceo y conflictos)
    composed: object            # ComposedSignals
    market: MarketData


def run_system(panel: pd.DataFrame, feats: pd.DataFrame, dates: pd.DatetimeIndex, spec: SystemSpec,
               composed=None, caches: dict | None = None) -> SystemRun:
    """Simula el portafolio sobre `dates` con régimen causal y θ según el calendario.

    `composed` permite reutilizar las señales cuando solo cambian costos, método o
    frecuencia de rebalanceo (las señales no dependen de eso).
    """
    from src.backtest import CostModel
    from src.portfolio import PortfolioSizer, compose_strategy_inputs

    schedule = [(dates[0] if start is None else start, model) for start, model in spec.regime_schedule]
    labels = R.classify(feats, schedule, panel.index).reindex(dates)
    if composed is None:
        composed = compose_strategy_inputs(panel, config.TICKERS, dates, spec.theta_for_day, labels,
                                           indicators=spec.indicators, min_agree=spec.min_agree, caches=caches)
    a = panel.index.get_loc(dates[0])
    market = MarketData.from_panel(panel, config.TICKERS).slice(a, a + len(dates))
    returns = panel["close"][config.TICKERS].pct_change().loc[dates].to_numpy()
    sizer = PortfolioSizer(returns, labels.to_numpy(), method=spec.method, rebalance_every=spec.rebalance_every,
                           band=spec.band, estimator=spec.estimator,
                           multipliers=dict(spec.multipliers or config.REGIME_MULTIPLIER))
    result = run_backtest(market, composed.inputs, sizer, spec.costs or CostModel())
    return SystemRun(result, labels, sizer, composed, market)


def run_individual_assets(run: SystemRun) -> dict:
    """Estrategia de cada activo sola con 100% del capital (mismas señales del sistema)."""
    from src.backtest import FullCapitalSizer
    out = {}
    for j, tk in enumerate(config.TICKERS):
        def col(x):
            return x[:, j:j + 1]
        inp = StrategyInputs(*(col(getattr(run.composed.inputs, f)) for f in
                               ("signal", "strength", "atr", "m_sl", "m_tp", "max_hold", "entry_mask")))
        mk = MarketData(run.market.dates, [tk], col(run.market.open), col(run.market.high), col(run.market.low),
                        col(run.market.close), col(run.market.adv))
        out[tk] = run_backtest(mk, inp, FullCapitalSizer())
    return out


def run_buy_hold(panel: pd.DataFrame, dates: pd.DatetimeIndex, costs=None):
    """Buy & Hold equiponderado: compra al open del primer día, vende al cierre del último."""
    from src.backtest import CostModel
    from src.portfolio import BuyHoldSizer, buy_hold_inputs
    a = panel.index.get_loc(dates[0])
    market = MarketData.from_panel(panel, config.TICKERS).slice(a, a + len(dates))
    return run_backtest(market, buy_hold_inputs(len(dates), len(config.TICKERS)), BuyHoldSizer(),
                        costs or CostModel(), force_close_last=True)


def table_lookup(table: dict):
    """θ para (fecha, ticker, régimen) desde una tabla fija {ticker: {régimen: θ}}."""
    return lambda d, tk, r: table[tk][config.REGIME_NAMES[r]]


def tested_windows(wf: dict) -> list[dict]:
    return [rec["ventana"] for rec in wf["ventanas"] if rec["ventana"]["test_start"]]


def wf_oos_spec(wf: dict, **kwargs) -> SystemSpec:
    """Spec de la simulación OOS continua: el mes de prueba k usa θ_k y el modelo de régimen_k."""
    table = theta_table(wf)
    tested = tested_windows(wf)
    starts = pd.DatetimeIndex([w["test_start"] for w in tested])
    ks = [w["k"] for w in tested]

    def theta_for_day(d, tk, r):
        k = ks[starts.searchsorted(d, side="right") - 1]
        return table[k][tk][config.REGIME_NAMES[r]]

    schedule = [(pd.Timestamp(w["test_start"]), wf["modelos"][w["k"]]) for w in tested]
    return SystemSpec(theta_for_day, schedule, **kwargs)


def wf_oos_dates(index: pd.DatetimeIndex, wf: dict) -> pd.DatetimeIndex:
    tested = tested_windows(wf)
    return index[(index >= pd.Timestamp(tested[0]["test_start"])) & (index <= pd.Timestamp(tested[-1]["test_end"]))]


def run_wf_is(panel: pd.DataFrame, feats: pd.DataFrame, wf: dict, **spec_kwargs) -> pd.DataFrame:
    """Desempeño in-sample del sistema en el train de cada ventana con su θ_k y régimen_k."""
    from src.metrics import cagr, max_drawdown, sharpe, daily_returns
    table = theta_table(wf)
    caches, rows = {}, []
    for w in tested_windows(wf):
        dates = panel.index[(panel.index >= pd.Timestamp(w["train_start"])) & (panel.index <= pd.Timestamp(w["train_end"]))]
        spec = SystemSpec(table_lookup(table[w["k"]]), [(dates[0], wf["modelos"][w["k"]])], **spec_kwargs)
        eq = run_system(panel, feats, dates, spec, caches=caches).result.equity
        rows.append({"k": w["k"], "train_inicio": w["train_start"], "train_fin": w["train_end"],
                     "retorno_is": eq.iloc[-1] / eq.iloc[0] - 1, "cagr_is": cagr(eq),
                     "sharpe_is": sharpe(daily_returns(eq)), "mdd_is": max_drawdown(eq), "calmar_is": calmar(eq)})
    return pd.DataFrame(rows)


def walk_forward_efficiency(oos_equity: pd.Series, wf_is: pd.DataFrame) -> dict:
    """WFE = rendimiento anualizado WF-OOS / promedio del anualizado WF-IS (y versión Calmar).

    WFE < ~0.5 indica que la mayor parte del desempeño in-sample es ruido ajustado.
    """
    from src.metrics import cagr
    is_cagr, is_calmar = float(wf_is["cagr_is"].mean()), float(wf_is["calmar_is"].mean())
    oos_cagr, oos_calmar = cagr(oos_equity), calmar(oos_equity)
    return {"cagr_oos": oos_cagr, "cagr_is_promedio": is_cagr, "wfe_cagr": oos_cagr / is_cagr if is_cagr else np.nan,
            "calmar_oos": oos_calmar, "calmar_is_promedio": is_calmar,
            "wfe_calmar": oos_calmar / is_calmar if is_calmar else np.nan}


def parameter_table(wf: dict) -> pd.DataFrame:
    """Tabla larga de θ por ventana, activo y estudio: robusto, argmax, Calmar y fallback."""
    rows = []
    for rec in wf["ventanas"]:
        w = rec["ventana"]
        units = {"compartido": rec["compartido"]} if "compartido" in rec else rec["activos"]
        for tk, u in units.items():
            for study, st in u["estudios"].items():
                row = {"k": w["k"], "train_inicio": w["train_start"], "train_fin": w["train_end"],
                       "activo": tk, "estudio": study, "valido": st.get("valido", False),
                       "motivo_fallback": st.get("motivo", ""),
                       "calmar_robusto": st.get("calmar_robusto"), "calmar_argmax": st.get("calmar_argmax")}
                for name in INT_PARAMS + ("bb_k", "m_sl", "m_tp"):
                    row[name] = (st.get("robusto") or {}).get(name)
                    row[f"argmax_{name}"] = (st.get("argmax") or {}).get(name)
                rows.append(row)
    return pd.DataFrame(rows)


PARAM_NAMES = INT_PARAMS + ("bb_k", "m_sl", "m_tp")


def perturb_theta(table: dict, param: str, delta: float) -> dict:
    """Copia de {ticker: {régimen: θ}} con `param` multiplicado por (1 + delta) en todos."""
    out = {}
    for tk, by_regime in table.items():
        out[tk] = {}
        for reg, th in by_regime.items():
            if th is None:
                out[tk][reg] = None
                continue
            th = dict(th)
            v = th[param] * (1 + delta)
            th[param] = max(1, int(round(v))) if param in INT_PARAMS else v
            out[tk][reg] = th
    return out


def sensitivity_analysis(panel, feats, dates, frozen_table: dict, schedule: list,
                         deltas=(-0.2, -0.1, 0.0, 0.1, 0.2), **spec_kwargs) -> pd.DataFrame:
    """Variación de cada parámetro (y de cada multiplicador de régimen) uno a la vez.

    El cambio se aplica a todos los activos y regímenes a la vez.
    """
    from src.metrics import max_drawdown, sharpe, daily_returns
    caches, rows = {}, []

    def evaluate(table, multipliers):
        spec = SystemSpec(table_lookup(table), schedule, multipliers=multipliers, **spec_kwargs)
        eq = run_system(panel, feats, dates, spec, caches=caches).result.equity
        return {"calmar": calmar(eq), "sharpe": sharpe(daily_returns(eq)), "mdd": max_drawdown(eq),
                "retorno_total": eq.iloc[-1] / eq.iloc[0] - 1}

    base = evaluate(frozen_table, None)
    for param in PARAM_NAMES:
        for d in deltas:
            m = base if d == 0 else evaluate(perturb_theta(frozen_table, param, d), None)
            rows.append({"tipo": "parametro", "parametro": param, "delta": d, **m})
    for regime in config.REGIME_MULTIPLIER:
        for d in deltas:
            mult = dict(config.REGIME_MULTIPLIER)
            mult[regime] = mult[regime] * (1 + d)
            m = base if d == 0 else evaluate(frozen_table, mult)
            rows.append({"tipo": "multiplicador", "parametro": f"M_{regime}", "delta": d, **m})
    return pd.DataFrame(rows)


def sensitivity_verdict(sens: pd.DataFrame) -> pd.DataFrame:
    """Por parámetro: Calmar base, rango en ±20%, caída y cambio relativo máximos.

    cambio relativo = max_δ |Calmar(δ) − Calmar(0)| / max(|Calmar(0)|, 0.25); el piso
    evita que un Calmar base cercano a 0 infle el cociente. Veredicto "pico" si el
    cambio relativo supera 50%, o si el Calmar cambia de signo partiendo de |Calmar| ≥ 0.25
    (cerca de 0, pasar de −0.01 a +0.01 no es un cambio material); si no, "meseta".
    """
    rows = []
    for (tipo, param), g in sens.groupby(["tipo", "parametro"], sort=False):
        base = g.loc[g["delta"] == 0, "calmar"].iloc[0]
        denom = max(abs(base), 0.25)
        change = float((g["calmar"] - base).abs().max() / denom)
        drop = float(max(base - g["calmar"].min(), 0.0) / denom)
        flips = bool((np.sign(g["calmar"]) != np.sign(base)).any()) and abs(base) >= 0.25
        rows.append({"tipo": tipo, "parametro": param, "calmar_base": base, "calmar_min": g["calmar"].min(),
                     "calmar_max": g["calmar"].max(), "caida_relativa_max": drop, "cambio_relativo_max": change,
                     "cambia_signo": flips, "veredicto": "pico" if (change > 0.5 or flips) else "meseta"})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Congelamiento de θ y candado del TEST
# --------------------------------------------------------------------------

def canonical_hash(content: dict) -> str:
    """SHA-256 del JSON canónico (claves ordenadas)."""
    import hashlib
    return hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def freeze_system(wf: dict, feature_names: list[str],
                  rebalance: tuple = (config.REBALANCE_EVERY, config.REBALANCE_BAND)) -> dict:
    """θ robusto y modelo de régimen de la última ventana de TRAIN, reglas y multiplicadores."""
    final = wf["ventanas"][-1]
    k = final["ventana"]["k"]
    content = {
        "variante": wf["variante"], "ventana_final": final["ventana"],
        "theta": theta_table(wf)[k], "modelo_regimen": wf["modelos"][k].to_dict(), "features_regimen": feature_names,
        "multiplicadores": config.REGIME_MULTIPLIER, "regla_entrada_por_regimen": config.REGIME_MIN_AGREE,
        "rebalanceo_dias": rebalance[0], "banda_rebalanceo": rebalance[1],
        "estimador_covarianza": config.COV_ESTIMATOR, "ventana_covarianza": config.COV_WINDOW,
        "correlacion_conflicto": config.CONFLICT_CORR, "fuerza_senal": "s_i = Σ votos / 3",
        "comision": config.COMMISSION,
    }
    return {"contenido": content, "sha256": canonical_hash(content)}


class LockViolationError(RuntimeError):
    """El TEST no puede correrse: θ alterado, árbol de git sucio o candado violado."""


def git_state(root) -> tuple[bool, str]:
    """(árbol limpio, commit HEAD) del repositorio."""
    import subprocess
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True).stdout
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    return status.strip() == "", head


def check_test_lock(frozen: dict, lock_path, tree_clean: bool, head: str, require_clean: bool = True) -> dict:
    """Aplica el candado del TEST y regresa el contenido del lock.

    - El hash guardado debe coincidir con el contenido de θ (detecta ediciones a mano).
    - Si ya existe un lock, el hash de θ debe ser el mismo: re-optimizar después de
      ver el TEST aborta; re-correr con los mismos parámetros sí se permite.
    - La primera corrida exige árbol de git limpio y registra el commit.
    """
    from datetime import datetime
    current = canonical_hash(frozen["contenido"])
    if current != frozen["sha256"]:
        raise LockViolationError("theta_congelado.json fue modificado: su hash no coincide con su contenido")
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock["sha256_theta"] != current:
            raise LockViolationError(f"El candado registra θ {lock['sha256_theta'][:12]} y el actual es {current[:12]}: "
                                "no se permite re-optimizar después de ver el TEST")
        return lock
    if require_clean and not tree_clean:
        raise LockViolationError("El working tree de git no está limpio: haz commit antes de correr el TEST")
    lock = {"sha256_theta": current, "commit": head, "fecha": datetime.now().isoformat(timespec="seconds"),
            "arbol_limpio": tree_clean}
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    return lock


def frozen_spec(frozen: dict, **kwargs) -> SystemSpec:
    """Spec del sistema congelado (θ fijo y modelo de régimen reconstruido del JSON)."""
    c = frozen["contenido"]
    model = R.FrozenRegimeModel.from_dict(c["modelo_regimen"])
    return SystemSpec(table_lookup(c["theta"]), [(None, model)], multipliers=c["multiplicadores"],
                      rebalance_every=c["rebalanceo_dias"], band=c["banda_rebalanceo"], **kwargs)


# --------------------------------------------------------------------------
# Diagnósticos de Optuna (pregunta 3)
# --------------------------------------------------------------------------

SEARCH_BOUNDS = {"ema_fast": (5, 30), "ema_gap": (10, 100), "rsi_window": (7, 28), "rsi_lower": (20, 40),
                 "rsi_upper": (60, 80), "bb_window": (10, 40), "bb_k": (1.5, 3.0), "m_sl": (1.0, 4.0),
                 "m_tp": (1.0, 6.0), "max_hold": (5, 40)}


def optuna_diagnostics(panel: pd.DataFrame, ticker: str, window: dict, n_trials: int, grid: int = 15) -> dict:
    """Re-ejecuta (misma semilla, mismo resultado) el estudio global de un activo en una ventana.

    Regresa trials (historia y slices), importancias (fANOVA) y una superficie del
    Calmar sobre los 2 parámetros más importantes con el resto fijo en el θ robusto:
    es un corte 2D de un espacio de 10 dimensiones.
    """
    i = config.TICKERS.index(ticker)
    ohlcv = panel.xs(ticker, axis=1, level=1)[["open", "high", "low", "close", "volume"]]
    a = panel.index.get_loc(pd.Timestamp(window["train_start"]))
    b = panel.index.get_loc(pd.Timestamp(window["train_end"])) + 1
    aw = AssetWindow(ohlcv, IndicatorCache(ohlcv), a, b)
    index = aw.market.dates

    def score(p):
        res = aw.backtest(p)
        return config.INVALID_OBJECTIVE if res.n_trades < config.N_MIN_GLOBAL else embargoed_calmar(res.equity_array, index)

    study = run_study(lambda tr: score(suggest_params(tr)), study_seed(window["k"], i, 3), n_trials)
    robust = robust_params(study)
    evaluator = optuna.importance.FanovaImportanceEvaluator(seed=config.SEED)
    importance = optuna.importance.get_param_importances(study, evaluator=evaluator)
    top2 = list(importance)[:2]
    surface = []
    for x in np.linspace(*SEARCH_BOUNDS[top2[0]], grid):
        for y in np.linspace(*SEARCH_BOUNDS[top2[1]], grid):
            th = dict(robust)
            for p, v in ((top2[0], x), (top2[1], y)):
                th[p] = int(round(v)) if p in INT_PARAMS else float(v)
            res = aw.backtest(th)
            surface.append({"x": th[top2[0]], "y": th[top2[1]], "calmar": embargoed_calmar(res.equity_array, index),
                            "n_operaciones": res.n_trades, "valido": res.n_trades >= config.N_MIN_GLOBAL})
    trials = study.trials_dataframe(attrs=("number", "value", "params"))
    trials.columns = [c.replace("params_", "") for c in trials.columns]
    trials.loc[trials["value"] <= config.INVALID_OBJECTIVE / 2, "value"] = np.nan
    return {"activo": ticker, "ventana": window, "trials": trials, "importancia": importance,
            "top2": top2, "superficie": pd.DataFrame(surface), "robusto": robust, "calmar_robusto": score(robust),
            "argmax": study.best_params, "calmar_argmax": study.best_value}


# --------------------------------------------------------------------------
# Robustez del sistema (preguntas 1, 4 y barridos)
# --------------------------------------------------------------------------

def _curve_metrics(eq: pd.Series, result=None) -> dict:
    from src.metrics import cagr, max_drawdown, sharpe, daily_returns
    row = {"retorno_total": eq.iloc[-1] / eq.iloc[0] - 1, "cagr": cagr(eq), "sharpe": sharpe(daily_returns(eq)),
           "mdd": max_drawdown(eq), "calmar": calmar(eq), "pnl_neto": eq.iloc[-1] - eq.iloc[0]}
    if result is not None:
        row.update({"n_operaciones": result.n_trades, "turnover_anual": result.turnover,
                    "costos_totales": result.total_costs, "pnl_bruto": row["pnl_neto"] + result.total_costs})
    return row


def single_indicator_comparison(panel, feats, dates, spec: SystemSpec, caches=None) -> pd.DataFrame:
    """2 de 3 vs. cada indicador solo (mismo θ, mismas salidas, mismo portafolio).

    Ceteris paribus: solo cambia la regla de entrada, así se aísla el efecto de la
    confirmación. Se reporta a nivel portafolio y por activo (estrategia individual).
    """
    from dataclasses import replace
    rows = []
    variants = [("2 de 3", ("ema", "rsi", "bb"), 2), ("solo EMA", ("ema",), 1),
                ("solo RSI", ("rsi",), 1), ("solo Bollinger", ("bb",), 1)]
    for name, indicators, k in variants:
        run = run_system(panel, feats, dates, replace(spec, indicators=indicators, min_agree=k), caches=caches)
        rows.append({"regla": name, "nivel": "portafolio", "activo": "portafolio",
                     **_curve_metrics(run.result.equity, run.result)})
        for tk, res in run_individual_assets(run).items():
            rows.append({"regla": name, "nivel": "activo", "activo": tk, **_curve_metrics(res.equity, res)})
    return pd.DataFrame(rows)


def cost_curve(panel, feats, dates, spec: SystemSpec, composed,
               commissions=tuple(np.round(np.arange(0, 0.005001, 0.00025), 6))) -> pd.DataFrame:
    """Retorno neto vs. comisión por lado (0% a 0.5% en pasos de 0.025%), señales fijas."""
    from dataclasses import replace
    from src.backtest import CostModel
    rows = []
    for c in commissions:
        run = run_system(panel, feats, dates, replace(spec, costs=CostModel(commission=float(c))), composed=composed)
        rows.append({"comision": float(c), **_curve_metrics(run.result.equity, run.result)})
    return pd.DataFrame(rows)


def break_even(curve: pd.DataFrame, official: float = config.COMMISSION) -> dict:
    """Comisión donde el PnL neto cruza 0 (interpolación lineal) y margen de seguridad.

    Margen = c_equilibrio − 0.125% (en puntos porcentuales) y c_equilibrio / 0.125%.
    Costo anual ≈ T̄·f·2c = rotación anual × 2c (fracción del capital por año).
    """
    c, pnl = curve["comision"].to_numpy(), curve["pnl_neto"].to_numpy()
    be = None
    if pnl[0] <= 0:
        be = 0.0
    else:
        for i in range(1, len(c)):
            if pnl[i] <= 0:
                be = float(c[i - 1] + (c[i] - c[i - 1]) * pnl[i - 1] / (pnl[i - 1] - pnl[i]))
                break
    row = curve.loc[np.isclose(curve["comision"], official)].iloc[0]
    return {"comision_oficial": official, "comision_equilibrio": be,
            "equilibrio_fuera_de_rango": be is None,
            "margen_puntos": None if be is None else be - official,
            "margen_multiplo": None if be is None else be / official,
            "turnover_anual": float(row["turnover_anual"]),
            "costo_anual_por_rotacion": float(row["turnover_anual"] * 2 * official),
            "costos_totales_oficial": float(row["costos_totales"]),
            "pnl_bruto_oficial": float(row["pnl_bruto"]), "pnl_neto_oficial": float(row["pnl_neto"])}


IMPACT_ETA = 0.1      # orden de magnitud de η en la literatura de impacto (Almgren et al., 2005)


def execution_scenarios(panel, feats, dates, spec: SystemSpec, composed,
                        slippages=(0.0, 2.5, 5.0, 10.0, 15.0, 20.0)) -> pd.DataFrame:
    """Advertencia de ejecución cuantificada.

    Escenario realista: comisión + spread de 2 bps (medio spread por operación) +
    borrow fee de 0.5% anual sobre cortos + impacto η·(|q|/ADV)^(2/3), η = 0.1.
    Además, barrido de slippage de 0 a 20 bps por operación sobre la comisión oficial.
    """
    from dataclasses import replace
    from src.backtest import CostModel
    rows = []
    scenarios = [("oficial (solo comisión)", CostModel()),
                 ("realista (spread+borrow+impacto)", CostModel(spread_bps=2.0, borrow_fee_annual=0.005,
                                                                impact_eta=IMPACT_ETA))]
    scenarios += [(f"slippage {s:g} bps", CostModel(slippage_bps=s)) for s in slippages if s > 0]
    for name, costs in scenarios:
        run = run_system(panel, feats, dates, replace(spec, costs=costs), composed=composed)
        rows.append({"escenario": name, **_curve_metrics(run.result.equity, run.result)})
    out = pd.DataFrame(rows)
    base = out.iloc[0]
    out["delta_cagr"] = out["cagr"] - base["cagr"]
    out["delta_calmar"] = out["calmar"] - base["calmar"]
    return out


REBALANCE_FREQUENCIES = (("diario", 1), ("semanal", 5), ("quincenal", 10), ("mensual", 21))
REBALANCE_BANDS = (0.0, 0.05, 0.10, 0.20)


def rebalance_sweep(panel, feats, dates, spec: SystemSpec, composed,
                    frequencies=REBALANCE_FREQUENCIES, bands=REBALANCE_BANDS) -> pd.DataFrame:
    """Malla frecuencia f × banda δ del rebalanceo híbrido (señales fijas).

    Por combinación: retorno bruto, costo total, retorno neto, turnover realizado,
    T̄ por rebalanceo, rebalanceos por año y costo anual ≈ T̄·f·2c.
    """
    from dataclasses import replace
    rows = []
    years = len(dates) / config.TRADING_DAYS
    for name, days in frequencies:
        for band in bands:
            run = run_system(panel, feats, dates, replace(spec, rebalance_every=days, band=band), composed=composed)
            m = _curve_metrics(run.result.equity, run.result)
            capital = run.result.equity.iloc[0]
            t_reb = [x[4] for x in run.sizer.log]
            t_bar = float(np.mean(t_reb)) if t_reb else 0.0
            f_year = len(t_reb) / years
            rows.append({"frecuencia": name, "dias": days, "banda": band, **m,
                         "retorno_bruto": m["pnl_bruto"] / capital, "costo_total_pct": m["costos_totales"] / capital,
                         "rebalanceos_por_anio": f_year, "turnover_por_rebalanceo": t_bar,
                         "costo_anual_rebalanceo": t_bar * f_year * 2 * config.COMMISSION,
                         "costos_vs_pnl_bruto": m["costos_totales"] / m["pnl_bruto"] if m["pnl_bruto"] > 0 else np.nan})
    return pd.DataFrame(rows)


def choose_rebalance(sweep: pd.DataFrame) -> tuple[int, float]:
    """(f, δ) con mayor Calmar en el WF-OOS de TRAIN; empate → menor turnover."""
    best = sweep.sort_values(["calmar", "turnover_anual"], ascending=[False, True]).iloc[0]
    return int(best["dias"]), float(best["banda"])


def weighting_comparison(panel, feats, dates, spec: SystemSpec, composed) -> tuple[pd.DataFrame, dict]:
    """Pesos iguales vs. RP naïve (1/σ) vs. RP optimizado, con las mismas señales y costos."""
    from dataclasses import replace
    from src.portfolio import WEIGHTING_METHODS
    rows, runs = [], {}
    for method, name in WEIGHTING_METHODS.items():
        run = run_system(panel, feats, dates, replace(spec, method=method), composed=composed)
        runs[method] = run
        rows.append({"metodo": method, "nombre": name, **_curve_metrics(run.result.equity, run.result)})
    return pd.DataFrame(rows), runs


def covariance_comparison(panel, dates, estimators=("muestral", "ewma", "ledoit_wolf"),
                          every: int = config.REBALANCE_EVERY) -> tuple[pd.DataFrame, pd.DataFrame]:
    """w^RP de los 6 activos con cada estimador de Σ en cada fecha de calendario (datos ≤ t).

    Estabilidad = ½·Σ|Δw^RP| promedio entre fechas consecutivas (turnover que generaría
    solo el estimador). Regresa (resumen por estimador, pesos en formato largo).
    """
    from src.portfolio import condition_number, estimate_covariance, risk_parity_weights
    returns = panel["close"][config.TICKERS].pct_change()
    pos = returns.index.get_indexer(dates)
    values = returns.to_numpy()
    rows, summary = [], []
    for est in estimators:
        weights, kappas = [], []
        for t in pos[::every]:
            window = values[t - config.COV_WINDOW + 1: t + 1]
            cov = estimate_covariance(window, est)
            weights.append(risk_parity_weights(cov))
            kappas.append(condition_number(cov))
            rows.append({"estimador": est, "fecha": returns.index[t], **dict(zip(config.TICKERS, weights[-1]))})
        w = np.array(weights)
        changes = 0.5 * np.abs(np.diff(w, axis=0)).sum(axis=1)
        summary.append({"estimador": est, "cambio_medio_pesos": float(changes.mean()),
                        "cambio_max_pesos": float(changes.max()), "kappa_mediana": float(np.median(kappas)),
                        **{f"peso_medio_{tk}": float(w[:, k].mean()) for k, tk in enumerate(config.TICKERS)}})
    return pd.DataFrame(summary), pd.DataFrame(rows)
