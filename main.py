"""Lab 02 MyST — Equipo 2 (Nivel C). Ejecuta el proyecto completo: `python main.py`.

Etapas: data -> train -> test -> report (`--stage all`, por omisión).
    data    descarga (solo si no hay datos congelados), auditoría y split.
    train   walk-forward en TRAIN, simulación WF-OOS, regímenes, portafolio, robustez y θ congelado.
    test    evalúa el sistema congelado en TEST una sola vez (con candado).
    report  figuras, tablas, borradores y PDFs leyendo docs/resultados/.
`--quick`: pocas ventanas y 20 trials; escribe en .cache/quick/ y ensaya el TEST
sobre los últimos 2 meses de TRAIN. Nunca toca docs/.

Este archivo solo orquesta: toda la lógica vive en src/.
"""
import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src import metrics as M
from src import optimize as O
from src import portfolio as P
from src import regimes as R
from src.data import load_clean, prepare_data

log = logging.getLogger("main")
QUICK_WINDOWS = 3
QUICK_TRIALS = 20
QUICK_TEST_MONTHS = 2
REPRESENTATIVE_ASSET = "NVDA"
SYSTEMS = ("rp", "ew")


@dataclass
class Context:
    """Rutas y parámetros de la corrida (oficial o --quick)."""
    quick: bool
    results: Path
    figures: Path
    docs: Path
    wf_cache: Path
    n_trials: int
    tree_clean: bool
    head: str

    @classmethod
    def build(cls, quick: bool) -> "Context":
        clean, head = O.git_state(config.ROOT)
        if quick:
            base = config.QUICK_DIR
            ctx = cls(True, base / "resultados", base / "figures", base, base / "wf", QUICK_TRIALS, clean, head)
        else:
            ctx = cls(False, config.RESULTS_DIR, config.FIGURES_DIR, config.DOCS_DIR, config.CACHE_DIR / "wf",
                      config.N_TRIALS, clean, head)
        for d in (ctx.results, ctx.figures, ctx.wf_cache):
            d.mkdir(parents=True, exist_ok=True)
        return ctx


def save_csv(df: pd.DataFrame, ctx: Context, name: str, index: bool = False) -> None:
    df.to_csv(ctx.results / name, index=index, float_format="%.10g")


def save_json(obj, ctx: Context, name: str) -> None:
    (ctx.results / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default),
                                    encoding="utf-8")


def _json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, (pd.Timestamp,)):
        return str(x.date())
    return str(x)


def splits_for(ctx: Context, splits: dict, index: pd.DatetimeIndex) -> dict:
    """En --quick, TRAIN termina 2 meses antes y el "TEST" de ensayo son esos 2 meses."""
    if not ctx.quick:
        return splits
    test_start = pd.Timestamp(splits["train_fin"]) - pd.DateOffset(months=QUICK_TEST_MONTHS)
    train = index[(index >= pd.Timestamp(splits["train_inicio"])) & (index < test_start)]
    test = index[(index >= test_start) & (index <= pd.Timestamp(splits["train_fin"]))]
    return {**splits, "train_fin": str(train[-1].date()), "test_inicio": str(test[0].date()),
            "test_fin": str(test[-1].date()), "dias_train": len(train), "dias_test": len(test)}


# ----------------------------------------------------------------------------
# Etapa train
# ----------------------------------------------------------------------------

def windows_for(ctx: Context, panel, splits, anchored: bool):
    first_test = "2016-01-01"
    windows = O.build_windows(panel.index, first_test, splits["train_fin"], anchored=anchored,
                              anchor_start="2015-07-01")
    if ctx.quick:
        windows = windows[-(QUICK_WINDOWS + 1):]
    return windows


def evaluate_variant(panel, feats, wf, composed=None, **spec_kwargs):
    """Simulación OOS continua (RP) y WF-IS de una variante, con su WFE."""
    dates = O.wf_oos_dates(panel.index, wf)
    run = O.run_system(panel, feats, dates, O.wf_oos_spec(wf, **spec_kwargs), composed=composed)
    wf_is = O.run_wf_is(panel, feats, wf, **spec_kwargs)
    return run, wf_is, O.walk_forward_efficiency(run.result.equity, wf_is)


def window_summaries(panel, feats, wf, method: str, **spec_kwargs) -> pd.DataFrame:
    """Métricas completas (summary) del sistema en el train de cada ventana (WF-IS)."""
    table = O.theta_table(wf)
    caches, rows = {}, []
    for w in O.tested_windows(wf):
        dates = panel.index[(panel.index >= pd.Timestamp(w["train_start"])) & (panel.index <= pd.Timestamp(w["train_end"]))]
        if method == "bh":
            res = O.run_buy_hold(panel, dates)
        else:
            spec = O.SystemSpec(O.table_lookup(table[w["k"]]), [(dates[0], wf["modelos"][w["k"]])], method=method,
                                **spec_kwargs)
            res = O.run_system(panel, feats, dates, spec, caches=caches).result
        rows.append(M.result_summary(res, f"WF-IS k={w['k']}"))
    return pd.DataFrame(rows)


def average_summary(rows: pd.DataFrame, name: str) -> dict:
    numeric = rows.select_dtypes("number").mean()
    return {"conjunto": name, "inicio": rows["inicio"].min(), "fin": rows["fin"].max(), **numeric.to_dict(),
            "n_ventanas": len(rows)}


def stage_train(ctx: Context) -> None:
    panel, splits0 = load_clean()
    splits = splits_for(ctx, splits0, panel.index)
    train_end = pd.Timestamp(splits["train_fin"])
    feats = R.regime_features(panel)

    # --- Régimen: selección de features solo con TRAIN ---
    names, subset_table = R.select_features(feats, train_end)
    save_csv(subset_table, ctx, "regimenes_seleccion_features.csv")
    log.info("Features de régimen: %s", names)

    # --- Walk-forward rolling: por activo y compartido ---
    windows = windows_for(ctx, panel, splits, anchored=False)
    t0 = time.perf_counter()
    wfs, evals = {}, {}
    for variant in ("por_activo", "compartido"):
        wfs[variant] = O.walk_forward(panel, windows, feats, names, variant, ctx.n_trials, ctx.wf_cache,
                                      keep_trials_for=None)
        evals[variant] = evaluate_variant(panel, feats, wfs[variant])
        log.info("%s: Calmar WF-OOS %.3f, WFE %.3f", variant, evals[variant][2]["calmar_oos"],
                 evals[variant][2]["wfe_cagr"])
    chosen = max(evals, key=lambda v: evals[v][2]["calmar_oos"])
    log.info("Variante elegida por Calmar WF-OOS: %s", chosen)

    # --- Anchored con la variante elegida ---
    anchored_windows = windows_for(ctx, panel, splits, anchored=True)
    wf_anch = O.walk_forward(panel, anchored_windows, feats, names, f"anclado_{chosen}", ctx.n_trials, ctx.wf_cache)
    wall = time.perf_counter() - t0
    wf = wfs[chosen]
    oos_dates = evals[chosen][0].result.dates

    # --- Rebalanceo híbrido: malla f × δ en el WF-OOS de TRAIN (señales fijas) ---
    sweep = O.rebalance_sweep(panel, feats, oos_dates, O.wf_oos_spec(wf), evals[chosen][0].composed)
    save_csv(sweep, ctx, "rebalanceo_barrido.csv")
    reb_every, reb_band = O.choose_rebalance(sweep)
    spec_kw = {"rebalance_every": reb_every, "band": reb_band}
    log.info("Rebalanceo elegido: cada %d días con banda δ = %.2f", reb_every, reb_band)
    evals = {v: evaluate_variant(panel, feats, wfs[v], evals[v][0].composed, **spec_kw) for v in evals}
    eval_anch = evaluate_variant(panel, feats, wf_anch, **spec_kw)
    run, wf_is, wfe = evals[chosen]
    spec = O.wf_oos_spec(wf, **spec_kw)

    # --- Resumen de optimización y variantes ---
    all_wf = {**wfs, f"anclado_{chosen}": wf_anch}
    comparison = []
    for name, (r, _, e) in {**evals, f"anclado_{chosen}": eval_anch}.items():
        comparison.append({"variante": name, **e, "mdd_oos": M.max_drawdown(r.result.equity),
                           "n_operaciones": r.result.n_trades, "elegida": name == chosen})
    save_csv(pd.DataFrame(comparison), ctx, "variantes_comparacion.csv")
    save_csv(pd.DataFrame({name: r.result.equity for name, (r, _, _) in {**evals, f"anclado_{chosen}": eval_anch}.items()}),
             ctx, "equity_variantes_wf_oos.csv", index=True)
    save_csv(pd.concat([O.parameter_table(w).assign(variante=n) for n, w in all_wf.items()]), ctx,
             "parametros_por_ventana.csv")
    save_csv(wf_is, ctx, "wf_is_por_ventana.csv")
    save_json({
        "segundos_optimizacion": sum(w["segundos"] for w in all_wf.values()),
        "segundos_etapa_walk_forward": wall,
        "configuraciones_evaluadas": sum(w["n_evaluadas"] for w in all_wf.values()),
        "por_variante": {n: {"segundos": w["segundos"], "configuraciones": w["n_evaluadas"],
                             "ventanas": len(w["ventanas"]), "fallbacks": O.fallback_counts(w)} for n, w in all_wf.items()},
        "trials_por_estudio": ctx.n_trials, "trials_aleatorios_iniciales": config.N_STARTUP_TRIALS,
        "ventanas_walk_forward": len(windows) - 1, "ventana_final_congelada": windows[-1].__dict__,
        "variante_elegida": chosen, "criterio": "mayor Calmar WF-OOS (portafolio RP) dentro de TRAIN",
        "rebalanceo_elegido": {"dias": reb_every, "banda": reb_band,
                               "criterio": "mayor Calmar WF-OOS en la malla f × δ dentro de TRAIN"},
        "wfe_rolling": wfe, "wfe_anclado": eval_anch[2], "features_regimen": names,
        "procesos": max(1, (__import__("os").cpu_count() or 2) - 1),
    }, ctx, "optimizacion_resumen.json")

    # --- θ congelado (respeta el candado si el TEST ya se corrió) ---
    frozen = O.freeze_system(wf, names, (reb_every, reb_band))
    lock = ctx.results / "test_lock.json"
    target = ctx.results / "theta_congelado.json"
    if lock.exists() and json.loads(lock.read_text(encoding="utf-8"))["sha256_theta"] != frozen["sha256"]:
        log.warning("El candado del TEST existe y θ cambió: se guarda en theta_reoptimizado.json (no se usa en TEST)")
        target = ctx.results / "theta_reoptimizado.json"
    target.write_text(json.dumps(frozen, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")

    # --- Sistema WF-OOS: EW, RP naïve, RP, Buy & Hold e individuales ---
    weighting, runs = O.weighting_comparison(panel, feats, oos_dates, spec, run.composed)
    save_csv(weighting, ctx, "ponderaciones_comparacion.csv")
    run_ew, run_iv = runs["ew"], runs["iv"]
    bh = O.run_buy_hold(panel, oos_dates)
    individual = O.run_individual_assets(run)
    export_system_outputs(ctx, "wf_oos", run, run_ew, bh, individual, run_iv)

    sets = [average_summary(window_summaries(panel, feats, wf, "rp", **spec_kw), "WF-IS") | {"sistema": "RP"},
            average_summary(window_summaries(panel, feats, wf, "ew", **spec_kw), "WF-IS") | {"sistema": "EW"},
            average_summary(window_summaries(panel, feats, wf, "bh"), "WF-IS") | {"sistema": "Buy & Hold"}]
    for name, res in (("RP", run.result), ("EW", run_ew.result), ("RP naïve", run_iv.result), ("Buy & Hold", bh)):
        sets.append(M.result_summary(res, "WF-OOS") | {"sistema": name})
    save_csv(pd.DataFrame(sets), ctx, "metricas_conjuntos.csv")

    # --- Regímenes: validación ---
    labels = run.regimes
    returns = panel["close"][config.TICKERS].pct_change()
    sils = [m.silhouette for m in wf["modelos"].values()]
    validation = {
        "features": names, "k": config.N_REGIMES, "ventana_features": config.REGIME_WINDOW,
        "actualizacion_cada_dias": config.REGIME_UPDATE_EVERY, "persistencia_actualizaciones": config.REGIME_PERSISTENCE,
        "silhouette": {"promedio_ventanas": float(np.mean(sils)), "minimo": float(np.min(sils)),
                       "maximo": float(np.max(sils)), "modelo_final": wf["modelos"][windows[-1].k].silhouette,
                       "objetivo": R.SILHOUETTE_TARGET},
        "persistencia_wf_oos": R.persistence_stats(labels),
        "matriz_transicion_wf_oos": R.transition_matrix(labels).round(6).to_dict(),
        "centroides_modelo_final": wf["modelos"][windows[-1].k].centroids().to_dict(),
        "kruskal_portafolio_rp": R.kruskal_by_regime(M.daily_returns(run.result.equity), labels.shift(1)),
        "kruskal_indice_ew": R.kruskal_by_regime(R.equal_weight_index(panel["close"][config.TICKERS]).pct_change(),
                                                 labels.shift(1)),
        "fallbacks": O.fallback_counts(wf),
    }
    transitions = P.transition_impact(labels, run.result.weights)
    save_csv(transitions, ctx, "regimen_transiciones.csv")
    validation["transiciones"] = {"n": len(transitions),
                                  "con_posiciones_abiertas": int((transitions["posiciones_abiertas"] > 0).sum()),
                                  "posiciones_afectadas": int(transitions["posiciones_abiertas"].sum()),
                                  "a_crisis": int((transitions["a"] == "Crisis").sum())}
    save_json(validation, ctx, "regimenes_validacion.json")
    ew_index = R.equal_weight_index(panel["close"][config.TICKERS])
    save_csv(feats.loc[oos_dates].assign(regimen=labels.map(config.REGIME_NAMES), indice_ew=ew_index.loc[oos_dates]),
             ctx, "regimenes_etiquetas_wf_oos.csv", index=True)
    save_csv(R.run_lengths(labels), ctx, "regimenes_rachas_wf_oos.csv")

    regime_tables = []
    for name, res in (("RP", run.result), ("EW", run_ew.result), ("Buy & Hold", bh)):
        regime_tables.append(regime_table(res.equity, labels, name))
    for tk, res in individual.items():
        regime_tables.append(regime_table(res.equity, labels, f"estrategia {tk}"))
    save_csv(pd.concat(regime_tables), ctx, "metricas_por_regimen.csv")

    corr = P.correlation_by_regime(returns, labels)
    save_csv(pd.concat({k: v for k, v in corr.items()}, names=["regimen", "activo"]), ctx,
             "correlacion_por_regimen.csv", index=True)
    save_csv(P.rolling_mean_correlation(returns).loc[oos_dates].to_frame(), ctx, "correlacion_promedio_movil.csv",
             index=True)

    # --- Portafolio: EW vs RP naïve vs RP, contribuciones, estimadores de Σ y resumen operativo ---
    rc = {name: P.risk_contribution_summary(r.result.weights, returns)
          for name, r in (("RP", run), ("EW", run_ew), ("RP naïve", run_iv))}
    save_csv(pd.concat(rc, names=["sistema", "activo"]), ctx, "rp_vs_ew_contribuciones.csv", index=True)
    save_json(portfolio_summary({"RP": run, "EW": run_ew, "RP naïve": run_iv}), ctx, "portafolio_resumen.json")
    cov_summary, cov_weights = O.covariance_comparison(panel, oos_dates)
    save_csv(cov_summary, ctx, "covarianza_estimadores.csv")
    save_csv(cov_weights, ctx, "covarianza_pesos_rp.csv")
    strength = pd.DataFrame(run.composed.raw_signal * run.composed.inputs.strength, index=oos_dates,
                            columns=config.TICKERS)
    save_csv(strength.resample("W-FRI").mean(), ctx, "fuerza_senal_semanal.csv", index=True)

    # --- Robustez ---
    save_csv(O.single_indicator_comparison(panel, feats, oos_dates, spec), ctx, "indicador_unico.csv")
    curve = O.cost_curve(panel, feats, oos_dates, spec, run.composed)
    save_csv(curve, ctx, "curva_costos.csv")
    save_json(O.break_even(curve), ctx, "costos_equilibrio.json")
    save_csv(O.execution_scenarios(panel, feats, oos_dates, spec, run.composed), ctx, "costos_ejecucion_wf_oos.csv")
    frozen_table = frozen["contenido"]["theta"]
    final_model = wf["modelos"][windows[-1].k]
    sens = O.sensitivity_analysis(panel, feats, oos_dates, frozen_table, [(oos_dates[0], final_model)], **spec_kw)
    save_csv(sens, ctx, "sensibilidad.csv")
    save_csv(O.sensitivity_verdict(sens), ctx, "sensibilidad_veredicto.csv")
    save_csv(signal_correlations(panel, oos_dates, frozen_table), ctx, "correlacion_senales.csv")

    # --- Diagnósticos de Optuna (activo representativo, última ventana) ---
    diag = O.optuna_diagnostics(panel, REPRESENTATIVE_ASSET, windows[-1].__dict__, ctx.n_trials)
    save_csv(diag["trials"], ctx, "optuna_trials.csv")
    save_csv(diag["superficie"], ctx, "optuna_superficie.csv")
    save_json({k: diag[k] for k in ("activo", "ventana", "importancia", "top2", "robusto", "calmar_robusto", "argmax", "calmar_argmax")},
              ctx, "optuna_diagnostico.json")
    log.info("Etapa train terminada")


def regime_table(equity: pd.Series, labels: pd.Series, name: str) -> pd.DataFrame:
    """Métricas por régimen + IC bootstrap de la media diaria + Kruskal-Wallis."""
    by = M.metrics_by_regime(equity, labels, config.REGIME_NAMES)
    ci = R.bootstrap_mean_ci(M.daily_returns(equity), labels.shift(1))[["regimen", "media_diaria", "ic_inf", "ic_sup"]]
    kw = R.kruskal_by_regime(M.daily_returns(equity), labels.shift(1))
    return by.merge(ci, on="regimen", how="left").assign(sistema=name, kruskal_p=kw["p_valor"])


def portfolio_summary(runs: dict) -> dict:
    """Conteos operativos de cada portafolio: rebalanceos, T̄, conflictos, apalancamiento, κ(Σ)."""
    out = {}
    for name, r in runs.items():
        logs = pd.DataFrame(r.sizer.log, columns=["t", "tipo", "kappa", "n_activos", "turnover"])
        years = len(r.result.dates) / config.TRADING_DAYS
        trades = r.result.trades
        entry_regime = trades["fecha_entrada"].map(r.regimes.map(config.REGIME_NAMES))
        out[name] = {
            "rebalanceos": logs["tipo"].value_counts().to_dict(),
            "rebalanceos_omitidos_por_banda": r.sizer.n_band_skips,
            "turnover_por_rebalanceo": float(logs["turnover"].mean()) if len(logs) else 0.0,
            "rebalanceos_por_anio": len(logs) / years,
            "costo_anual_rebalanceo": (float(logs["turnover"].mean()) if len(logs) else 0.0) * len(logs) / years
                                      * 2 * config.COMMISSION,
            "conflictos": r.sizer.n_conflicts, "conflictos_empatados": r.sizer.n_conflict_ties,
            "kappa_mediana": float(logs["kappa"].median()), "kappa_max": float(logs["kappa"].max()),
            "max_exposicion_post_ejecucion": float(r.result.exec_gross.max()),
            "max_exposicion_al_cierre": float(r.result.gross_exposure.max()),
            "exposicion_promedio": float(r.result.gross_exposure.mean()),
            "recortes_apalancamiento": r.result.n_leverage_scalings,
            "operaciones_largas": int((trades["direccion"] == "largo").sum()),
            "operaciones_cortas": int((trades["direccion"] == "corto").sum()),
            "entradas_por_regimen": entry_regime.value_counts().to_dict(),
            "motivos_salida": trades["motivo_salida"].value_counts().to_dict(),
            "turnover_anual": r.result.turnover, "costos_totales": r.result.total_costs,
        }
    return out


def signal_correlations(panel, dates, frozen_table) -> pd.DataFrame:
    """Correlación entre los votos de los 3 indicadores por activo (θ congelado de Tendencia)."""
    from src.signals import generate_signals, signal_correlation
    rows = []
    for tk in config.TICKERS:
        th = frozen_table[tk]["Tendencia"] or next(t for t in frozen_table[tk].values() if t)
        sig = generate_signals(panel.xs(tk, axis=1, level=1), th).loc[dates]
        c = signal_correlation(sig)
        rows.append({"activo": tk, "ema_rsi": c.loc["s_ema", "s_rsi"], "ema_bb": c.loc["s_ema", "s_bb"],
                     "rsi_bb": c.loc["s_rsi", "s_bb"]})
    return pd.DataFrame(rows)


def export_system_outputs(ctx: Context, tag: str, run, run_ew, bh, individual, run_iv) -> None:
    """Curvas, pesos, operaciones y tablas de rendimientos de un conjunto (wf_oos o test)."""
    curves = pd.DataFrame({"RP": run.result.equity, "EW": run_ew.result.equity, "RP naïve": run_iv.result.equity,
                           "Buy & Hold": bh.equity})
    for tk, res in individual.items():
        curves[f"estrategia {tk}"] = res.equity
    curves["regimen"] = run.regimes.map(config.REGIME_NAMES)
    save_csv(curves, ctx, f"equity_{tag}.csv", index=True)
    save_csv(run.result.weights, ctx, f"pesos_rp_{tag}.csv", index=True)
    save_csv(run_ew.result.weights, ctx, f"pesos_ew_{tag}.csv", index=True)
    save_csv(run.result.trades, ctx, f"operaciones_rp_{tag}.csv")
    for period in ("mensual", "trimestral", "anual"):
        table = pd.DataFrame({n: M.returns_table(e)[period] for n, e in
                              (("RP", run.result.equity), ("EW", run_ew.result.equity), ("Buy & Hold", bh.equity))})
        save_csv(table, ctx, f"retornos_{period}_{tag}.csv", index=True)
    rows = [M.result_summary(res, tk) | {"tipo": "estrategia individual"} for tk, res in individual.items()]
    rows.append(M.result_summary(run.result, "Portafolio RP") | {"tipo": "portafolio"})
    rows.append(M.result_summary(run_ew.result, "Portafolio EW") | {"tipo": "portafolio"})
    save_csv(pd.DataFrame(rows), ctx, f"metricas_por_activo_{tag}.csv")


# ----------------------------------------------------------------------------
# Etapa test
# ----------------------------------------------------------------------------

def stage_test(ctx: Context, strict: bool) -> bool:
    """Evalúa el sistema congelado en TEST. Regresa False si se omitió (árbol sucio en --stage all)."""
    panel, splits0 = load_clean()
    splits = splits_for(ctx, splits0, panel.index)
    frozen = json.loads((ctx.results / "theta_congelado.json").read_text(encoding="utf-8"))
    try:
        lock = O.check_test_lock(frozen, ctx.results / "test_lock.json", ctx.tree_clean, ctx.head,
                                 require_clean=not ctx.quick)
    except O.LockViolationError as e:
        if strict or "limpio" not in str(e):
            raise
        log.warning("TEST pendiente: %s", e)
        return False
    log.info("Candado OK (θ %s…, commit %s)", lock["sha256_theta"][:12], lock["commit"][:10])

    feats = R.regime_features(panel)
    dates = panel.index[(panel.index >= pd.Timestamp(splits["test_inicio"])) & (panel.index <= pd.Timestamp(splits["test_fin"]))]
    spec = O.frozen_spec(frozen)
    run = O.run_system(panel, feats, dates, spec)
    run_ew = O.run_system(panel, feats, dates, replace(spec, method="ew"), composed=run.composed)
    run_iv = O.run_system(panel, feats, dates, replace(spec, method="iv"), composed=run.composed)
    bh = O.run_buy_hold(panel, dates)
    individual = O.run_individual_assets(run)
    export_system_outputs(ctx, "test", run, run_ew, bh, individual, run_iv)

    rows = [M.result_summary(res, "TEST") | {"sistema": name}
            for name, res in (("RP", run.result), ("EW", run_ew.result), ("RP naïve", run_iv.result), ("Buy & Hold", bh))]
    save_csv(pd.DataFrame(rows), ctx, "metricas_test.csv")
    tables = [regime_table(r.equity, run.regimes, n) for n, r in (("RP", run.result), ("EW", run_ew.result), ("Buy & Hold", bh))]
    save_csv(pd.concat(tables), ctx, "metricas_por_regimen_test.csv")
    save_csv(O.execution_scenarios(panel, feats, dates, spec, run.composed), ctx, "costos_ejecucion_test.csv")
    rc = {name: P.risk_contribution_summary(r.result.weights, panel["close"][config.TICKERS].pct_change())
          for name, r in (("RP", run), ("EW", run_ew), ("RP naïve", run_iv))}
    save_csv(pd.concat(rc, names=["sistema", "activo"]), ctx, "rp_vs_ew_contribuciones_test.csv", index=True)

    # Estabilidad del régimen fuera de muestra: TRAIN (WF-OOS) vs TEST
    train_labels = pd.read_csv(ctx.results / "regimenes_etiquetas_wf_oos.csv", index_col=0, parse_dates=True)["regimen"]
    codes = {v: k for k, v in config.REGIME_NAMES.items()}
    save_json({"train_wf_oos": R.persistence_stats(train_labels.map(codes)),
               "test": R.persistence_stats(run.regimes)}, ctx, "regimenes_estabilidad.json")
    ew_index = R.equal_weight_index(panel["close"][config.TICKERS])
    save_csv(feats.loc[dates].assign(regimen=run.regimes.map(config.REGIME_NAMES), indice_ew=ew_index.loc[dates]),
             ctx, "regimenes_etiquetas_test.csv", index=True)

    # Evaluación secundaria: el walk-forward continúa sobre TEST (re-optimización mensual)
    secondary_windows = O.build_windows(panel.index, str(dates[0].replace(day=1).date()), splits["test_fin"])[:-1]
    secondary_windows = [w for w in secondary_windows if pd.Timestamp(w.test_start) >= dates[0]]
    if secondary_windows:
        names = frozen["contenido"]["features_regimen"]
        variant = frozen["contenido"]["variante"]
        wf_test = O.walk_forward(panel, secondary_windows, feats, names, variant, ctx.n_trials,
                                 ctx.wf_cache / "test_continuo")
        sec = O.run_system(panel, feats, O.wf_oos_dates(panel.index, wf_test), O.wf_oos_spec(wf_test))
        save_csv(pd.DataFrame([M.result_summary(sec.result, "TEST (WF continuo)") | {"sistema": "RP"}]), ctx,
                 "metricas_test_secundario.csv")
        save_csv(sec.result.equity.to_frame("RP WF continuo"), ctx, "equity_test_secundario.csv", index=True)
    log.info("Etapa test terminada")
    return True


# ----------------------------------------------------------------------------
# Etapa report
# ----------------------------------------------------------------------------

def stage_report(ctx: Context) -> None:
    from src import plots, report
    plots.make_all_figures(ctx.results, ctx.figures)
    report.build_all(ctx.results, ctx.figures, ctx.docs)
    log.info("Etapa report terminada")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Lab 02 MyST — Equipo 2 (Nivel C)")
    parser.add_argument("--stage", choices=["data", "train", "test", "report", "all"], default="all")
    parser.add_argument("--quick", action="store_true", help="ensayo rápido en .cache/quick/ (no toca docs/)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stdout)
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    ctx = Context.build(args.quick)
    log.info("Corrida %s, etapa %s (git limpio: %s)", "QUICK" if ctx.quick else "OFICIAL", args.stage, ctx.tree_clean)
    t0 = time.perf_counter()
    if args.stage in ("data", "all"):
        prepare_data(ctx.results)
    if args.stage in ("train", "all"):
        stage_train(ctx)
    if args.stage in ("test", "all"):
        stage_test(ctx, strict=args.stage == "test")
    if args.stage in ("report", "all"):
        stage_report(ctx)
    log.info("Listo en %.1f minutos", (time.perf_counter() - t0) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
