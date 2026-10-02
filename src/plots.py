"""Figuras del proyecto (solo grafican; los cálculos viven en los demás módulos).

Todas en español, con título, ejes etiquetados, leyenda (o barra de color) y
fuentes grandes para leerse desde el fondo del salón. PNG a 150 dpi.
`make_all_figures` lee docs/resultados/ y escribe docs/figures/.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.metrics import drawdown  # noqa: E402

plt.rcParams.update({"font.size": 14, "axes.titlesize": 17, "axes.labelsize": 15, "legend.fontsize": 12,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "figure.dpi": 100, "savefig.dpi": 150,
                     "axes.grid": True, "grid.alpha": 0.3, "savefig.bbox": "tight",
                     "svg.hashsalt": "lab02", "pdf.fonttype": 42})

REGIME_COLORS = {"Tendencia": "#2a9d8f", "Reversión": "#e9c46a", "Crisis": "#e76f51"}
SYSTEM_COLORS = {"RP": "#1d3557", "EW": "#e63946", "RP naïve": "#f4a261", "Buy & Hold": "#8d99ae"}
ASSET_COLORS = dict(zip(config.TICKERS, plt.cm.tab10.colors))
MONTHS = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]


def _date_axis(ax) -> None:
    """Fechas compactas sin traslape, sin importar la longitud del periodo."""
    locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _save(fig, path: Path) -> None:
    fig.savefig(path, metadata={"Software": None})
    plt.close(fig)


def _read(results: Path, name: str, **kw) -> pd.DataFrame | None:
    path = results / name
    return pd.read_csv(path, **kw) if path.exists() else None


def _shade_regimes(ax, regimes: pd.Series, alpha: float = 0.18) -> None:
    """Sombrea el fondo según el régimen vigente (rachas consecutivas)."""
    change = regimes.ne(regimes.shift()).cumsum()
    for _, seg in regimes.groupby(change):
        ax.axvspan(seg.index[0], seg.index[-1], color=REGIME_COLORS.get(seg.iloc[0], "white"), alpha=alpha, lw=0)
    for name, color in REGIME_COLORS.items():
        ax.fill_between([], [], color=color, alpha=0.4, label=f"Régimen {name}")


def _pending(ax, text="TEST pendiente:\nse corre una sola vez con el candado") -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, fontsize=15, color="gray")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.plot([], [], color="gray", label="sin datos todavía")
    ax.legend(loc="lower right")


def portfolio_value(curves: dict[str, pd.DataFrame], path: Path) -> None:
    """Figura 1: valor del portafolio RP vs. EW y Buy & Hold en entrenamiento (WF-OOS) y prueba."""
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5), gridspec_kw={"width_ratios": [3, 1.3]})
    for ax, (title, df) in zip(axes, curves.items()):
        ax.set_title(title)
        if df is None:
            _pending(ax)
            continue
        for name in ("RP", "EW", "Buy & Hold"):
            ax.plot(df.index, df[name] / 1e6, label=f"{name}", color=SYSTEM_COLORS[name], lw=2 if name == "RP" else 1.5)
        ax.set_yscale("log")
        ax.set_xlabel("Fecha")
        _date_axis(ax)
        ax.set_ylabel("Valor (millones USD, escala log)")
        ax.legend(loc="upper left")
    fig.suptitle("Valor del portafolio: Risk Parity vs. benchmarks (capital inicial $1,000,000)", fontsize=18)
    _save(fig, path)


def drawdowns(curves: dict[str, pd.DataFrame], path: Path) -> None:
    """Figura 2: curva de drawdown de cada sistema."""
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.5), gridspec_kw={"width_ratios": [3, 1.3]})
    for ax, (title, df) in zip(axes, curves.items()):
        ax.set_title(title)
        if df is None:
            _pending(ax)
            continue
        for name in ("RP", "EW", "Buy & Hold"):
            dd = -drawdown(df[name]) * 100
            ax.plot(dd.index, dd, label=name, color=SYSTEM_COLORS[name], lw=1.6)
        ax.fill_between(df.index, -drawdown(df["RP"]) * 100, 0, color=SYSTEM_COLORS["RP"], alpha=0.15)
        ax.set_xlabel("Fecha")
        _date_axis(ax)
        ax.set_ylabel("Drawdown (%)")
        ax.legend(loc="lower left")
    fig.suptitle("Drawdown: caída desde el máximo previo", fontsize=18)
    _save(fig, path)


def returns_heatmap(monthly: pd.Series, annual: pd.Series, title: str, path: Path) -> None:
    """Figura 3a: rendimientos mensuales (año × mes) con columna anual."""
    df = monthly.to_frame("r")
    df["anio"], df["mes"] = df.index.year, df.index.month
    grid = df.pivot(index="anio", columns="mes", values="r").reindex(columns=range(1, 13))
    grid.columns = MONTHS
    grid["Año"] = annual.groupby(annual.index.year).first()
    values = grid.to_numpy() * 100
    fig, ax = plt.subplots(figsize=(16, 0.55 * len(grid) + 2.5))
    lim = np.nanpercentile(np.abs(values), 95)
    im = ax.imshow(values, cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")
    for (i, j), v in np.ndenumerate(values):
        if not np.isnan(v):
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=10.5,
                    fontweight="bold" if j == 12 else "normal")
    ax.set_xticks(range(13), grid.columns)
    ax.set_yticks(range(len(grid)), grid.index)
    ax.set_xlabel("Mes (última columna: año completo)")
    ax.set_ylabel("Año")
    ax.set_title(title)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Rendimiento (%)")
    _save(fig, path)


def quarterly_table(quarterly: pd.DataFrame, title: str, path: Path) -> None:
    """Figura 3b: rendimientos trimestrales por sistema (barras agrupadas)."""
    q = quarterly * 100
    labels = [f"{d.year}-T{(d.month - 1) // 3 + 1}" for d in q.index]
    fig, ax = plt.subplots(figsize=(18, 6))
    x = np.arange(len(q))
    width = 0.8 / len(q.columns)
    for k, col in enumerate(q.columns):
        ax.bar(x + k * width, q[col], width, label=col, color=SYSTEM_COLORS.get(col))
    step = max(1, len(x) // 16)
    ax.set_xticks(x[::step] + 0.4, labels[::step], rotation=45, ha="right")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Trimestre")
    ax.set_ylabel("Rendimiento trimestral (%)")
    ax.set_title(title)
    ax.legend()
    _save(fig, path)


def sensitivity(sens: pd.DataFrame, path: Path) -> None:
    """Figura 4: Calmar ante variaciones de ±20% de cada parámetro (uno a la vez)."""
    params = list(dict.fromkeys(sens["parametro"]))
    ncols = 4
    nrows = int(np.ceil(len(params) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.6 * nrows), sharey=True)
    base = sens.loc[sens["delta"] == 0, "calmar"].iloc[0]
    for ax, p in zip(axes.flat, params):
        g = sens[sens["parametro"] == p].sort_values("delta")
        color = "#7b2cbf" if g["tipo"].iloc[0] == "multiplicador" else "#1d3557"
        ax.plot(g["delta"] * 100, g["calmar"], "o-", color=color, lw=2, label="Calmar")
        ax.axhline(base, color="gray", ls="--", lw=1, label="base")
        ax.set_title(p, fontsize=14)
        ax.set_xticks([-20, -10, 0, 10, 20])
        ax.set_xlabel("Variación (%)", fontsize=12)
    for ax in axes[:, 0]:
        ax.set_ylabel("Calmar")
    for ax in list(axes.flat)[len(params):]:
        ax.axis("off")
    handles = [plt.Line2D([], [], color="#1d3557", marker="o", lw=2), plt.Line2D([], [], color="#7b2cbf", marker="o", lw=2),
               plt.Line2D([], [], color="gray", ls="--")]
    fig.legend(handles, ["Calmar (parámetro de θ)", "Calmar (multiplicador de régimen)", "Calmar base (θ congelado)"],
               loc="lower right", bbox_to_anchor=(0.98, 0.06), fontsize=13)
    fig.suptitle("Sensibilidad del Calmar a ±20% en cada parámetro (θ congelado, portafolio RP)", fontsize=18)
    fig.tight_layout()
    _save(fig, path)


def cost_curve(curve: pd.DataFrame, be: dict, path: Path) -> None:
    """Figura 5: retorno neto total vs. comisión por lado, con 0.125% y equilibrio."""
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.plot(curve["comision"] * 100, curve["retorno_total"] * 100, "o-", color="#1d3557", lw=2,
            label="Retorno neto total (WF-OOS)")
    ax.axhline(0, color="black", lw=1)
    ax.axvline(config.COMMISSION * 100, color="#e63946", ls="--", lw=2, label="Comisión oficial 0.125%")
    if be["comision_equilibrio"] is not None:
        ax.axvline(be["comision_equilibrio"] * 100, color="#2a9d8f", ls=":", lw=2.5,
                   label=f"Equilibrio ≈ {be['comision_equilibrio'] * 100:.3f}%")
    ax.set_xlabel("Comisión por lado (% del nocional)")
    ax.set_ylabel("Retorno neto total (%)")
    ax.set_title("Retorno neto contra nivel de costo de transacción")
    ax.legend()
    _save(fig, path)


def regime_timeline(index_price: pd.Series, regimes: pd.Series, title: str, path: Path) -> None:
    """Figura 6a: índice equiponderado con el régimen sombreado."""
    fig, ax = plt.subplots(figsize=(17, 6))
    _shade_regimes(ax, regimes)
    ax.plot(index_price.index, index_price, color="black", lw=1.4, label="Índice equiponderado (6 activos)")
    ax.set_yscale("log")
    ax.set_xlabel("Fecha")
    _date_axis(ax)
    ax.set_ylabel("Nivel del índice (escala log)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    _save(fig, path)


def feature_distributions(feats: pd.DataFrame, used: list[str], path: Path) -> None:
    """Figura 6b: distribución de cada feature por régimen (boxplots)."""
    cols = [c for c in feats.columns if c not in ("regimen", "indice_ew")]
    fig, axes = plt.subplots(1, len(cols), figsize=(4.2 * len(cols), 6))
    names = list(REGIME_COLORS)
    for ax, c in zip(axes, cols):
        data = [feats.loc[feats["regimen"] == n, c].dropna() for n in names]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
        for patch, n in zip(bp["boxes"], names):
            patch.set_facecolor(REGIME_COLORS[n])
        ax.set_xticks(range(1, 4), names, rotation=20)
        ax.set_title(c + (" *" if c in used else ""), fontsize=14)
        ax.set_xlabel("Régimen", fontsize=12)
    axes[0].set_ylabel("Valor de la feature")
    handles = [plt.Rectangle((0, 0), 1, 1, color=REGIME_COLORS[n]) for n in names]
    fig.legend(handles, names, loc="upper right", ncol=3)
    fig.suptitle("Distribución de las variables de régimen (* = usadas en K-means)", fontsize=18, y=1.02)
    fig.tight_layout()
    _save(fig, path)


def equity_with_regimes(equity: pd.Series, regimes: pd.Series, path: Path) -> None:
    """Figura 6c: valor del portafolio RP con los regímenes superpuestos."""
    fig, ax = plt.subplots(figsize=(17, 6))
    _shade_regimes(ax, regimes)
    ax.plot(equity.index, equity / 1e6, color=SYSTEM_COLORS["RP"], lw=2, label="Portafolio RP")
    ax.set_xlabel("Fecha")
    _date_axis(ax)
    ax.set_ylabel("Valor (millones USD)")
    ax.set_title("Valor del portafolio con los regímenes superpuestos (WF-OOS)")
    ax.legend(loc="upper left")
    _save(fig, path)


def risk_contributions(rc: pd.DataFrame, path: Path) -> None:
    """Figura 7a: contribución promedio al riesgo por activo: pesos iguales, RP naïve y RP."""
    data = rc.reset_index()
    data = data[data["activo"].isin(config.TICKERS)]
    pivot = data.pivot(index="activo", columns="sistema", values="rc_promedio_abs").reindex(config.TICKERS) * 100
    systems = [s for s in ("EW", "RP naïve", "RP") if s in pivot.columns]
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(pivot))
    width = 0.8 / len(systems)
    for k, s in enumerate(systems):
        ax.bar(x + (k - (len(systems) - 1) / 2) * width, pivot[s], width, label=s, color=SYSTEM_COLORS[s])
    ax.axhline(100 / 6, color="black", ls="--", lw=1.3, label="Paridad perfecta (1/6)")
    ax.set_xticks(x, pivot.index)
    ax.set_xlabel("Activo")
    ax.set_ylabel("Contribución promedio al riesgo (%)")
    ax.set_title("Contribución al riesgo por activo: pesos iguales vs. RP naïve (1/σ) vs. Risk Parity")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4)
    _save(fig, path)


def signal_heatmap(strength: pd.DataFrame, path: Path) -> None:
    """Figura 7b: fuerza de señal con signo (−3 a +3), promedio semanal, por activo."""
    fig, ax = plt.subplots(figsize=(18, 5))
    im = ax.imshow(strength.T.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3,
                   extent=[0, len(strength), len(strength.columns), 0], interpolation="nearest")
    ax.set_yticks(np.arange(len(strength.columns)) + 0.5, strength.columns)
    years = strength.index.year
    ticks = [i for i in range(len(years)) if i == 0 or years[i] != years[i - 1]]
    ax.set_xticks(ticks, [str(years[i]) for i in ticks])
    ax.set_xlabel("Fecha (semanas)")
    ax.set_ylabel("Activo")
    ax.set_title("Mapa de calor de la fuerza de señal (rojo = corto, azul = largo; ±2 o ±3 votos)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Dirección × votos (promedio semanal)")
    _save(fig, path)


def correlation_by_regime(corr: pd.DataFrame, rolling: pd.Series, regimes: pd.Series, path: Path) -> None:
    """Figura 7c: correlación entre activos en cada régimen y su evolución en el tiempo."""
    names = [n for n in REGIME_COLORS if n in corr.index.get_level_values(0)]
    fig = plt.figure(figsize=(18, 11))
    grid = fig.add_gridspec(2, len(names), height_ratios=[1.2, 1])
    for k, n in enumerate(names):
        ax = fig.add_subplot(grid[0, k])
        m = corr.loc[n][config.TICKERS].reindex(config.TICKERS)
        im = ax.imshow(m.to_numpy(), cmap="viridis", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(m.to_numpy()):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color="white" if v < 0.6 else "black")
        ax.set_xticks(range(6), config.TICKERS, rotation=45)
        ax.set_yticks(range(6), config.TICKERS)
        ax.set_title(f"{n}: corr. promedio {(m.to_numpy().sum() - 6) / 30:.2f}", fontsize=14)
        ax.grid(False)
    fig.colorbar(im, ax=fig.axes[: len(names)], label="Correlación")
    ax = fig.add_subplot(grid[1, :])
    _shade_regimes(ax, regimes)
    ax.plot(rolling.index, rolling, color="black", lw=1.6, label="Correlación promedio entre pares (126 días)")
    ax.set_xlabel("Fecha")
    _date_axis(ax)
    ax.set_ylabel("Correlación promedio")
    ax.legend(loc="lower left", ncol=2)
    fig.suptitle("Evolución de la matriz de correlación entre regímenes", fontsize=18)
    _save(fig, path)


def rebalance_sweep(sweep: pd.DataFrame, chosen: dict, path: Path) -> None:
    """Figura 7d: retorno bruto, costo total, retorno neto y turnover según f y δ (paso 8.3)."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    freqs = list(dict.fromkeys(sweep["frecuencia"]))
    x = np.arange(len(freqs))
    panels = [("retorno_bruto", "Retorno bruto (%)", 100), ("costo_total_pct", "Costo total (% del capital)", 100),
              ("retorno_total", "Retorno neto (%)", 100), ("turnover_anual", "Turnover realizado (veces/año)", 1)]
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, sweep["banda"].nunique()))
    for ax, (col, label, scale) in zip(axes, panels):
        for color, (band, g) in zip(colors, sweep.groupby("banda")):
            g = g.set_index("frecuencia").reindex(freqs)
            ax.plot(x, g[col] * scale, "o-", color=color, lw=2, label=f"δ = {band:.2f}")
        best = sweep[(sweep["dias"] == chosen["dias"]) & np.isclose(sweep["banda"], chosen["banda"])].iloc[0]
        ax.scatter([freqs.index(best["frecuencia"])], [best[col] * scale], s=220, facecolors="none",
                   edgecolors="#e63946", lw=2.5, label="elegido", zorder=5)
        ax.set_xticks(x, freqs)
        ax.set_xlabel("Frecuencia de revisión f")
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=14)
    axes[0].legend(fontsize=10)
    fig.suptitle("Rebalanceo híbrido (calendario + banda ‖w − w*‖₁ > δ): retorno, costo y turnover (WF-OOS)",
                 fontsize=18)
    fig.tight_layout()
    _save(fig, path)


def covariance_estimators(weights: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    """w^RP en el tiempo con cada estimador de Σ y su estabilidad (paso 6)."""
    names = list(summary["estimador"])
    fig, axes = plt.subplots(1, len(names), figsize=(20, 5.5), sharey=True)
    for ax, est in zip(axes, names):
        w = weights[weights["estimador"] == est].set_index("fecha")[config.TICKERS]
        ax.stackplot(w.index, w.T.to_numpy(), labels=config.TICKERS,
                     colors=[ASSET_COLORS[t] for t in config.TICKERS], alpha=0.85)
        change = summary.set_index("estimador").loc[est, "cambio_medio_pesos"]
        ax.set_title(f"{est}: ½Σ|Δw| medio = {change:.3f}", fontsize=14)
        ax.set_xlabel("Fecha")
        _date_axis(ax)
    axes[0].set_ylabel("Peso Risk Parity")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("Pesos de Risk Parity con tres estimadores de Σ (126 días, solo datos ≤ t)", fontsize=18)
    fig.tight_layout()
    _save(fig, path)


def single_indicator(df: pd.DataFrame, path: Path) -> None:
    """Regla 2 de 3 vs. cada indicador solo: operaciones y Calmar (portafolio)."""
    d = df[df["nivel"] == "portafolio"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    colors = ["#1d3557" if r == "2 de 3" else "#a8dadc" for r in d["regla"]]
    axes[0].bar(d["regla"], d["n_operaciones"], color=colors, label="Operaciones")
    axes[0].set_ylabel("Número de operaciones")
    axes[1].bar(d["regla"], d["calmar"], color=colors, label="Calmar")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_ylabel("Calmar (WF-OOS)")
    for ax in axes:
        ax.set_xlabel("Regla de entrada")
        ax.legend()
    fig.suptitle("¿Qué aporta la confirmación 2 de 3? (mismo θ, mismas salidas)", fontsize=18)
    fig.tight_layout()
    _save(fig, path)


def regime_transitions(matrix: pd.DataFrame, persistence: dict, path: Path) -> None:
    """Matriz de transición diaria y duración observada vs. esperada E[D] = 1/(1 − A_jj)."""
    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    im = axes[0].imshow(matrix.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(matrix.to_numpy()):
        axes[0].text(j, i, "—" if np.isnan(v) else f"{v:.3f}", ha="center", va="center",
                     color="white" if v > 0.6 else "black")
    axes[0].set_xticks(range(3), matrix.columns)
    axes[0].set_yticks(range(3), matrix.index)
    axes[0].set_xlabel("Régimen en t+1")
    axes[0].set_ylabel("Régimen en t")
    axes[0].set_title("Matriz de transición A_ij")
    axes[0].grid(False)
    fig.colorbar(im, ax=axes[0], label="Probabilidad")
    names = list(persistence["por_regimen"])
    obs = [persistence["por_regimen"][n]["duracion_promedio_observada"] or 0 for n in names]
    exp = [persistence["por_regimen"][n]["duracion_esperada_markov"] or 0 for n in names]
    x = np.arange(len(names))
    axes[1].bar(x - 0.2, obs, 0.4, label="Observada", color=[REGIME_COLORS[n] for n in names])
    axes[1].bar(x + 0.2, exp, 0.4, label="Esperada 1/(1−A_jj)", color="gray")
    axes[1].axhline(10, color="black", ls="--", label="Objetivo ≥ 10 días")
    axes[1].set_xticks(x, names)
    axes[1].set_xlabel("Régimen")
    axes[1].set_ylabel("Duración promedio (días hábiles)")
    axes[1].set_title("Persistencia de los regímenes")
    axes[1].legend()
    fig.suptitle("Análisis de transiciones de régimen (WF-OOS)", fontsize=18)
    fig.tight_layout()
    _save(fig, path)


def portfolio_vs_assets(curves: pd.DataFrame, path: Path) -> None:
    """Portafolio RP vs. la estrategia de cada activo sola (100% del capital)."""
    fig, ax = plt.subplots(figsize=(17, 6.5))
    for tk in config.TICKERS:
        ax.plot(curves.index, curves[f"estrategia {tk}"] / 1e6, lw=1.2, color=ASSET_COLORS[tk], alpha=0.85,
                label=f"Estrategia {tk}")
    ax.plot(curves.index, curves["RP"] / 1e6, color="black", lw=2.8, label="Portafolio RP")
    ax.set_yscale("log")
    ax.set_xlabel("Fecha")
    _date_axis(ax)
    ax.set_ylabel("Valor (millones USD, escala log)")
    ax.set_title("Portafolio vs. estrategia individual de cada activo (WF-OOS)")
    ax.legend(ncol=4, loc="upper left")
    _save(fig, path)


def costs_vs_gross(per_asset: pd.DataFrame, path: Path) -> None:
    """PnL bruto, costos y PnL neto por sistema y estrategia individual."""
    d = per_asset.set_index("conjunto")
    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(d))
    ax.bar(x - 0.27, d["pnl_bruto"] / 1e3, 0.27, label="PnL bruto", color="#2a9d8f")
    ax.bar(x, -d["costos_totales"] / 1e3, 0.27, label="Costos (negativo)", color="#e63946")
    ax.bar(x + 0.27, d["pnl_neto"] / 1e3, 0.27, label="PnL neto", color="#1d3557")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, d.index, rotation=20)
    ax.set_xlabel("Sistema")
    ax.set_ylabel("Miles USD")
    ax.set_title("Costos totales contra retorno bruto (WF-OOS)")
    ax.legend()
    _save(fig, path)


def optuna_panels(trials: pd.DataFrame, diag: dict, surface: pd.DataFrame, figures: Path) -> None:
    """Diagnósticos de Optuna: historia, importancia, slices y superficie 2D (corte de 10D)."""
    who = f"{diag['activo']}, ventana final ({diag['ventana']['train_start']} a {diag['ventana']['train_end']})"
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(trials["number"], trials["value"], "o", alpha=0.6, label="Calmar del trial")
    ax.plot(trials["number"], trials["value"].cummax(), color="#e63946", lw=2.5, label="Mejor hasta el momento")
    ax.axvline(config.N_STARTUP_TRIALS - 0.5, color="gray", ls="--", label="Fin de la fase aleatoria (30)")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Calmar (train, con embargo)")
    ax.set_title(f"Historia de optimización — {who}")
    ax.legend()
    _save(fig, figures / "12a_optuna_historia.png")

    imp = pd.Series(diag["importancia"]).sort_values()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(imp.index, imp.values, color="#457b9d", label="Importancia fANOVA")
    ax.set_xlabel("Importancia relativa")
    ax.set_ylabel("Parámetro")
    ax.set_title(f"Importancia de parámetros — {diag['activo']}")
    ax.legend(loc="lower right")
    _save(fig, figures / "12b_optuna_importancia.png")

    params = list(imp.index[::-1])
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=True)
    for ax, p in zip(axes.flat, params):
        ax.scatter(trials[p], trials["value"], c=trials["number"], cmap="viridis", s=22, label="trials")
        ax.axvline(diag["robusto"][p], color="#e63946", lw=2, label="θ robusto")
        ax.set_xlabel(p, fontsize=13)
        ax.legend(fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Calmar")
    fig.suptitle("Slice plots: Calmar vs. cada parámetro (color = número de trial)", fontsize=18)
    fig.tight_layout()
    _save(fig, figures / "12c_optuna_slices.png")

    x, y = diag["top2"]
    piv = surface.pivot_table(index="y", columns="x", values="calmar")
    X, Y = np.meshgrid(piv.columns, piv.index)
    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(projection="3d")
    surf = ax.plot_surface(X, Y, piv.to_numpy(), cmap="viridis", edgecolor="none", alpha=0.85)
    bad = surface[~surface["valido"].astype(bool)]
    ax.scatter(bad["x"], bad["y"], bad["calmar"], color="gray", marker="x", s=25,
               label=f"< {config.N_MIN_GLOBAL} operaciones (inválido)")
    ax.scatter([diag["robusto"][x]], [diag["robusto"][y]], [diag["calmar_robusto"]], color="red", s=90,
               label="θ robusto")
    ax.set_xlabel(x, labelpad=12)
    ax.set_ylabel(y, labelpad=12)
    ax.set_zlabel("Calmar", labelpad=10)
    ax.set_title(f"Superficie del Calmar sobre {x} y {y} — {diag['activo']}\n"
                 "(corte 2D de un espacio de 10D; resto en θ robusto)")
    ax.legend(loc="upper left")
    fig.colorbar(surf, ax=ax, shrink=0.6, label="Calmar")
    _save(fig, figures / "12d_optuna_superficie_3d.png")


def rolling_vs_anchored(curves: pd.DataFrame, comparison: pd.DataFrame, path: Path) -> None:
    """Curvas WF-OOS de las variantes y su WFE."""
    fig, ax = plt.subplots(figsize=(16, 6.5))
    for col in curves.columns:
        row = comparison.set_index("variante").loc[col]
        ax.plot(curves.index, curves[col] / 1e6, lw=2,
                label=f"{col} (WFE retorno = {row['wfe_cagr']:.2f}, Calmar OOS = {row['calmar_oos']:.2f})")
    ax.set_xlabel("Fecha")
    _date_axis(ax)
    ax.set_ylabel("Valor (millones USD)")
    ax.set_title("Walk-forward: rolling (6 meses) vs. anchored (train creciente) y variantes de θ")
    ax.legend()
    _save(fig, path)


def signal_correlation(df: pd.DataFrame, path: Path) -> None:
    """Correlación entre los votos de los 3 indicadores por activo."""
    m = df.set_index("activo")[["ema_rsi", "ema_bb", "rsi_bb"]]
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(m.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(m.to_numpy()):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=13)
    ax.set_xticks(range(3), ["EMA–RSI", "EMA–Bollinger", "RSI–Bollinger"])
    ax.set_yticks(range(len(m)), m.index)
    ax.set_xlabel("Par de indicadores")
    ax.set_ylabel("Activo")
    ax.set_title("Correlación entre las señales de los 3 indicadores\n(> 0.7 indicaría redundancia)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Correlación de Pearson")
    _save(fig, path)


def make_all_figures(results: Path, figures: Path) -> list[Path]:
    """Genera todas las figuras a partir de los archivos de resultados."""
    figures.mkdir(parents=True, exist_ok=True)
    rd = lambda name, **kw: _read(results, name, **kw)  # noqa: E731
    oos = rd("equity_wf_oos.csv", index_col=0, parse_dates=True)
    test = rd("equity_test.csv", index_col=0, parse_dates=True)
    curves = {"Entrenamiento (walk-forward OOS concatenado)": oos, "Prueba (TEST, sistema congelado)": test}
    portfolio_value(curves, figures / "01_valor_portafolio.png")
    drawdowns(curves, figures / "02_drawdown.png")

    monthly = rd("retornos_mensual_wf_oos.csv", index_col=0, parse_dates=True)
    annual = rd("retornos_anual_wf_oos.csv", index_col=0, parse_dates=True)
    quarterly = rd("retornos_trimestral_wf_oos.csv", index_col=0, parse_dates=True)
    returns_heatmap(monthly["RP"], annual["RP"], "Rendimientos mensuales y anuales (%) — portafolio RP, WF-OOS",
                    figures / "03a_retornos_mensuales_anuales.png")
    quarterly_table(quarterly, "Rendimientos trimestrales — WF-OOS", figures / "03b_retornos_trimestrales.png")
    if test is not None:
        tm = rd("retornos_mensual_test.csv", index_col=0, parse_dates=True)
        ta = rd("retornos_anual_test.csv", index_col=0, parse_dates=True)
        returns_heatmap(tm["RP"], ta["RP"], "Rendimientos mensuales y anuales (%) — portafolio RP, TEST",
                        figures / "03c_retornos_mensuales_test.png")

    sensitivity(rd("sensibilidad.csv"), figures / "04_sensibilidad.png")
    be = json.loads((results / "costos_equilibrio.json").read_text(encoding="utf-8"))
    cost_curve(rd("curva_costos.csv"), be, figures / "05_curva_costos.png")

    labels = rd("regimenes_etiquetas_wf_oos.csv", index_col=0, parse_dates=True)
    validation = json.loads((results / "regimenes_validacion.json").read_text(encoding="utf-8"))
    regimes = labels["regimen"]
    timeline = labels[["indice_ew", "regimen"]]
    if test is not None:
        timeline = pd.concat([timeline, rd("regimenes_etiquetas_test.csv", index_col=0, parse_dates=True)[["indice_ew", "regimen"]]])
    regime_timeline(timeline["indice_ew"], timeline["regimen"], "Línea de tiempo de regímenes sobre el índice equiponderado",
                    figures / "06a_regimenes_linea_tiempo.png")
    feature_distributions(labels, validation["features"], figures / "06b_distribuciones_features.png")
    equity_with_regimes(oos["RP"], regimes, figures / "06c_valor_con_regimenes.png")

    risk_contributions(rd("rp_vs_ew_contribuciones.csv", index_col=[0, 1]), figures / "07a_contribuciones_riesgo.png")
    signal_heatmap(rd("fuerza_senal_semanal.csv", index_col=0, parse_dates=True), figures / "07b_fuerza_senal.png")
    correlation_by_regime(rd("correlacion_por_regimen.csv", index_col=[0, 1]),
                          rd("correlacion_promedio_movil.csv", index_col=0, parse_dates=True).iloc[:, 0],
                          regimes, figures / "07c_correlacion_por_regimen.png")
    chosen = json.loads((results / "optimizacion_resumen.json").read_text(encoding="utf-8"))["rebalanceo_elegido"]
    rebalance_sweep(rd("rebalanceo_barrido.csv"), chosen, figures / "07d_barrido_rebalanceo.png")
    covariance_estimators(rd("covarianza_pesos_rp.csv", parse_dates=["fecha"]), rd("covarianza_estimadores.csv"),
                          figures / "15_estimadores_covarianza.png")

    single_indicator(rd("indicador_unico.csv"), figures / "08_indicador_unico.png")
    regime_transitions(pd.DataFrame(validation["matriz_transicion_wf_oos"]),
                       validation["persistencia_wf_oos"], figures / "09_transiciones_regimen.png")
    portfolio_vs_assets(oos, figures / "10_portafolio_vs_activos.png")
    costs_vs_gross(rd("metricas_por_activo_wf_oos.csv"), figures / "11_costos_vs_bruto.png")
    diag = json.loads((results / "optuna_diagnostico.json").read_text(encoding="utf-8"))
    optuna_panels(rd("optuna_trials.csv"), diag, rd("optuna_superficie.csv"), figures)
    rolling_vs_anchored(rd("equity_variantes_wf_oos.csv", index_col=0, parse_dates=True),
                        rd("variantes_comparacion.csv"), figures / "13_rolling_vs_anclado.png")
    signal_correlation(rd("correlacion_senales.csv"), figures / "14_correlacion_senales.png")
    return sorted(figures.glob("*.png"))
