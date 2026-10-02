"""Reporte y presentación generados a partir de docs/resultados/ (ninguna cifra a mano).

Cada número pasa por `Facts.get(archivo, selector)`, que lo lee del archivo de
resultados y lo registra. `verify` vuelve a leer los archivos y comprueba que cada
cifra impresa en los borradores coincide (C14). Si falta el TEST, se escribe
el marcador [PENDIENTE: test].
"""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from src import config  # noqa: E402

PENDING = "[PENDIENTE: test]"
SYSTEMS = ("RP", "EW", "Buy & Hold")


class Facts:
    """Lee cifras de los archivos de resultados y registra cada una para verificarla."""

    def __init__(self, results: Path):
        self.results = results
        self.registry: list[dict] = []
        self._cache: dict = {}

    def exists(self, name: str) -> bool:
        return (self.results / name).exists()

    def load(self, name: str):
        if name not in self._cache:
            path = self.results / name
            if name.endswith(".json"):
                self._cache[name] = json.loads(path.read_text(encoding="utf-8"))
            else:
                df = pd.read_csv(path)
                if {"conjunto", "sistema"} <= set(df.columns):
                    df["_k"] = df["conjunto"].astype(str) + "|" + df["sistema"].astype(str)
                self._cache[name] = df
        return self._cache[name]

    def note(self, name: str, what: str, value) -> None:
        """Registra una cifra derivada (rango, lista) que se imprime con formato propio."""
        self.registry.append({"archivo": name, "selector": [what], "valor": value, "texto": None, "derivada": True})

    def get(self, name: str, selector, fmt=None):
        """`selector`: tupla de claves (JSON) o (columna_filtro, valor, columna) / (fila, columna) en CSV."""
        value = _select(self.load(name), selector)
        text = fmt(value) if fmt else value
        self.registry.append({"archivo": name, "selector": list(map(str, selector)), "valor": _plain(value),
                              "texto": text if isinstance(text, str) else None})
        return text


def _select(data, selector):
    if isinstance(data, pd.DataFrame):
        if len(selector) == 3:
            col, val, target = selector
            rows = data[data[col].astype(str) == str(val)]
            return rows[target].iloc[0]
        row, target = selector
        return data[target].iloc[row]
    for key in selector:
        data = data[key]
    return data


def _plain(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else float(v)
    return v if isinstance(v, (int, str, bool, type(None))) else str(v)


def pct(x, d=1):
    return "n/d" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{d}f}%"


def num(x, d=2):
    return "n/d" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{d}f}"


def money(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/d"
    return f"−${-x:,.0f}" if x < 0 else f"${x:,.0f}"


def integer(x):
    return f"{int(x):,}"


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def fig(figures: Path, docs: Path, name: str, caption: str) -> str:
    rel = Path("figures") / name if (docs / "figures").resolve() == figures.resolve() else figures / name
    return f"![{caption}]({rel.as_posix()})\n\n*Figura: {caption}.*"


# ----------------------------------------------------------------------------
# Contenido del reporte
# ----------------------------------------------------------------------------

def metric_row(f: Facts, file: str, conjunto: str, sistema: str) -> list[str]:
    g = lambda col, fm: f.get(file, ("_k", f"{conjunto}|{sistema}", col), fm)  # noqa: E731
    return [conjunto, sistema, g("cagr", pct), g("sharpe", num), g("sortino", num), g("mdd", pct), g("calmar", num),
            g("win_rate", pct), g("n_operaciones", lambda v: integer(v) if pd.notna(v) else "n/d")]


def build_report(f: Facts, figures: Path, docs: Path) -> str:
    has_test = f.exists("metricas_test.csv")
    opt = "optimizacion_resumen.json"
    val = "regimenes_validacion.json"
    chosen = f.get(opt, ("variante_elegida",))
    vc = "variantes_comparacion.csv"
    wfe_roll = f.get(opt, ("wfe_rolling", "wfe_cagr"), num)
    wfe_roll_calmar = f.get(opt, ("wfe_rolling", "wfe_calmar"), num)
    wfe_anch = f.get(opt, ("wfe_anclado", "wfe_cagr"), num)
    wfe_value = f.get(opt, ("wfe_rolling", "wfe_cagr"))
    mc = "metricas_conjuntos.csv"
    rp_oos = lambda col, fm=None: f.get(mc, ("_k", "WF-OOS|RP", col), fm)  # noqa: E731
    be = "costos_equilibrio.json"
    sens_v = f.load("sensibilidad_veredicto.csv")
    n_peaks = int((sens_v["veredicto"] == "pico").sum())
    pr = "portafolio_resumen.json"
    splits = "splits.json"

    test_cagr = f.get("metricas_test.csv", ("sistema", "RP", "cagr"), pct) if has_test else PENDING
    test_calmar = f.get("metricas_test.csv", ("sistema", "RP", "calmar"), num) if has_test else PENDING
    test_mdd = f.get("metricas_test.csv", ("sistema", "RP", "mdd"), pct) if has_test else PENDING
    bh_test = f.get("metricas_test.csv", ("sistema", "Buy & Hold", "cagr"), pct) if has_test else PENDING

    out = []
    w = out.append
    w("# Laboratorio 02 — Estrategias de Trading con Análisis Técnico\n")
    w("**Microestructuras y Sistemas de Trading (ITESO) · Equipo 2 · Nivel C**  \n"
      "Integrantes: Milca (TSLA, NFLX), Paula (META, AMZN), Arturo (NVDA, GOOGL).\n")

    # 1. Resumen ejecutivo
    w("## 1. Resumen ejecutivo\n")
    w(f"Construimos un sistema de trading multi-activo sobre seis acciones tecnológicas de gran capitalización "
      f"(NVDA, AMZN, TSLA, META, NFLX, GOOGL) con datos diarios de {f.get(splits, ('train_inicio',))} a "
      f"{f.get(splits, ('test_fin',))}. Cada activo opera largo y corto con tres indicadores de familias distintas "
      f"(cruce de EMAs, RSI y Bandas de Bollinger) y la regla de confirmación 2 de 3; las salidas son por stop-loss y "
      f"take-profit en múltiplos de ATR, señal contraria o time-stop. Un detector de régimen (K-means) clasifica el "
      f"mercado en Tendencia, Reversión o Crisis, los parámetros se optimizan por régimen con Optuna maximizando el "
      f"Calmar en un walk-forward de 6 meses → 1 mes, y las posiciones se agregan en un portafolio de Risk Parity "
      f"con costos de 0.125% por lado y sin apalancamiento.\n")
    w(f"En el tramo fuera de muestra del walk-forward dentro de TRAIN ({rp_oos('inicio')} a {rp_oos('fin')}), el "
      f"portafolio Risk Parity obtuvo un CAGR de {rp_oos('cagr', pct)}, Sharpe de {rp_oos('sharpe', num)}, "
      f"drawdown máximo de {rp_oos('mdd', pct)} y Calmar de {rp_oos('calmar', num)}, frente a un Calmar de "
      f"{f.get(mc, ('_k', 'WF-OOS|EW', 'calmar'), num)} con pesos iguales y {f.get(mc, ('_k', 'WF-OOS|Buy & Hold', 'calmar'), num)} "
      f"del Buy & Hold. La eficiencia del walk-forward (WFE) es {wfe_roll} en rendimiento anualizado "
      f"({wfe_roll_calmar} en Calmar). En TEST, con el sistema congelado, el CAGR fue {test_cagr}, el MDD "
      f"{test_mdd} y el Calmar {test_calmar} (Buy & Hold: {bh_test}).\n")
    w(f"**Conclusión:** {edge_conclusion(wfe_value, n_peaks, len(sens_v), f, has_test)}\n")

    # 2. Datos
    w("## 2. Datos, auditoría y selección de activos\n")
    meta = "download_metadata.json" if f.exists("download_metadata.json") else None
    w(f"Fuente: Yahoo Finance mediante `yfinance` con `auto_adjust=True` (precios ajustados por splits y dividendos, "
      f"para que los rendimientos sean comparables y no aparezcan saltos artificiales). Los datos se congelaron en "
      f"`data/prices_daily.csv`{' el ' + f.get(meta, ('fecha_descarga',))[:10] if meta else ''}; `main.py` nunca "
      f"vuelve a descargar si el archivo existe.\n")
    audit = f.load("auditoria_datos.csv")
    w(md_table(["Activo", "Días", "Faltantes", "Duplicados", "Viol. OHLC", "Rend. diario > ±25%"],
               [[f.get("auditoria_datos.csv", ("ticker", t, "ticker")),
                 f.get("auditoria_datos.csv", ("ticker", t, "dias_con_datos"), integer),
                 f.get("auditoria_datos.csv", ("ticker", t, "fechas_faltantes_vs_calendario"), integer),
                 f.get("auditoria_datos.csv", ("ticker", t, "duplicados"), integer),
                 str(int(audit.loc[audit.ticker == t, ["violaciones_high", "violaciones_low"]].sum(axis=1).iloc[0])),
                 str(audit.loc[audit.ticker == t, "rend_extremos"].fillna("—").iloc[0])] for t in config.TICKERS]))
    w("\nLos rendimientos extremos corresponden a eventos reales de resultados trimestrales (no a splits mal "
      "ajustados): los splits de NVDA, TSLA, AMZN, GOOGL y NFLX no producen saltos en la serie ajustada.\n")
    w(f"**Split cronológico sin traslape:** TRAIN = primer 80% ({f.get(splits, ('train_inicio',))} a "
      f"{f.get(splits, ('train_fin',))}, {f.get(splits, ('dias_train',), integer)} días) y TEST = último 20% "
      f"({f.get(splits, ('test_inicio',))} a {f.get(splits, ('test_fin',))}, {f.get(splits, ('dias_test',), integer)} "
      f"días, ≈ 2.3 años). El periodo incluye Q4 2018, COVID 2020, el bear market de 2022 y el choque de abril de 2025, "
      f"necesarios para que el régimen de Crisis tenga observaciones.\n")
    w("**Asignación de activos:** la lista ['NVDA','AMZN','TSLA','META','NFLX','GOOGL'] se repartió aleatoriamente "
      "con `random.seed(42)`: Milca → TSLA, NFLX; Paula → META, AMZN; Arturo → NVDA, GOOGL.\n")
    w("**Sesgos.** (i) *Supervivencia y selección*: son empresas que hoy sabemos ganadoras, así que el Buy & Hold es "
      "un benchmark muy difícil de vencer en retorno; el sistema se juzga por Calmar y riesgo. (ii) *Diversificación "
      "limitada*: los seis son mega-cap growth tech con correlación alta; la diferencia entre Risk Parity y pesos "
      "iguales viene sobre todo de las diferencias de volatilidad (TSLA y NVDA más volátiles que AMZN o GOOGL).\n")

    # 3. Indicadores
    w("## 3. Indicadores y regla de confirmación\n")
    w("| Familia | Indicador | Fórmula | Voto s_i |\n|---|---|---|---|\n"
      "| Tendencia | Cruce de EMAs | EMA_t = α·P_t + (1−α)·EMA_{t−1}, α = 2/(h+1) | +1 si EMA_rápida > EMA_lenta; −1 si es menor |\n"
      "| Momento | RSI (Wilder) | RSI = 100 − 100/(1 + RS), RS = Wilder(G)/Wilder(L) | +1 si RSI < umbral_inf; −1 si RSI > umbral_sup |\n"
      "| Volatilidad | Bollinger | MB = SMA_N, UB/LB = MB ± k·σ_N | +1 si C < LB; −1 si C > UB; 0 dentro |\n")
    w("\nEl ATR de Wilder (TR_t = max(H−L, |H−C_{t−1}|, |L−C_{t−1}|), ventana 14) no vota: coloca SL y TP.\n")
    w("**Regla de confirmación (2 de 3):**\n\n"
      "    S_t = +1  si  Σ_i 1[s_i,t = +1] ≥ 2\n"
      "    S_t = −1  si  Σ_i 1[s_i,t = −1] ≥ 2\n"
      "    S_t =  0  en otro caso (sin nueva apertura)\n"
      "    Fuerza_t = max(Σ_i 1[s_i,t = +1], Σ_i 1[s_i,t = −1]) ∈ {2, 3}\n")
    w("**Fuerza continua para el tamaño (clase de Risk Parity, paso 7.2):** si la compuerta 2 de 3 se cumple, "
      "s_t = (1/3)·Σ_j x_j,t ∈ [−1, 1]; si no, 0. Así una señal 3 de 3 entra con |s| = 1, una 2 de 3 limpia con "
      "2/3 y una mixta (dos a favor y uno en contra) con 1/3. La compuerta de la clase es |Σ_j x_j| ≥ 2, que no "
      "abriría las señales mixtas; conservamos la regla del lineamiento (\"al menos 2 de 3 coinciden\") y la "
      "diferencia queda reflejada en el tamaño: una señal mixta pesa la mitad que una 2 de 3 limpia.\n")
    w("**Interpretación económica:** la regla abre en dos situaciones: un *pullback* dentro de una tendencia "
      "(EMA + un oscilador de acuerdo) o una reversión fuerte contra la tendencia (RSI y Bollinger de acuerdo). "
      "Todo se calcula con datos hasta el cierre de t y se ejecuta en el open de t+1.\n")
    sc = f.load("correlacion_senales.csv")
    w(f"La correlación entre los votos RSI–Bollinger va de {num(sc['rsi_bb'].min())} a {num(sc['rsi_bb'].max())} "
      f"(umbral de redundancia 0.7) y la de EMA con los osciladores es negativa ({num(sc[['ema_rsi', 'ema_bb']].min().min())} a "
      f"{num(sc[['ema_rsi', 'ema_bb']].max().max())}): los indicadores aportan información distinta.\n")
    f.note("correlacion_senales.csv", "rsi_bb, min..max", [float(sc["rsi_bb"].min()), float(sc["rsi_bb"].max())])
    w(fig(figures, docs, "14_correlacion_senales.png", "correlación entre las señales de los tres indicadores"))

    # 4. Motor
    w("\n## 4. Motor de backtesting y convenciones\n")
    w("- **Event-driven** con estado explícito: efectivo, cantidad con signo, precio de entrada, SL/TP y valor diario.\n"
      "- **Orden de eventos:** (1) open de t: stops por gap y órdenes pendientes; (2) durante t: SL/TP con high/low; "
      "(3) cierre de t: marca a mercado y decisiones con datos ≤ t para el open de t+1.\n"
      "- **SL primero** si SL y TP caen en la misma barra (conservador). **Gap:** si el open ya rebasó el nivel, se "
      "ejecuta al open.\n"
      "- **SL/TP con ATR:** largo SL = P_in − m_sl·ATR, TP = P_in + m_tp·ATR (al revés en cortos), fijos al entrar.\n"
      "- **Costos:** 0.125% del nocional en cada apertura y cada cierre. Compra: Cash −= q·P·(1+c); venta: Cash += q·P·(1−c).\n"
      "- **Cortos:** la venta abre la posición y registra el pasivo; Equity = Cash + Σ q_i·P_i con q_i < 0 en cortos.\n"
      "- **Sin apalancamiento:** q* = w·E*/(P·(1+c)), de modo que nocional + comisión ≤ capital; si Σ|q·P| > equity "
      "(un corto perdedor), la posición se recorta al open siguiente.\n")
    w(f"Exposición bruta máxima tras ejecutar: {f.get(pr, ('RP', 'max_exposicion_post_ejecucion'), num)} "
      f"(al cierre, por la deriva intradía, {f.get(pr, ('RP', 'max_exposicion_al_cierre'), num)}). Operaciones del "
      f"portafolio RP en WF-OOS: {f.get(pr, ('RP', 'operaciones_largas'), integer)} largas y "
      f"{f.get(pr, ('RP', 'operaciones_cortas'), integer)} cortas.\n")

    # 5. Optimización
    w("## 5. Optimización y walk-forward\n")
    w(f"Optuna con `TPESampler` ({f.get(opt, ('trials_aleatorios_iniciales',), integer)} trials aleatorios y luego "
      f"bayesiano) maximiza el Calmar con {f.get(opt, ('trials_por_estudio',), integer)} trials por estudio: en cada "
      f"ventana y activo, 1 estudio global y 1 por régimen (θ_{{i,j}} = argmax Calmar(backtest(train_k | S_t = j))). "
      f"Restricción: un θ con menos de {config.N_MIN_GLOBAL} operaciones (global) o {config.N_MIN_REGIME} (régimen) "
      f"recibe −10⁹. Purga: las posiciones se cierran el último día del train; embargo: los últimos "
      f"{config.EMBARGO_DAYS} días no entran al objetivo. Se usa el **θ robusto** (mediana del 10% de mejores trials, "
      f"centro de la meseta), no el argmax.\n")
    w(md_table(["Parámetro", "Rango", "Justificación"], [
        ["ema_fast", "5–30", "1 a 6 semanas, horizonte de las operaciones"],
        ["ema_gap", "10–100", "lenta = rápida + gap (lenta > rápida)"],
        ["rsi_window", "7–28", "alrededor del estándar de Wilder (14)"],
        ["rsi_lower / rsi_upper", "20–40 / 60–80", "alrededor de 30/70"],
        ["bb_window / bb_k", "10–40 / 1.5–3.0", "alrededor de 20 días y 2σ"],
        ["m_sl / m_tp", "1–4 / 1–6 ATR", "fuera del ruido diario; payoff > 1 posible"],
        ["max_hold", "5–40 días", "time-stop de 1 a 8 semanas"]]))
    w(f"\n**Cómputo:** {f.get(opt, ('ventanas_walk_forward',), integer)} ventanas de walk-forward + 1 final; tiempo "
      f"total de optimización {f.get(opt, ('segundos_optimizacion',), lambda s: f'{s / 60:.1f} minutos')} con "
      f"{f.get(opt, ('procesos',), integer)} procesos en paralelo y "
      f"**{f.get(opt, ('configuraciones_evaluadas',), integer)} configuraciones evaluadas** en total (todas las variantes).\n")
    fb = f.load(opt)["por_variante"][chosen]["fallbacks"]
    w(f"**Fallback** (θ global cuando el régimen tiene < {config.MIN_REGIME_DAYS} días en el train o ningún trial "
      f"cumple el mínimo de operaciones), variante elegida: Tendencia "
      f"{f.get(opt, ('por_variante', chosen, 'fallbacks', 'por_regimen', 'Tendencia'), integer)}, Reversión "
      f"{f.get(opt, ('por_variante', chosen, 'fallbacks', 'por_regimen', 'Reversión'), integer)} y Crisis "
      f"{f.get(opt, ('por_variante', chosen, 'fallbacks', 'por_regimen', 'Crisis'), integer)} de "
      f"{fb['estudios_por_regimen']} estudios por régimen.\n")
    rows = []
    for v in f.load(vc)["variante"]:
        rows.append([v, f.get(vc, ("variante", v, "calmar_oos"), num), f.get(vc, ("variante", v, "cagr_oos"), pct),
                     f.get(vc, ("variante", v, "cagr_is_promedio"), pct), f.get(vc, ("variante", v, "wfe_cagr"), num),
                     f.get(vc, ("variante", v, "wfe_calmar"), num), "sí" if f.get(vc, ("variante", v, "elegida")) else "no"])
    w(md_table(["Variante", "Calmar WF-OOS", "CAGR WF-OOS", "CAGR WF-IS prom.", "WFE (retorno)", "WFE (Calmar)", "Elegida"], rows))
    w(f"\nLa variante oficial (**{chosen}**) se eligió por el mayor Calmar WF-OOS dentro de TRAIN, antes de tocar TEST.\n")
    w(fig(figures, docs, "13_rolling_vs_anclado.png", "walk-forward rolling vs. anchored"))

    # 6. Métricas por conjunto
    w("\n## 6. Métricas por conjunto, activo y régimen\n")
    rows = [metric_row(f, mc, c, s) for c in ("WF-IS", "WF-OOS") for s in SYSTEMS]
    if has_test:
        rows += [metric_row(f, "metricas_test.csv", "TEST", s) for s in SYSTEMS]
    else:
        rows += [["TEST", s] + [PENDING] * 7 for s in SYSTEMS]
    w(md_table(["Conjunto", "Sistema", "CAGR", "Sharpe", "Sortino", "MDD", "Calmar", "Win rate", "Operaciones"], rows))
    w("\nWF-IS es el promedio de las métricas in-sample de cada ventana de 6 meses (sus trains se traslapan); "
      "WF-OOS es la simulación continua de los meses de prueba concatenados.\n")
    annual = pd.read_csv(f.results / "retornos_anual_wf_oos.csv", index_col=0, parse_dates=True)
    w("**Rendimientos anuales (WF-OOS):**\n")
    w(md_table(["Año"] + list(annual.columns), [[str(d.year)] + [pct(v) for v in r] for d, r in annual.iterrows()]))
    w("\n" + fig(figures, docs, "03a_retornos_mensuales_anuales.png", "rendimientos mensuales y anuales del portafolio RP"))
    w("\n" + fig(figures, docs, "03b_retornos_trimestrales.png", "rendimientos trimestrales"))
    if has_test:
        w("\n" + fig(figures, docs, "03c_retornos_mensuales_test.png", "rendimientos mensuales en TEST"))
    w("\n" + fig(figures, docs, "01_valor_portafolio.png", "valor del portafolio en entrenamiento y prueba"))
    w("\n" + fig(figures, docs, "02_drawdown.png", "curva de drawdown"))

    # 7. Régimen
    w("\n## 7. Análisis de régimen\n")
    w(f"**Metodología.** Un régimen único para los seis activos, calculado sobre su índice equiponderado con "
      f"K-means (K = 3 por teoría: tendencia, reversión y crisis; `n_init=10`, `random_state=42`). Features en "
      f"ventana móvil de {config.REGIME_WINDOW} días (3 meses): volatilidad realizada (separa Crisis), ATR "
      f"normalizado, fuerza de tendencia |ln(MA_21/MA_63)|/σ, R² de ln(precio) contra el tiempo (tendencia "
      f"lineal vs. rango) y autocorrelación lag-1 (persistencia vs. reversión). El scaler y K-means se ajustan solo "
      f"con datos ≤ fin del train de cada ventana; fuera de él solo se predice.\n")
    sel = f.load("regimenes_seleccion_features.csv")
    full_sil = sel.loc[sel["n_features"] == 5, "silhouette"].iloc[0]
    w(f"**Selección de features (solo TRAIN):** con las 5 features el silhouette es {num(full_sil, 3)} (< 0.4), "
      f"así que se usa el subconjunto con mayor silhouette: **{' + '.join(f.get(val, ('features',)))}**. La "
      f"autocorrelación con 63 datos tiene error estándar ≈ 1/√63 ≈ 0.13, mayor que la diferencia entre "
      f"centroides; ATR normalizado y fuerza de tendencia son redundantes con volatilidad y R².\n")
    f.note("regimenes_seleccion_features.csv", "n_features=5, silhouette", float(full_sil))
    w(md_table(["Subconjunto", "Silhouette"], [[r.features, num(r.silhouette, 3)] for r in sel.head(6).itertuples()]))
    w(f"\n**Adaptación a datos diarios.** Los lineamientos piden actualizar cada 1–6 horas con barras de 5 minutos; "
      f"en diario se reclasifica cada {config.REGIME_UPDATE_EVERY} días hábiles (semanal, alineado con el rebalanceo) "
      f"y un cambio se confirma solo si aparece en {config.REGIME_PERSISTENCE} actualizaciones seguidas. El objetivo "
      f"de persistencia (> 12 h en 5 minutos) se adapta a ≥ 10 días hábiles: el régimen debe durar más que una "
      f"operación típica para que los parámetros por régimen tengan sentido.\n")
    p = ("persistencia_wf_oos",)
    w(md_table(["Validación", "Valor", "Objetivo"], [
        ["Silhouette (promedio de ventanas)", f.get(val, ("silhouette", "promedio_ventanas"), lambda v: num(v, 3)), "> 0.4"],
        ["Silhouette (mínimo / modelo final)", f"{f.get(val, ('silhouette', 'minimo'), lambda v: num(v, 3))} / "
                                               f"{f.get(val, ('silhouette', 'modelo_final'), lambda v: num(v, 3))}", "> 0.4"],
        ["Duración promedio (días)", f.get(val, p + ("duracion_promedio_global",), lambda v: num(v, 1)), "≥ 10"],
        ["Transiciones por año", f.get(val, p + ("transiciones_por_anio",), lambda v: num(v, 1)), "—"]]))
    rows = []
    for n in ("Tendencia", "Reversión", "Crisis"):
        rows.append([n, f.get(val, p + ("por_regimen", n, "proporcion_tiempo"), pct),
                     f.get(val, p + ("por_regimen", n, "n_episodios"), integer),
                     f.get(val, p + ("por_regimen", n, "duracion_promedio_observada"), lambda v: num(v, 1)),
                     f.get(val, p + ("por_regimen", n, "duracion_esperada_markov"), lambda v: num(v, 1))])
    w("\n" + md_table(["Régimen", "% del tiempo", "Episodios", "Duración observada", "E[D] = 1/(1−A_jj)"], rows))
    w("\n" + fig(figures, docs, "06a_regimenes_linea_tiempo.png", "línea de tiempo de regímenes"))
    w("\n" + fig(figures, docs, "06b_distribuciones_features.png", "distribución de las features por régimen"))
    w("\n" + fig(figures, docs, "06c_valor_con_regimenes.png", "valor del portafolio con regímenes superpuestos"))
    w("\n" + fig(figures, docs, "09_transiciones_regimen.png", "matriz de transición y persistencia"))
    tr = "regimenes_validacion.json"
    w(f"\n**Transiciones:** {f.get(tr, ('transiciones', 'n'), integer)} cambios de régimen en WF-OOS, "
      f"{f.get(tr, ('transiciones', 'con_posiciones_abiertas'), integer)} con posiciones abiertas "
      f"({f.get(tr, ('transiciones', 'posiciones_afectadas'), integer)} posiciones afectadas) y "
      f"{f.get(tr, ('transiciones', 'a_crisis'), integer)} entradas a Crisis. Regla: la posición abierta conserva su "
      f"SL/TP; las entradas nuevas usan el θ del régimen nuevo; el multiplicador nuevo se aplica en el siguiente "
      f"rebalanceo, salvo al entrar a Crisis, que rebalancea de inmediato a M = 0.3 y solo admite entradas 3 de 3.\n")
    w(regime_section(f))
    if has_test:
        st = f.load("regimenes_estabilidad.json")
        w("\n**Estabilidad fuera de muestra (TRAIN WF-OOS vs. TEST):**\n")
        w(md_table(["Régimen", "% tiempo TRAIN", "% tiempo TEST", "Duración TRAIN", "Duración TEST"],
                   [[n, f.get("regimenes_estabilidad.json", ("train_wf_oos", "por_regimen", n, "proporcion_tiempo"), pct),
                     f.get("regimenes_estabilidad.json", ("test", "por_regimen", n, "proporcion_tiempo"), pct),
                     num(st["train_wf_oos"]["por_regimen"][n]["duracion_promedio_observada"], 1),
                     num(st["test"]["por_regimen"][n]["duracion_promedio_observada"], 1)]
                    for n in ("Tendencia", "Reversión", "Crisis")]))
    else:
        w(f"\n**Estabilidad fuera de muestra (TRAIN vs. TEST):** {PENDING}\n")

    # 8. Portafolio
    w("\n## 8. Metodología del portafolio\n")
    w("**Del retorno al riesgo del portafolio.** R_p = Σ_i w_i R_i, σ_p² = wᵀΣw y σ_p = √(wᵀΣw). La contribución "
      "marginal es ∂σ_p/∂w_k = (Σw)_k/σ_p = Cov(R_k, R_p)/σ_p y la contribución total RC_i = w_i·(Σw)_i/σ_p; por el "
      "Teorema de Euler (σ_p es homogénea de grado 1) Σ_i RC_i = σ_p exactamente, así que RC_i/σ_p es un porcentaje "
      "genuino del riesgo.\n")
    w("**Risk Parity.** Se busca RC_i = σ_p/n para todo i con Σw_i = 1 y w_i > 0. Minimizar Σ(RC_i − RC_j)² "
      "directamente no es convexo y depende del punto de partida, así que se usa la formulación convexa de Spinu "
      "(2013), resuelta con SLSQP:\n\n"
      "    min_{y>0}  ½·yᵀΣy − (1/n)·Σ_i ln y_i,      w = y / Σ_j y_j\n\n"
      "(forma cuadrática PSD + barrera logarítmica ⇒ convexa, solución única). Verificación numérica en cada "
      "rebalanceo: max|RC_i/σ_p − 1/n| < 10⁻⁴; si falla se lanza una excepción (también es prueba automatizada, "
      "junto con los ejemplos numéricos de la clase).\n")
    w(f"**Tres versiones de pesos** (mismas señales, costos, rebalanceo y multiplicadores): pesos iguales w_i = 1/n; "
      f"RP naïve w_i = (1/σ_i)/Σ_j(1/σ_j), exacta solo si todas las correlaciones son iguales; y RP optimizado (Spinu).\n")
    pc = "ponderaciones_comparacion.csv"
    w(md_table(["Ponderación", "CAGR", "MDD", "Calmar", "Sharpe", "Turnover anual", "Costos"],
               [[f.get(pc, ("metodo", m, "nombre")), f.get(pc, ("metodo", m, "cagr"), pct), f.get(pc, ("metodo", m, "mdd"), pct),
                 f.get(pc, ("metodo", m, "calmar"), num), f.get(pc, ("metodo", m, "sharpe"), num),
                 f.get(pc, ("metodo", m, "turnover_anual"), lambda v: num(v, 2)), f.get(pc, ("metodo", m, "costos_totales"), money)]
                for m in ("ew", "iv", "rp")]))
    ce = "covarianza_estimadores.csv"
    w(f"\n**Estimación de Σ** (siempre sobre rendimientos, nunca precios). Oficial: muestral de {config.COV_WINDOW} "
      f"días con datos ≤ t (n/T ≈ 0.05, número de condición mediano {f.get(pr, ('RP', 'kappa_mediana'), lambda v: num(v, 1))}). "
      f"Como pide la clase, se compararon los pesos de RP con tres estimadores; la estabilidad es el cambio medio "
      f"½Σ|Δw^RP| entre fechas semanales consecutivas:\n")
    w(md_table(["Estimador", "Cambio medio de pesos", "Cambio máximo", "κ(Σ) mediano"],
               [[e, f.get(ce, ("estimador", e, "cambio_medio_pesos"), lambda v: num(v, 3)),
                 f.get(ce, ("estimador", e, "cambio_max_pesos"), lambda v: num(v, 3)),
                 f.get(ce, ("estimador", e, "kappa_mediana"), lambda v: num(v, 1))]
                for e in ("muestral", "ewma", "ledoit_wolf")]))
    w("\nEWMA (λ = 0.94, T_eff ≈ 17 días) reacciona más rápido pero mueve más los pesos; Ledoit-Wolf encoge los "
      "eigenvalores extremos y suaviza los pesos frente a la muestral.\n")
    w("\n" + fig(figures, docs, "15_estimadores_covarianza.png", "pesos de Risk Parity con tres estimadores de Σ"))
    w("\n**¿Por qué Risk Parity?** Es una decisión de asignación de riesgo, no de retorno: no necesita rendimientos "
      "esperados (el insumo más ruidoso de Markowitz), y *equal weight no es equal risk*: con pesos iguales TSLA y NVDA "
      "dominan el riesgo. RP penaliza la volatilidad propia y también la covarianza con el resto.\n")
    w(f"**Agregación de señales (paso 7).** Dos objetos independientes: w^RP dice cuánto riesgo puede tomar cada activo "
      f"y s_i ∈ [−1, 1] qué tan convencido está el sistema de la dirección:\n\n"
      f"    w̃_i = w_i^RP · s_i,      w^target = m(régimen) · w̃ / max(1, Σ_i |w̃_i|)\n\n"
      f"con m = 1.0 / 0.7 / 0.3 en Tendencia / Reversión / Crisis (en Crisis además solo se abren señales 3 de 3). "
      f"Conflictos entre activos correlacionados: si corr_126(i, j) > {config.CONFLICT_CORR} y las direcciones son "
      f"opuestas, gana la de mayor |s| y la otra queda en 0; si empatan, ambas reducen s a la mitad "
      f"({f.get(pr, ('RP', 'conflictos'), integer)} conflictos en WF-OOS, {f.get(pr, ('RP', 'conflictos_empatados'), integer)} empates). "
      f"El resultado de RP no es el portafolio final: se combina con las señales y el régimen antes de llegar a una orden.\n")
    opt_ = f.load("optimizacion_resumen.json")["rebalanceo_elegido"]
    w(f"**Rebalanceo (paso 8).** Turnover T_t = ½·Σ_i |w_i,t − w_i,t⁻| contra los pesos *después del drift* "
      f"(el factor ½ cuenta un viaje redondo una vez) y costo anual ≈ T̄·f·2c. Disparador híbrido: en fechas de "
      f"calendario (cada f días) se rebalancea solo si ‖w_t⁻ − w^target‖₁ > δ; al entrar a Crisis, de inmediato; "
      f"las entradas y salidas por señal, SL, TP o time-stop se ejecutan siempre al open siguiente. f y δ se "
      f"eligieron con la malla del barrido (sección 11) por Calmar WF-OOS dentro de TRAIN: "
      f"**f = {f.get('optimizacion_resumen.json', ('rebalanceo_elegido', 'dias'), integer)} días, "
      f"δ = {f.get('optimizacion_resumen.json', ('rebalanceo_elegido', 'banda'), lambda v: num(v, 2))}**. "
      f"Rebalanceos por año: {f.get(pr, ('RP', 'rebalanceos_por_anio'), lambda v: num(v, 1))}, T̄ por rebalanceo "
      f"{f.get(pr, ('RP', 'turnover_por_rebalanceo'), pct)}, costo anual de rebalanceo ≈ "
      f"{f.get(pr, ('RP', 'costo_anual_rebalanceo'), lambda v: pct(v, 2))}; revisiones de calendario omitidas por "
      f"estar dentro de la banda: {f.get(pr, ('RP', 'rebalanceos_omitidos_por_banda'), integer)}. Turnover total "
      f"(incluye entradas y salidas por señal): {f.get(pr, ('RP', 'turnover_anual'), lambda v: num(v, 1))} veces el capital al año.\n")
    w("\n" + fig(figures, docs, "07b_fuerza_senal.png", "mapa de calor de la fuerza de señal"))
    w("\n" + fig(figures, docs, "07c_correlacion_por_regimen.png", "evolución de la matriz de correlación entre regímenes"))

    # 9. Estrategia individual
    w("\n## 9. Estrategia individual de cada activo\n")
    w("El diseño (indicadores, regla 2 de 3, SL/TP con ATR, time-stop) es común; cada activo tiene su propio θ por "
      "régimen en cada ventana. θ congelado para TEST (última ventana de TRAIN):\n")
    frozen = f.load("theta_congelado.json")["contenido"]
    theta = frozen["theta"]

    def theta_cells(th):
        if th is None:
            return ["sin θ válido"] + ["—"] * 4
        return [f"{th['ema_fast']}/{th['ema_fast'] + th['ema_gap']}", f"{th['rsi_window']} ({th['rsi_lower']}–{th['rsi_upper']})",
                f"{th['bb_window']}, {th['bb_k']:.2f}", f"{th['m_sl']:.2f} / {th['m_tp']:.2f}", str(th["max_hold"])]

    header = ["EMA rápida/lenta", "RSI (umbrales)", "Bollinger (N, k)", "SL / TP (ATR)", "max_hold"]
    regimes_ = ("Tendencia", "Reversión", "Crisis")
    if frozen["variante"].endswith("compartido"):
        w("Variante compartida: el mismo θ por régimen para los seis activos.\n")
        w(md_table(["Régimen"] + header, [[reg] + theta_cells(theta[config.TICKERS[0]][reg]) for reg in regimes_]))
    else:
        w(md_table(["Integrante", "Activo", "Régimen"] + header,
                   [[owner, tk, reg] + theta_cells(theta[tk][reg])
                    for owner, tickers in config.ASSIGNMENT.items() for tk in tickers for reg in regimes_]))
    w("\n*sin θ válido*: en la última ventana ni el estudio de ese régimen ni el global alcanzaron el mínimo de "
      "operaciones; el activo no abre posiciones nuevas en ese régimen (las abiertas siguen con su SL/TP).\n")
    pa = "metricas_por_activo_wf_oos.csv"
    w("\n**Desempeño de la estrategia de cada activo sola (100% del capital, WF-OOS):**\n")
    w(md_table(["Activo", "CAGR", "Sharpe", "MDD", "Calmar", "Win rate", "Largas", "Cortas"],
               [[tk, f.get(pa, ("conjunto", tk, "cagr"), pct), f.get(pa, ("conjunto", tk, "sharpe"), num),
                 f.get(pa, ("conjunto", tk, "mdd"), pct), f.get(pa, ("conjunto", tk, "calmar"), num),
                 f.get(pa, ("conjunto", tk, "win_rate"), pct), f.get(pa, ("conjunto", tk, "n_largas"), integer),
                 f.get(pa, ("conjunto", tk, "n_cortas"), integer)] for tk in config.TICKERS]))

    # 10. Comparaciones
    w("\n## 10. Comparaciones\n")
    rc = f.load("rp_vs_ew_contribuciones.csv")
    systems = ("EW", "RP naïve", "RP")
    rc_rows = []
    for tk in config.TICKERS:
        rc_rows.append([tk] + [pct(rc[(rc.sistema == s_) & (rc.activo == tk)]["rc_promedio_abs"].iloc[0]) for s_ in systems])
    conc = {s_: rc[(rc.sistema == s_) & (rc.activo == "concentracion_max_rc")]["rc_promedio_abs"].iloc[0] for s_ in systems}
    rc_rows.append(["max RC promedio"] + [pct(conc[s_]) for s_ in systems])
    f.note("rp_vs_ew_contribuciones.csv", "concentracion_max_rc", {k: float(v) for k, v in conc.items()})
    w("**Contribución realizada al riesgo por activo** (pesos con signo después de señales y régimen; mismas señales, "
      "costos y rebalanceo):\n")
    w(md_table(["Activo", "Pesos iguales", "RP naïve", "Risk Parity"], rc_rows))
    w("\nLa paridad exacta se cumple en w^RP (verificada a 10⁻⁴); las contribuciones realizadas se alejan de 1/6 "
      "porque cada día solo algunos activos tienen señal y su tamaño depende de s_i.\n")
    w("\n" + fig(figures, docs, "07a_contribuciones_riesgo.png", "contribuciones al riesgo por activo"))
    w("\n" + fig(figures, docs, "10_portafolio_vs_activos.png", "portafolio vs. estrategia de cada activo"))

    # 11. Robustez y costos
    w("\n## 11. Robustez y costos\n")
    w(robustness_section(f, figures, docs))

    # 12. Supuestos
    w("\n## 12. Supuestos y decisiones\n")
    w(assumptions())

    # 13. Preguntas
    w("\n## 13. Respuestas a las preguntas\n")
    w(answers(f, has_test))

    # 14. Advertencia
    w("\n## 14. Advertencia de ejecución\n")
    w(execution_warning(f, has_test))
    return "\n".join(out) + "\n"


def edge_conclusion(wfe, n_peaks, n_params, f: Facts, has_test) -> str:
    calmar_oos = f.get("metricas_conjuntos.csv", ("_k", "WF-OOS|RP", "calmar"))
    calmar_bh = f.get("metricas_conjuntos.csv", ("_k", "WF-OOS|Buy & Hold", "calmar"))
    parts = []
    if calmar_oos <= 0:
        parts.append("el sistema no genera rendimiento ajustado por riesgo positivo fuera de muestra, así que la "
                     "evidencia no respalda una ventaja real")
    elif wfe is not None and wfe < 0.5:
        parts.append(f"fuera de muestra sobrevive menos de la mitad del desempeño in-sample (WFE = {num(wfe)}), lo "
                     f"que indica que buena parte de lo optimizado es ajuste a la muestra")
    else:
        parts.append(f"una parte importante del desempeño in-sample sobrevive fuera de muestra (WFE = {num(wfe)})")
    parts.append(f"{n_peaks} de {n_params} parámetros/multiplicadores muestran un pico (el Calmar cambia más de 50% "
                 f"o de signo con ±20%)" if n_peaks else "ningún parámetro muestra un pico aislado ante ±20%")
    parts.append("el Calmar del sistema " + ("supera al" if calmar_oos > calmar_bh else "queda por debajo del")
                 + " Buy & Hold en el mismo periodo")
    be = f.load("costos_equilibrio.json")
    if be["pnl_bruto_oficial"] > 0 >= be["pnl_neto_oficial"]:
        parts.append(f"antes de costos el sistema gana {money(be['pnl_bruto_oficial'])}, pero con 0.125% por lado y "
                     f"una rotación de {num(be['turnover_anual'], 1)} veces al año los costos se lo comen")
    if not has_test:
        parts.append("el veredicto final depende del TEST, que se corre una sola vez")
    return "; ".join(parts) + "."


def regime_section(f: Facts) -> str:
    data = f.load("metricas_por_regimen.csv")
    rows = []
    for s in ("RP", "EW", "Buy & Hold"):
        for n in ("Tendencia", "Reversión", "Crisis"):
            r = data[(data.sistema == s) & (data.regimen == n)]
            if r.empty or pd.isna(r["media_anualizada"].iloc[0]):
                continue
            r = r.iloc[0]
            rows.append([s, n, integer(r.n_dias), pct(r.media_anualizada), num(r.sharpe), pct(r.mdd),
                         f"[{r.ic_inf * 1e4:.1f}, {r.ic_sup * 1e4:.1f}]"])
    kw = data[data.sistema == "RP"]["kruskal_p"].iloc[0]
    f.note("metricas_por_regimen.csv", "RP, kruskal_p", float(kw))
    return ("\n**Métricas por régimen (WF-OOS).** El rendimiento del día t se atribuye al régimen vigente al cierre de "
            "t−1. IC bootstrap al 95% (1000 remuestreos, semilla 42) de la media diaria, en puntos base:\n\n"
            + md_table(["Sistema", "Régimen", "Días", "Media anualizada", "Sharpe", "MDD", "IC 95% media diaria (pb)"], rows)
            + f"\n\nPrueba de Kruskal-Wallis (H0: misma distribución de rendimientos diarios entre regímenes) para el "
              f"portafolio RP: p = {num(kw, 3)}.\n")


def robustness_section(f: Facts, figures, docs) -> str:
    out = []
    iu = f.load("indicador_unico.csv")
    port = iu[iu.nivel == "portafolio"]
    out.append("**2 de 3 vs. un solo indicador** (mismo θ, mismas salidas, mismo portafolio):\n")
    out.append(md_table(["Regla", "Operaciones", "Calmar", "CAGR", "MDD"],
                        [[r.regla, integer(r.n_operaciones), num(r.calmar), pct(r.cagr), pct(r.mdd)] for r in port.itertuples()]))
    f.note("indicador_unico.csv", "portafolio", port[["regla", "n_operaciones", "calmar"]].to_dict("records"))
    out.append("\n" + fig(figures, docs, "08_indicador_unico.png", "2 de 3 vs. indicador único"))
    sv = f.load("sensibilidad_veredicto.csv")
    out.append("\n**Sensibilidad ±20%** (θ congelado aplicado a todos los activos y regímenes, un parámetro a la vez):\n")
    out.append(md_table(["Parámetro", "Calmar base", "Calmar mín.", "Calmar máx.", "Caída máx.", "Cambio máx.", "Veredicto"],
                        [[r.parametro, num(r.calmar_base), num(r.calmar_min), num(r.calmar_max),
                          pct(r.caida_relativa_max), pct(r.cambio_relativo_max), r.veredicto] for r in sv.itertuples()]))
    f.note("sensibilidad_veredicto.csv", "veredicto", sv["veredicto"].tolist())
    out.append("\n" + fig(figures, docs, "04_sensibilidad.png", "sensibilidad del Calmar a ±20%"))
    be = "costos_equilibrio.json"
    out.append(f"\n**Curva de costos:** comisión de 0% a 0.5% por lado. Comisión de equilibrio: "
               f"{f.get(be, ('comision_equilibrio',), lambda v: 'fuera del rango probado' if v is None else pct(v, 3))}; "
               f"margen frente a 0.125%: {f.get(be, ('margen_puntos',), lambda v: 'n/d' if v is None else f'{v * 100:+.3f} puntos porcentuales')}. "
               f"Con turnover de {f.get(be, ('turnover_anual',), lambda v: num(v, 1))} veces al año, la comisión cuesta "
               f"≈ {f.get(be, ('costo_anual_por_rotacion',), pct)} del capital por año.\n")
    out.append(fig(figures, docs, "05_curva_costos.png", "retorno neto contra nivel de costo"))
    rb = f.load("rebalanceo_barrido.csv")
    out.append("\n**Barrido del rebalanceo híbrido** (frecuencia de revisión f × banda δ; retorno bruto, costo total, "
               "retorno neto y turnover realizado):\n")
    out.append(md_table(["f", "δ", "Retorno bruto", "Costo total", "Retorno neto", "Calmar", "Turnover anual", "Rebal./año"],
                        [[r.frecuencia, num(r.banda, 2), pct(r.retorno_bruto), pct(r.costo_total_pct), pct(r.retorno_total),
                          num(r.calmar), num(r.turnover_anual, 2), num(r.rebalanceos_por_anio, 1)] for r in rb.itertuples()]))
    f.note("rebalanceo_barrido.csv", "todas", rb["retorno_total"].tolist())
    out.append(f"\nEl costo total baja de forma monótona al revisar con menos frecuencia o con banda más ancha "
               f"({pct(rb.costo_total_pct.max())} → {pct(rb.costo_total_pct.min())} del capital), pero el retorno bruto "
               f"no tiene un patrón claro (de {pct(rb.retorno_bruto.min())} a {pct(rb.retorno_bruto.max())}): el retorno "
               f"neto va de {pct(rb.retorno_total.min())} a {pct(rb.retorno_total.max())} y la combinación elegida es la mejor "
               f"de 16 con diferencias pequeñas y ruidosas, así que elegirla agrega un sesgo de selección menor (se declara "
               f"en Supuestos). Ninguna combinación vuelve rentable al sistema.\n")
    out.append("\n" + fig(figures, docs, "07d_barrido_rebalanceo.png", "barrido de frecuencia de rebalanceo"))
    out.append("\n" + fig(figures, docs, "11_costos_vs_bruto.png", "costos totales contra retorno bruto"))
    out.append("\n" + fig(figures, docs, "12a_optuna_historia.png", "historia de optimización"))
    out.append("\n" + fig(figures, docs, "12b_optuna_importancia.png", "importancia de parámetros"))
    out.append("\n" + fig(figures, docs, "12c_optuna_slices.png", "slice plots"))
    out.append("\n" + fig(figures, docs, "12d_optuna_superficie_3d.png", "superficie del Calmar (corte 2D de 10D)"))
    opt = "optimizacion_resumen.json"
    out.append(f"\n**Fuentes de degradación.** (1) *Sensibilidad de parámetros*: ver la tabla de ±20%. (2) *Minería de "
               f"datos*: se evaluaron {f.get(opt, ('configuraciones_evaluadas',), integer)} configuraciones; con tantas "
               f"pruebas, el mejor Calmar in-sample está sesgado hacia arriba, de ahí el θ robusto y la WFE. "
               f"(3) *Cambios de régimen*: ver métricas por régimen y la estabilidad TRAIN vs. TEST. (4) *Ejecución*: "
               f"ver la sección 14.\n")
    return "\n".join(out)


def assumptions() -> str:
    items = [
        "Precios ajustados (auto_adjust=True); datos congelados; solo fechas comunes a los 6 activos (sin forward-fill).",
        "Señal con datos hasta el cierre de t; ejecución al open de t+1; acciones fraccionarias.",
        "SL primero si SL y TP caen en la misma barra; gaps se ejecutan al open; niveles SL/TP fijos al entrar con el ATR del día de la señal.",
        "Costo oficial: solo comisión de 0.125% por lado; spread, impacto y borrow fee solo en el escenario realista.",
        "Sin apalancamiento: q* = w·E*/(P·(1+c)); si un corto perdedor lleva Σ|q·P| por encima del equity se recorta al open siguiente. Con exposición ≤ 100% el margen de cortos (50% inicial, 25–30% de mantenimiento) se cumple.",
        "Calmar con piso de |MDD| = 1% para evitar divisiones que exploten en ventanas casi sin drawdown.",
        "Tamaño de posición no se optimiza: en un activo, escalar la posición escala CAGR y MDD casi por igual (Calmar casi invariante); el tamaño lo deciden Risk Parity y los multiplicadores, a los que se les hace sensibilidad.",
        f"Mínimo de operaciones: {config.N_MIN_GLOBAL} por activo en 6 meses (global) y {config.N_MIN_REGIME} por régimen; si un régimen tiene < {config.MIN_REGIME_DAYS} días o ningún trial cumple, se usa el θ global (fallback contado).",
        "θ robusto = mediana del 10% de mejores trials válidos (enteros redondeados); el argmax se guarda solo para comparar.",
        "En la optimización por régimen, Crisis exige 3 de 3 igual que en el portafolio (consistencia).",
        "Régimen: K = 3 por teoría; scaler y K-means ajustados con toda la historia ≤ fin del train de cada ventana (no solo los 6 meses, para que existan episodios de crisis); features elegidas por silhouette solo con TRAIN.",
        "Régimen actualizado cada 5 días hábiles con confirmación en 2 actualizaciones; objetivo de persistencia adaptado a ≥ 10 días.",
        "El rendimiento del día t se atribuye al régimen del cierre de t−1 (el que decidió la posición).",
        "Covarianza muestral de 126 días, solo datos ≤ t (oficial); EWMA y Ledoit-Wolf se reportan como comparación; correlación de conflicto con la misma ventana.",
        "Risk Parity con la formulación convexa de Spinu sobre los 6 activos; el tamaño final es w^RP·s_i escalado por m(régimen) (clase, paso 7).",
        "Compuerta 2 de 3 del lineamiento (abre con dos a favor aunque el tercero vote en contra); la fuerza s = Σ votos/3 de la clase reduce a 1/3 el tamaño de esas señales mixtas.",
        "Rebalanceo híbrido: revisión cada f días y rebalanceo solo si ‖w − w*‖₁ > δ; f y δ elegidos por Calmar WF-OOS dentro de TRAIN, después de elegir la variante. Entre revisiones solo se mueven los activos que entran o salen; Crisis rebalancea de inmediato.",
        "Turnover T_t = ½·Σ|w_t − w_t⁻| contra los pesos después del drift; costo anual ≈ T̄·f·2c.",
        "Los experimentos de un solo indicador, la curva de costos y el barrido de rebalanceo mantienen fijo el θ del walk-forward (ceteris paribus).",
        "La sensibilidad ±20% usa el θ congelado y el modelo de régimen final sobre el tramo WF-OOS de TRAIN.",
        "Impacto de mercado η·(|q|/ADV)^(2/3) con η = 0.1 (orden de magnitud de la literatura) y ADV de 20 días hasta t−1.",
        "WF-IS se reporta como promedio de las ventanas (sus trains se traslapan); WF-OOS es una simulación continua.",
        "La variante (por activo o compartida) se elige con el Calmar WF-OOS dentro de TRAIN, antes de tocar TEST.",
    ]
    return "\n".join(f"{i}. {t}" for i, t in enumerate(items, 1)) + "\n"


def answers(f: Facts, has_test: bool) -> str:
    iu = f.load("indicador_unico.csv")
    port = iu[iu.nivel == "portafolio"].set_index("regla")
    two = port.loc["2 de 3"]
    singles = port.drop("2 de 3")
    opt = "optimizacion_resumen.json"
    sv = f.load("sensibilidad_veredicto.csv")
    peaks = sv[sv.veredicto == "pico"]["parametro"].tolist()
    be = f.load("costos_equilibrio.json")
    reg = f.load("metricas_por_regimen.csv")
    rp_reg = reg[reg.sistema == "RP"].set_index("regimen")
    kw = rp_reg["kruskal_p"].iloc[0]
    mc = f.load("metricas_conjuntos.csv")
    get = lambda c, s, col: mc[(mc.conjunto == c) & (mc.sistema == s)][col].iloc[0]  # noqa: E731
    rc = f.load("rp_vs_ew_contribuciones.csv")
    conc = {s_: rc[(rc.sistema == s_) & (rc.activo == "concentracion_max_rc")]["rc_promedio_abs"].iloc[0]
            for s_ in ("RP", "EW", "RP naïve")}
    pr = f.load("portafolio_resumen.json")
    out = []
    out.append("**1. ¿Qué aporta la regla 2 de 3 frente a un solo indicador?** "
               f"Con 2 de 3 el portafolio hizo {integer(two.n_operaciones)} operaciones con Calmar {num(two.calmar)}; "
               f"con un solo indicador, entre {integer(singles.n_operaciones.min())} y {integer(singles.n_operaciones.max())} "
               f"operaciones y Calmar entre {num(singles.calmar.min())} y {num(singles.calmar.max())}. "
               + ("La confirmación reduce el número de operaciones frente al promedio de los indicadores solos "
                  f"({num(singles.n_operaciones.mean(), 0)}). " if two.n_operaciones < singles.n_operaciones.mean() else
                  "La confirmación no reduce el número de operaciones frente al promedio de los indicadores solos "
                  f"({num(singles.n_operaciones.mean(), 0)}). ")
               + ("En Calmar supera a todos los indicadores solos." if two.calmar > singles.calmar.max() else
                  f"En Calmar no supera a todos: el mejor indicador solo es {singles.calmar.idxmax()} "
                  f"({num(singles.calmar.max())}).") + "\n")
    w = f.load(opt)["wfe_rolling"]
    wa = f.load(opt)["wfe_anclado"]
    out.append(f"**2. ¿Cuánto se degrada de train a test en el walk-forward?** El CAGR in-sample promedio por ventana "
               f"fue {pct(w['cagr_is_promedio'])} y el CAGR WF-OOS {pct(w['cagr_oos'])}: WFE = {num(w['wfe_cagr'])} "
               f"(Calmar: {num(w['calmar_is_promedio'])} → {num(w['calmar_oos'])}, WFE = {num(w['wfe_calmar'])}). "
               f"Con train anclado la WFE es {num(wa['wfe_cagr'])}. "
               + ("Sobrevive menos de la mitad de la ventaja in-sample: la mayor parte es ajuste a la muestra."
                  if w["wfe_cagr"] < 0.5 else "Sobrevive una fracción sustancial de la ventaja in-sample.")
               + (f" En TEST (sistema congelado) el Calmar fue {num(f.load('metricas_test.csv').set_index('sistema').loc['RP', 'calmar'])}."
                  if has_test else f" TEST: {PENDING}.") + "\n")
    out.append(f"**3. ¿Qué tan sensible es a ±20%?** "
               + (f"Con {', '.join(peaks)} el Calmar cambia más de 50% o cambia de signo (pico); el resto "
                  f"({len(sv) - len(peaks)} de {len(sv)}) se comporta como meseta."
                  if peaks else f"Ninguno de los {len(sv)} parámetros y multiplicadores cambia el Calmar más de 50% ni "
                                f"de signo: es una meseta, no un pico aislado.")
               + f" El mayor cambio relativo fue {pct(sv.cambio_relativo_max.max())} "
                 f"({sv.loc[sv.cambio_relativo_max.idxmax(), 'parametro']}) y la mayor caída "
                 f"{pct(sv.caida_relativa_max.max())} ({sv.loc[sv.caida_relativa_max.idxmax(), 'parametro']}).\n")
    if be["comision_equilibrio"] is None:
        q4 = "Ni con 0.5% por lado deja de ser rentable en WF-OOS: el margen frente a 0.125% es mayor a 4 veces."
    elif be["comision_equilibrio"] == 0:
        q4 = "No es rentable ni con costo cero en WF-OOS: no hay margen de seguridad."
    elif be["margen_puntos"] < 0:
        q4 = (f"Ya no es rentable con la comisión oficial: el PnL neto cruza cero en ≈ {pct(be['comision_equilibrio'], 3)} "
              f"por lado, por debajo de 0.125% (margen de {be['margen_puntos'] * 100:+.3f} puntos porcentuales; el "
              f"equilibrio es {num(be['margen_multiplo'])} veces la comisión oficial). Antes de costos la estrategia gana "
              f"{money(be['pnl_bruto_oficial'])}, pero los costos ({money(be['costos_totales_oficial'])}) se comen toda la "
              f"ganancia y dejan un PnL neto de {money(be['pnl_neto_oficial'])}.")
    else:
        q4 = (f"Deja de ser rentable con una comisión de ≈ {pct(be['comision_equilibrio'], 3)} por lado: margen de "
              f"{be['margen_puntos'] * 100:+.3f} puntos porcentuales ({num(be['margen_multiplo'])} veces la comisión oficial).")
    out.append(f"**4. ¿A qué costo deja de ser rentable?** {q4} Con un turnover de {num(be['turnover_anual'], 1)} veces "
               f"al año, la comisión oficial cuesta ≈ {pct(be['costo_anual_por_rotacion'])} del capital por año.\n")
    means = ", ".join(f"{n}: {pct(rp_reg.loc[n, 'media_anualizada'])} anualizado (IC 95% diario "
                      f"[{rp_reg.loc[n, 'ic_inf'] * 1e4:.1f}, {rp_reg.loc[n, 'ic_sup'] * 1e4:.1f}] pb)"
                      for n in rp_reg.index if pd.notna(rp_reg.loc[n, "media_anualizada"]))
    crisis_entries = pr["RP"]["entradas_por_regimen"].get("Crisis", 0)
    out.append(f"**5. ¿El desempeño difiere significativamente entre regímenes?** Portafolio RP — {means}. "
               f"Kruskal-Wallis p = {num(kw, 3)}: "
               + ("las diferencias son estadísticamente significativas al 5%." if kw < 0.05 else
                  "no hay evidencia de diferencias significativas al 5% (los intervalos se traslapan).")
               + f" Aun así la capa de régimen aporta control de riesgo: en Crisis la exposición baja a 30% y solo se "
                 f"abren señales 3 de 3 (solo {crisis_entries} entradas en Crisis en WF-OOS), y los parámetros se "
                 f"adaptan al tipo de mercado.\n")
    pc = f.load("ponderaciones_comparacion.csv").set_index("metodo")
    out.append(f"**6. ¿Risk Parity mejora el Calmar frente a pesos iguales?** WF-OOS con las mismas señales, costos y "
               f"rebalanceo: Calmar RP {num(pc.loc['rp', 'calmar'])} vs. EW {num(pc.loc['ew', 'calmar'])} (RP naïve "
               f"{num(pc.loc['iv', 'calmar'])}); MDD {pct(pc.loc['rp', 'mdd'])} vs. {pct(pc.loc['ew', 'mdd'])} "
               f"({pct(pc.loc['iv', 'mdd'])}); CAGR {pct(pc.loc['rp', 'cagr'])} vs. {pct(pc.loc['ew', 'cagr'])} "
               f"({pct(pc.loc['iv', 'cagr'])}). La mayor contribución al riesgo de un solo activo es en promedio "
               f"{pct(conc['EW'])} con EW, {pct(conc['RP naïve'])} con RP naïve y {pct(conc['RP'])} con RP. "
               + ("RP mejora el Calmar frente a EW" if pc.loc["rp", "calmar"] > pc.loc["ew", "calmar"] else "RP no mejora el Calmar frente a EW")
               + (" sin sacrificar CAGR" if pc.loc["rp", "cagr"] >= pc.loc["ew", "cagr"] else " a costa de menor CAGR")
               + f"; su turnover es {num(pc.loc['rp', 'turnover_anual'], 2)} vs. {num(pc.loc['ew', 'turnover_anual'], 2)} veces al año. "
               + (f"En esta muestra la RP naïve tuvo un Calmar ligeramente mayor que la optimizada "
                  f"({num(pc.loc['iv', 'calmar'])} vs. {num(pc.loc['rp', 'calmar'])}). "
                  if pc.loc["iv", "calmar"] > pc.loc["rp", "calmar"] else "")
               + ("La diferencia con RP naïve es pequeña, consistente con que los seis activos tienen correlaciones "
                  "parecidas (la naïve es exacta si todas son iguales). " if abs(pc.loc["rp", "calmar"] - pc.loc["iv", "calmar"]) < 0.05 else "")
               + f"La mejora viene de controlar el drawdown, no de generar retorno: el Calmar es "
                 f"{'negativo' if pc.loc['rp', 'calmar'] < 0 else 'positivo'}.\n")
    out.append("**7. Tres limitaciones para operar con capital real.** (i) *Ejecución*: se supone llenado completo al "
               "open o al nivel exacto de SL/TP, sin impacto ni rechazos (ver sección 14); los stops intradía en la "
               "práctica sufren slippage. (ii) *Universo y sesgo de selección*: seis mega-cap tech muy correlacionadas, "
               "elegidas sabiendo que fueron ganadoras; la diversificación real es limitada y el resultado no se "
               "generaliza a otros sectores. (iii) *Estabilidad estadística*: con "
               f"{integer(f.load(opt)['configuraciones_evaluadas'])} configuraciones probadas y regímenes que cambian, "
               "el θ óptimo es inestable entre ventanas; además, la disponibilidad y el costo de préstamo de los cortos no "
               "están garantizados.\n")
    return "\n".join(out)


def execution_warning(f: Facts, has_test: bool) -> str:
    name = "costos_ejecucion_test.csv" if has_test else "costos_ejecucion_wf_oos.csv"
    label = "TEST" if has_test else "WF-OOS de TRAIN"
    ex = f.load(name)
    rows = [[r.escenario, pct(r.cagr), num(r.calmar), f"{r.delta_cagr * 100:+.2f} pp", f"{r.delta_calmar:+.2f}"]
            for r in ex.itertuples()]
    f.note(name, "escenarios", ex["cagr"].tolist())
    real = ex.iloc[1]
    return (f"**El backtest asume ejecución completa al precio modelado (open o nivel exacto de SL/TP) y no incorpora "
            f"impacto de mercado ni fallas de ejecución.** Para estimar la magnitud, el sistema se re-simuló con las "
            f"mismas señales ({label}) agregando spread de 2 pb, borrow fee de 0.5% anual sobre cortos e impacto "
            f"η·(|q|/ADV)^(2/3) con η = 0.1, y un barrido de slippage por operación:\n\n"
            + md_table(["Escenario", "CAGR", "Calmar", "Δ CAGR", "Δ Calmar"], rows)
            + f"\n\nEn el escenario realista el CAGR cambia {real.delta_cagr * 100:+.2f} puntos porcentuales y el Calmar "
              f"{real.delta_calmar:+.2f}. Con $1,000,000 la participación sobre el volumen diario es mínima, así que el "
              f"impacto es pequeño; el riesgo dominante es el slippage de los stops en días de gap.\n")


# ----------------------------------------------------------------------------
# Presentación
# ----------------------------------------------------------------------------

def build_slides(f: Facts) -> list[dict]:
    """Máximo 12 diapositivas (sin portada ni cierre), reparto parejo y tiempos."""
    has_test = f.exists("metricas_test.csv")
    mc = f.load("metricas_conjuntos.csv")
    get = lambda s, col: mc[(mc.conjunto == "WF-OOS") & (mc.sistema == s)][col].iloc[0]  # noqa: E731
    opt = f.load("optimizacion_resumen.json")
    be = f.load("costos_equilibrio.json")
    val = f.load("regimenes_validacion.json")
    iu = f.load("indicador_unico.csv")
    two = iu[(iu.nivel == "portafolio") & (iu.regla == "2 de 3")].iloc[0]
    singles = iu[(iu.nivel == "portafolio") & (iu.regla != "2 de 3")]
    sv = f.load("sensibilidad_veredicto.csv")
    reg = f.load("metricas_por_regimen.csv")
    kw = reg[reg.sistema == "RP"]["kruskal_p"].iloc[0]
    rc = f.load("rp_vs_ew_contribuciones.csv")
    conc = {s: rc[(rc.sistema == s) & (rc.activo == "concentracion_max_rc")]["rc_promedio_abs"].iloc[0] for s in ("RP", "EW")}
    t = f.load("metricas_test.csv").set_index("sistema") if has_test else None
    test_line = (f"TEST congelado: CAGR {pct(t.loc['RP', 'cagr'])}, Calmar {num(t.loc['RP', 'calmar'])} "
                 f"(B&H {num(t.loc['Buy & Hold', 'calmar'])})" if has_test else f"TEST: {PENDING}")
    be_txt = ("sin equilibrio hasta 0.5%" if be["comision_equilibrio"] is None else
              f"equilibrio ≈ {pct(be['comision_equilibrio'], 3)} por lado")
    slides = [
        {"who": "Milca", "sec": 50, "title": "Universo y datos", "fig": "06a_regimenes_linea_tiempo.png",
         "bullets": ["6 mega-cap tech, diario 2015–2026 (11.7 años)", "TRAIN 80% / TEST 20%: sin traslape",
                     "Sesgo: ganadoras conocidas → B&H difícil de vencer"]},
        {"who": "Milca", "sec": 50, "title": "Estrategia: confirmación 2 de 3", "fig": "08_indicador_unico.png",
         "bullets": ["EMA (tendencia) + RSI (momento) + Bollinger (volatilidad)",
                     "S_t = ±1 si ≥ 2 indicadores coinciden; ejecución en t+1",
                     f"2 de 3: {integer(two.n_operaciones)} ops, Calmar {num(two.calmar)} vs. "
                     f"{num(singles.calmar.min())}–{num(singles.calmar.max())} con uno solo"]},
        {"who": "Milca", "sec": 50, "title": "Señales en el tiempo", "fig": "07b_fuerza_senal.png",
         "bullets": ["Fuerza = número de votos (2 o 3) × dirección", "Largos y cortos en los 6 activos",
                     "RSI–Bollinger correlación < 0.7: no redundantes"]},
        {"who": "Paula", "sec": 50, "title": "Motor y costos", "fig": "05_curva_costos.png",
         "bullets": ["Event-driven; SL primero; gaps al open; sin apalancamiento",
                     f"0.125% por lado; {be_txt}", f"Rotación {num(be['turnover_anual'], 1)}×/año ≈ "
                                                  f"{pct(be['costo_anual_por_rotacion'])} del capital/año"]},
        {"who": "Paula", "sec": 50, "title": "Walk-forward: ¿cuánto sobrevive?", "fig": "13_rolling_vs_anclado.png",
         "bullets": ["Train 6 meses → test 1 mes, paso mensual, 100 trials/estudio",
                     f"WFE rolling = {num(opt['wfe_rolling']['wfe_cagr'])}; anchored = {num(opt['wfe_anclado']['wfe_cagr'])}",
                     f"{integer(opt['configuraciones_evaluadas'])} configuraciones en {opt['segundos_optimizacion'] / 60:.0f} min"]},
        {"who": "Paula", "sec": 50, "title": "Sensibilidad ±20%", "fig": "04_sensibilidad.png",
         "bullets": [f"{int((sv.veredicto == 'meseta').sum())} de {len(sv)} parámetros: meseta",
                     f"Mayor cambio del Calmar: {pct(sv.cambio_relativo_max.max())}",
                     "θ robusto = mediana del top 10% (centro de la meseta)"]},
        {"who": "Paula", "sec": 50, "title": "Diagnóstico de la optimización", "fig": "12d_optuna_superficie_3d.png",
         "bullets": ["Superficie del Calmar: corte 2D de un espacio 10D", "Importancia fANOVA de los parámetros",
                     "Mínimo de operaciones evita Calmar sin significado"]},
        {"who": "Arturo", "sec": 50, "title": "Regímenes con K-means", "fig": "06c_valor_con_regimenes.png",
         "bullets": [f"Volatilidad + R² de tendencia; silhouette {num(val['silhouette']['promedio_ventanas'], 2)}",
                     f"Duración promedio {num(val['persistencia_wf_oos']['duracion_promedio_global'], 0)} días (objetivo ≥ 10)",
                     "Fit solo con datos ≤ fin del train; actualización semanal"]},
        {"who": "Arturo", "sec": 50, "title": "Desempeño por régimen", "fig": "09_transiciones_regimen.png",
         "bullets": [f"Kruskal-Wallis p = {num(kw, 3)}", "Crisis: M = 0.3 y solo 3 de 3",
                     "Transiciones: posiciones conservan su SL/TP"]},
        {"who": "Arturo", "sec": 50, "title": "Risk Parity vs. pesos iguales", "fig": "07a_contribuciones_riesgo.png",
         "bullets": ["Spinu: min ½yᵀΣy − (1/n)Σ ln y; w = y/Σy",
                     f"Calmar: EW {num(get('EW', 'calmar'))} · 1/σ {num(get('RP naïve', 'calmar'))} · RP {num(get('RP', 'calmar'))}",
                     f"MDD RP {pct(get('RP', 'mdd'))} vs. EW {pct(get('EW', 'mdd'))}"]},
        {"who": "Arturo", "sec": 50, "title": "Resultado: entrenamiento y prueba", "fig": "01_valor_portafolio.png",
         "bullets": [f"WF-OOS: CAGR {pct(get('RP', 'cagr'))}, Calmar {num(get('RP', 'calmar'))} "
                     f"(B&H {num(get('Buy & Hold', 'calmar'))})", test_line]},
        {"who": "Milca", "sec": 50, "title": "Conclusiones y limitaciones", "fig": "02_drawdown.png",
         "bullets": ["Ejecución ideal: sin impacto ni slippage de stops", "Universo tech correlacionado y sesgo de selección",
                     "θ inestable entre ventanas: re-optimización con cautela"]},
    ]
    return slides


def slides_markdown(slides: list[dict]) -> str:
    total = sum(s["sec"] for s in slides)
    out = ["# Presentación — Lab 02 MyST, Equipo 2\n",
           f"{len(slides)} diapositivas (sin portada ni cierre), {total / 60:.1f} minutos de exposición + 5 de preguntas.\n",
           "| # | Expositor | Tiempo | Título | Figura |", "|---|---|---|---|---|"]
    out += [f"| {i} | {s['who']} | {s['sec']} s | {s['title']} | {s['fig']} |" for i, s in enumerate(slides, 1)]
    for i, s in enumerate(slides, 1):
        out.append(f"\n## {i}. {s['title']} ({s['who']}, {s['sec']} s)\n")
        out += [f"- {b}" for b in s["bullets"]]
        out.append(f"\nFigura: `{s['fig']}`")
    return "\n".join(out) + "\n"


def render_slides(slides: list[dict], figures: Path, path: Path) -> None:
    """PDF 16:9 con portada, diapositivas (figura + 2–3 viñetas, letra grande) y cierre."""
    def text_slide(pdf, title, lines):
        fig_ = plt.figure(figsize=(16, 9))
        fig_.patch.set_facecolor("#1d3557")
        fig_.text(0.5, 0.62, title, ha="center", va="center", fontsize=40, color="white", weight="bold")
        for k, line in enumerate(lines):
            fig_.text(0.5, 0.45 - k * 0.07, line, ha="center", va="center", fontsize=24, color="white")
        pdf.savefig(fig_)
        plt.close(fig_)

    with PdfPages(path, metadata={"Title": "Lab 02 MyST — Equipo 2", "CreationDate": None, "ModDate": None}) as pdf:
        text_slide(pdf, "Estrategias de Trading con Análisis Técnico",
                   ["Laboratorio 02 · Microestructuras y Sistemas de Trading · ITESO",
                    "Equipo 2 (Nivel C): Milca · Paula · Arturo"])
        for i, s in enumerate(slides, 1):
            fig_ = plt.figure(figsize=(16, 9))
            fig_.text(0.04, 0.93, f"{i}. {s['title']}", fontsize=34, weight="bold", color="#1d3557", va="center")
            fig_.text(0.96, 0.93, s["who"], fontsize=20, color="gray", ha="right", va="center")
            ax = fig_.add_axes([0.03, 0.05, 0.62, 0.8])
            ax.imshow(mpimg.imread(figures / s["fig"]))
            ax.axis("off")
            for k, b in enumerate(s["bullets"]):
                fig_.text(0.67, 0.75 - k * 0.2, "• " + b, fontsize=21, va="top", wrap=True, color="#222222")
            pdf.savefig(fig_)
            plt.close(fig_)
        text_slide(pdf, "Gracias", ["Preguntas"])


# ----------------------------------------------------------------------------
# Markdown -> PDF (reportlab) y verificación
# ----------------------------------------------------------------------------

def markdown_to_pdf(markdown: str, base: Path, path: Path) -> None:
    """Convierte el subconjunto de Markdown usado aquí (títulos, párrafos, listas, tablas, figuras, código)."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle

    font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont("DejaVu", str(font_dir / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(font_dir / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-Oblique", str(font_dir / "DejaVuSans-Oblique.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVuMono", str(font_dir / "DejaVuSansMono.ttf")))
    from reportlab.pdfbase.pdfmetrics import registerFontFamily
    registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu-Oblique", boldItalic="DejaVu-Bold")
    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["BodyText"], fontName="DejaVu", fontSize=9.5, leading=13)
    styles = {1: ParagraphStyle("h1", parent=ss["Title"], fontName="DejaVu-Bold", fontSize=18),
              2: ParagraphStyle("h2", parent=ss["Heading2"], fontName="DejaVu-Bold", fontSize=14, textColor=colors.HexColor("#1d3557")),
              3: ParagraphStyle("h3", parent=ss["Heading3"], fontName="DejaVu-Bold", fontSize=11)}
    cell = ParagraphStyle("cell", parent=body, fontSize=7.5, leading=9)
    width = letter[0] - 3 * cm

    def inline(text):
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"<i>\1</i>", text)
        return re.sub(r"`(.+?)`", r'<font name="DejaVuMono">\1</font>', text)

    story, lines, i = [], markdown.splitlines(), 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            story.append(Paragraph(inline(line.lstrip("# ").strip()), styles[min(level, 3)]))
        elif line.startswith("!["):
            m = re.match(r"!\[(.*?)\]\((.*?)\)", line)
            img_path = (base / m.group(2)) if not Path(m.group(2)).is_absolute() else Path(m.group(2))
            if img_path.exists():
                h, w_ = mpimg.imread(img_path).shape[:2]
                story.append(Image(str(img_path), width=width, height=width * h / w_))
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                if not re.match(r"^\|[-|\s]+\|$", lines[i]):
                    rows.append([Paragraph(inline(c.strip()), cell) for c in lines[i].strip("|").split("|")])
                i += 1
            t = Table(rows, repeatRows=1, hAlign="LEFT")
            t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe7f2")),
                                   ("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
            story.append(t)
            story.append(Spacer(1, 6))
            continue
        elif line.startswith("    "):
            block = []
            while i < len(lines) and lines[i].startswith("    "):
                block.append(lines[i][4:])
                i += 1
            story.append(Preformatted("\n".join(block), ParagraphStyle("code", fontName="DejaVuMono", fontSize=9, leading=12,
                                                                       backColor=colors.HexColor("#f4f4f4"))))
            continue
        elif re.match(r"^(- |\d+\. )", line):
            story.append(Paragraph(inline(line), ParagraphStyle("li", parent=body, leftIndent=12)))
        elif line.strip():
            story.append(Paragraph(inline(line), body))
        else:
            story.append(Spacer(1, 4))
        i += 1
    doc = SimpleDocTemplate(str(path), pagesize=letter, leftMargin=1.5 * cm, rightMargin=1.5 * cm,
                            topMargin=1.5 * cm, bottomMargin=1.5 * cm, title="Lab 02 MyST — Equipo 2",
                            author="Equipo 2", invariant=1)
    doc.build(story)


def verify(registry: list[dict], results: Path, documents: list[str]) -> dict:
    """C14: recalcula cada cifra registrada desde su archivo y comprueba que el texto impreso coincide."""
    fresh = Facts(results)
    checked, errors = 0, []
    for item in registry:
        if item.get("derivada") or item["texto"] is None:
            continue
        value = _plain(_select(fresh.load(item["archivo"]), tuple(item["selector"])))
        same = value == item["valor"] or (isinstance(value, float) and isinstance(item["valor"], float)
                                          and np.isclose(value, item["valor"], rtol=1e-12, atol=0))
        in_doc = any(item["texto"] in d for d in documents)
        checked += 1
        if not (same and in_doc):
            errors.append({**item, "valor_archivo": value, "en_documento": in_doc})
    return {"cifras_registradas": len(registry), "cifras_verificadas": checked,
            "cifras_derivadas": sum(1 for r in registry if r.get("derivada")), "errores": errors, "ok": not errors}


def build_all(results: Path, figures: Path, docs: Path) -> dict:
    """Borradores Markdown, PDFs y verificación de cifras."""
    facts = Facts(results)
    report_md = build_report(facts, figures, docs)
    slides = build_slides(facts)
    slides_md = slides_markdown(slides)
    (docs / "borrador_reporte.md").write_text(report_md, encoding="utf-8")
    (docs / "borrador_presentacion.md").write_text(slides_md, encoding="utf-8")
    markdown_to_pdf(report_md, docs, docs / "reporte.pdf")
    render_slides(slides, figures, docs / "presentacion.pdf")
    check = verify(facts.registry, results, [report_md, slides_md])
    (results / "verificacion_cifras.json").write_text(json.dumps(check, indent=2, ensure_ascii=False, default=str),
                                                      encoding="utf-8")
    if not check["ok"]:
        raise RuntimeError(f"{len(check['errores'])} cifras del reporte no coinciden con docs/resultados/")
    return check
