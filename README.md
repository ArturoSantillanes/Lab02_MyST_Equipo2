# Lab02_MyST_Equipo2 — Estrategias de Trading con Análisis Técnico

**Microestructuras y Sistemas de Trading · ITESO · Laboratorio 02 · Nivel de alcance: C**

## Integrantes y activos

| Integrante | Activos | Módulos de los que es responsable |
|------------|---------|-----------------------------------|
| Milca  | TSLA, NFLX  | `src/data.py`, `src/signals.py`, pruebas de causalidad y confirmación |
| Paula  | META, AMZN  | `src/backtest.py`, `src/metrics.py`, `src/optimize.py`, pruebas de contabilidad, restricciones y candado |
| Arturo | NVDA, GOOGL | `src/regimes.py`, `src/portfolio.py`, pruebas de régimen y Risk Parity |
| Los 3  | —           | `src/config.py`, `src/plots.py`, `src/report.py`, `main.py`, notebook, reporte y presentación |

Los activos se asignaron al azar con `random.seed(42)` sobre la lista
`['NVDA','AMZN','TSLA','META','NFLX','GOOGL']` (2n = 6 activos para n = 3 integrantes).

## Descripción

Sistema de trading sistemático multi-activo con datos diarios de seis acciones tecnológicas (2015-01-02 a
2026-08-31). Cada activo opera largo y corto con tres indicadores de familias distintas (cruce de EMAs, RSI de
Wilder y Bandas de Bollinger) bajo una regla de confirmación 2 de 3, con stop-loss y take-profit en múltiplos de
ATR, en un motor event-driven con comisión de 0.125% por lado y sin apalancamiento. Un detector de régimen
(K-means sobre volatilidad y R² de tendencia del índice equiponderado) distingue Tendencia, Reversión y Crisis;
los parámetros se optimizan por activo y por régimen con Optuna maximizando el Calmar en un walk-forward de
6 meses → 1 mes, y las posiciones se agregan en un portafolio de Risk Parity con multiplicadores de riesgo por
régimen. El último 20% de los datos (TEST) se evalúa una sola vez con el sistema congelado.

## Instalación (Windows)

Requiere Python ≥ 3.10 (desarrollado con Python 3.14.7).

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducir todos los resultados

```bat
python main.py
```

Ejecuta todas las etapas: datos → auditoría → walk-forward (por activo, compartido y anclado) → regímenes →
portafolio → robustez → TEST (con candado) → figuras, tablas, borradores y PDFs en `docs/`.

- Por etapa: `python main.py --stage {data,train,test,report}`.
- Ensayo rápido (pocas ventanas, 20 trials; escribe en `.cache/quick/`, nunca en `docs/`): `python main.py --quick`.
- Pruebas: `pytest -v`.
- Notebook (solo lee resultados y grafica): `jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb`.

**Candado del TEST.** La etapa `test` exige el working tree de git limpio la primera vez, registra el commit en
`docs/resultados/test_lock.json` y aborta si después cambia el θ congelado (`docs/resultados/theta_congelado.json`,
protegido con SHA-256). Volver a correr con el mismo θ sí está permitido porque es determinista. Si el árbol no
está limpio y el TEST no se ha corrido, `python main.py` lo deja como pendiente y continúa.

**Tiempo aproximado de ejecución** (8 hilos, 7 procesos): ≈ 15 min el walk-forward por activo, ≈ 6 min el
compartido, ≈ 20 min el anclado y ≈ 15 min los análisis y figuras: **≈ 1 hora** desde cero. Las ventanas ya
optimizadas se guardan en `.cache/wf/` y una corrida interrumpida se retoma desde ahí.

## Semilla

`SEED = 42`, definida una sola vez en `src/config.py`. Cada estudio de Optuna usa una semilla derivada
determinísticamente de `SEED` y de (ventana, activo, régimen) (`src/optimize.py:study_seed`), así que las
corridas en paralelo son reproducibles. K-means, bootstrap e importancias usan la misma `SEED`.

## Datos

Fuente: Yahoo Finance mediante `yfinance` 1.7.0 con `auto_adjust=True` (precios ajustados por splits y
dividendos). Descargados el 2026-09-30 y congelados en `data/prices_daily.csv` (formato largo) con su metadata
en `data/download_metadata.json`. `main.py` no vuelve a descargar si el CSV existe, porque yfinance puede cambiar
datos históricos.

## Estructura

```
main.py              orquesta las etapas (sin lógica de modelo)
src/config.py        constantes, semilla y rutas
src/data.py          descarga, auditoría, limpieza y split
src/signals.py       EMA, RSI, Bollinger, ATR y regla 2 de 3
src/backtest.py      motor event-driven con costos, SL/TP y sin apalancamiento
src/metrics.py       Sharpe, Sortino, Calmar, MDD, win rate, tablas de rendimientos
src/optimize.py      Optuna, walk-forward, θ robusto, sensibilidad, costos, candado del TEST
src/regimes.py       features de régimen, K-means causal, validación
src/portfolio.py     Risk Parity, agregación de señales, rebalanceo, benchmarks
src/plots.py         figuras (solo grafican)
src/report.py        borradores y PDFs con cifras leídas de docs/resultados/ (verificación automática)
tests/               pruebas con pytest
notebooks/           analysis.ipynb (sin lógica)
docs/                figuras, resultados, checklist, reporte y presentación
```

## Uso de asistencia de IA

El código, las pruebas, el notebook y los borradores del reporte y la presentación se desarrollaron con
asistencia de IA (Claude Code), a partir de un prompt de especificación escrito por el equipo con base en los
lineamientos del laboratorio. Las cifras del reporte y de la presentación no se escribieron a mano: las genera
`src/report.py` leyendo `docs/resultados/`. Cada integrante revisó línea por línea los módulos de los que es
responsable (tabla de arriba), hizo sus propios commits y responde por ellos; los tres son responsables de todo
el código entregado.
