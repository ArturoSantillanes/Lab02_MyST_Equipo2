# Checklist de cumplimiento — Lab 02 MyST, Equipo 2 (Nivel C)

Exposición: 7 de octubre de 2026. Último commit válido: 6 de octubre de 2026, 23:59.

† = queda PENDIENTE hasta que el equipo corra el TEST real.

| ID | Requisito | Estado | Evidencia |
|----|-----------|--------|-----------|
| A1 | Repo `Lab02_MyST_Equipo2`, público | PASS | `git remote -v` → github.com/ArturoSantillanes/Lab02_MyST_Equipo2; captura de GitHub: repositorio *Public* |
| A2 | Estructura exacta; ningún archivo suelto extra en la raíz | PASS | Raíz: README.md, requirements.txt, .gitignore, main.py, data/, src/, tests/, notebooks/, docs/ (.venv/ y .cache/ ignorados) |
| A3 | `python main.py` corre sin errores desde entorno limpio (`--quick` y `--stage train`) | PASS | `--quick` desde venv limpio (pip install -r requirements.txt) → exit 0 en 2.3 min; `python main.py --stage train` → exit 0 (16.8 min con ventanas en caché); `--stage report` → exit 0. TEST pendiente por diseño (candado) |
| A4 | `requirements.txt` con `==`; `.gitignore` adecuado sin ignorar `data/` ni `docs/` | PASS | `requirements.txt`: 13 paquetes con `==`; `.gitignore` no ignora `data/` ni `docs/` |
| A5 | README completo (sección 18) | PASS | `README.md`: integrantes/activos/semilla 42, nivel C, párrafo, instalación, `python main.py`, `pytest -v`, semilla, fuente y fecha de datos, tiempo ≈ 1 h, uso de IA |
| A6 | Semilla en un único lugar; dos corridas `--quick` dan hashes idénticos | PASS | `SEED = 42` solo en `src/config.py`; dos corridas `--quick` desde cero: 50 CSV con SHA-256 idénticos (ambas desde el venv limpio) |
| A7 | Datos congelados en `data/`; `main.py` no descarga si existen | PASS | `data/prices_daily.csv` + `download_metadata.json`; `src/data.py:download_prices` no descarga si existe el CSV |
| A8 | Notebook sin lógica; corre con nbconvert | PASS | `jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb` → 0 errores, 23 figuras; solo importa `src`, lee resultados y llama `plots.make_all_figures` |
| A9 | Toda la lógica en `src/`, en el archivo correcto | PASS | Lógica en `src/` (data, signals, backtest, metrics, optimize, regimes, portfolio, plots, report); `main.py` solo orquesta |
| A10 | Plan de commits por integrante con mensajes descriptivos | PASS | `../Lab02_estudio/plan_commits.md`: 12 bloques con dueño, archivos y mensaje `tipo(módulo): …` |
| B1 | Comisión 0.125% en cada entrada y salida | PASS | `config.COMMISSION = 0.00125`; `test_contabilidad` (golden file al centavo y Σ costos = c·Σ nocional) |
| B2 | Sin apalancamiento (Σ|w| ≤ 1 verificado sobre la serie completa) | PASS | Σ|w| máximo tras ejecutar = 0.6679 (WF-OOS RP); `test_sin_apalancamiento_en_toda_la_serie`, `test_portafolio_rp_sin_apalancamiento…` |
| B3 | Existen operaciones largas y cortas en los resultados | PASS | WF-OOS RP: 248 largas y 376 cortas (`portafolio_resumen.json`) |
| B4 | Capital inicial $1,000,000 | PASS | `config.INITIAL_CAPITAL = 1_000_000.0` |
| B5 | ≥ 3 indicadores de ≥ 2 familias, documentados | PASS | EMA (tendencia), RSI (momento), Bollinger (volatilidad) en `src/signals.py` con fórmulas en docstrings |
| B6 | Regla 2 de 3 implementada y como fórmula en el reporte | PASS | `signals.confirm`; fórmula en `docs/borrador_reporte.md` sección 3 |
| B7 | Señal al cierre de t, ejecución en t+1 | PASS | `test_causalidad` (5 valores de t) y `test_ejecucion_en_t_mas_1_al_open` |
| B8 | Motor event-driven con estado explícito | PASS | `backtest.run_backtest`: loop diario con efectivo, q, dirección, SL/TP y equity |
| B9 | SL/TP; SL primero; gaps; convención declarada | PASS | `test_sl_primero…`, `test_corto_sl_primero`, `test_gap_se_ejecuta_al_open`; convención en docstring y reporte §4 |
| B10 | Optuna maximizando Calmar | PASS | `optimize.make_objective` → `embargoed_calmar`; `direction="maximize"` |
| B11 | 100 trials por régimen por ventana | PASS | `trials_por_estudio = 100` (global + 3 regímenes por ventana y activo) |
| B12 | Mínimo de operaciones declarado y aplicado | PASS | N_MIN_GLOBAL = 10, N_MIN_REGIME = 5 → −1e9 (`optimize.make_objective`); reporte §5 y supuesto 8 |
| B13 | Walk-forward 6m / 1m / mensual | PASS | 100 ventanas 6m→1m, paso mensual + ventana final (`optimize.build_windows`) |
| B14 | Tiempo total y número de configuraciones registrados | PASS | 63.1 min y 460,800 configuraciones (`optimizacion_resumen.json`) |
| B15 | 3 regímenes con K-means; features en ventana de 63 días | PASS | K-means K=3, ventana 63 días; features volatilidad, r2_tendencia (`regimes.py`) |
| B16 | Frecuencia de actualización adaptada y justificada | PASS | Actualización cada 5 días + persistencia 2; justificación en reporte §7 |
| B17 | Sin look-ahead en regímenes (fit solo en train) | PASS | `fit_regime_model(feats, fit_end)`; `test_etiqueta_en_t_no_cambia…`, `test_ajuste_no_usa_datos_posteriores…` |
| B18 | θ por régimen en cada ventana; fallback declarado y contado | PASS | θ por régimen por ventana; fallbacks: Tendencia 277, Reversión 337, Crisis 602 de 606 |
| B19 | Reglas de transición implementadas y declaradas | PASS | Reglas en `portfolio.PortfolioSizer` y reporte §7; 41 transiciones, 78 posiciones afectadas |
| B20 | 6 activos diarios, ≥ 6 años, completos y traslapados | PASS | `auditoria_datos.csv`: 6 activos × 2,932 días, 11.7 años, `completo_y_traslapado = True` |
| B21 | Risk Parity con RC iguales verificadas numéricamente | PASS | `risk_parity_weights` (Spinu convexo, clase paso 5) lanza `RiskParityError` si max|RC/σ_p − 1/n| ≥ 1e-4; `test_risk_parity` incluye Euler y los ejemplos numéricos de la clase |
| B22 | Covarianza declarada, justificada y sin datos futuros | PASS | Muestral 126 días ≤ t (oficial), comparada con EWMA y Ledoit-Wolf en `covarianza_estimadores.csv`; κ mediano 34.5; figura 15 |
| B23 | Agregación: confianza, correlación, régimen y política de conflictos | PASS | s_i = Σ votos/3, w^target = m·(w^RP·s)/max(1,Σ|w̃|) (clase paso 7), correlación 126 d; 26 conflictos (5 empates) |
| B24 | Rebalanceo: frecuencia, disparadores, turnover y multiplicadores | PASS | Híbrido calendario + banda (f = 10 d, δ = 0.20) + Crisis; T_t = ½Σ|Δw|; 2.0 rebalanceos/año; costo ≈ T̄·f·2c = 0.05%/año |
| B25 | Benchmark EW con mismas señales, costos y rebalanceo | PASS | EW y RP naïve (1/σ) con el mismo `composed`, costos y rebalanceo (`optimize.weighting_comparison`, `ponderaciones_comparacion.csv`) |
| B26 | "Supuestos y decisiones" completo en el reporte | PASS | `docs/borrador_reporte.md` §12: 20 supuestos |
| B27 | θ robusto (meseta) usado en vez del argmax; argmax guardado | PASS | `robust_params`; `parametros_por_ventana.csv` con columnas robustas y `argmax_*` |
| B28 | Rolling vs. anchored comparados, con WFE de cada uno | PASS | por_activo: WFE -0.07; compartido: WFE -0.46; anclado_por_activo: WFE -0.31 (`variantes_comparacion.csv`, figura 13) |
| C1 | Métricas para WF-IS, WF-OOS y TEST† por separado | PENDIENTE | WF-IS y WF-OOS en `metricas_conjuntos.csv` (RP WF-OOS Calmar -0.04); TEST† pendiente |
| C2 | Tablas de retornos mensuales, trimestrales y anuales | PASS | `retornos_{mensual,trimestral,anual}_wf_oos.csv`; figuras 03a y 03b |
| C3 | Sensibilidad ±20% de cada parámetro, conclusión meseta/pico | PASS | `sensibilidad_veredicto.csv`: 12 mesetas, 1 pico; figura 04 |
| C4 | Curva retorno vs. costo con punto de equilibrio y margen | PASS | Equilibrio 0.060%, margen -0.065 pp; figura 05 |
| C5 | Fuentes de degradación discutidas con cifras | PASS | Reporte §11 «Fuentes de degradación» con cifras |
| C6 | 2 de 3 vs. indicador único | PASS | `indicador_unico.csv` (portafolio y por activo); figura 08 |
| C7 | Validación de régimen completa, incluyendo TRAIN vs. TEST† | PENDIENTE | Silhouette 0.468, duración 49.8 d, matriz, bootstrap; TRAIN vs TEST† pendiente |
| C8 | Métricas por régimen (estrategia y portafolio) | PASS | `metricas_por_regimen.csv` (RP, EW, B&H y 6 estrategias) con IC bootstrap y Kruskal-Wallis |
| C9 | RP vs. EW con contribuciones al riesgo y drawdown | PASS | `rp_vs_ew_contribuciones.csv`, reporte §10 y respuesta 6; figura 07a |
| C10 | Portafolio vs. cada activo individual | PASS | `metricas_por_activo_wf_oos.csv`; figura 10 |
| C11 | Barrido de rebalanceo; turnover; costos vs. retorno bruto | PASS | `rebalanceo_barrido.csv` (malla 4 f × 4 δ: retorno bruto, costo, neto, turnover); figuras 07d y 11 |
| C12 | Preguntas 1–7 respondidas con cifras propias† | PENDIENTE | Preguntas 1–7 respondidas con cifras de TRAIN (reporte §13); la parte de TEST† queda `[PENDIENTE: test]` |
| C13 | Advertencia de ejecución con magnitud estimada | PASS | `costos_ejecucion_wf_oos.csv`: escenario realista y slippage 0–20 pb; reporte §14 |
| C14 | Cifras de reporte y presentación coinciden con `docs/resultados/` (verificación automática) | PASS | `verificacion_cifras.json`: 222 cifras verificadas, 0 errores |
| C15 | Candado del test funcionando (en `--quick`: modificar θ → aborta) | PASS | `--quick`: θ alterado → `LockViolationError` (exit 1); θ restaurado → TEST corre; `tests/test_candado.py` (4 casos) |
| F1 | Figura: valor del portafolio (WF-OOS y TEST) con EW y B&H | PASS | `docs/figures/01_valor_portafolio.png` |
| F2 | Figura: curva de drawdown | PASS | `docs/figures/02_drawdown.png` |
| F3 | Figura: retornos mensuales, trimestrales y anuales | PASS | `docs/figures/03a_retornos_mensuales_anuales.png`, `03b_retornos_trimestrales.png` |
| F4 | Figura: sensibilidad ±20% por parámetro | PASS | `docs/figures/04_sensibilidad.png` |
| F5 | Figura: retorno neto vs. costo con 0.125% y equilibrio | PASS | `docs/figures/05_curva_costos.png` |
| F6 | Figuras de régimen: línea de tiempo, distribuciones, equity sombreado | PASS | `06a_regimenes_linea_tiempo.png`, `06b_distribuciones_features.png`, `06c_valor_con_regimenes.png` |
| F7 | Figuras de portafolio: RC RP vs EW, fuerza de señal, correlación por régimen, barrido de rebalanceo | PASS | `07a_contribuciones_riesgo.png`, `07b_fuerza_senal.png`, `07c_correlacion_por_regimen.png`, `07d_barrido_rebalanceo.png` |
| F8 | Todas las figuras con título, ejes, leyenda y fuente legible | PASS | 24 PNG revisados visualmente: título, ejes, leyenda o barra de color, fuente ≥ 12 pt |
| D1 | test_causalidad pasa | PASS | `pytest tests/test_causalidad.py` → 7 passed |
| D2 | test_confirmacion pasa | PASS | `pytest tests/test_confirmacion.py` → 14 passed |
| D3 | test_contabilidad pasa (golden file al centavo) | PASS | `pytest tests/test_contabilidad.py` → 4 passed (golden file a ±$0.01) |
| D4 | test_regimen_causal pasa | PASS | `pytest tests/test_regimen_causal.py` → 10 passed |
| D5 | test_risk_parity pasa | PASS | `pytest tests/test_risk_parity.py` → 18 passed |
| D6 | test_restricciones pasa | PASS | `pytest tests/test_restricciones.py` → 8 passed; total `pytest -v` → 65 passed (venv limpio) |
| D7 | Responsabilidad única, nombres descriptivos, docstrings con fórmulas | PASS | Un módulo por responsabilidad; docstrings con fórmulas en indicadores, métricas, RP, régimen y motor |
| D8 | Sin código muerto, comentado ni duplicado | PASS | Búsqueda de definiciones sin referencias → 0 (se eliminaron `data.field`, `data.ohlcv` y la duplicación de Bollinger) |
| D9 | Uso de IA declarado en el README | PASS | `README.md` sección «Uso de asistencia de IA» |
| E1 | `docs/reporte.pdf` generado y completo† | PENDIENTE | `docs/reporte.pdf` (22 páginas) generado; marcadores `[PENDIENTE: test]` hasta correr el TEST† |
| E2 | `docs/presentacion.pdf` ≤ 12 diapositivas, sin capturas de código† | PENDIENTE | `docs/presentacion.pdf`: portada + 12 diapositivas + cierre, sin capturas de código; cifras de TEST† pendientes |
| E3 | Reparto de la exposición entre Milca, Paula y Arturo (10 min) | PASS | Milca 4, Paula 4, Arturo 4 diapositivas de 50 s (10 min) en `borrador_presentacion.md` |
| E4 | Material de estudio en `../Lab02_estudio/` | PASS | `../Lab02_estudio/`: guia_revision.md, plan_commits.md, notas_orador.md |
| E5 | Recordatorio en `plan_commits.md`: último commit antes de 23:59 del 6-oct-2026 | PASS | Recordatorio al inicio de `plan_commits.md`: último commit 6-oct-2026 23:59 |

## Ciclo de verificación

Ítems: 75 — PASS 70, FAIL 0, PENDIENTE 5 (todos los PENDIENTE dependen del TEST real, marcados con †).

| Pasada | Qué se corrió | Hallazgos | Cambios |
|---|---|---|---|
| 1 | `--quick` completo, revisión de figuras y borrador | superficie 3D vacía, fechas encimadas, leyendas tapando curvas, veredicto de sensibilidad solo veía caídas, RC n/d en tramos cortos, tabla de θ repetida, redacción de P1/P6, código muerto (`data.field`, `data.ohlcv`) y Bollinger duplicado | corregidos en `plots.py`, `optimize.py`, `portfolio.py`, `report.py`, `signals.py`, `data.py` |
| 2 | `--stage train` y `--stage report` oficiales | Buy & Hold aplanaba la figura 1, formato de montos negativos, tabla de auditoría rota por `|r|`, P4 sin aclarar que 0.125% ya no es rentable | escala log, `money`, encabezado, redacción de P4 |
| 3 | Material de clase de Risk Parity | faltaban Spinu como solver principal, RP naïve, fuerza continua s_i, composición con m(régimen), turnover ½Σ|Δw| y rebalanceo híbrido, comparación de estimadores de Σ | `portfolio.py`, `optimize.py`, `backtest.py`, `main.py`, `plots.py`, `report.py`, pruebas con los ejemplos de la clase |
| 4 (final) | venv limpio + `pytest -v` (65 passed), dos `--quick` desde cero (50 CSV con hashes idénticos), candado (θ alterado → aborta), `--stage report`, notebook, candado | ninguno | ninguno (código con el mismo SHA-256 antes y después) |
