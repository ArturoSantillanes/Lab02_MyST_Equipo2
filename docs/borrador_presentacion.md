# Presentación — Lab 02 MyST, Equipo 2

12 diapositivas (sin portada ni cierre), 10.0 minutos de exposición + 5 de preguntas.

| # | Expositor | Tiempo | Título | Figura |
|---|---|---|---|---|
| 1 | Milca | 50 s | Universo y datos | 06a_regimenes_linea_tiempo.png |
| 2 | Milca | 50 s | Estrategia: confirmación 2 de 3 | 08_indicador_unico.png |
| 3 | Milca | 50 s | Señales en el tiempo | 07b_fuerza_senal.png |
| 4 | Paula | 50 s | Motor y costos | 05_curva_costos.png |
| 5 | Paula | 50 s | Walk-forward: ¿cuánto sobrevive? | 13_rolling_vs_anclado.png |
| 6 | Paula | 50 s | Sensibilidad ±20% | 04_sensibilidad.png |
| 7 | Paula | 50 s | Diagnóstico de la optimización | 12d_optuna_superficie_3d.png |
| 8 | Arturo | 50 s | Regímenes con K-means | 06c_valor_con_regimenes.png |
| 9 | Arturo | 50 s | Desempeño por régimen | 09_transiciones_regimen.png |
| 10 | Arturo | 50 s | Risk Parity vs. pesos iguales | 07a_contribuciones_riesgo.png |
| 11 | Arturo | 50 s | Resultado: entrenamiento y prueba | 01_valor_portafolio.png |
| 12 | Milca | 50 s | Conclusiones y limitaciones | 02_drawdown.png |

## 1. Universo y datos (Milca, 50 s)

- 6 mega-cap tech, diario 2015–2026 (11.7 años)
- TRAIN 80% / TEST 20%: sin traslape
- Sesgo: ganadoras conocidas → B&H difícil de vencer

Figura: `06a_regimenes_linea_tiempo.png`

## 2. Estrategia: confirmación 2 de 3 (Milca, 50 s)

- EMA (tendencia) + RSI (momento) + Bollinger (volatilidad)
- S_t = ±1 si ≥ 2 indicadores coinciden; ejecución en t+1
- 2 de 3: 624 ops, Calmar -0.04 vs. -0.10–0.13 con uno solo

Figura: `08_indicador_unico.png`

## 3. Señales en el tiempo (Milca, 50 s)

- Fuerza = número de votos (2 o 3) × dirección
- Largos y cortos en los 6 activos
- RSI–Bollinger correlación < 0.7: no redundantes

Figura: `07b_fuerza_senal.png`

## 4. Motor y costos (Paula, 50 s)

- Event-driven; SL primero; gaps al open; sin apalancamiento
- 0.125% por lado; equilibrio ≈ 0.060% por lado
- Rotación 5.3×/año ≈ 1.3% del capital/año

Figura: `05_curva_costos.png`

## 5. Walk-forward: ¿cuánto sobrevive? (Paula, 50 s)

- Train 6 meses → test 1 mes, paso mensual, 100 trials/estudio
- WFE rolling = -0.07; anchored = -0.31
- 460,800 configuraciones en 63 min

Figura: `13_rolling_vs_anclado.png`

## 6. Sensibilidad ±20% (Paula, 50 s)

- 12 de 13 parámetros: meseta
- Mayor cambio del Calmar: 69.7%
- θ robusto = mediana del top 10% (centro de la meseta)

Figura: `04_sensibilidad.png`

## 7. Diagnóstico de la optimización (Paula, 50 s)

- Superficie del Calmar: corte 2D de un espacio 10D
- Importancia fANOVA de los parámetros
- Mínimo de operaciones evita Calmar sin significado

Figura: `12d_optuna_superficie_3d.png`

## 8. Regímenes con K-means (Arturo, 50 s)

- Volatilidad + R² de tendencia; silhouette 0.47
- Duración promedio 50 días (objetivo ≥ 10)
- Fit solo con datos ≤ fin del train; actualización semanal

Figura: `06c_valor_con_regimenes.png`

## 9. Desempeño por régimen (Arturo, 50 s)

- Kruskal-Wallis p = 0.094
- Crisis: M = 0.3 y solo 3 de 3
- Transiciones: posiciones conservan su SL/TP

Figura: `09_transiciones_regimen.png`

## 10. Risk Parity vs. pesos iguales (Arturo, 50 s)

- Spinu: min ½yᵀΣy − (1/n)Σ ln y; w = y/Σy
- Calmar: EW -0.07 · 1/σ -0.02 · RP -0.04
- MDD RP 15.8% vs. EW 18.5%

Figura: `07a_contribuciones_riesgo.png`

## 11. Resultado: entrenamiento y prueba (Arturo, 50 s)

- WF-OOS: CAGR -0.7%, Calmar -0.04 (B&H 0.76)
- TEST congelado: CAGR -0.5%, Calmar -0.06 (B&H 0.99)

Figura: `01_valor_portafolio.png`

## 12. Conclusiones y limitaciones (Milca, 50 s)

- Ejecución ideal: sin impacto ni slippage de stops
- Universo tech correlacionado y sesgo de selección
- θ inestable entre ventanas: re-optimización con cautela

Figura: `02_drawdown.png`
