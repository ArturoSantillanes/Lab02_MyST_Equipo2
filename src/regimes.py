"""Detección de régimen de mercado con K-means sobre un índice equiponderado.

Régimen único para los 6 activos, con features en ventana móvil de 63 días
(3 meses, especificación Nivel C). Tres regímenes: Tendencia, Reversión, Crisis.

Causalidad
- Las features en t usan solo datos ≤ t (ventanas hacia atrás).
- El scaler y K-means se ajustan solo con features ≤ fin del train de cada
  ventana del walk-forward; fuera de ese train solo se hace `predict`.
- La etiqueta se actualiza cada 5 días hábiles (días con índice global múltiplo
  de 5) y un cambio se confirma solo si aparece en 2 actualizaciones seguidas.
"""
import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src import config

TREND, REVERSION, CRISIS = 0, 1, 2
FEATURES = ["volatilidad", "atr_norm", "fuerza_tendencia", "r2_tendencia", "autocorr_1"]
MA_FAST = 21
SILHOUETTE_TARGET = 0.4


def equal_weight_index(close: pd.DataFrame) -> pd.Series:
    """Índice equiponderado rebalanceado diario: r_I,t = (1/n)·Σ_i r_i,t,  I_t = Π(1 + r_I)."""
    r = close.pct_change().mean(axis=1).fillna(0.0)
    return (1 + r).cumprod().rename("indice_ew")


def regime_features(panel: pd.DataFrame, window: int = config.REGIME_WINDOW) -> pd.DataFrame:
    """Features de régimen sobre ventana móvil de `window` días (todas causales).

    volatilidad      σ_t = √(1/n·Σ r²_{I,t−i})·√252           -> separa Crisis
    atr_norm         media_63 de (1/n_a)·Σ_i TR_i,t / C_i,t−1   -> amplitud intradía comparable
    fuerza_tendencia |ln(MA_21 / MA_63)| / σ_diaria              -> desplazamiento direccional
    r2_tendencia     R² de ln(I) contra el tiempo en la ventana  -> qué tan lineal es la tendencia
    autocorr_1       ρ₁ de r_I en la ventana                     -> persistencia (>0) o reversión (<0)
    """
    close = panel["close"][config.TICKERS]
    index = equal_weight_index(close)
    r = index.pct_change()
    vol_daily = np.sqrt((r ** 2).rolling(window).mean())

    prev_close = close.shift(1)
    tr = np.maximum(panel["high"][config.TICKERS] - panel["low"][config.TICKERS],
                    np.maximum((panel["high"][config.TICKERS] - prev_close).abs(),
                               (panel["low"][config.TICKERS] - prev_close).abs()))
    natr = (tr / prev_close).mean(axis=1).rolling(window).mean()

    ma_fast, ma_slow = index.rolling(MA_FAST).mean(), index.rolling(window).mean()
    trend_strength = np.log(ma_fast / ma_slow).abs() / vol_daily

    log_i = np.log(index)
    time = pd.Series(np.arange(len(index), dtype=float), index=index.index)
    r2 = log_i.rolling(window).corr(time) ** 2
    autocorr = r.rolling(window).corr(r.shift(1))

    feats = pd.DataFrame({"volatilidad": vol_daily * np.sqrt(config.TRADING_DAYS), "atr_norm": natr,
                          "fuerza_tendencia": trend_strength, "r2_tendencia": r2,
                          "autocorr_1": autocorr})
    return feats.dropna()


def label_map(centers_scaled: np.ndarray, feature_names: list[str]) -> dict[int, int]:
    """Re-etiquetado reproducible de clusters (resuelve el label switching).

    Crisis = centroide con mayor volatilidad (o ATR normalizado si no está la volatilidad).
    De los dos restantes, Tendencia = mayor promedio estandarizado de fuerza de
    tendencia y R² (o ρ₁ si no están); el otro es Reversión.
    Solo depende de los centroides, así que es invariante a permutar los IDs.
    """
    idx = {f: j for j, f in enumerate(feature_names)}
    vol_col = idx.get("volatilidad", idx.get("atr_norm"))
    crisis = int(np.argmax(centers_scaled[:, vol_col]))
    rest = [k for k in range(len(centers_scaled)) if k != crisis]
    trend_cols = [idx[f] for f in ("fuerza_tendencia", "r2_tendencia") if f in idx] or [idx["autocorr_1"]]
    score = {k: centers_scaled[k, trend_cols].mean() for k in rest}
    trend = max(rest, key=score.get)
    reversion = min(rest, key=score.get)
    return {crisis: CRISIS, trend: TREND, reversion: REVERSION}


@dataclass
class RegimeModel:
    """Scaler + K-means congelados y su mapeo cluster -> régimen."""
    feature_names: list[str]
    scaler: StandardScaler
    kmeans: KMeans
    mapping: dict[int, int]
    fit_end: pd.Timestamp
    silhouette: float

    def predict(self, feats: pd.DataFrame) -> np.ndarray:
        clusters = self.kmeans.predict(self.scaler.transform(feats[self.feature_names].to_numpy()))
        return np.vectorize(self.mapping.get)(clusters)

    def centroids(self) -> pd.DataFrame:
        """Centroides en unidades originales, indexados por nombre de régimen."""
        raw = self.scaler.inverse_transform(self.kmeans.cluster_centers_)
        names = [config.REGIME_NAMES[self.mapping[k]] for k in range(len(raw))]
        return pd.DataFrame(raw, index=names, columns=self.feature_names)

    def to_dict(self) -> dict:
        """Representación JSON (para congelar el modelo del TEST)."""
        return {"features": self.feature_names, "fit_end": str(self.fit_end.date()),
                "scaler_mean": self.scaler.mean_.tolist(), "scaler_scale": self.scaler.scale_.tolist(),
                "centers_scaled": self.kmeans.cluster_centers_.tolist(),
                "mapping": {str(k): config.REGIME_NAMES[v] for k, v in self.mapping.items()},
                "silhouette": self.silhouette}


@dataclass
class FrozenRegimeModel:
    """Modelo de régimen reconstruido de `RegimeModel.to_dict()` (el que se congela para TEST).

    predict = centroide más cercano en el espacio estandarizado, idéntico a KMeans.predict.
    """
    feature_names: list[str]
    mean: np.ndarray
    scale: np.ndarray
    centers: np.ndarray
    mapping: dict[int, int]

    @classmethod
    def from_dict(cls, d: dict) -> "FrozenRegimeModel":
        codes = {v: k for k, v in config.REGIME_NAMES.items()}
        return cls(d["features"], np.array(d["scaler_mean"]), np.array(d["scaler_scale"]),
                   np.array(d["centers_scaled"]), {int(k): codes[v] for k, v in d["mapping"].items()})

    def predict(self, feats: pd.DataFrame) -> np.ndarray:
        x = (feats[self.feature_names].to_numpy() - self.mean) / self.scale
        clusters = np.argmin(((x[:, None, :] - self.centers[None, :, :]) ** 2).sum(axis=2), axis=1)
        return np.array([self.mapping[c] for c in clusters])


def fit_regime_model(feats: pd.DataFrame, fit_end: pd.Timestamp,
                     feature_names: list[str] = FEATURES) -> RegimeModel:
    """Ajusta scaler y K-means (K=3 por teoría) con features ≤ `fit_end` únicamente."""
    x_raw = feats.loc[:fit_end, feature_names].to_numpy()
    scaler = StandardScaler().fit(x_raw)
    x = scaler.transform(x_raw)
    km = KMeans(n_clusters=config.N_REGIMES, init="k-means++", n_init=10, random_state=config.SEED).fit(x)
    sil = float(silhouette_score(x, km.labels_, sample_size=min(len(x), 3000), random_state=config.SEED))
    return RegimeModel(list(feature_names), scaler, km, label_map(km.cluster_centers_, feature_names),
                       pd.Timestamp(fit_end), sil)


def feature_subset_table(feats: pd.DataFrame, fit_end: pd.Timestamp) -> pd.DataFrame:
    """Silhouette de K-means para cada subconjunto válido de features (datos ≤ `fit_end`).

    Válido = al menos una feature de volatilidad (para identificar Crisis) y una de
    tendencia/persistencia (para separar Tendencia de Reversión). Ordenado de mejor a peor.
    """
    vol, trend = {"volatilidad", "atr_norm"}, {"fuerza_tendencia", "r2_tendencia", "autocorr_1"}
    rows = []
    for k in range(2, len(FEATURES) + 1):
        for combo in itertools.combinations(FEATURES, k):
            if vol & set(combo) and trend & set(combo):
                model = fit_regime_model(feats, fit_end, list(combo))
                rows.append({"features": "+".join(combo), "n_features": k, "silhouette": model.silhouette})
    return pd.DataFrame(rows).sort_values(["silhouette", "n_features"], ascending=[False, True]).reset_index(drop=True)


def select_features(feats: pd.DataFrame, fit_end: pd.Timestamp) -> tuple[list[str], pd.DataFrame]:
    """Usa las 5 features si su silhouette ≥ 0.4; si no, el subconjunto con mayor silhouette.

    Se decide una sola vez con datos de TRAIN (nunca con TEST) y se documenta.
    """
    table = feature_subset_table(feats, fit_end)
    full = table.loc[table["n_features"] == len(FEATURES), "silhouette"].iloc[0]
    chosen = FEATURES if full >= SILHOUETTE_TARGET else table.loc[0, "features"].split("+")
    return list(chosen), table


def update_days(index: pd.DatetimeIndex, full_index: pd.DatetimeIndex,
                every: int = config.REGIME_UPDATE_EVERY) -> np.ndarray:
    """Máscara de días de actualización: posición en el calendario completo múltiplo de `every`.

    Anclar al calendario completo (no al recorte) evita que el calendario de
    actualizaciones cambie al agregar o quitar datos.
    """
    pos = full_index.get_indexer(index)
    return pos % every == 0


def classify(feats: pd.DataFrame, schedule: list[tuple[pd.Timestamp, RegimeModel]],
             full_index: pd.DatetimeIndex, every: int = config.REGIME_UPDATE_EVERY,
             persistence: int = config.REGIME_PERSISTENCE) -> pd.Series:
    """Etiqueta diaria de régimen con modelos que cambian según el calendario.

    `schedule`: lista ordenada de (fecha desde la que rige, modelo). En cada día de
    actualización se predice con el modelo vigente; el régimen cambia solo si la
    nueva etiqueta aparece en `persistence` actualizaciones seguidas. Entre
    actualizaciones se mantiene la última etiqueta confirmada.
    """
    starts = [s for s, _ in schedule]
    feats = feats.loc[feats.index >= starts[0]]
    raw = np.full(len(feats), -1)
    for k, (start, model) in enumerate(schedule):
        end = starts[k + 1] if k + 1 < len(schedule) else None
        sel = (feats.index >= start) & ((feats.index < end) if end is not None else True)
        if sel.any():
            raw[sel] = model.predict(feats.loc[sel])
    is_update = update_days(feats.index, full_index, every)

    labels = np.empty(len(feats), dtype=int)
    current, candidate, streak = None, None, 0
    for t, (r, upd) in enumerate(zip(raw, is_update)):
        if upd or current is None:
            if current is None:
                current = r
            elif r != current:
                streak = streak + 1 if r == candidate else 1
                candidate = r
                if streak >= persistence:
                    current, candidate, streak = r, None, 0
            else:
                candidate, streak = None, 0
        labels[t] = current
    return pd.Series(labels, index=feats.index, name="regimen")


def run_lengths(labels: pd.Series) -> pd.DataFrame:
    """Rachas consecutivas de cada régimen: (régimen, inicio, fin, duración en días)."""
    change = labels.ne(labels.shift()).cumsum()
    g = labels.groupby(change)
    return pd.DataFrame({"regimen": g.first().map(config.REGIME_NAMES), "inicio": g.apply(lambda s: s.index[0]),
                         "fin": g.apply(lambda s: s.index[-1]), "duracion": g.size()}).reset_index(drop=True)


def transition_matrix(labels: pd.Series) -> pd.DataFrame:
    """Matriz empírica A_ij = P(régimen_{t+1} = j | régimen_t = i) con etiquetas diarias."""
    names = [config.REGIME_NAMES[k] for k in range(config.N_REGIMES)]
    counts = pd.crosstab(labels.iloc[:-1].to_numpy(), labels.iloc[1:].to_numpy())
    counts = counts.reindex(index=range(config.N_REGIMES), columns=range(config.N_REGIMES), fill_value=0)
    probs = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    probs.index, probs.columns = names, names
    return probs


def persistence_stats(labels: pd.Series) -> dict:
    """Duración promedio observada vs. esperada E[D_j] = 1/(1 − A_jj), transiciones y proporciones."""
    runs = run_lengths(labels)
    A = transition_matrix(labels)
    years = len(labels) / config.TRADING_DAYS
    out = {"n_transiciones": int(len(runs) - 1), "transiciones_por_anio": float((len(runs) - 1) / years),
           "duracion_promedio_global": float(runs["duracion"].mean()), "por_regimen": {}}
    for k, name in config.REGIME_NAMES.items():
        r = runs.loc[runs["regimen"] == name, "duracion"]
        a_jj = A.loc[name, name]
        out["por_regimen"][name] = {
            "proporcion_tiempo": float((labels == k).mean()),
            "n_episodios": int(len(r)),
            "duracion_promedio_observada": float(r.mean()) if len(r) else None,
            "duracion_esperada_markov": float(1 / (1 - a_jj)) if pd.notna(a_jj) and a_jj < 1 else None,
        }
    return out


def bootstrap_mean_ci(returns: pd.Series, labels: pd.Series, n_boot: int = config.BOOTSTRAP_SAMPLES,
                      seed: int = config.SEED, alpha: float = 0.05) -> pd.DataFrame:
    """IC bootstrap (percentil) de la media del rendimiento diario por régimen.

    Remuestreo i.i.d. con reemplazo dentro de cada régimen. Se reporta además la
    media anualizada (×252) para lectura.
    """
    rng = np.random.default_rng(seed)
    aligned = pd.concat([returns.rename("r"), labels.rename("regimen")], axis=1, join="inner").dropna()
    rows = []
    for k, name in config.REGIME_NAMES.items():
        x = aligned.loc[aligned["regimen"] == k, "r"].to_numpy()
        if len(x) < 2:
            rows.append({"regimen": name, "n_dias": len(x)})
            continue
        means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
        lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
        rows.append({"regimen": name, "n_dias": len(x), "media_diaria": x.mean(), "ic_inf": lo, "ic_sup": hi,
                     "media_anualizada": x.mean() * config.TRADING_DAYS,
                     "vol_anualizada": x.std(ddof=1) * np.sqrt(config.TRADING_DAYS)})
    return pd.DataFrame(rows)


def kruskal_by_regime(returns: pd.Series, labels: pd.Series) -> dict:
    """Prueba de Kruskal-Wallis: ¿la distribución de rendimientos difiere entre regímenes?

    No paramétrica (no supone normalidad, robusta a colas pesadas). H0: misma distribución.
    """
    from scipy.stats import kruskal
    aligned = pd.concat([returns.rename("r"), labels.rename("g")], axis=1, join="inner").dropna()
    groups = [g["r"].to_numpy() for _, g in aligned.groupby("g") if len(g) > 1]
    if len(groups) < 2:
        return {"estadistico": None, "p_valor": None}
    h, p = kruskal(*groups)
    return {"estadistico": float(h), "p_valor": float(p)}
