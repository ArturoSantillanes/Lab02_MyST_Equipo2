"""Régimen causal: la etiqueta en t no cambia al agregar datos posteriores; re-etiquetado invariante."""
import numpy as np
import pandas as pd
import pytest

from conftest import synthetic_panel
from src import regimes as R

NAMES = ["volatilidad", "r2_tendencia"]


@pytest.fixture(scope="module")
def setup():
    panel = synthetic_panel()
    feats = R.regime_features(panel)
    fit_end = feats.index[250]
    model = R.fit_regime_model(feats, fit_end, NAMES)
    return panel, feats, model, fit_end


def test_features_causales(setup):
    panel, feats, _, _ = setup
    t = 500
    truncated = R.regime_features(panel.iloc[: t + 1])
    pd.testing.assert_frame_equal(feats.loc[: truncated.index[-1]], truncated)


@pytest.mark.parametrize("t", [350, 450, 650, 699])
def test_etiqueta_en_t_no_cambia_con_datos_futuros(setup, t):
    panel, feats, model, fit_end = setup
    start = feats.index[feats.index > fit_end][0]
    full = R.classify(feats, [(start, model)], panel.index)
    truncated_feats = R.regime_features(panel.iloc[: t + 1])
    partial = R.classify(truncated_feats, [(start, model)], panel.index)
    date = panel.index[t]
    assert partial.loc[date] == full.loc[date]
    pd.testing.assert_series_equal(full.loc[:date], partial)


def test_ajuste_no_usa_datos_posteriores_a_fit_end(setup):
    """Alterar las features después de fit_end no cambia el modelo ajustado."""
    _, feats, model, fit_end = setup
    shocked = feats.copy()
    shocked.loc[shocked.index > fit_end] *= 10
    other = R.fit_regime_model(shocked, fit_end, NAMES)
    np.testing.assert_allclose(model.kmeans.cluster_centers_, other.kmeans.cluster_centers_)


def test_reetiquetado_invariante_a_permutaciones(setup):
    _, _, model, _ = setup
    centers = model.kmeans.cluster_centers_
    base = R.label_map(centers, NAMES)
    for perm in [(1, 2, 0), (2, 0, 1), (0, 2, 1)]:
        permuted = R.label_map(centers[list(perm)], NAMES)
        # el cluster que quedó en la posición k es el original perm[k]: mismo régimen
        assert {k: permuted[k] for k in range(3)} == {k: base[perm[k]] for k in range(3)}


def test_crisis_es_el_cluster_de_mayor_volatilidad(setup):
    _, _, model, _ = setup
    c = model.centroids()
    assert c["volatilidad"].idxmax() == "Crisis"
    assert c.loc["Tendencia", "r2_tendencia"] > c.loc["Reversión", "r2_tendencia"]


def test_persistencia_exige_dos_actualizaciones():
    """Un régimen que aparece en una sola actualización no se confirma."""
    class Fixed:
        def __init__(self, seq): self.seq = seq
        def predict(self, feats): return self.seq[: len(feats)]
    idx = pd.bdate_range("2020-01-01", periods=30)
    raw = np.array([0] * 10 + [2] * 5 + [0] * 5 + [1] * 10)   # Crisis solo en 1 actualización (día 10)
    labels = R.classify(pd.DataFrame(index=idx), [(idx[0], Fixed(raw))], idx)
    assert (labels.iloc[:20] == 0).all()                       # el pico aislado de Crisis se ignora
    assert labels.iloc[24] == 0 and labels.iloc[25] == 1       # 1 visto en días 20 y 25: se confirma el 25


def test_modelo_congelado_predice_igual(setup):
    """El modelo reconstruido desde el JSON congelado da las mismas etiquetas que el original."""
    _, feats, model, _ = setup
    frozen = R.FrozenRegimeModel.from_dict(model.to_dict())
    np.testing.assert_array_equal(frozen.predict(feats), model.predict(feats))
