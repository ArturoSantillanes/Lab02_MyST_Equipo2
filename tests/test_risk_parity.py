"""Risk Parity y agregación: RC iguales (tolerancia 1e-4) y ejemplos numéricos de la clase."""
import numpy as np
import pytest

from src import config
from src.portfolio import (RiskParityError, compose_target, estimate_covariance, inverse_vol_weights,
                           risk_contributions, risk_parity_weights, signal_strength, turnover)

TOL = config.RP_TOLERANCE


def synthetic_cov(n=6, seed=config.SEED):
    rng = np.random.default_rng(seed)
    vols = rng.uniform(0.015, 0.045, n)               # vol diaria heterogénea (TSLA/NVDA vs GOOGL)
    corr = 0.6 + 0.3 * rng.uniform(size=(n, n))
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)
    corr = corr + n * 1e-3 * np.eye(n)                 # definida positiva
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    return np.outer(vols, vols) * corr


@pytest.mark.parametrize("seed", [1, 2, 3, 42])
def test_contribuciones_iguales(seed):
    cov = synthetic_cov(seed=seed)
    w = risk_parity_weights(cov)
    rc = risk_contributions(w, cov)
    assert np.max(np.abs(rc - 1 / 6)) < TOL
    assert w.sum() == pytest.approx(1.0) and (w > 0).all()


def test_definicion_rc_igual_sigma_p_entre_n_y_euler():
    """RC_i = w_i·(Σw)_i/σ_p = σ_p/n y Σ_i RC_i = σ_p exacto (Teorema de Euler)."""
    cov = synthetic_cov()
    w = risk_parity_weights(cov)
    sigma_p = np.sqrt(w @ cov @ w)
    rc_abs = w * (cov @ w) / sigma_p
    np.testing.assert_allclose(rc_abs, sigma_p / 6, rtol=0, atol=TOL * sigma_p)
    assert rc_abs.sum() == pytest.approx(sigma_p, rel=1e-12)


def test_spinu_cumple_la_condicion_de_primer_orden():
    """En el óptimo de ½yᵀΣy − (1/n)Σ ln y_i se cumple y_i·(Σy)_i = 1/n para todo i."""
    cov = synthetic_cov()
    w = risk_parity_weights(cov)
    y = w / np.sqrt(w @ cov @ w)                       # y = c·w con c² = 1/(wᵀΣw)
    np.testing.assert_allclose(y * (cov @ y), 1 / 6, atol=1e-6)


def test_ilusion_del_50_50_de_la_clase():
    """σ1 = 5%, σ2 = 25%, ρ = 0, w = (0.5, 0.5): σ_p ≈ 12.75% y el activo 2 aporta ≈ 96% del riesgo."""
    cov = np.diag([0.05 ** 2, 0.25 ** 2])
    w = np.array([0.5, 0.5])
    assert np.sqrt(w @ cov @ w) == pytest.approx(0.12748, abs=1e-5)
    assert risk_contributions(w, cov)[1] == pytest.approx(0.9615, abs=1e-4)


def test_volatilidad_inversa_ejemplo_de_la_clase():
    w = inverse_vol_weights(np.diag([0.05 ** 2, 0.25 ** 2]))
    np.testing.assert_allclose(w, [20 / 24, 4 / 24])


def test_correlacion_constante_rp_igual_a_volatilidad_inversa():
    """Con ρ_ij iguales, Risk Parity coincide exactamente con 1/σ."""
    vols = np.array([0.04, 0.035, 0.02, 0.03, 0.025, 0.018])
    corr = np.full((6, 6), 0.7)
    np.fill_diagonal(corr, 1.0)
    cov = np.outer(vols, vols) * corr
    np.testing.assert_allclose(risk_parity_weights(cov), inverse_vol_weights(cov), atol=1e-5)


def test_presupuestos_de_riesgo():
    cov = synthetic_cov()
    budgets = np.array([1, 2 / 3, 1, 2 / 3, 1, 1])
    w = risk_parity_weights(cov, budgets)
    np.testing.assert_allclose(risk_contributions(w, cov), budgets / budgets.sum(), atol=TOL)


def test_composicion_ejemplo_integrador_de_la_clase():
    """w^RP = (0.5, 0.3, 0.2), votos A=(1,1,1), B=(1,1,0), C sin señal, m = 0.7 -> (0.35, 0.14, 0)."""
    s = signal_strength(np.array([1, 1, 0]), np.array([3, 2, 0]))
    np.testing.assert_allclose(s, [1.0, 2 / 3, 0.0])
    np.testing.assert_allclose(compose_target(np.array([0.5, 0.3, 0.2]), s, 0.7), [0.35, 0.14, 0.0])


def test_senal_mixta_entra_con_fuerza_un_tercio():
    """Con la compuerta 2 de 3 del laboratorio, (1,−1,−1) abre corto pero con |s| = 1/3."""
    np.testing.assert_allclose(signal_strength(np.array([-1]), np.array([-1])), [-1 / 3])


def test_composicion_nunca_apalanca():
    w = compose_target(np.array([0.6, 0.6, 0.6]), np.array([1.0, -1.0, 1.0]), 1.0)
    assert np.abs(w).sum() == pytest.approx(1.0)


def test_turnover_contra_drift_ejemplo_de_la_clase():
    """Target viejo (0.4, 0.6), drift A +20% y B −10%, nuevo target (0.5, 0.5)."""
    drift = np.array([48_000, 54_000]) / 102_000
    assert turnover([0.5, 0.5], [0.4, 0.6]) == pytest.approx(0.10)
    assert turnover([0.5, 0.5], drift) == pytest.approx(0.0294, abs=1e-4)


@pytest.mark.parametrize("estimator", ["muestral", "ewma", "ledoit_wolf"])
def test_estimadores_de_covarianza_dan_rp_valido(estimator):
    rng = np.random.default_rng(config.SEED)
    window = rng.multivariate_normal(np.zeros(6), synthetic_cov(), size=126)
    cov = estimate_covariance(window, estimator)
    assert np.allclose(cov, cov.T) and np.linalg.eigvalsh(cov).min() > 0
    w = risk_parity_weights(cov)
    assert np.max(np.abs(risk_contributions(w, cov) - 1 / 6)) < TOL


def test_falla_si_no_se_alcanza_la_tolerancia():
    with pytest.raises(RiskParityError):
        risk_parity_weights(synthetic_cov(), tol=0.0)
