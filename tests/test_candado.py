"""Candado del TEST: θ alterado o re-optimizado después de ver el TEST aborta."""
import copy

import pytest

from src.optimize import LockViolationError, canonical_hash, check_test_lock


def frozen(value=10):
    content = {"theta": {"NVDA": {"Tendencia": {"ema_fast": value}}}}
    return {"contenido": content, "sha256": canonical_hash(content)}


def test_primera_corrida_exige_arbol_limpio(tmp_path):
    with pytest.raises(LockViolationError, match="limpio"):
        check_test_lock(frozen(), tmp_path / "lock.json", tree_clean=False, head="abc")
    assert not (tmp_path / "lock.json").exists()


def test_rerun_con_mismo_theta_se_permite(tmp_path):
    lock = check_test_lock(frozen(), tmp_path / "lock.json", tree_clean=True, head="abc")
    again = check_test_lock(frozen(), tmp_path / "lock.json", tree_clean=False, head="def")
    assert again == lock and lock["commit"] == "abc"


def test_reoptimizar_despues_del_test_aborta(tmp_path):
    check_test_lock(frozen(10), tmp_path / "lock.json", tree_clean=True, head="abc")
    with pytest.raises(LockViolationError, match="re-optimizar"):
        check_test_lock(frozen(12), tmp_path / "lock.json", tree_clean=True, head="abc")


def test_editar_theta_a_mano_aborta(tmp_path):
    f = frozen()
    edited = copy.deepcopy(f)
    edited["contenido"]["theta"]["NVDA"]["Tendencia"]["ema_fast"] = 11
    with pytest.raises(LockViolationError, match="modificado"):
        check_test_lock(edited, tmp_path / "lock.json", tree_clean=True, head="abc")
