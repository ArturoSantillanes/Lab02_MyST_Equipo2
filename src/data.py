"""Datos: descarga congelada, carga, auditoría, limpieza y split cronológico.

Flujo: `download_prices` (solo si no existe el CSV) -> `load_prices` -> `audit_data`
-> `clean_data` -> `chronological_split`. `prepare_data` encadena todo.

Convención de panel: DataFrame indexado por fecha con columnas MultiIndex
(campo, ticker), campos en minúsculas: open, high, low, close, volume.
"""
import json
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src import config

log = logging.getLogger(__name__)

FIELDS = ["open", "high", "low", "close", "volume"]
EXTREME_RETURN = 0.25


def download_prices(force: bool = False) -> None:
    """Descarga OHLCV diario ajustado (auto_adjust=True) y lo congela en data/.

    No hace nada si el CSV ya existe (yfinance puede cambiar históricos), salvo
    `force=True`. Guarda formato largo y un JSON con la metadata de la descarga.
    """
    if config.PRICES_CSV.exists() and not force:
        log.info("Datos congelados encontrados en %s; no se descarga.", config.PRICES_CSV)
        return
    import yfinance as yf

    # yfinance trata `end` como exclusivo: se suma un día para incluir END_DATE.
    end_exclusive = (pd.Timestamp(config.END_DATE) + timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(config.TICKERS, start=config.START_DATE, end=end_exclusive,
                      auto_adjust=True, group_by="ticker", progress=False, threads=False)
    frames = []
    for ticker in config.TICKERS:
        sub = raw[ticker].rename(columns=str.lower)[FIELDS].dropna(how="all")
        sub = sub.rename_axis("date").reset_index()
        sub.insert(1, "ticker", ticker)
        frames.append(sub)
    long = pd.concat(frames, ignore_index=True)
    long["date"] = pd.to_datetime(long["date"]).dt.strftime("%Y-%m-%d")

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    long.to_csv(config.PRICES_CSV, index=False)
    metadata = {
        "fuente": "Yahoo Finance vía yfinance (auto_adjust=True)",
        "fecha_descarga": datetime.now().isoformat(timespec="seconds"),
        "version_yfinance": yf.__version__,
        "tickers": config.TICKERS,
        "inicio": config.START_DATE,
        "fin_inclusivo": config.END_DATE,
        "filas": len(long),
    }
    config.METADATA_JSON.write_text(json.dumps(metadata, indent=2, ensure_ascii=False),
                                    encoding="utf-8")
    log.info("Descargadas %d filas y congeladas en %s", len(long), config.PRICES_CSV)


def load_prices() -> pd.DataFrame:
    """Carga el CSV congelado como panel (campo, ticker) sobre el calendario unión.

    Se usa la unión de fechas para que los faltantes queden visibles como NaN y
    la auditoría los cuente; `clean_data` después se queda con las fechas comunes.
    """
    long = pd.read_csv(config.PRICES_CSV, parse_dates=["date"])
    panel = long.pivot_table(index="date", columns="ticker", values=FIELDS, aggfunc="first")
    panel = panel.reindex(columns=pd.MultiIndex.from_product([FIELDS, config.TICKERS]))
    return panel.sort_index()


def count_duplicates() -> pd.Series:
    """Número de filas (fecha, ticker) duplicadas por ticker en el CSV crudo."""
    long = pd.read_csv(config.PRICES_CSV)
    dup = long.duplicated(subset=["date", "ticker"], keep="first")
    return long[dup].groupby("ticker").size().reindex(config.TICKERS, fill_value=0)


def audit_data(panel: pd.DataFrame) -> pd.DataFrame:
    """Auditoría por ticker: faltantes, NaNs, duplicados, OHLC, no positivos y |r| > 25%.

    Consistencia OHLC: high >= max(open, close) y low <= min(open, close).
    Rendimientos extremos: r_t = C_t / C_{t-1} - 1 con |r_t| > 25%; se listan con
    fecha para decidir si son eventos reales o splits mal ajustados.
    """
    duplicates = count_duplicates()
    rows = []
    for t in config.TICKERS:
        o, h, l, c, v = (panel[(f, t)] for f in FIELDS)
        present = c.notna()
        ret = c.pct_change(fill_method=None)
        extreme = ret[ret.abs() > EXTREME_RETURN]
        rows.append({
            "ticker": t,
            "primera_fecha": c[present].index.min().date(),
            "ultima_fecha": c[present].index.max().date(),
            "dias_con_datos": int(present.sum()),
            "fechas_faltantes_vs_calendario": int((~present).sum()),
            "nans_en_filas_presentes": int(panel.xs(t, axis=1, level=1)[present].isna().sum().sum()),
            "duplicados": int(duplicates[t]),
            "violaciones_high": int((h < np.maximum(o, c) - 1e-9).sum()),
            "violaciones_low": int((l > np.minimum(o, c) + 1e-9).sum()),
            "precios_no_positivos": int((panel.xs(t, axis=1, level=1)[["open", "high", "low", "close"]] <= 0).sum().sum()),
            "volumen_no_positivo": int((v[present] <= 0).sum()),
            "n_rend_extremos": len(extreme),
            "rend_extremos": "; ".join(f"{d.date()}:{r:+.1%}" for d, r in extreme.items()),
        })
    audit = pd.DataFrame(rows)
    audit["completo_y_traslapado"] = (
        (audit["fechas_faltantes_vs_calendario"] == 0)
        & (audit["nans_en_filas_presentes"] == 0)
        & (audit["primera_fecha"] == audit["primera_fecha"].max())
        & (audit["ultima_fecha"] == audit["ultima_fecha"].min())
    )
    return audit


def clean_data(panel: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Conserva solo fechas con datos completos en los 6 activos (sin forward-fill).

    Regresa el panel limpio y el número de filas eliminadas.
    """
    complete = panel.notna().all(axis=1)
    return panel[complete].copy(), int((~complete).sum())


def chronological_split(index: pd.DatetimeIndex, train_fraction: float = config.TRAIN_FRACTION) -> dict:
    """Split sin traslape: TRAIN = primer `train_fraction` de días, TEST = el resto."""
    n_train = int(np.floor(len(index) * train_fraction))
    return {
        "train_inicio": str(index[0].date()),
        "train_fin": str(index[n_train - 1].date()),
        "test_inicio": str(index[n_train].date()),
        "test_fin": str(index[-1].date()),
        "dias_train": n_train,
        "dias_test": len(index) - n_train,
        "fraccion_train": train_fraction,
    }


def prepare_data(results_dir=config.RESULTS_DIR) -> tuple[pd.DataFrame, dict]:
    """Descarga si hace falta, audita, limpia y guarda auditoría y split en `results_dir`."""
    download_prices()
    panel = load_prices()
    results_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_data(panel)
    audit.to_csv(results_dir / "auditoria_datos.csv", index=False)
    clean, dropped = clean_data(panel)
    log.info("Limpieza: %d fechas eliminadas por no tener los 6 activos", dropped)
    splits = chronological_split(clean.index)
    splits["fechas_eliminadas_limpieza"] = dropped
    (results_dir / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    return clean, splits


def load_clean() -> tuple[pd.DataFrame, dict]:
    """Panel limpio y su split (determinista: se recalcula igual que en `prepare_data`)."""
    clean, dropped = clean_data(load_prices())
    splits = chronological_split(clean.index)
    splits["fechas_eliminadas_limpieza"] = dropped
    return clean, splits
