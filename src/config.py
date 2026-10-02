"""Configuración central del Lab 02 (Equipo 2, Nivel C).

Única fuente de constantes, semilla y rutas. Todos los módulos importan de aquí;
ningún otro archivo define parámetros fijos del laboratorio.
"""
from pathlib import Path

# --- Reproducibilidad -------------------------------------------------------
SEED = 42

# --- Rutas (relativas a la raíz del repo) ------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"
FIGURES_DIR = DOCS_DIR / "figures"
RESULTS_DIR = DOCS_DIR / "resultados"
CACHE_DIR = ROOT / ".cache"
QUICK_DIR = CACHE_DIR / "quick"

PRICES_CSV = DATA_DIR / "prices_daily.csv"
METADATA_JSON = DATA_DIR / "download_metadata.json"

# --- Universo y asignación (random.seed(42) sobre la lista original) ---------
TICKERS = ["NVDA", "AMZN", "TSLA", "META", "NFLX", "GOOGL"]
ASSIGNMENT = {
    "Milca": ["TSLA", "NFLX"],
    "Paula": ["META", "AMZN"],
    "Arturo": ["NVDA", "GOOGL"],
}

# --- Datos -------------------------------------------------------------------
START_DATE = "2015-01-01"
END_DATE = "2026-08-31"          # inclusivo
TRAIN_FRACTION = 0.80            # TRAIN = primer 80%, TEST = último 20%

# --- Parámetros fijos del laboratorio (no se optimizan) ----------------------
INITIAL_CAPITAL = 1_000_000.0
COMMISSION = 0.00125             # 0.125% del nocional en entrada y salida
MAX_GROSS_EXPOSURE = 1.0         # sin apalancamiento: sum|w_i| <= 1
MIN_AGREE = 2                    # confirmación 2 de 3
TRADING_DAYS = 252
RISK_FREE = 0.0
ATR_WINDOW = 14
CALMAR_MDD_FLOOR = 0.01          # piso de |MDD| para Calmar

# --- Walk-forward y optimización ---------------------------------------------
WF_TRAIN_MONTHS = 6
WF_TEST_MONTHS = 1
N_TRIALS = 100                   # por régimen por ventana
N_STARTUP_TRIALS = 30            # exploración aleatoria antes de TPE
N_MIN_GLOBAL = 10                # operaciones mínimas por activo en 6 meses
N_MIN_REGIME = 5                 # operaciones mínimas por activo en un régimen
MIN_REGIME_DAYS = 20             # días mínimos del régimen para optimizarlo
EMBARGO_DAYS = 5                 # días finales excluidos del objetivo
ROBUST_TOP_FRACTION = 0.10       # θ robusto = mediana del top 10% de trials
INVALID_OBJECTIVE = -1e9

# --- Regímenes ---------------------------------------------------------------
REGIME_WINDOW = 63               # 3 meses de días hábiles
REGIME_UPDATE_EVERY = 5          # reclasificación semanal
REGIME_PERSISTENCE = 2           # confirmaciones seguidas para cambiar
N_REGIMES = 3
REGIME_NAMES = {0: "Tendencia", 1: "Reversión", 2: "Crisis"}
BOOTSTRAP_SAMPLES = 1000

# --- Portafolio --------------------------------------------------------------
COV_WINDOW = 126                 # 6 meses para covarianza y correlación
REBALANCE_EVERY = 5              # calendario semanal
REBALANCE_BAND = 0.05            # δ: rebalancear solo si ‖w − w*‖₁ > δ (híbrido)
COV_ESTIMATOR = "muestral"       # alternativas comparadas: "ewma", "ledoit_wolf"
EWMA_LAMBDA = 0.94
CONFLICT_CORR = 0.70
REGIME_MULTIPLIER = {"Tendencia": 1.0, "Reversión": 0.7, "Crisis": 0.3}
REGIME_MIN_AGREE = {"Tendencia": 2, "Reversión": 2, "Crisis": 3}
RP_TOLERANCE = 1e-4
