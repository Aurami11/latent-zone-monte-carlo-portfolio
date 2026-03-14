import os

# ======================================================
# OUTPUT / CACHE
# ======================================================
OUTPUT_DIR = "bt_topk_horizon_outputs"
CACHE_DIR = "yf_cache"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ======================================================
# UNIVERSE / DATA
# ======================================================
TICKERS = [
    "SPY", "QQQ", "DIA", "IWM", "MDY", "VTI", "RSP", "MTUM", "QUAL", "USMV",
    "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLRE", "XLC",
    "EFA", "EEM", "VWO", "VGK", "EWJ", "EWU", "EWC", "EWA",
    "BIL", "SHY", "IEF", "TLT", "TIP", "LQD", "HYG", "EMB", "BND",
    "VNQ", "GLD", "SLV", "DBC", "USO",
]

START = "2016-01-01"
END = None
INTERVAL = "1d"
BACKTEST_START = "2019-01-01"


# ======================================================
# PORTFOLIO
# ======================================================
HORIZON_STEPS = 10
TOP_K = 30
INITIAL_CAPITAL = 100_000.0
FEE_BPS = 0.0
FALLBACK_ASSET = "SHY"
ANNUALIZATION = 252

PROBA_POWER = 0.8
UPSIDE_POWER = 0.5
REQUIRE_PUP_GT_PDN = True

BENCH_6040 = {"SPY": 0.60, "IEF": 0.40}


# ======================================================
# TUBES / STATE
# ======================================================
TUBE_LEVELS = [
    (0.40, 0.60),
    (0.25, 0.75),
    (0.10, 0.90),
    (0.02, 0.98),
]
TUBE_WINDOW = 252
TUBE_SHIFT = 1

VOL_WINDOW = 60
VOL_SHIFT = 1

VOL_SHELLS_ENABLED = True
VOL_ALPHA_INNER = 1.0
VOL_ALPHA_OUTER = 2.0
VOL_SHELL_USE_SQRT_H = True
TAIL_SIGMA_MULTIPLIER = 2.5


# ======================================================
# LATENT ZONE FILTER
# ======================================================
ENERGY_QUANTILE = 0.98
ENERGY_LOOKBACK = 504
MIN_ZONE_OBS_FOR_Q = 80

ZONE_P_MIN = 0.02
ZONE_P_MAX = 0.98
ZONE_KAPPA = 1.0
ZONE_DIST_CAP = 3

ENERGY_THRESHOLDS_TUBE = [1.0, 2.0, 4.0, 6.0]
ENERGY_THRESHOLDS_VOL = [6.0, 8.0]


# ======================================================
# LOCAL DYNAMICS / MONTE CARLO
# ======================================================
N_PATHS = 2500
SEED = 42
FIT_WINDOW = 504
REFIT_EVERY = 21

DRIFT_MODE = "linear"
TANH_LAMBDA = 1.0
POSITION_EPS = 1e-3