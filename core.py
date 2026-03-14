import numpy as np
import pandas as pd
import config
from utility_func import sigmoid
from yfinance_data import download_ohlc_cached

def annual_stats_from_equity(equity: pd.Series) -> tuple[float, float, float, float, float]:
    equity = equity.dropna()
    if equity.empty:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    ret = equity.pct_change().fillna(0.0)
    n_days = len(equity)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)

    if n_days > 1 and equity.iloc[0] > 0:
        ann_return = float((equity.iloc[-1] / equity.iloc[0]) ** (config.ANNUALIZATION / n_days) - 1.0)
        ann_vol = float(ret.std(ddof=1) * np.sqrt(config.ANNUALIZATION))
    else:
        ann_return = np.nan
        ann_vol = np.nan

    sharpe = ann_return / ann_vol if (np.isfinite(ann_return) and np.isfinite(ann_vol) and ann_vol > 0) else np.nan

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_drawdown = float(drawdown.min()) if len(drawdown) else np.nan
    return total_return, ann_return, ann_vol, sharpe, max_drawdown


def horizon_rebalance_dates(
    common_idx: pd.DatetimeIndex,
    signal_df: pd.DataFrame,
    backtest_start: str | None,
    horizon_steps: int,
) -> pd.DatetimeIndex:
    start_dt = pd.Timestamp(backtest_start) if backtest_start is not None else common_idx[0]
    first_signal_dt = pd.Timestamp(signal_df["entry_date"].min())
    start_dt = max(start_dt, first_signal_dt)

    eligible = common_idx[common_idx >= start_dt]
    if len(eligible) == 0:
        raise ValueError("Aucune date de réallocation disponible.")

    return pd.DatetimeIndex(eligible[::horizon_steps])

# ======================================================
# TUBES / STATE
# ======================================================
def build_multi_tube(
    df: pd.DataFrame,
    window: int,
    levels: list[tuple[float, float]],
    shift: int = 0,
) -> list[pd.DataFrame]:
    x = np.log(df["Close"])
    tubes: list[pd.DataFrame] = []

    for q_low, q_high in levels:
        upper = x.rolling(window).quantile(q_high)
        lower = x.rolling(window).quantile(q_low)
        if shift:
            upper = upper.shift(shift)
            lower = lower.shift(shift)

        tube = pd.DataFrame({"U": np.exp(upper), "L": np.exp(lower)}).dropna()
        tube = tube.reindex(df.index, method="ffill")
        tubes.append(tube)

    return tubes


def compute_state(df: pd.DataFrame, vol_window: int, vol_shift: int) -> pd.DataFrame:
    out = df[["Close"]].copy()
    out["x"] = np.log(out["Close"])
    out["r1"] = out["x"].diff()

    vol = out["r1"].rolling(vol_window).std()
    if vol_shift:
        vol = vol.shift(vol_shift)
    out["vol"] = vol

    out["energy"] = (out["r1"] / out["vol"]) ** 2
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out


def align_state_and_tubes(
    state: pd.DataFrame,
    tubes: list[pd.DataFrame],
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    idx = state.index
    for tube in tubes:
        idx = idx.intersection(tube.dropna().index)

    state2 = state.loc[idx].copy()
    tubes2 = [tube.loc[idx].copy() for tube in tubes]
    return state2, tubes2


# ======================================================
# VOLATILITY SHELLS
# ======================================================
def build_vol_shell_bounds(
    state: pd.DataFrame,
    outer_tube: pd.DataFrame,
    alpha_inner: float,
    alpha_outer: float,
    use_sqrt_h: bool,
    horizon: int,
) -> pd.DataFrame:
    idx = state.index
    upper_outer = outer_tube.loc[idx, "U"].astype(float)
    lower_outer = outer_tube.loc[idx, "L"].astype(float)

    sig = state.loc[idx, "vol"].astype(float).clip(lower=1e-12)
    scale = np.sqrt(horizon) if use_sqrt_h else 1.0

    u4 = np.exp(np.log(upper_outer) + alpha_inner * sig * scale)
    u5 = np.exp(np.log(upper_outer) + alpha_outer * sig * scale)
    d4 = np.exp(np.log(lower_outer) - alpha_inner * sig * scale)
    d5 = np.exp(np.log(lower_outer) - alpha_outer * sig * scale)

    return pd.DataFrame(
        {
            "U_top": upper_outer,
            "U4": u4,
            "U5": u5,
            "L_bot": lower_outer,
            "D4": d4,
            "D5": d5,
        },
        index=idx,
    )


# ======================================================
# ZONES
# ======================================================
def energy_threshold_for_tube(level_idx: int) -> float:
    kk = min(max(int(level_idx), 0), len(config.ENERGY_THRESHOLDS_TUBE) - 1)
    return float(config.ENERGY_THRESHOLDS_TUBE[kk])


def energy_threshold_for_vol(level_idx: int) -> float:
    jj = 0 if int(level_idx) == 4 else 1
    jj = min(max(jj, 0), len(config.ENERGY_THRESHOLDS_VOL) - 1)
    return float(config.ENERGY_THRESHOLDS_VOL[jj])


def build_boundaries_for_date(
    dt: pd.Timestamp,
    tubes: list[pd.DataFrame],
    vol_shells: pd.DataFrame | None,
) -> tuple[np.ndarray, list[tuple[str, int]]]:
    n = len(tubes)
    if n < 1:
        raise ValueError("Il faut au moins un tube.")

    lower = []
    lower_meta = []

    if vol_shells is not None and config.VOL_SHELLS_ENABLED:
        lower.append(float(vol_shells.loc[dt, "D5"]))
        lower_meta.append(("Dvol", 5))
        lower.append(float(vol_shells.loc[dt, "D4"]))
        lower_meta.append(("Dvol", 4))

    for k in range(n - 1, -1, -1):
        lower.append(float(tubes[k].loc[dt, "L"]))
        lower_meta.append(("L", k))

    upper = []
    upper_meta = []

    for k in range(0, n):
        upper.append(float(tubes[k].loc[dt, "U"]))
        upper_meta.append(("U", k))

    if vol_shells is not None and config.VOL_SHELLS_ENABLED:
        upper.append(float(vol_shells.loc[dt, "U4"]))
        upper_meta.append(("Uvol", 4))
        upper.append(float(vol_shells.loc[dt, "U5"]))
        upper_meta.append(("Uvol", 5))

    boundaries = np.array(lower + upper, dtype=float)
    meta = lower_meta + upper_meta

    order = np.argsort(boundaries)
    boundaries = boundaries[order]
    meta = [meta[i] for i in order]

    boundaries_log = np.log(np.clip(boundaries, 1e-300, np.inf))
    return boundaries_log, meta


def classify_zone_from_boundaries(x_log: float, boundaries_log: np.ndarray) -> int:
    return int(np.searchsorted(boundaries_log, x_log, side="right"))


def zone_bounds_from_boundaries(
    dt: pd.Timestamp,
    zone: int,
    boundaries_log: np.ndarray,
    meta: list[tuple[str, int]],
    state: pd.DataFrame,
) -> tuple[float, float, str, float]:
    zmax = len(boundaries_log)
    sig = float(state.loc[dt, "vol"])
    scale = np.sqrt(config.HORIZON_STEPS) if config.VOL_SHELL_USE_SQRT_H else 1.0
    sig = max(sig, 1e-12)

    if 0 < zone < zmax:
        lower_log = float(boundaries_log[zone - 1])
        upper_log = float(boundaries_log[zone])

        lower = float(np.exp(lower_log))
        upper = float(np.exp(upper_log))

        kind = "tube"
        t_l, lvl_l = meta[zone - 1]
        t_u, lvl_u = meta[zone]

        if (t_l in ("Dvol", "Uvol")) or (t_u in ("Dvol", "Uvol")):
            kind = "vol"

        def theta_for_meta(tag: str, level: int) -> float:
            if tag in ("L", "U"):
                return energy_threshold_for_tube(level)
            if tag in ("Dvol", "Uvol"):
                return energy_threshold_for_vol(level)
            return max(energy_threshold_for_tube(999), energy_threshold_for_vol(999))

        theta_fixed = float(max(theta_for_meta(t_l, lvl_l), theta_for_meta(t_u, lvl_u)))
        return upper, lower, kind, theta_fixed

    if zone <= 0:
        upper_log = float(boundaries_log[0])
        upper = float(np.exp(upper_log))
        lower = float(upper * np.exp(-config.TAIL_SIGMA_MULTIPLIER * sig * scale))
        t0, lvl0 = meta[0]
        theta_fixed = energy_threshold_for_vol(5) if t0 in ("Dvol", "Uvol") else energy_threshold_for_tube(lvl0)
        return upper, lower, "tail", float(theta_fixed)

    lower_log = float(boundaries_log[-1])
    lower = float(np.exp(lower_log))
    upper = float(lower * np.exp(config.TAIL_SIGMA_MULTIPLIER * sig * scale))
    t1, lvl1 = meta[-1]
    theta_fixed = energy_threshold_for_vol(5) if t1 in ("Dvol", "Uvol") else energy_threshold_for_tube(lvl1)
    return upper, lower, "tail", float(theta_fixed)


def get_zone_interval(
    dt: pd.Timestamp,
    zone: int,
    boundaries_log: np.ndarray,
    meta: list[tuple[str, int]],
    state: pd.DataFrame,
) -> tuple[float, float]:
    upper, lower, _kind, _theta = zone_bounds_from_boundaries(
        dt=dt,
        zone=zone,
        boundaries_log=boundaries_log,
        meta=meta,
        state=state,
    )
    return float(lower), float(upper)


def map_position_between_zones(
    x_obs_log: float,
    lower_obs: float,
    upper_obs: float,
    lower_latent: float,
    upper_latent: float,
    eps: float = config.POSITION_EPS,
) -> float:
    lower_obs_log = np.log(lower_obs)
    upper_obs_log = np.log(upper_obs)
    lower_latent_log = np.log(lower_latent)
    upper_latent_log = np.log(upper_latent)

    width_obs = max(upper_obs_log - lower_obs_log, 1e-12)
    q_obs = (x_obs_log - lower_obs_log) / width_obs
    q_obs = float(np.clip(q_obs, eps, 1.0 - eps))

    x_start_log = lower_latent_log + q_obs * (upper_latent_log - lower_latent_log)
    return float(x_start_log)


# ======================================================
# LATENT ZONE FILTER
# ======================================================
def compute_latent_zone_path(
    state: pd.DataFrame,
    tubes: list[pd.DataFrame],
    vol_shells: pd.DataFrame | None = None,
    seed: int = 123,
) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = state.index
    x_all = state["x"].to_numpy()
    e_all = state["energy"].to_numpy()

    zone_obs_list: list[int] = []
    zone_latent_list: list[int] = []

    boundaries0, _meta0 = build_boundaries_for_date(idx[0], tubes, vol_shells)
    z0 = classify_zone_from_boundaries(float(x_all[0]), boundaries0)
    zone_obs_list.append(int(z0))
    zone_latent_list.append(int(z0))

    for i in range(1, len(idx)):
        dt = idx[i]
        x = float(x_all[i])
        energy = float(e_all[i])

        boundaries, meta = build_boundaries_for_date(dt, tubes, vol_shells)
        z_obs = classify_zone_from_boundaries(x, boundaries)
        z_prev = zone_latent_list[-1]

        if z_obs > z_prev:
            direction = 1
        elif z_obs < z_prev:
            direction = -1
        else:
            direction = 0

        z_new = z_prev

        if direction != 0:
            _upper_prev, _lower_prev, _kind_prev, theta_fixed = zone_bounds_from_boundaries(
                dt, z_prev, boundaries, meta, state
            )

            j0 = max(0, i - int(config.ENERGY_LOOKBACK))
            hist_z = np.array(zone_latent_list[j0:i], dtype=int)
            hist_e = e_all[j0:i]
            energy_in_zone = hist_e[hist_z == z_prev]

            if energy_in_zone.size >= config.MIN_ZONE_OBS_FOR_Q:
                theta = float(np.quantile(energy_in_zone, config.ENERGY_QUANTILE))
            else:
                theta = float(theta_fixed)

            p_move = config.ZONE_P_MIN + (config.ZONE_P_MAX - config.ZONE_P_MIN) * sigmoid(config.ZONE_KAPPA * (energy - theta))
            dist = min(config.ZONE_DIST_CAP, max(1, int(abs(z_obs - z_prev))))
            p_move = 1.0 - (1.0 - p_move) ** dist

            if rng.random() < p_move:
                z_new = z_prev + direction

        zone_obs_list.append(int(z_obs))
        zone_latent_list.append(int(z_new))

    zone_obs = pd.Series(zone_obs_list, index=idx, name="zone_obs")
    zone_latent = pd.Series(zone_latent_list, index=idx, name="zone_latent")
    return zone_latent, zone_obs


# ======================================================
# LOCAL DYNAMICS / MONTE CARLO
# ======================================================
def fit_local_dynamics(
    state: pd.DataFrame,
    bounds_df: pd.DataFrame,
    idx: pd.DatetimeIndex,
    drift_mode: str = "linear",
    tanh_lambda: float = 1.0,
) -> tuple[float, float, float]:
    x = state.loc[idx, "x"]
    r_next = state["r1"].shift(-1).loc[idx]

    upper = bounds_df.loc[idx, "U"].astype(float)
    lower = bounds_df.loc[idx, "L"].astype(float)

    ok = (upper > 0) & (lower > 0) & np.isfinite(upper) & np.isfinite(lower)
    x = x[ok]
    r_next = r_next[ok]
    upper = upper[ok]
    lower = lower[ok]

    if len(x) < 50:
        rv = r_next.to_numpy()
        a = float(np.nanmean(rv)) if rv.size else 0.0
        slope = 0.0
        resid = rv - a
        sigma = float(np.nanstd(resid, ddof=1)) if resid.size > 1 else 1e-4
        return float(a), float(slope), float(max(sigma, 1e-8))

    center = 0.5 * (np.log(upper) + np.log(lower))
    half_width = 0.5 * (np.log(upper) - np.log(lower))
    half_width = half_width.clip(lower=1e-12)

    y = (x - center) / half_width
    mask = np.isfinite(y.to_numpy()) & np.isfinite(r_next.to_numpy())
    yv = y.to_numpy()[mask]
    rv = r_next.to_numpy()[mask]

    if rv.size < 50:
        a = float(np.nanmean(rv)) if rv.size else 0.0
        slope = 0.0
        resid = rv - a
        sigma = float(np.nanstd(resid, ddof=1)) if resid.size > 1 else 1e-4
        return float(a), float(slope), float(max(sigma, 1e-8))

    if drift_mode == "tanh":
        xreg = np.column_stack([np.ones_like(yv), np.tanh(tanh_lambda * yv)])
    else:
        xreg = np.column_stack([np.ones_like(yv), yv])

    beta, *_ = np.linalg.lstsq(xreg, rv, rcond=None)
    a, slope = beta

    fitted = xreg @ beta
    resid = rv - fitted
    sigma = float(np.std(resid, ddof=1)) if resid.size > 1 else 1e-8
    return float(a), float(slope), float(max(sigma, 1e-8))


def mc_prob_hit_bounds(
    x0_log: float,
    lower: float,
    upper: float,
    a: float,
    slope: float,
    sigma: float,
    horizon: int,
    n_paths: int,
    seed: int,
    drift_mode: str = "linear",
    tanh_lambda: float = 1.0,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)

    upper_log = np.log(upper)
    lower_log = np.log(lower)

    center = 0.5 * (upper_log + lower_log)
    half_width = max(0.5 * (upper_log - lower_log), 1e-12)

    q = (x0_log - lower_log) / (upper_log - lower_log + 1e-12)
    q = float(np.clip(q, config.POSITION_EPS, 1.0 - config.POSITION_EPS))
    x0_log = lower_log + q * (upper_log - lower_log)

    x = np.full(n_paths, x0_log, dtype=float)
    alive = np.ones(n_paths, dtype=bool)
    hit_up = np.zeros(n_paths, dtype=bool)
    hit_down = np.zeros(n_paths, dtype=bool)

    for _ in range(horizon):
        if not alive.any():
            break

        y = (x[alive] - center) / half_width

        if drift_mode == "tanh":
            mu = a + slope * np.tanh(tanh_lambda * y)
        else:
            mu = a + slope * y

        x[alive] += mu + sigma * rng.standard_normal(alive.sum())

        idx_alive = np.where(alive)[0]
        up = x[alive] >= upper_log
        down = x[alive] <= lower_log

        hit_up[idx_alive[up]] = True
        hit_down[idx_alive[down]] = True
        alive[idx_alive[up | down]] = False

    p_up = float(hit_up.mean())
    p_down = float(hit_down.mean())
    p_none = float(1.0 - p_up - p_down)
    return p_up, p_down, p_none


# ======================================================
# SIGNALS
# ======================================================
def build_daily_signal_table(
    ticker: str,
    drift_mode: str = "linear",
    tanh_lambda: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = download_ohlc_cached(ticker)

    tubes = build_multi_tube(df, config.TUBE_WINDOW, config.TUBE_LEVELS, shift=config.TUBE_SHIFT)
    state = compute_state(df, vol_window=config.VOL_WINDOW, vol_shift=config.VOL_SHIFT)
    state, tubes = align_state_and_tubes(state, tubes)

    vol_shells = None
    if config.VOL_SHELLS_ENABLED:
        vol_shells = build_vol_shell_bounds(
            state=state,
            outer_tube=tubes[-1],
            alpha_inner=config.VOL_ALPHA_INNER,
            alpha_outer=config.VOL_ALPHA_OUTER,
            use_sqrt_h=config.VOL_SHELL_USE_SQRT_H,
            horizon=config.HORIZON_STEPS,
        )

    zone_latent, zone_obs = compute_latent_zone_path(
        state=state,
        tubes=tubes,
        vol_shells=vol_shells,
        seed=config.SEED + 999,
    )

    bounds_rows = []
    for dt in state.index:
        boundaries_log, meta = build_boundaries_for_date(dt, tubes, vol_shells)
        zone = int(zone_latent.loc[dt])
        upper, lower, zone_kind, theta_fixed = zone_bounds_from_boundaries(dt, zone, boundaries_log, meta, state)
        bounds_rows.append(
            {
                "date": dt,
                "zone_latent": zone,
                "zone_obs": int(zone_obs.loc[dt]),
                "U": upper,
                "L": lower,
                "zone_kind": zone_kind,
                "theta_fixed": theta_fixed,
            }
        )
    bounds_df = pd.DataFrame(bounds_rows).set_index("date")

    params_cache: dict[int, tuple[float, float, float]] = {}
    last_fit_pos: dict[int, int] = {}
    rows = []

    state_idx = state.index
    for t in range(config.FIT_WINDOW, len(state) - config.HORIZON_STEPS - 1):
        signal_dt = state_idx[t]
        entry_dt = state_idx[t + 1]
        exit_dt = state_idx[t + config.HORIZON_STEPS]

        boundaries_log, meta = build_boundaries_for_date(signal_dt, tubes, vol_shells)

        zone_lat = int(bounds_df.loc[signal_dt, "zone_latent"])
        zone_obs_now = int(bounds_df.loc[signal_dt, "zone_obs"])

        lower_lat, upper_lat = get_zone_interval(signal_dt, zone_lat, boundaries_log, meta, state)
        lower_obs, upper_obs = get_zone_interval(signal_dt, zone_obs_now, boundaries_log, meta, state)

        x_obs = float(state["x"].iloc[t])
        x_start = map_position_between_zones(
            x_obs_log=x_obs,
            lower_obs=lower_obs,
            upper_obs=upper_obs,
            lower_latent=lower_lat,
            upper_latent=upper_lat,
            eps=config.POSITION_EPS,
        )

        p_obs = float(np.exp(x_obs))
        p_start = float(np.exp(x_start))
        energy = float(state["energy"].iloc[t])

        upper = upper_lat
        lower = lower_lat
        zone_kind = str(bounds_df.loc[signal_dt, "zone_kind"])

        upper_log = np.log(upper)
        lower_log = np.log(lower)
        center = 0.5 * (upper_log + lower_log)
        half_width = max(0.5 * (upper_log - lower_log), 1e-12)
        y_start = (x_start - center) / half_width

        need_refit = (zone_lat not in params_cache) or ((t - last_fit_pos.get(zone_lat, -10**9)) >= config.REFIT_EVERY)
        if need_refit:
            start = max(0, t - config.FIT_WINDOW)
            idx_all = state.index[start:t]
            idx_zone = idx_all[zone_latent.loc[idx_all] == zone_lat]
            idx_fit = idx_zone if len(idx_zone) >= max(80, int(0.25 * len(idx_all))) else idx_all

            a, slope, sigma = fit_local_dynamics(
                state=state,
                bounds_df=bounds_df,
                idx=idx_fit,
                drift_mode=drift_mode,
                tanh_lambda=tanh_lambda,
            )
            params_cache[zone_lat] = (a, slope, sigma)
            last_fit_pos[zone_lat] = t
        else:
            a, slope, sigma = params_cache[zone_lat]

        p_up, p_down, p_none = mc_prob_hit_bounds(
            x0_log=x_start,
            lower=lower,
            upper=upper,
            a=a,
            slope=slope,
            sigma=sigma,
            horizon=config.HORIZON_STEPS,
            n_paths=config.N_PATHS,
            seed=config.SEED + t,
            drift_mode=drift_mode,
            tanh_lambda=tanh_lambda,
        )

        rows.append(
            {
                "signal_date": signal_dt,
                "entry_date": entry_dt,
                "exit_date": exit_dt,
                "ticker": ticker,
                "zone_latent": zone_lat,
                "zone_obs": zone_obs_now,
                "zone_kind": zone_kind,
                "P_obs": p_obs,
                "P_start": p_start,
                "energy": energy,
                "U": upper,
                "L": lower,
                "y_start": y_start,
                "p_up": p_up,
                "p_down": p_down,
                "p_none": p_none,
                "a": a,
                "slope": slope,
                "sigma": sigma,
            }
        )

    signals = pd.DataFrame(rows).set_index("signal_date")
    return df.loc[state.index].copy(), signals


# ======================================================
# MULTI-ASSET INPUTS
# ======================================================
def build_multi_asset_inputs(
    tickers: list[str],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    ohlc_map: dict[str, pd.DataFrame] = {}
    signal_tables: list[pd.DataFrame] = []

    for ticker in tickers:
        print(f"Build signals: {ticker}")
        ohlc, signals = build_daily_signal_table(
            ticker=ticker,
            drift_mode=config.DRIFT_MODE,
            tanh_lambda=config.TANH_LAMBDA,
        )
        ohlc_map[ticker] = ohlc[["Open", "Close"]].copy()
        signal_tables.append(signals.reset_index())

    signal_df = pd.concat(signal_tables, axis=0, ignore_index=True)
    signal_df["signal_date"] = pd.to_datetime(signal_df["signal_date"])
    signal_df["entry_date"] = pd.to_datetime(signal_df["entry_date"])
    signal_df["exit_date"] = pd.to_datetime(signal_df["exit_date"])

    return ohlc_map, signal_df.sort_values(["entry_date", "ticker"]).reset_index(drop=True)


# ======================================================
# COMMON PANELS
# ======================================================
def build_common_panels(
    ohlc_map: dict[str, pd.DataFrame],
    tickers: list[str],
    backtest_start: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    common_idx = None
    for ticker in tickers:
        idx_t = ohlc_map[ticker].index
        common_idx = idx_t if common_idx is None else common_idx.intersection(idx_t)

    if common_idx is None or len(common_idx) == 0:
        raise ValueError("Pas de calendrier commun exploitable.")

    if backtest_start is not None:
        common_idx = common_idx[common_idx >= pd.Timestamp(backtest_start)]

    if len(common_idx) == 0:
        raise ValueError("Pas de données après BACKTEST_START.")

    open_panel = pd.concat({t: ohlc_map[t].loc[common_idx, "Open"] for t in tickers}, axis=1)
    close_panel = pd.concat({t: ohlc_map[t].loc[common_idx, "Close"] for t in tickers}, axis=1)
    return open_panel, close_panel, common_idx


# ======================================================
# TOP-K ALLOCATION
# ======================================================
def upside_to_upper_from_row(row: pd.Series) -> float:
    p_start = float(row["P_start"])
    upper = float(row["U"])
    if not np.isfinite(p_start) or p_start <= 0:
        return 0.0
    return float(max(upper / p_start - 1.0, 0.0))


def selection_score_from_row(row: pd.Series) -> float:
    p_up = float(row["p_up"])
    upside = float(row["upside_to_upper"])

    if p_up <= 0 or upside <= 0:
        return 0.0

    return float((p_up ** config.PROBA_POWER) * (upside ** config.UPSIDE_POWER))


def build_topk_weight_schedule(
    signal_df: pd.DataFrame,
    tickers: list[str],
    rebal_dates: pd.DatetimeIndex,
    top_k: int = config.TOP_K,
    fallback_asset: str = config.FALLBACK_ASSET,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal_df = signal_df.copy()
    signal_df = signal_df[signal_df["ticker"].isin(tickers)].copy()

    rows_w = []
    rows_diag = []

    if fallback_asset not in tickers:
        fallback_asset = "IEF" if "IEF" in tickers else tickers[0]

    for dt in rebal_dates:
        candidates = signal_df.loc[signal_df["entry_date"] == dt].copy()
        candidates = candidates.sort_values(["ticker", "signal_date"]).drop_duplicates("ticker", keep="last")

        n_candidates_before_filters = int(len(candidates))

        if not candidates.empty:
            candidates["upside_to_upper"] = candidates.apply(upside_to_upper_from_row, axis=1)
            candidates["score"] = candidates.apply(selection_score_from_row, axis=1)

            filt = pd.Series(True, index=candidates.index)
            if config.REQUIRE_PUP_GT_PDN:
                filt &= candidates["p_up"] > candidates["p_down"]

            candidates = candidates.loc[filt].copy()

        row_w = {"date": dt}
        for ticker in tickers:
            row_w[ticker] = 0.0

        if candidates.empty:
            row_w[fallback_asset] = 1.0
            rows_w.append(row_w)
            rows_diag.append(
                {
                    "date": dt,
                    "n_candidates_before_filters": n_candidates_before_filters,
                    "n_candidates_after_filters": 0,
                    "n_selected": 1,
                    "sum_w": 1.0,
                    "selected_assets": fallback_asset,
                    "selected_avg_p_up": np.nan,
                    "selected_avg_upside": np.nan,
                    "fallback_used": 1,
                }
            )
            continue

        candidates = candidates.sort_values(["score", "p_up", "upside_to_upper"], ascending=False).head(top_k).copy()
        score_sum = float(candidates["score"].sum())

        if (not np.isfinite(score_sum)) or score_sum <= 0:
            weights = np.full(len(candidates), 1.0 / len(candidates), dtype=float)
        else:
            weights = candidates["score"].to_numpy(dtype=float) / score_sum

        for ticker, weight in zip(candidates["ticker"], weights):
            row_w[str(ticker)] = float(weight)

        rows_w.append(row_w)
        rows_diag.append(
            {
                "date": dt,
                "n_candidates_before_filters": n_candidates_before_filters,
                "n_candidates_after_filters": int(len(candidates)),
                "n_selected": int(len(candidates)),
                "sum_w": float(np.sum(weights)),
                "selected_assets": "|".join(candidates["ticker"].astype(str).tolist()),
                "selected_avg_p_up": float(candidates["p_up"].mean()),
                "selected_avg_upside": float(candidates["upside_to_upper"].mean()),
                "fallback_used": 0,
            }
        )

    target_weights = pd.DataFrame(rows_w).set_index("date")
    diag_df = pd.DataFrame(rows_diag).set_index("date")
    return target_weights, diag_df


def build_equal_weight_schedule(
    tickers: list[str],
    rebal_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    n = len(tickers)
    weight = 1.0 / n if n > 0 else 0.0
    rows = []
    for dt in rebal_dates:
        row = {"date": dt}
        for ticker in tickers:
            row[ticker] = weight
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def build_constant_weight_schedule(
    tickers: list[str],
    rebal_dates: pd.DatetimeIndex,
    weights_map: dict[str, float],
) -> pd.DataFrame:
    rows = []
    for dt in rebal_dates:
        row = {"date": dt}
        for ticker in tickers:
            row[ticker] = float(weights_map.get(ticker, 0.0))
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


# ======================================================
# BACKTEST
# ======================================================
def simulate_from_target_weights(
    open_panel: pd.DataFrame,
    close_panel: pd.DataFrame,
    target_weights: pd.DataFrame,
    initial_capital: float = config.INITIAL_CAPITAL,
    fee_bps: float = config.FEE_BPS,
    label: str = "portfolio",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tickers = list(close_panel.columns)
    target_weights = target_weights.reindex(columns=tickers).fillna(0.0).copy()

    shares = pd.Series(0.0, index=tickers, dtype=float)
    cash = float(initial_capital)

    equity_rows = []
    weight_rows = []
    rebal_rows = []

    for dt in close_panel.index:
        fee_paid = 0.0
        turnover = 0.0

        if dt in target_weights.index:
            tgt_w = target_weights.loc[dt].astype(float)
            open_px = open_panel.loc[dt].astype(float)

            current_value = shares * open_px
            equity_open = float(cash + current_value.sum())

            target_value = tgt_w * equity_open
            delta_value = target_value - current_value

            sell_value = (-delta_value.clip(upper=0.0)).astype(float)
            buy_value = (delta_value.clip(lower=0.0)).astype(float)

            if sell_value.sum() > 0:
                sell_fee = sell_value * fee_bps / 1e4
                for ticker in tickers:
                    sv = float(sell_value[ticker])
                    px = float(open_px[ticker])
                    if sv <= 0 or (not np.isfinite(px)) or px <= 0:
                        continue
                    shares[ticker] -= sv / px
                    cash += sv - float(sell_fee[ticker])

                fee_paid += float(sell_fee.sum())
                turnover += float(sell_value.sum())

            if buy_value.sum() > 0:
                buy_fee = buy_value * fee_bps / 1e4
                total_need = float(buy_value.sum() + buy_fee.sum())

                scale = 1.0
                if total_need > max(cash, 0.0) and total_need > 0:
                    scale = max(cash, 0.0) / total_need

                buy_exec = buy_value * scale
                buy_fee_exec = buy_exec * fee_bps / 1e4

                for ticker in tickers:
                    bv = float(buy_exec[ticker])
                    px = float(open_px[ticker])
                    if bv <= 0 or (not np.isfinite(px)) or px <= 0:
                        continue
                    shares[ticker] += bv / px
                    cash -= bv + float(buy_fee_exec[ticker])

                fee_paid += float(buy_fee_exec.sum())
                turnover += float(buy_exec.sum())

            if abs(cash) < 1e-10:
                cash = 0.0

            rebal_row = {
                "date": dt,
                "label": label,
                "fee_paid": fee_paid,
                "turnover": turnover / max(equity_open, 1e-12),
            }
            for ticker in tickers:
                rebal_row[ticker] = float(tgt_w[ticker])
            rebal_rows.append(rebal_row)

        close_px = close_panel.loc[dt].astype(float)
        position_value = shares * close_px
        equity = float(cash + position_value.sum())
        gross_exposure = float(position_value.abs().sum())

        eq_row = {
            "date": dt,
            "label": label,
            "cash": float(cash),
            "gross_exposure": gross_exposure,
            "equity": equity,
            "cash_weight": float(cash / equity) if abs(equity) > 1e-12 else np.nan,
        }
        equity_rows.append(eq_row)

        weight_row = {"date": dt, "label": label}
        if abs(equity) > 1e-12:
            for ticker in tickers:
                weight_row[ticker] = float(position_value[ticker] / equity)
        else:
            for ticker in tickers:
                weight_row[ticker] = np.nan
        weight_rows.append(weight_row)

    equity_df = pd.DataFrame(equity_rows).set_index("date")
    weights_daily = pd.DataFrame(weight_rows).set_index("date")
    rebal_df = pd.DataFrame(rebal_rows).set_index("date") if rebal_rows else pd.DataFrame()
    return equity_df, weights_daily, rebal_df


def simulate_buyhold_single_asset(
    open_panel: pd.DataFrame,
    close_panel: pd.DataFrame,
    ticker: str,
    initial_capital: float = config.INITIAL_CAPITAL,
    fee_bps: float = config.FEE_BPS,
) -> pd.DataFrame:
    if ticker not in close_panel.columns:
        raise ValueError(f"{ticker} absent des données.")

    idx = close_panel.index
    first_dt = idx[0]
    entry_px = float(open_panel.loc[first_dt, ticker])
    if not np.isfinite(entry_px) or entry_px <= 0:
        raise ValueError(f"Open invalide pour {ticker}.")

    alloc = initial_capital / (1.0 + fee_bps / 1e4)
    entry_fee = alloc * fee_bps / 1e4
    shares = alloc / entry_px
    cash = initial_capital - alloc - entry_fee

    rows = []
    for dt in idx:
        close_px = float(close_panel.loc[dt, ticker])
        equity = cash + shares * close_px
        rows.append(
            {
                "date": dt,
                "label": f"{ticker}_buyhold",
                "cash": float(cash),
                "gross_exposure": float(shares * close_px),
                "equity": float(equity),
                "cash_weight": float(cash / equity) if abs(equity) > 1e-12 else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("date")