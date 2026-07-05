
"""
synthetic_lab.py
=================
Génère un univers synthétique multi-actifs (GBM / OU / régimes / saisonnalité)
et le branche directement sur core.py pour tester le score composite
(zone_obs - zone_latent) + (p_up - p_down), avec deux méthodes de suivi
de capital (cohorte / capital unique) et une allocation multi-actifs
avec seuil individuel par actif.

Dépendances: core.py, config.py, numpy, pandas.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import config
import core


# ======================================================
# 1) GENERATEUR D'ACTIFS SYNTHETIQUES
# ======================================================

def _make_dates(n: int, start: str = "2015-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def _to_ohlc(prices: np.ndarray, dates: pd.DatetimeIndex, noise: float = 0.0015,
             seed: int = 0) -> pd.DataFrame:
    """Transforme une série de Close en DataFrame OHLC minimal (Open, Close)
    compatible avec les fonctions de core.py (build_multi_tube, compute_state, ...).
    Open[t] = Close[t-1] * (1 + petit bruit) pour simuler l'écart open/close."""
    rng = np.random.default_rng(seed)
    close = pd.Series(prices, index=dates, name="Close")
    gap = rng.normal(0.0, noise, size=len(close))
    open_ = close.shift(1).fillna(close.iloc[0]) * (1.0 + gap)
    df = pd.DataFrame({"Open": open_.values, "Close": close.values}, index=dates)
    return df


def gbm_path(n: int, mu: float, sigma: float, s0: float = 100.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    log_ret = (mu - 0.5 * sigma ** 2) / 252.0 + sigma * np.sqrt(1 / 252.0) * z
    x = np.log(s0) + np.cumsum(log_ret)
    return np.exp(x)


def gbm_regime_switch_path(n: int, mu_before: float, mu_after: float, sigma: float,
                            switch_frac: float = 0.7, s0: float = 100.0, seed: int = 0,
                            crash_shock: float = -0.25) -> np.ndarray:
    """Rendement nul (ou faible) puis rupture de régime négative + saut (crash)."""
    rng = np.random.default_rng(seed)
    switch_idx = int(n * switch_frac)
    z = rng.standard_normal(n)
    mu_path = np.where(np.arange(n) < switch_idx, mu_before, mu_after)
    log_ret = (mu_path - 0.5 * sigma ** 2) / 252.0 + sigma * np.sqrt(1 / 252.0) * z
    log_ret[switch_idx] += crash_shock  # saut discret au moment du crash
    x = np.log(s0) + np.cumsum(log_ret)
    return np.exp(x)


def seasonal_path(n: int, mu: float, sigma: float, amplitude: float, period: int = 63,
                   s0: float = 100.0, seed: int = 0) -> np.ndarray:
    """Brownien + composante sinusoïdale (hausse/baisse cyclique, ex: trimestrielle)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    seasonal_drift = amplitude * np.sin(2 * np.pi * t / period) / 252.0
    z = rng.standard_normal(n)
    log_ret = (mu - 0.5 * sigma ** 2) / 252.0 + seasonal_drift + sigma * np.sqrt(1 / 252.0) * z
    x = np.log(s0) + np.cumsum(log_ret)
    return np.exp(x)


def ou_range_path(n: int, kappa: float, sigma: float, mean_level: float = 100.0,
                   s0: float = 100.0, seed: int = 0) -> np.ndarray:
    """Processus Ornstein-Uhlenbeck sur le log-prix: actif qui 'range' autour d'un niveau."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    x = np.zeros(n)
    x[0] = np.log(s0)
    mu_log = np.log(mean_level)
    for i in range(1, n):
        x[i] = x[i - 1] + kappa * (mu_log - x[i - 1]) * dt + sigma * np.sqrt(dt) * rng.standard_normal()
    return np.exp(x)


def riskfree_path(n: int, rate: float = 0.03, s0: float = 100.0) -> np.ndarray:
    t = np.arange(n)
    return s0 * (1.0 + rate / 252.0) ** t


ASSET_PROFILES = {
    "RISKFREE":   dict(kind="riskfree", rate=0.03),
    "HIGHVOL":    dict(kind="gbm", mu=0.0, sigma=0.55),
    "NODRIFT":    dict(kind="gbm", mu=0.0, sigma=0.18),
    "CRASH":      dict(kind="regime", mu_before=0.0, mu_after=-0.35, sigma=0.20,
                        switch_frac=0.65, crash_shock=-0.25),
    "BULL":       dict(kind="gbm", mu=0.18, sigma=0.16),
    # "Vrai" bull: drift fort + volatilite tres faible -> quasi monotone croissant,
    # contrairement a BULL ci-dessus qui, avec sigma=0.16, produit souvent des
    # phases de repli visibles (bull-bear-bull) malgre un drift positif.
    #"BULL_TRUE":  dict(kind="gbm", mu=0.30, sigma=0.06),
    "SEASONAL":   dict(kind="seasonal", mu=0.02, sigma=0.15, amplitude=0.35, period=63),
    "RANGE":      dict(kind="ou", kappa=6.0, sigma=0.20, mean_level=100.0),
}


def build_synthetic_universe(n_days: int = 1500, start: str = "2015-01-01",
                              seed: int = 42, s0: float = 100.0) -> dict[str, pd.DataFrame]:
    """Construit les 7 actifs synthétiques et retourne un dict {ticker: OHLC df}
    directement injectable dans core.py."""
    dates = _make_dates(n_days, start=start)
    universe: dict[str, pd.DataFrame] = {}

    for i, (name, params) in enumerate(ASSET_PROFILES.items()):
        kind = params["kind"]
        s = seed + i * 1000

        if kind == "riskfree":
            prices = riskfree_path(n_days, rate=params["rate"], s0=s0)
        elif kind == "gbm":
            prices = gbm_path(n_days, params["mu"], params["sigma"], s0=s0, seed=s)
        elif kind == "regime":
            prices = gbm_regime_switch_path(
                n_days, params["mu_before"], params["mu_after"], params["sigma"],
                switch_frac=params["switch_frac"], s0=s0, seed=s,
                crash_shock=params["crash_shock"],
            )
        elif kind == "seasonal":
            prices = seasonal_path(n_days, params["mu"], params["sigma"],
                                    params["amplitude"], period=params["period"], s0=s0, seed=s)
        elif kind == "ou":
            prices = ou_range_path(n_days, params["kappa"], params["sigma"],
                                    mean_level=params["mean_level"], s0=s0, seed=s)
        else:
            raise ValueError(f"kind inconnu: {kind}")

        universe[name] = _to_ohlc(prices, dates, seed=s + 1)

    return universe


# ======================================================
# 2) BRANCHEMENT SUR core.py (bypass yfinance)
# ======================================================

def build_daily_signal_table_from_df(ticker: str, df: pd.DataFrame,
                                      drift_mode: str = "linear",
                                      tanh_lambda: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Réplique core.build_daily_signal_table mais part d'un DataFrame OHLC
    déjà en mémoire (synthétique) au lieu d'appeler download_ohlc_cached."""
    tubes = core.build_multi_tube(df, config.TUBE_WINDOW, config.TUBE_LEVELS, shift=config.TUBE_SHIFT)
    state = core.compute_state(df, vol_window=config.VOL_WINDOW, vol_shift=config.VOL_SHIFT)
    state, tubes = core.align_state_and_tubes(state, tubes)

    vol_shells = None
    if config.VOL_SHELLS_ENABLED:
        vol_shells = core.build_vol_shell_bounds(
            state=state, outer_tube=tubes[-1],
            alpha_inner=config.VOL_ALPHA_INNER, alpha_outer=config.VOL_ALPHA_OUTER,
            use_sqrt_h=config.VOL_SHELL_USE_SQRT_H, horizon=config.HORIZON_STEPS,
        )

    zone_latent, zone_obs = core.compute_latent_zone_path(
        state=state, tubes=tubes, vol_shells=vol_shells, seed=config.SEED + 999,
    )

    bounds_rows = []
    for dt in state.index:
        boundaries_log, meta = core.build_boundaries_for_date(dt, tubes, vol_shells)
        zone = int(zone_latent.loc[dt])
        upper, lower, zone_kind, theta_fixed = core.zone_bounds_from_boundaries(
            dt, zone, boundaries_log, meta, state
        )
        bounds_rows.append({
            "date": dt, "zone_latent": zone, "zone_obs": int(zone_obs.loc[dt]),
            "U": upper, "L": lower, "zone_kind": zone_kind, "theta_fixed": theta_fixed,
        })
    bounds_df = pd.DataFrame(bounds_rows).set_index("date")

    params_cache: dict[int, tuple[float, float, float]] = {}
    last_fit_pos: dict[int, int] = {}
    rows = []
    state_idx = state.index

    for t in range(config.FIT_WINDOW, len(state) - config.HORIZON_STEPS - 1):
        signal_dt = state_idx[t]
        entry_dt = state_idx[t + 1]
        exit_dt = state_idx[t + config.HORIZON_STEPS]

        boundaries_log, meta = core.build_boundaries_for_date(signal_dt, tubes, vol_shells)
        zone_lat = int(bounds_df.loc[signal_dt, "zone_latent"])
        zone_obs_now = int(bounds_df.loc[signal_dt, "zone_obs"])

        lower_lat, upper_lat = core.get_zone_interval(signal_dt, zone_lat, boundaries_log, meta, state)
        lower_obs, upper_obs = core.get_zone_interval(signal_dt, zone_obs_now, boundaries_log, meta, state)

        x_obs = float(state["x"].iloc[t])
        x_start = core.map_position_between_zones(
            x_obs_log=x_obs, lower_obs=lower_obs, upper_obs=upper_obs,
            lower_latent=lower_lat, upper_latent=upper_lat, eps=config.POSITION_EPS,
        )

        p_obs = float(np.exp(x_obs))
        p_start = float(np.exp(x_start))
        energy = float(state["energy"].iloc[t])
        upper, lower = upper_lat, lower_lat
        zone_kind = str(bounds_df.loc[signal_dt, "zone_kind"])

        upper_log, lower_log = np.log(upper), np.log(lower)
        center = 0.5 * (upper_log + lower_log)
        half_width = max(0.5 * (upper_log - lower_log), 1e-12)
        y_start = (x_start - center) / half_width

        need_refit = (zone_lat not in params_cache) or (
            (t - last_fit_pos.get(zone_lat, -10**9)) >= config.REFIT_EVERY
        )
        if need_refit:
            start = max(0, t - config.FIT_WINDOW)
            idx_all = state.index[start:t]
            idx_zone = idx_all[zone_latent.loc[idx_all] == zone_lat]
            idx_fit = idx_zone if len(idx_zone) >= max(80, int(0.25 * len(idx_all))) else idx_all
            a, slope, sigma = core.fit_local_dynamics(
                state=state, bounds_df=bounds_df, idx=idx_fit,
                drift_mode=drift_mode, tanh_lambda=tanh_lambda,
            )
            params_cache[zone_lat] = (a, slope, sigma)
            last_fit_pos[zone_lat] = t
        else:
            a, slope, sigma = params_cache[zone_lat]

        p_up, p_down, p_none = core.mc_prob_hit_bounds(
            x0_log=x_start, lower=lower, upper=upper, a=a, slope=slope, sigma=sigma,
            horizon=config.HORIZON_STEPS, n_paths=config.N_PATHS, seed=config.SEED + t,
            drift_mode=drift_mode, tanh_lambda=tanh_lambda,
        )

        rows.append({
            "signal_date": signal_dt, "entry_date": entry_dt, "exit_date": exit_dt,
            "ticker": ticker, "zone_latent": zone_lat, "zone_obs": zone_obs_now,
            "zone_kind": zone_kind, "P_obs": p_obs, "P_start": p_start, "energy": energy,
            "U": upper, "L": lower, "y_start": y_start, "p_up": p_up, "p_down": p_down,
            "p_none": p_none, "a": a, "slope": slope, "sigma": sigma,
        })

    signals = pd.DataFrame(rows).set_index("signal_date")
    return df.loc[state.index].copy(), signals


def build_multi_asset_inputs_synthetic(universe: dict[str, pd.DataFrame],
                                        drift_mode: str = None,
                                        tanh_lambda: float = None) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    drift_mode = drift_mode or config.DRIFT_MODE
    tanh_lambda = tanh_lambda or config.TANH_LAMBDA
    ohlc_map: dict[str, pd.DataFrame] = {}
    signal_tables = []
    for ticker, df in universe.items():
        print(f"[synthetic] build signals: {ticker}")
        ohlc, signals = build_daily_signal_table_from_df(ticker, df, drift_mode, tanh_lambda)
        ohlc_map[ticker] = ohlc[["Open", "Close"]].copy()
        signal_tables.append(signals.reset_index())
    signal_df = pd.concat(signal_tables, axis=0, ignore_index=True)
    for col in ("signal_date", "entry_date", "exit_date"):
        signal_df[col] = pd.to_datetime(signal_df[col])
    return ohlc_map, signal_df.sort_values(["entry_date", "ticker"]).reset_index(drop=True)


# ======================================================
# 3) SCORE COMPOSITE
# ======================================================

def add_composite_score(signal_df: pd.DataFrame, w_gap: float = 1.0, w_dir: float = 1.0,
                         use_continuous_gap: bool = True) -> pd.DataFrame:
    """score = -w_gap * gap + w_dir * (p_up - p_down)
    gap continu (recommandé) = log(P_obs) - log(P_start)
    gap discret (alternatif) = zone_obs - zone_latent
    """
    df = signal_df.copy()
    if use_continuous_gap:
        df["gap"] = np.log(df["P_obs"]) - np.log(df["P_start"])
    else:
        df["gap"] = (df["zone_obs"] - df["zone_latent"]).astype(float)
    df["p_spread"] = df["p_up"] - df["p_down"]
    df["score"] = -w_gap * df["gap"] + w_dir * df["p_spread"]
    return df


# ======================================================
# 4) BACKTEST PAR COHORTE (chaque entrée = position isolée)
# ======================================================

def add_expanding_score_threshold(signal_df: pd.DataFrame, quantile: float = 0.7,
                                   min_obs: int = 200) -> pd.DataFrame:
    """Calcule un seuil de score PAR DATE, base uniquement sur l'historique
    disponible jusqu'a cette date (expanding quantile), pour eviter tout
    look-ahead. Avant min_obs observations, le seuil est +inf (aucune entree)."""
    df = signal_df.sort_values("signal_date").copy()
    scores_sorted = df["score"].to_numpy()
    thresholds = np.full(len(df), np.inf)
    for i in range(len(df)):
        if i + 1 >= min_obs:
            thresholds[i] = np.quantile(scores_sorted[: i + 1], quantile)
    df["score_threshold_expanding"] = thresholds
    return df.sort_index()


def backtest_cohort(ohlc_map: dict[str, pd.DataFrame], signal_df: pd.DataFrame,
                     threshold: float | None = None, horizon_steps: int = None,
                     initial_capital: float = None, fee_bps: float = None,
                     max_concurrent_per_asset: int | None = None) -> pd.DataFrame:
    """Pour chaque signal valide (score >= threshold), simule une position
    isolee (capital dedie, pas de partage entre cohortes) tenue horizon_steps jours.
    threshold peut etre: un float (seuil fixe, risque de look-ahead si calibre sur
    tout l'historique), None (pas de filtre), ou le nom d'une colonne de seuil
    deja calculee point-in-time (ex: "score_threshold_expanding") pour comparer
    score vs seuil ligne a ligne sans fuite d'information.
    max_concurrent_per_asset: si fixe, limite le nombre de positions simultanees
    ouvertes sur un meme actif (repro un peu la contrainte de capital sans
    modeliser tout le portefeuille)."""
    horizon_steps = horizon_steps or config.HORIZON_STEPS
    initial_capital = initial_capital if initial_capital is not None else config.INITIAL_CAPITAL
    fee_bps = fee_bps if fee_bps is not None else config.FEE_BPS

    if threshold is None:
        valid = signal_df.copy()
    elif isinstance(threshold, str):
        valid = signal_df[signal_df["score"] >= signal_df[threshold]].copy()
    else:
        valid = signal_df[signal_df["score"] >= threshold].copy()
    if max_concurrent_per_asset is not None and not valid.empty:
        valid = valid.sort_values(["ticker", "entry_date"]).copy()
        keep_idx = []
        open_until: dict[str, list[pd.Timestamp]] = {}
        for idx_row, sig in valid.iterrows():
            t = sig["ticker"]
            entry_dt = pd.Timestamp(sig["entry_date"])
            exits = open_until.get(t, [])
            exits = [e for e in exits if e > entry_dt]
            if len(exits) < max_concurrent_per_asset:
                exits.append(pd.Timestamp(sig["exit_date"]))
                keep_idx.append(idx_row)
            open_until[t] = exits
        valid = valid.loc[keep_idx]

    rows = []
    for _, sig in valid.iterrows():
        ticker = sig["ticker"]
        df = ohlc_map[ticker]
        idx = df.index
        pos = idx.get_indexer([sig["entry_date"]])[0]
        if pos < 0 or pos + horizon_steps >= len(idx):
            continue
        entry_dt = idx[pos]
        exit_dt = idx[pos + horizon_steps]

        entry_px = float(df.loc[entry_dt, "Open"])
        exit_px = float(df.loc[exit_dt, "Open"])
        if entry_px <= 0:
            continue

        fee = fee_bps / 1e4
        gross_ret = exit_px / entry_px - 1.0
        net_ret = (1 - fee) * (1 + gross_ret) * (1 - fee) - 1.0

        rows.append({
            "ticker": ticker, "signal_date": sig.name, "entry_date": entry_dt,
            "exit_date": exit_dt, "score": sig["score"], "gross_return": gross_ret,
            "net_return": net_ret, "pnl": initial_capital * net_ret,
        })
    return pd.DataFrame(rows)


def backtest_cohort_baselines(ohlc_map: dict[str, pd.DataFrame], signal_df: pd.DataFrame,
                               threshold: float | None = None, horizon_steps: int = None,
                               rng_seed: int = 7, use_expanding_threshold: bool = True,
                               expanding_quantile: float = 0.7,
                               expanding_min_obs: int = 200) -> dict[str, pd.DataFrame]:
    """Compare le score composite a 4 baselines: entree immediate (J0),
    entree aleatoire, gap seul, p_spread seul.

    Par defaut (use_expanding_threshold=True), le seuil de selection est un
    quantile EXPANDING (point-in-time, calcule uniquement sur l'historique
    disponible a la date du signal) plutot qu'un seuil fixe calibre sur tout
    l'historique -- ce qui evite un biais de look-ahead qui avantagerait
    artificiellement le cohorte par rapport aux strategies capital unique.
    Si threshold est fourni explicitement, il prime sur le mode expanding."""
    horizon_steps = horizon_steps or config.HORIZON_STEPS
    out = {}

    def _resolve(df: pd.DataFrame, fixed_threshold: float | None):
        if fixed_threshold is not None:
            return df, fixed_threshold
        if use_expanding_threshold:
            df2 = add_expanding_score_threshold(df, quantile=expanding_quantile, min_obs=expanding_min_obs)
            return df2, "score_threshold_expanding"
        return df, df["score"].quantile(0.7)

    df_comp, thr_comp = _resolve(signal_df, threshold)
    out["composite"] = backtest_cohort(ohlc_map, df_comp, thr_comp, horizon_steps)

    df_gap = signal_df.copy()
    df_gap["score"] = -df_gap["gap"]
    df_gap, thr_gap = _resolve(df_gap, None)
    out["gap_only"] = backtest_cohort(ohlc_map, df_gap, thr_gap, horizon_steps=horizon_steps)

    df_dir = signal_df.copy()
    df_dir["score"] = df_dir["p_spread"]
    out["p_spread_only"] = backtest_cohort(ohlc_map, df_dir, threshold=0.0,
                                            horizon_steps=horizon_steps)

    rng = np.random.default_rng(rng_seed)
    df_rand = signal_df.copy()
    df_rand["score"] = rng.random(len(df_rand))
    df_rand, thr_rand = _resolve(df_rand, None)
    out["random_entry"] = backtest_cohort(ohlc_map, df_rand, thr_rand, horizon_steps=horizon_steps)

    df_immediate = signal_df.copy()
    df_immediate["score"] = 1.0
    out["buy_and_hold_immediate"] = backtest_cohort(ohlc_map, df_immediate, threshold=0.5,
                                                      horizon_steps=horizon_steps)
    return out


def summarize_cohort_results(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in results.items():
        if df.empty:
            rows.append({"method": name, "n_trades": 0, "avg_return": np.nan,
                         "sharpe_like": np.nan, "hit_rate": np.nan})
            continue
        avg = df["net_return"].mean()
        std = df["net_return"].std(ddof=1) if len(df) > 1 else np.nan
        rows.append({
            "method": name, "n_trades": len(df), "avg_return": avg,
            "sharpe_like": avg / std if std and std > 0 else np.nan,
            "hit_rate": (df["net_return"] > 0).mean(),
        })
    return pd.DataFrame(rows)


def summarize_cohort_results_by_ticker(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Meme resume que summarize_cohort_results mais decompose par (methode, ticker),
    pour verifier ou le score composite apporte vraiment de la valeur (attendu: RANGE,
    SEASONAL) versus ou il ne devrait rien apporter (attendu: NODRIFT, RISKFREE)."""
    rows = []
    for name, df in results.items():
        if df.empty:
            continue
        for ticker, sub in df.groupby("ticker"):
            avg = sub["net_return"].mean()
            std = sub["net_return"].std(ddof=1) if len(sub) > 1 else np.nan
            rows.append({
                "method": name, "ticker": ticker, "n_trades": len(sub), "avg_return": avg,
                "sharpe_like": avg / std if std and std > 0 else np.nan,
                "hit_rate": (sub["net_return"] > 0).mean(),
            })
    return pd.DataFrame(rows)


# ======================================================
# 4bis) EQUITY CURVE RECONSTRUITE A PARTIR DU COHORTE
# ======================================================

def build_cohort_equity_curve(cohort_trades: pd.DataFrame, common_idx: pd.DatetimeIndex) -> pd.Series:
    """Reconstruit une equity curve journaliere a partir des trades de cohorte
    (positions isolees, capital dedie par entree). Chaque trade contribue un
    rendement geometrique journalier lisse sur sa periode de detention; a chaque
    date, on moyenne (equal-weight) les rendements des trades ouverts ce jour-la.
    Si aucun trade n'est ouvert, le rendement du jour est nul (equivalent cash).
    Cela donne une courbe comparable aux strategies capital-unique et buy&hold."""
    if cohort_trades.empty:
        return pd.Series(1.0, index=common_idx)

    daily_rets = pd.DataFrame(0.0, index=common_idx, columns=range(len(cohort_trades)))
    active_mask = pd.DataFrame(False, index=common_idx, columns=range(len(cohort_trades)))

    for i, (_, tr) in enumerate(cohort_trades.reset_index(drop=True).iterrows()):
        entry_dt = pd.Timestamp(tr["entry_date"])
        exit_dt = pd.Timestamp(tr["exit_date"])
        mask = (common_idx >= entry_dt) & (common_idx <= exit_dt)
        n_days = int(mask.sum())
        if n_days < 1:
            continue
        r_daily = (1.0 + float(tr["net_return"])) ** (1.0 / n_days) - 1.0
        daily_rets.loc[mask, i] = r_daily
        active_mask.loc[mask, i] = True

    n_active = active_mask.sum(axis=1)
    port_ret = daily_rets.sum(axis=1) / n_active.replace(0, np.nan)
    port_ret = port_ret.fillna(0.0)
    equity = (1.0 + port_ret).cumprod()
    equity.iloc[0] = 1.0
    return equity


# ======================================================
# 5) BACKTEST CAPITAL UNIQUE (allocation multi-actifs)
# ======================================================

def build_score_target_weights(signal_df: pd.DataFrame, tickers: list[str],
                                rebal_dates: pd.DatetimeIndex,
                                thresholds: dict[str, float] | float | None = 0.0,
                                score_power: float = 1.0) -> pd.DataFrame:
    """Poids proportionnels au score parmi les actifs dont score >= seuil individuel.
    thresholds: dict {ticker: seuil} ou seuil unique appliqué à tous."""
    no_threshold = thresholds is None
    if not no_threshold and not isinstance(thresholds, dict):
        thresholds = {t: thresholds for t in tickers}

    rows = []
    for dt in rebal_dates:
        cand = signal_df[signal_df["entry_date"] == dt].copy()
        cand = cand[cand["ticker"].isin(tickers)]
        cand = cand.sort_values(["ticker", "signal_date"]).drop_duplicates("ticker", keep="last")

        row = {"date": dt}
        for t in tickers:
            row[t] = 0.0

        if cand.empty:
            rows.append(row)
            continue

        if no_threshold:
            eligible = cand.copy()
        else:
            mask = cand.apply(lambda r: r["score"] >= thresholds.get(r["ticker"], 0.0), axis=1)
            eligible = cand.loc[mask]
        if not eligible.empty:
            eligible = eligible.loc[eligible["score"] > 0]

        if not eligible.empty:
            w = np.power(eligible["score"].to_numpy(dtype=float), score_power)
            w = w / w.sum()
            for ticker, weight in zip(eligible["ticker"], w):
                row[str(ticker)] = float(weight)

        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def backtest_single_capital(ohlc_map: dict[str, pd.DataFrame], signal_df: pd.DataFrame,
                             tickers: list[str], thresholds: dict[str, float] | float | None = 0.0,
                             backtest_start: str = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Réutilise directement les fonctions de core.py (build_common_panels,
    horizon_rebalance_dates, simulate_from_target_weights) pour une gestion
    en capital unique partagé entre tous les actifs."""
    open_panel, close_panel, common_idx = core.build_common_panels(
        ohlc_map, tickers, backtest_start=backtest_start
    )
    rebal_dates = core.horizon_rebalance_dates(
        common_idx, signal_df, backtest_start=backtest_start, horizon_steps=config.HORIZON_STEPS
    )
    target_weights = build_score_target_weights(signal_df, tickers, rebal_dates, thresholds=thresholds)
    equity_df, weights_daily, rebal_df = core.simulate_from_target_weights(
        open_panel, close_panel, target_weights, label="score_composite_portfolio"
    )
    return equity_df, weights_daily


# ======================================================
# 4ter) RECONSTRUCTION DE "TRADES" IMPLICITES POUR STRATEGIES CAPITAL UNIQUE
# ======================================================

def extract_implicit_trades_from_weights(weights_daily: pd.DataFrame, close_panel: pd.DataFrame,
                                          tickers: list[str], method_name: str) -> pd.DataFrame:
    """Pour les strategies capital unique / risk-aware, il n'existe pas de notion
    explicite de 'trade' (l'allocation est continue). On reconstruit des episodes
    de detention implicites: pour chaque actif, un episode commence quand son
    poids passe de 0 (ou NaN) a >0, et se termine quand le poids repasse a 0 (ou
    a la fin de la periode). Le rendement de l'episode est le rendement close-to-
    close cumule de l'actif sur cette fenetre, pondere approximativement par le
    poids moyen detenu (pour rester comparable au 'net_return' du cohorte, qui
    lui correspond a 100% du capital dedie sur toute la duree du trade).
    Cela permet de calculer un rendement moyen par 'trade' comparable au cohorte,
    pour TOUTES les strategies (pas seulement le cohorte)."""
    w = weights_daily[tickers].fillna(0.0)
    idx = w.index.intersection(close_panel.index)
    w = w.loc[idx]
    px = close_panel.loc[idx, tickers]

    rows = []
    for t in tickers:
        active = w[t] > 1e-6
        if not active.any():
            continue
        change = active.astype(int).diff().fillna(active.iloc[0].astype(int))
        starts = idx[change == 1]
        if active.iloc[0]:
            starts = idx[[0]].append(starts) if len(starts) else idx[[0]]
        ends = idx[change == -1]

        s_list = list(starts)
        e_list = list(ends)
        episodes = []
        ei = 0
        for s in s_list:
            while ei < len(e_list) and e_list[ei] <= s:
                ei += 1
            e = e_list[ei] if ei < len(e_list) else idx[-1]
            episodes.append((s, e))
            ei += 1

        for s, e in episodes:
            p_start = float(px.loc[s, t])
            p_end = float(px.loc[e, t])
            if p_start <= 0:
                continue
            gross_ret = p_end / p_start - 1.0
            avg_w = float(w.loc[s:e, t].mean())
            net_return = gross_ret * avg_w if avg_w > 0 else gross_ret
            rows.append({
                "ticker": t, "entry_date": s, "exit_date": e,
                "net_return": net_return, "gross_return": gross_ret,
                "avg_weight": avg_w, "method": method_name,
            })

    return pd.DataFrame(rows)


def summarize_trades_by_ticker(trades: pd.DataFrame, method_name: str) -> pd.DataFrame:
    """Meme resume que summarize_cohort_results_by_ticker (n_trades, avg_return,
    sharpe_like, hit_rate) mais applique a un DataFrame de trades generique
    (cohorte OU episodes implicites extraits des poids capital-unique/risk-aware)."""
    if trades.empty:
        return pd.DataFrame(columns=["method", "ticker", "n_trades", "avg_return", "sharpe_like", "hit_rate"])
    rows = []
    for ticker, sub in trades.groupby("ticker"):
        avg = sub["net_return"].mean()
        std = sub["net_return"].std(ddof=1) if len(sub) > 1 else np.nan
        rows.append({
            "method": method_name, "ticker": ticker, "n_trades": len(sub), "avg_return": avg,
            "sharpe_like": avg / std if std and std > 0 else np.nan,
            "hit_rate": (sub["net_return"] > 0).mean(),
        })
    return pd.DataFrame(rows)


def build_all_strategies_summary_by_ticker(
    cohort_trades: pd.DataFrame,
    weights_capital_unique: pd.DataFrame, weights_meanvar: pd.DataFrame,
    weights_mcap: pd.DataFrame, weights_equal: pd.DataFrame,
    close_panel: pd.DataFrame, tickers: list[str],
) -> pd.DataFrame:
    """Empile le resume (rendement moyen par trade/episode, par actif) pour
    TOUTES les strategies: cohorte, capital unique, risk-aware, buy&hold market
    cap, buy&hold equal-weight -- pour comparaison directe sur un meme graphe."""
    parts = [summarize_trades_by_ticker(cohort_trades, "cohorte")]

    specs = [
        (weights_capital_unique, "capital_unique"),
        (weights_meanvar, "risk_aware"),
        (weights_mcap, "buy_hold_marketcap"),
        (weights_equal, "buy_hold_equalweight"),
    ]
    for weights, name in specs:
        trades = extract_implicit_trades_from_weights(weights, close_panel, tickers, name)
        parts.append(summarize_trades_by_ticker(trades, name))

    return pd.concat(parts, axis=0, ignore_index=True)


# ======================================================
# 5bis) CONTRIBUTION PAR ACTIF AU RENDEMENT FINAL
# ======================================================

def compute_capital_unique_contribution(weights_daily: pd.DataFrame, close_panel: pd.DataFrame,
                                         tickers: list[str]) -> pd.Series:
    """Contribution de chaque actif au rendement total de la strategie capital unique.
    Methode standard d'attribution de performance: contribution_i = somme_t( w_i,t-1 * r_i,t ),
    ou r_i,t est le rendement journalier simple (close-to-close) de l'actif i, et w_i,t-1
    le poids detenu la veille. La somme des contributions approxime le rendement total
    du portefeuille sur la periode (decomposition arithmetique, standard en attribution)."""
    idx = weights_daily.index.intersection(close_panel.index)
    w = weights_daily.loc[idx, tickers].fillna(0.0).shift(1).fillna(0.0)
    r = close_panel.loc[idx, tickers].pct_change().fillna(0.0)
    contrib = (w * r).sum()
    return contrib.sort_values(ascending=False)


def compute_cohort_contribution(cohort_trades: pd.DataFrame) -> pd.Series:
    """Contribution de chaque actif au P&L total (en devise) du cohorte, obtenue en
    sommant le pnl dollar de tous les trades par ticker (chaque trade a un capital
    dedie identique, donc cette somme reflete a la fois le nombre de trades et leur
    qualite moyenne pour cet actif)."""
    if cohort_trades.empty:
        return pd.Series(dtype=float)
    return cohort_trades.groupby("ticker")["pnl"].sum().sort_values(ascending=False)


# ======================================================
# 5ter) ETUDE DE SENSIBILITE A L'HORIZON DE HOLD
# ======================================================

def run_single_horizon_experiment(horizon_steps: int, n_days: int = 1500, seed: int = 42,
                                   thresholds: dict[str, float] | None = None,
                                   tickers_filter: list[str] | None = None,
                                   risk_aversion: float = 5.0, cov_lookback: int = 60,
                                   max_weight: float = 0.6) -> dict[str, float]:
    """Reconstruit tout le pipeline (univers, signaux, cohorte, capital unique,
    risk-aware, buy&hold) pour UNE valeur d'horizon donnee (config.HORIZON_STEPS
    est temporairement modifie puis restaure), et retourne les multiples
    d'equity finaux (valeur finale / valeur initiale) pour chaque methode.

    tickers_filter: si fourni, restreint l'univers a ce sous-ensemble d'actifs
    (ex: ["HIGHVOL", "RANGE"]) au lieu des 8 profils complets."""
    old_horizon = config.HORIZON_STEPS
    try:
        config.HORIZON_STEPS = horizon_steps

        universe = build_synthetic_universe(n_days=n_days, seed=seed)
        if tickers_filter is not None:
            universe = {k: v for k, v in universe.items() if k in tickers_filter}
        tickers = list(universe.keys())
        ohlc_map, signal_df = build_multi_asset_inputs_synthetic(universe)
        signal_df = add_composite_score(signal_df, w_gap=1.0, w_dir=1.0, use_continuous_gap=True)

        if signal_df.empty:
            raise ValueError(f"Aucun signal genere pour horizon={horizon_steps} (fenetre trop courte).")

        common_start = pd.Timestamp(signal_df["entry_date"].min())
        thr = thresholds if thresholds is not None else {t: 0.0 for t in tickers}
        thr = {k: v for k, v in thr.items() if k in tickers} if isinstance(thr, dict) else thr

        cohort_results = backtest_cohort_baselines(ohlc_map, signal_df, threshold=None,
                                                     horizon_steps=horizon_steps)
        cohort_trades = cohort_results["composite"]

        equity_strategy, weights_daily = backtest_single_capital(
            ohlc_map, signal_df, tickers, thresholds=thr, backtest_start=str(common_start.date())
        )
        equity_meanvar, _ = backtest_meanvar_capital(
            ohlc_map, signal_df, tickers, thresholds=thr, risk_aversion=risk_aversion,
            cov_lookback=cov_lookback, max_weight=max_weight, backtest_start=str(common_start.date())
        )
        open_panel, close_panel, common_idx = core.build_common_panels(
            ohlc_map, tickers, backtest_start=str(common_start.date())
        )
        cohort_equity = build_cohort_equity_curve(cohort_trades, common_idx)

        p0 = close_panel.iloc[0]
        mcap_weights = (p0 / p0.sum()).to_dict()
        mcap_target = pd.DataFrame([{**{"date": common_idx[0]}, **mcap_weights}]).set_index("date")
        equity_mcap, _, _ = core.simulate_from_target_weights(
            open_panel, close_panel, mcap_target, label="buy_hold_marketcap"
        )
        equal_target = pd.DataFrame([{**{"date": common_idx[0]},
                                       **{t: 1.0 / len(tickers) for t in tickers}}]).set_index("date")
        equity_equal, _, _ = core.simulate_from_target_weights(
            open_panel, close_panel, equal_target, label="buy_hold_equalweight"
        )

        return {
            "horizon_steps": horizon_steps,
            "n_signals": int(len(signal_df)),
            "n_cohort_trades": int(len(cohort_trades)),
            "final_multiple_cohort": float(cohort_equity.iloc[-1] / cohort_equity.iloc[0]),
            "final_multiple_capital_unique": float(equity_strategy["equity"].iloc[-1] / equity_strategy["equity"].iloc[0]),
            "final_multiple_risk_aware": float(equity_meanvar["equity"].iloc[-1] / equity_meanvar["equity"].iloc[0]),
            "final_multiple_buy_hold_mcap": float(equity_mcap["equity"].iloc[-1] / equity_mcap["equity"].iloc[0]),
            "final_multiple_buy_hold_equal": float(equity_equal["equity"].iloc[-1] / equity_equal["equity"].iloc[0]),
        }
    finally:
        config.HORIZON_STEPS = old_horizon


def run_horizon_sweep(horizons: list[int], n_days: int = 1500, seed: int = 42,
                       thresholds: dict[str, float] | None = None,
                       tickers_filter: list[str] | None = None,
                       risk_aversion: float = 5.0, cov_lookback: int = 60,
                       max_weight: float = 0.6) -> pd.DataFrame:
    """Boucle run_single_horizon_experiment sur une liste d'horizons et empile
    les resultats dans un DataFrame pret pour le graphe 'performance vs horizon'.
    tickers_filter permet de restreindre le sweep a un sous-univers (ex: ["HIGHVOL", "RANGE"])."""
    rows = []
    for h in horizons:
        print(f"[horizon_sweep] horizon_steps={h}")
        try:
            rows.append(run_single_horizon_experiment(
                h, n_days=n_days, seed=seed, thresholds=thresholds, tickers_filter=tickers_filter,
                risk_aversion=risk_aversion, cov_lookback=cov_lookback, max_weight=max_weight,
            ))
        except Exception as exc:
            print(f"  -> echec pour horizon={h}: {exc}")
    return pd.DataFrame(rows)


# ======================================================
# 5quater) ALLOCATION RISK-AWARE (SCORE vs VOLATILITE)
# ======================================================

def _covariance_matrix(close_panel: pd.DataFrame, tickers: list[str], dt: pd.Timestamp,
                        lookback: int = 60) -> np.ndarray:
    """Matrice de covariance des rendements log journaliers sur une fenetre glissante
    se terminant juste avant dt (pas de look-ahead)."""
    idx_all = close_panel.index
    pos = idx_all.searchsorted(dt)
    start = max(0, pos - lookback)
    window = close_panel[tickers].iloc[start:pos]
    if len(window) < 10:
        return np.eye(len(tickers)) * 1e-4
    rets = np.log(window).diff().dropna()
    cov = rets.cov().to_numpy()
    cov = cov + np.eye(len(tickers)) * 1e-8
    return cov


def build_meanvar_target_weights(signal_df: pd.DataFrame, close_panel: pd.DataFrame,
                                  tickers: list[str], rebal_dates: pd.DatetimeIndex,
                                  thresholds: dict[str, float] | float | None = 0.0,
                                  risk_aversion: float = 5.0, cov_lookback: int = 60,
                                  long_only: bool = True, max_weight: float = 0.6,
                                  annualize_cov: bool = True) -> pd.DataFrame:
    """Allocation qui maximise score'w - risk_aversion * w' Sigma w, sous contrainte
    sum(w) = 1, 0 <= w <= max_weight, et w_i = 0 si score_i < seuil_i.

    ATTENTION D'ECHELLE : Sans annualisation, le terme de
    risque w'Sigma w est negligeable face au terme de score, et le programme
    devient quasi-lineaire en w -> l'optimum est un sommet du simplexe (100% sur
    le seul actif au score max), ce qui produit une allocation extremement
    concentree et un whipsaw permanent (changement brutal d'actif a chaque
    rebalancement, frais + volatilite qui detruisent l'equity).

    Deux corrections appliquees:
    1) annualize_cov=True multiplie Sigma par 252 pour la ramener a une echelle
       annuelle comparable a celle du score (le risque penalise redevient
       significatif face au terme de score).
    2) max_weight plafonne le poids max par actif (garde-fou de diversification
       qui empeche la concentration totale, quel que soit le reglage de gamma).
    """
    from scipy.optimize import minimize

    no_threshold = thresholds is None
    if not no_threshold and not isinstance(thresholds, dict):
        thresholds = {t: thresholds for t in tickers}

    cov_scale = 252.0 if annualize_cov else 1.0
    rows = []
    for dt in rebal_dates:
        cand = signal_df[signal_df["entry_date"] == dt].copy()
        cand = cand[cand["ticker"].isin(tickers)]
        cand = cand.sort_values(["ticker", "signal_date"]).drop_duplicates("ticker", keep="last")

        row = {"date": dt}
        for t in tickers:
            row[t] = 0.0

        if cand.empty:
            rows.append(row)
            continue

        if no_threshold:
            eligible = cand.copy()
        else:
            mask = cand.apply(lambda r: r["score"] >= thresholds.get(r["ticker"], 0.0), axis=1)
            eligible = cand.loc[mask]
        eligible = eligible.loc[eligible["score"] > 0]

        if eligible.empty:
            rows.append(row)
            continue

        elig_tickers = eligible["ticker"].tolist()
        scores = eligible["score"].to_numpy(dtype=float)
        n = len(elig_tickers)

        full_cov = _covariance_matrix(close_panel, tickers, dt, lookback=cov_lookback) * cov_scale
        idx_map = [tickers.index(t) for t in elig_tickers]
        cov = full_cov[np.ix_(idx_map, idx_map)]

        mw = max_weight if n > 1 else 1.0
        mw = max(mw, 1.0 / n)  # garantit la faisabilite si max_weight * n < 1

        def objective(w, scores=scores, cov=cov, gamma=risk_aversion):
            return -(w @ scores - gamma * (w @ cov @ w))

        w0 = np.full(n, 1.0 / n)
        bounds = [(0.0, mw)] * n if long_only else [(-mw, mw)] * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints,
                        options={"maxiter": 300, "ftol": 1e-10})
        w_opt = res.x if res.success else w0
        w_opt = np.clip(w_opt, 0.0, None)
        if w_opt.sum() > 0:
            w_opt = w_opt / w_opt.sum()

        for ticker, weight in zip(elig_tickers, w_opt):
            row[str(ticker)] = float(weight)

        rows.append(row)

    return pd.DataFrame(rows).set_index("date")


def backtest_meanvar_capital(ohlc_map: dict[str, pd.DataFrame], signal_df: pd.DataFrame,
                              tickers: list[str], thresholds: dict[str, float] | float | None = 0.0,
                              risk_aversion: float = 5.0, cov_lookback: int = 60,
                              max_weight: float = 0.6, annualize_cov: bool = True,
                              backtest_start: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Meme logique que backtest_single_capital, mais l'allocation est calculee
    par build_meanvar_target_weights (score/risque) au lieu d'un poids
    proportionnel au score seul."""
    open_panel, close_panel, common_idx = core.build_common_panels(
        ohlc_map, tickers, backtest_start=backtest_start
    )
    rebal_dates = core.horizon_rebalance_dates(
        common_idx, signal_df, backtest_start=backtest_start, horizon_steps=config.HORIZON_STEPS
    )
    target_weights = build_meanvar_target_weights(
        signal_df, close_panel, tickers, rebal_dates, thresholds=thresholds,
        risk_aversion=risk_aversion, cov_lookback=cov_lookback,
        max_weight=max_weight, annualize_cov=annualize_cov,
    )
    equity_df, weights_daily, rebal_df = core.simulate_from_target_weights(
        open_panel, close_panel, target_weights, label="score_meanvar_portfolio"
    )
    return equity_df, weights_daily


# ======================================================
# 6) POINT D'ENTREE / EXEMPLE D'UTILISATION
# ======================================================

if __name__ == "__main__":
    universe = build_synthetic_universe(n_days=1500, seed=42)
    tickers = list(universe.keys())

    ohlc_map, signal_df = build_multi_asset_inputs_synthetic(universe)
    signal_df = add_composite_score(signal_df, w_gap=1.0, w_dir=1.0, use_continuous_gap=True)

    # --- Test 1: backtest par cohorte, avec baselines ---
    threshold_global = signal_df["score"].quantile(0.7)
    cohort_results = backtest_cohort_baselines(ohlc_map, signal_df, threshold=threshold_global)
    summary = summarize_cohort_results(cohort_results)
    print(summary)

    # --- Test 2: backtest capital unique, seuils individuels par actif ---
    # Ex: seuil plus strict sur l'actif très volatile, plus permissif sur le range
    thresholds = {
        "RISKFREE": 0.05, "HIGHVOL": 0.15, "NODRIFT": 0.08, "CRASH": 0.10,
        "BULL": 0.05, "SEASONAL": 0.08, "RANGE": 0.02,
    }
    equity_df, weights_daily = backtest_single_capital(ohlc_map, signal_df, tickers, thresholds=thresholds)
    stats = core.annual_stats_from_equity(equity_df["equity"])
    print("total_return, ann_return, ann_vol, sharpe, max_dd =", stats)
