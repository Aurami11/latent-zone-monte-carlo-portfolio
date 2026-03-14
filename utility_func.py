import re
import pandas as pd
import numpy as np
import math
import config

def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def today_end_exclusive() -> pd.Timestamp:
    return pd.Timestamp.today().normalize() + pd.Timedelta(days=1)


def normalize_ohlc_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    return df


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


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