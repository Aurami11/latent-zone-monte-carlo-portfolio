import os
import yfinance as yf
import pandas as pd
import config
from utility_func import safe_name, today_end_exclusive, normalize_ohlc_columns


def _resolve_start_end(interval: str):
    end = config.END if config.END is not None else None

    if interval == "1h":
        max_lookback_days = 729
        today = pd.Timestamp.now().normalize()
        dynamic_start = (today - pd.Timedelta(days=max_lookback_days))
        configured_start = pd.Timestamp(config.START) if config.START else dynamic_start
        start = max(configured_start, dynamic_start)
        return start.strftime("%Y-%m-%d"), end

    return config.START, end


def _strip_tz(ts):
    if ts.tzinfo is not None:
        return ts.tz_localize(None)
    return ts


def _normalize_index_for_compare(index):
    if index.tz is not None:
        return index.tz_localize(None)
    return index


def _cache_covers_range(df, start, end):
    if df.empty:
        return False
    idx_naive = _normalize_index_for_compare(df.index)
    start_ts = _strip_tz(pd.Timestamp(start))
    if idx_naive.min() > start_ts:
        return False
    if end is not None:
        end_ts = _strip_tz(pd.Timestamp(end))
        if idx_naive.max() < end_ts:
            return False
    return True


def _filter_range(df, start, end):
    tz = df.index.tz
    start_ts = pd.Timestamp(start)
    if tz is not None and start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize(tz)
    elif tz is None and start_ts.tzinfo is not None:
        start_ts = start_ts.tz_localize(None)

    out = df[df.index >= start_ts].copy()

    if end is not None:
        end_ts = pd.Timestamp(end)
        if tz is not None and end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize(tz)
        elif tz is None and end_ts.tzinfo is not None:
            end_ts = end_ts.tz_localize(None)
        out = out[out.index <= end_ts].copy()

    return out


def download_ohlc_cached(ticker: str) -> pd.DataFrame:
    interval = config.INTERVAL
    start, end = _resolve_start_end(interval)

    path = os.path.join(config.CACHE_DIR, f"{safe_name(ticker)}_{interval}.parquet")

    if os.path.exists(path):
        cached = pd.read_parquet(path)
        if _cache_covers_range(cached, start, end):
            return _filter_range(cached, start, end)

    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    df = normalize_ohlc_columns(df)
    df = df[["Open", "Close", "High", "Low", "Volume"]].dropna().copy()

    os.makedirs(config.CACHE_DIR, exist_ok=True)
    df.to_parquet(path)

    return _filter_range(df, start, end)


if __name__ == "__main__":
    ticker = "GOOG"
    df = download_ohlc_cached(ticker)
    print(df.head())
    print(df.tail())
    print(f"n rows = {len(df)}, from {df.index.min()} to {df.index.max()}")