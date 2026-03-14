import os
import yfinance as yf
import pandas as pd
import config
from utility_func import safe_name, today_end_exclusive, normalize_ohlc_columns

def download_ohlc_cached(ticker: str) -> pd.DataFrame:
    path = os.path.join(config.CACHE_DIR, f"{safe_name(ticker)}_{config.INTERVAL}.parquet")

    if os.path.exists(path):
        df = pd.read_parquet(path)
        return df[df.index >= pd.Timestamp(config.START)].copy()

    end = config.END if config.END is not None else str(today_end_exclusive().date())
    df = yf.download(
        ticker,
        start=config.START,
        end=end,
        interval=config.INTERVAL,
        auto_adjust=True,
        progress=False,
    )
    df = normalize_ohlc_columns(df)
    df = df[["Open", "Close"]].dropna().copy()
    df.to_parquet(path)
    return df[df.index >= pd.Timestamp(config.START)].copy()