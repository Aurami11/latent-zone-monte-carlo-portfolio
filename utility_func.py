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