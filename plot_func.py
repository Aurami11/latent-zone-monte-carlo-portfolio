import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from core import annual_stats_from_equity

# ======================================================
# SUMMARY / PLOTS
# ======================================================
def summarize_equity_curve(name: str, equity: pd.Series) -> dict:
    total_return, ann_return, ann_vol, sharpe, max_dd = annual_stats_from_equity(equity)
    return {
        "name": name,
        "final_equity": float(equity.iloc[-1]),
        "total_return": float(total_return) if np.isfinite(total_return) else np.nan,
        "ann_return": float(ann_return) if np.isfinite(ann_return) else np.nan,
        "ann_vol": float(ann_vol) if np.isfinite(ann_vol) else np.nan,
        "sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "max_drawdown": float(max_dd) if np.isfinite(max_dd) else np.nan,
    }


def save_equity_comparison_plot(curves: pd.DataFrame, filepath: str) -> None:
    if curves.empty:
        return

    plt.figure(figsize=(13, 7))
    for col in curves.columns:
        plt.plot(curves.index, curves[col], linewidth=2, label=col)

    plt.title("Portfolio Top-K Horizon vs Benchmarks")
    plt.xlabel("Date")
    plt.ylabel("Capital")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


def save_weights_plot(weights_df: pd.DataFrame, filepath: str, max_assets: int = 12) -> None:
    df = weights_df.copy()
    if df.empty:
        return

    avg_abs = df.abs().mean().sort_values(ascending=False)
    keep = list(avg_abs.head(max_assets).index)
    other = [c for c in df.columns if c not in keep]

    plot_df = df[keep].copy()
    if other:
        plot_df["OTHER"] = df[other].sum(axis=1)

    plot_df = plot_df.fillna(0.0)

    plt.figure(figsize=(14, 7))
    plt.stackplot(plot_df.index, [plot_df[c].to_numpy() for c in plot_df.columns], labels=plot_df.columns)
    plt.title("Daily Portfolio Weights")
    plt.xlabel("Date")
    plt.ylabel("Weight")
    plt.legend(loc="upper left", ncol=2)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()