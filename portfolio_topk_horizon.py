import os
import pandas as pd
from core import horizon_rebalance_dates, build_multi_asset_inputs, build_common_panels, build_topk_weight_schedule, build_equal_weight_schedule, build_constant_weight_schedule, simulate_from_target_weights, simulate_buyhold_single_asset
from plot_func import summarize_equity_curve, save_equity_comparison_plot, save_weights_plot
import config

# ======================================================
# RUN
# ======================================================
def run_topk_horizon_portfolio() -> None:
    print("=== Build multi-asset inputs ===")
    ohlc_map, signal_df = build_multi_asset_inputs(config.TICKERS)

    print("=== Build common panels ===")
    open_panel, close_panel, common_idx = build_common_panels(
        ohlc_map=ohlc_map,
        tickers=config.TICKERS,
        backtest_start=config.BACKTEST_START,
    )

    print("=== Build rebalance dates ===")
    rebal_dates = horizon_rebalance_dates(
        common_idx=common_idx,
        signal_df=signal_df,
        backtest_start=config.BACKTEST_START,
        horizon_steps=config.HORIZON_STEPS,
    )

    print("=== Build strategy Top-K schedule ===")
    strategy_target_weights, strategy_diag = build_topk_weight_schedule(
        signal_df=signal_df,
        tickers=config.TICKERS,
        rebal_dates=rebal_dates,
        top_k=config.TOP_K,
        fallback_asset=config.FALLBACK_ASSET,
    )

    print("=== Build benchmark schedules ===")
    ew_target_weights = build_equal_weight_schedule(
        tickers=config.TICKERS,
        rebal_dates=rebal_dates,
    )

    bench6040_target_weights = build_constant_weight_schedule(
        tickers=config.TICKERS,
        rebal_dates=rebal_dates,
        weights_map=config.BENCH_6040,
    )

    print("=== Backtest strategy ===")
    strategy_equity, strategy_daily_weights, strategy_rebal = simulate_from_target_weights(
        open_panel=open_panel,
        close_panel=close_panel,
        target_weights=strategy_target_weights,
        initial_capital=config.INITIAL_CAPITAL,
        fee_bps=config.FEE_BPS,
        label="strategy_topk_horizon",
    )

    print("=== Backtest equal-weight ===")
    ew_equity, ew_daily_weights, ew_rebal = simulate_from_target_weights(
        open_panel=open_panel,
        close_panel=close_panel,
        target_weights=ew_target_weights,
        initial_capital=config.INITIAL_CAPITAL,
        fee_bps=config.FEE_BPS,
        label="equal_weight",
    )

    print("=== Backtest SPY/IEF 60/40 ===")
    bench6040_equity, bench6040_daily_weights, bench6040_rebal = simulate_from_target_weights(
        open_panel=open_panel,
        close_panel=close_panel,
        target_weights=bench6040_target_weights,
        initial_capital=config.INITIAL_CAPITAL,
        fee_bps=config.FEE_BPS,
        label="spy_ief_60_40",
    )

    print("=== Backtest SPY buy & hold ===")
    spy_bh_equity = simulate_buyhold_single_asset(
        open_panel=open_panel,
        close_panel=close_panel,
        ticker="SPY",
        initial_capital=config.INITIAL_CAPITAL,
        fee_bps=config.FEE_BPS,
    )

    curves = pd.DataFrame(
        {
            "strategy_topk_horizon": strategy_equity["equity"],
            "equal_weight": ew_equity["equity"],
            "spy_ief_60_40": bench6040_equity["equity"],
            "spy_buyhold": spy_bh_equity["equity"],
        }
    ).dropna(how="all")

    summary_rows = [
        summarize_equity_curve("strategy_topk_horizon", curves["strategy_topk_horizon"]),
        summarize_equity_curve("equal_weight", curves["equal_weight"]),
        summarize_equity_curve("spy_ief_60_40", curves["spy_ief_60_40"]),
        summarize_equity_curve("spy_buyhold", curves["spy_buyhold"]),
    ]
    summary_df = pd.DataFrame(summary_rows).sort_values("name").reset_index(drop=True)

    invested_df = strategy_equity[["cash_weight"]].copy()
    invested_df["invested_weight"] = 1.0 - invested_df["cash_weight"]

    subdir = os.path.join(config.OUTPUT_DIR, "topk_horizon_portfolio")
    os.makedirs(subdir, exist_ok=True)

    signal_df.to_csv(os.path.join(subdir, "all_signals.csv"), index=False)

    strategy_target_weights.to_csv(os.path.join(subdir, "strategy_target_weights.csv"))
    strategy_diag.to_csv(os.path.join(subdir, "strategy_rebalance_diagnostics.csv"))

    strategy_equity.to_csv(os.path.join(subdir, "strategy_equity.csv"))
    strategy_daily_weights.to_csv(os.path.join(subdir, "strategy_daily_weights.csv"))
    invested_df.to_csv(os.path.join(subdir, "strategy_invested_weight.csv"))
    if not strategy_rebal.empty:
        strategy_rebal.to_csv(os.path.join(subdir, "strategy_rebalances.csv"))

    ew_equity.to_csv(os.path.join(subdir, "equal_weight_equity.csv"))
    ew_daily_weights.to_csv(os.path.join(subdir, "equal_weight_daily_weights.csv"))
    if not ew_rebal.empty:
        ew_rebal.to_csv(os.path.join(subdir, "equal_weight_rebalances.csv"))

    bench6040_equity.to_csv(os.path.join(subdir, "spy_ief_60_40_equity.csv"))
    bench6040_daily_weights.to_csv(os.path.join(subdir, "spy_ief_60_40_daily_weights.csv"))
    if not bench6040_rebal.empty:
        bench6040_rebal.to_csv(os.path.join(subdir, "spy_ief_60_40_rebalances.csv"))

    spy_bh_equity.to_csv(os.path.join(subdir, "spy_buyhold_equity.csv"))
    curves.to_csv(os.path.join(subdir, "benchmark_curves.csv"))
    summary_df.to_csv(os.path.join(subdir, "performance_summary.csv"), index=False)

    save_equity_comparison_plot(
        curves=curves,
        filepath=os.path.join(subdir, "portfolio_vs_benchmarks.png"),
    )
    save_weights_plot(
        weights_df=strategy_daily_weights.drop(columns=["label"], errors="ignore"),
        filepath=os.path.join(subdir, "strategy_daily_weights.png"),
        max_assets=20,
    )

    print("\n=== SUMMARY ===")
    print(summary_df)

    print("\n=== LAST REALLOCATIONS ===")
    print(strategy_diag.tail(10)[[
        "n_candidates_before_filters",
        "n_candidates_after_filters",
        "n_selected",
        "sum_w",
        "selected_assets",
        "selected_avg_p_up",
        "selected_avg_upside",
        "fallback_used",
    ]])

    print("\nAverage invested weight:", float(invested_df["invested_weight"].mean()))
    print("Last invested weight:", float(invested_df["invested_weight"].iloc[-1]))
    print("\nSaved in:", subdir)


if __name__ == "__main__":
    run_topk_horizon_portfolio()