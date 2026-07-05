
"""
synthetic_report.py
====================
Genere un rapport HTML interactif (Plotly) a partir des sorties de synthetic_lab.py :
- trajectoires des actifs simules
- zone_obs vs zone_latent par actif
- score composite dans le temps
- composition du portefeuille (poids) au cours du temps
- comparaison equity: buy&hold market-cap-weighted vs strategie cohorte vs strategie capital unique

Usage:
    python synthetic_report.py
Genere: synthetic_report.html
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config
import core
from synthetic_lab import (
    build_synthetic_universe, build_multi_asset_inputs_synthetic,
    add_composite_score, backtest_cohort_baselines, summarize_cohort_results,
    summarize_cohort_results_by_ticker, build_score_target_weights, backtest_single_capital,
    build_cohort_equity_curve, compute_capital_unique_contribution, compute_cohort_contribution,
    run_horizon_sweep, backtest_meanvar_capital, build_all_strategies_summary_by_ticker,
)

pio.templates.default = "perplexity" if "perplexity" in pio.templates else pio.templates.default


# ======================================================
# 1) PIPELINE DE DONNEES
# ======================================================

def run_pipeline(n_days: int = 1500, seed: int = 42, tickers_filter: list[str] | None = None,
                  horizon_steps: int | None = None, risk_aversion: float = 5.0,
                  cov_lookback: int = 60, max_weight: float = 0.6):
    # Fixe explicitement l'horizon utilise par TOUT le pipeline (au lieu de
    # dependre implicitement de la valeur globale config.HORIZON_STEPS au
    # moment de l'appel). Garantit que le graphe d'equity principal utilise
    # exactement le meme horizon que le point correspondant du sweep.
    if horizon_steps is not None:
        config.HORIZON_STEPS = horizon_steps
    resolved_horizon = config.HORIZON_STEPS

    universe = build_synthetic_universe(n_days=n_days, seed=seed)
    if tickers_filter is not None:
        universe = {k: v for k, v in universe.items() if k in tickers_filter}
    tickers = list(universe.keys())

    ohlc_map, signal_df = build_multi_asset_inputs_synthetic(universe)
    signal_df = add_composite_score(signal_df, w_gap=1.0, w_dir=1.0, use_continuous_gap=True)

    # threshold=None -> seuil expanding point-in-time (pas de look-ahead), voir
    # backtest_cohort_baselines dans synthetic_lab.py
    cohort_results = backtest_cohort_baselines(ohlc_map, signal_df, threshold=None)
    cohort_summary = summarize_cohort_results(cohort_results)
    cohort_summary_by_ticker = summarize_cohort_results_by_ticker(cohort_results)

    thresholds = {
        "RISKFREE": 0.05, "HIGHVOL": 0.15, "NODRIFT": 0.08, "CRASH": 0.10,
        "BULL": 0.05, "BULL_TRUE": 0.05, "SEASONAL": 0.08, "RANGE": 0.02,
    }
    thresholds = {k: v for k, v in thresholds.items() if k in tickers}

    # Premiere date ou la strategie peut effectivement produire un signal (fin de warm-up:
    # tube/vol/fit windows). On aligne TOUTES les courbes (strategie + buy&hold) sur cette
    # meme date de depart pour une comparaison equitable (sinon le buy&hold profite d'un
    # historique de prix auquel la strategie n'a pas encore acces).
    common_start = pd.Timestamp(signal_df["entry_date"].min())

    equity_strategy, weights_daily = backtest_single_capital(
        ohlc_map, signal_df, tickers, thresholds=thresholds, backtest_start=str(common_start.date())
    )

    # Allocation risk-aware: maximise score'w - risk_aversion * w' Sigma w
    # (analogue Markowitz du score composite, penalise la volatilite/correlation).
    equity_meanvar, weights_meanvar_daily = backtest_meanvar_capital(
        ohlc_map, signal_df, tickers, thresholds=thresholds, risk_aversion=risk_aversion,
        cov_lookback=cov_lookback, max_weight=max_weight, backtest_start=str(common_start.date())
    )

    open_panel, close_panel, common_idx = core.build_common_panels(
        ohlc_map, tickers, backtest_start=str(common_start.date())
    )

    # Buy & hold "market-cap-like": poids proportionnels au prix a la date de depart commune
    # (proxy de taille), fixes sur toute la periode (pas de rebalancement).
    p0 = close_panel.iloc[0]
    mcap_weights = (p0 / p0.sum()).to_dict()
    mcap_target = pd.DataFrame([{**{"date": common_idx[0]}, **mcap_weights}]).set_index("date")
    equity_mcap, weights_mcap_daily, _ = core.simulate_from_target_weights(
        open_panel, close_panel, mcap_target, label="buy_hold_marketcap"
    )

    # Equal-weight buy & hold classique (reference supplementaire)
    equal_target = pd.DataFrame([{**{"date": common_idx[0]},
                                   **{t: 1.0 / len(tickers) for t in tickers}}]).set_index("date")
    equity_equal, weights_equal_daily, _ = core.simulate_from_target_weights(
        open_panel, close_panel, equal_target, label="buy_hold_equalweight"
    )

    # Equity curve reconstruite a partir de la strategie COHORTE (composite, seuil q70),
    # pour comparer sur le meme graphe: cohorte / capital unique / buy&hold.
    cohort_trades = results_cohort_for_equity = cohort_results["composite"]
    cohort_equity = build_cohort_equity_curve(cohort_trades, common_idx)
    cohort_equity_df = pd.DataFrame({"equity": cohort_equity}, index=common_idx)

    # Rebase a 1.0 sur la date de depart commune pour comparer des trajectoires normalisees.
    equity_strategy = equity_strategy.copy()
    equity_mcap = equity_mcap.copy()
    equity_equal = equity_equal.copy()
    equity_strategy["equity_norm"] = equity_strategy["equity"] / equity_strategy["equity"].iloc[0]
    equity_mcap["equity_norm"] = equity_mcap["equity"] / equity_mcap["equity"].iloc[0]
    equity_equal["equity_norm"] = equity_equal["equity"] / equity_equal["equity"].iloc[0]
    cohort_equity_df["equity_norm"] = cohort_equity_df["equity"] / cohort_equity_df["equity"].iloc[0]
    equity_meanvar = equity_meanvar.copy()
    equity_meanvar["equity_norm"] = equity_meanvar["equity"] / equity_meanvar["equity"].iloc[0]

    # Composition moyenne du portefeuille strategie sur toute la periode (hors dates 100% cash).
    w = weights_daily[tickers].fillna(0.0)
    avg_weights = w.mean().sort_values(ascending=False)

    # Contribution de chaque actif au rendement total (attribution de performance).
    contrib_capital_unique = compute_capital_unique_contribution(weights_daily, close_panel, tickers)
    contrib_cohort = compute_cohort_contribution(cohort_trades)

    # Rendement moyen par trade/episode et par actif, pour TOUTES les strategies
    # (pas seulement le cohorte): cohorte, capital unique, risk-aware, buy&hold.
    all_strategies_summary = build_all_strategies_summary_by_ticker(
        cohort_trades=cohort_trades,
        weights_capital_unique=weights_daily, weights_meanvar=weights_meanvar_daily,
        weights_mcap=weights_mcap_daily, weights_equal=weights_equal_daily,
        close_panel=close_panel, tickers=tickers,
    )

    # Etude de sensibilite a l'horizon de hold: on reconstruit tout le pipeline
    # pour plusieurs horizons et on compare les multiples d'equity finaux.
    horizon_sweep_df = run_horizon_sweep(
        horizons=[3, 5, 10, 15, 20, 30], n_days=n_days, seed=seed, thresholds=thresholds,
        tickers_filter=tickers_filter, risk_aversion=risk_aversion,
        cov_lookback=cov_lookback, max_weight=max_weight,
    )

    return {
        "universe": universe, "ohlc_map": ohlc_map, "signal_df": signal_df,
        "tickers": tickers, "cohort_results": cohort_results, "cohort_summary": cohort_summary,
        "cohort_summary_by_ticker": cohort_summary_by_ticker,
        "equity_strategy": equity_strategy, "weights_daily": weights_daily,
        "equity_mcap": equity_mcap, "equity_equal": equity_equal, "equity_cohort": cohort_equity_df,
        "equity_meanvar": equity_meanvar, "weights_meanvar_daily": weights_meanvar_daily,
        "close_panel": close_panel, "mcap_weights": mcap_weights,
        "avg_weights": avg_weights, "common_start": common_start, "horizon_steps": resolved_horizon,
        "contrib_capital_unique": contrib_capital_unique, "contrib_cohort": contrib_cohort,
        "horizon_sweep_df": horizon_sweep_df, "all_strategies_summary": all_strategies_summary,
    }


# ======================================================
# 2) FIGURES
# ======================================================

def fig_trajectories(close_panel: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for t in close_panel.columns:
        fig.add_trace(go.Scatter(x=close_panel.index, y=close_panel[t], mode="lines", name=t))
    fig.update_layout(
        title={"text": "Trajectoires des 7 actifs synthetiques (2015-2020)<br>"
                        "<span style='font-size: 16px; font-weight: normal;'>"
                        "Source: simulation interne | GBM, OU, saisonnalite, regime switch</span>",
               "y": 0.97},
        legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="center", x=0.5),
        margin=dict(t=140, l=70, r=40, b=60),
        height=520,
    )
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Prix", type="log")
    fig.update_traces(cliponaxis=False)
    return fig


def fig_zones(signal_df: pd.DataFrame, tickers: list[str]) -> go.Figure:
    n = len(tickers)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=False,
                         subplot_titles=[f"<b>{t}</b>" for t in tickers],
                         vertical_spacing=max(0.35 / n, 0.06))
    for i, t in enumerate(tickers, start=1):
        sub = signal_df[signal_df["ticker"] == t].sort_index()
        fig.add_trace(go.Scatter(x=sub.index, y=sub["zone_obs"], mode="lines",
                                  name="zone_obs", line=dict(width=1, color="#636EFA"),
                                  showlegend=(i == 1), legendgroup="obs"), row=i, col=1)
        fig.add_trace(go.Scatter(x=sub.index, y=sub["zone_latent"], mode="lines",
                                  name="zone_latent", line=dict(width=1, dash="dash", color="#EF553B"),
                                  showlegend=(i == 1), legendgroup="lat"), row=i, col=1)
        fig.update_yaxes(title_text="Zone", row=i, col=1, title_font=dict(size=11))
        fig.update_xaxes(title_text="Date" if i == n else None, row=i, col=1,
                          tickfont=dict(size=9))

    for ann in fig.layout.annotations:
        ann.font = dict(size=13)
        ann.y = ann.y + 0.012

    fig.update_layout(
        height=260 * n,
        title={"text": "Zone observee vs zone latente par actif<br>"
                        "<span style='font-size: 15px; font-weight: normal;'>"
                        "Source: core.py compute_latent_zone_path | ecart = signal de sur/sous-evaluation</span>",
               "y": 0.995},
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="center", x=0.5,
                    font=dict(size=11)),
        margin=dict(t=90, l=70, r=40, b=50),
    )
    fig.update_traces(cliponaxis=False)
    return fig


def fig_score(signal_df: pd.DataFrame, tickers: list[str]) -> go.Figure:
    fig = go.Figure()
    for t in tickers:
        sub = signal_df[signal_df["ticker"] == t].sort_index()
        fig.add_trace(go.Scatter(x=sub.index, y=sub["score"], mode="lines", name=t, line=dict(width=1)))
    fig.update_layout(
        title={"text": "Score composite dans le temps par actif<br>"
                        "<span style='font-size: 16px; font-weight: normal;'>"
                        "Source: calcul interne | score = -(gap) + (p_up - p_down)</span>",
               "y": 0.97},
        legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="center", x=0.5),
        margin=dict(t=140, l=70, r=40, b=60),
        height=520,
    )
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Score")
    fig.update_traces(cliponaxis=False)
    return fig


def fig_weights(weights_daily: pd.DataFrame, tickers: list[str]) -> go.Figure:
    fig = go.Figure()
    w = weights_daily[tickers].fillna(0.0)
    for t in tickers:
        fig.add_trace(go.Scatter(x=w.index, y=w[t], mode="lines", name=t, stackgroup="one"))
    fig.update_layout(
        title={"text": "Composition du portefeuille strategie (capital unique)<br>"
                        "<span style='font-size: 16px; font-weight: normal;'>"
                        "Source: core.simulate_from_target_weights | poids proportionnels au score</span>",
               "y": 0.97},
        legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="center", x=0.5),
        margin=dict(t=140, l=70, r=40, b=60),
        height=520,
    )
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Poids (%)", tickformat=".0%")
    fig.update_traces(cliponaxis=False)
    return fig


def fig_avg_weights(avg_weights: pd.Series) -> go.Figure:
    aw = avg_weights[avg_weights > 1e-6]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=aw.index.tolist(), y=aw.values, name="Poids moyen"))
    fig.update_layout(
        title={"text": "Composition moyenne du portefeuille sur toute la periode<br>"
                        "<span style='font-size: 16px; font-weight: normal;'>"
                        "Source: moyenne des poids quotidiens | strategie score composite</span>",
               "y": 0.95},
        showlegend=False,
        margin=dict(t=120, l=70, r=40, b=60),
        height=480,
    )
    fig.update_xaxes(title_text="Actif")
    fig.update_yaxes(title_text="Poids moyen (%)", tickformat=".1%")
    fig.update_traces(cliponaxis=False)
    return fig


def fig_equity_comparison(equity_strategy: pd.DataFrame, equity_mcap: pd.DataFrame,
                           equity_equal: pd.DataFrame, equity_cohort: pd.DataFrame,
                           equity_meanvar: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity_strategy.index, y=equity_strategy["equity_norm"],
                              mode="lines", name="Strategie capital unique (score seul)"))
    fig.add_trace(go.Scatter(x=equity_meanvar.index, y=equity_meanvar["equity_norm"],
                              mode="lines", name="Strategie risk-aware (score/volatilite)"))
    fig.add_trace(go.Scatter(x=equity_cohort.index, y=equity_cohort["equity_norm"],
                              mode="lines", name="Strategie cohorte (positions isolees)"))
    fig.add_trace(go.Scatter(x=equity_mcap.index, y=equity_mcap["equity_norm"],
                              mode="lines", name="Buy&Hold poids ~ market cap"))
    fig.add_trace(go.Scatter(x=equity_equal.index, y=equity_equal["equity_norm"],
                              mode="lines", name="Buy&Hold equal-weight"))
    fig.update_layout(
        title={"text": "Equity normalisee (base 1.0): cohorte vs capital unique vs buy-and-hold<br>"
                        "<span style='font-size: 15px; font-weight: normal;'>"
                        "Source: core.simulate_from_target_weights + reconstruction cohorte | meme date de depart</span>",
               "y": 0.97},
        legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="center", x=0.5),
        margin=dict(t=140, l=70, r=40, b=60),
        height=520,
    )
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Equity (base 1.0)")
    fig.update_traces(cliponaxis=False)
    return fig


def fig_cohort_summary_by_ticker(cohort_summary_by_ticker: pd.DataFrame) -> go.Figure:
    """Barres groupees: rendement moyen par trade (cohorte), une couleur par
    baseline (composite/gap/p_spread/random/immediat), un groupe par actif."""
    df = cohort_summary_by_ticker.copy()
    method_order = ["composite", "gap_only", "p_spread_only", "random_entry", "buy_and_hold_immediate"]
    df["method"] = pd.Categorical(df["method"], categories=method_order, ordered=True)
    df = df.sort_values(["ticker", "method"])

    fig = go.Figure()
    for m in method_order:
        sub = df[df["method"] == m]
        if sub.empty:
            continue
        fig.add_trace(go.Bar(x=sub["ticker"], y=sub["avg_return"], name=m))

    fig.update_layout(
        barmode="group",
        title={"text": "Rendement moyen par trade (cohorte), decompose par actif<br>"
                        "<span style='font-size: 15px; font-weight: normal;'>"
                        "Source: backtest cohorte | baselines du signal d'entree</span>",
               "y": 0.95},
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5, font=dict(size=11)),
        margin=dict(t=150, l=70, r=40, b=60),
        height=560,
    )
    fig.update_xaxes(title_text="Actif")
    fig.update_yaxes(title_text="Rendement moy.", tickformat=".2%")
    fig.update_traces(cliponaxis=False)
    return fig


def fig_all_strategies_by_ticker(all_strategies_summary: pd.DataFrame) -> go.Figure:
    """Barres groupees: rendement moyen par trade/episode, une couleur par
    STRATEGIE (cohorte, capital unique, risk-aware, buy&hold market cap,
    buy&hold equal-weight), un groupe par actif. Permet de comparer directement
    comment chaque strategie performe actif par actif, au-dela du score seul."""
    df = all_strategies_summary.copy()
    method_order = ["cohorte", "capital_unique", "risk_aware", "buy_hold_marketcap", "buy_hold_equalweight"]
    df["method"] = pd.Categorical(df["method"], categories=method_order, ordered=True)
    df = df.sort_values(["ticker", "method"])

    fig = go.Figure()
    for m in method_order:
        sub = df[df["method"] == m]
        if sub.empty:
            continue
        fig.add_trace(go.Bar(x=sub["ticker"], y=sub["avg_return"], name=m))

    fig.update_layout(
        barmode="group",
        title={"text": "Rendement moyen par trade/episode, TOUTES strategies, par actif<br>"
                        "<span style='font-size: 15px; font-weight: normal;'>"
                        "Source: cohorte + episodes de detention implicites | comparaison directe des strategies</span>",
               "y": 0.95},
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5, font=dict(size=11)),
        margin=dict(t=150, l=70, r=40, b=60),
        height=560,
    )
    fig.update_xaxes(title_text="Actif")
    fig.update_yaxes(title_text="Rendement moy.", tickformat=".2%")
    fig.update_traces(cliponaxis=False)
    return fig


def fig_cohort_summary(cohort_summary: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=cohort_summary["method"], y=cohort_summary["avg_return"],
                          name="Rendement moyen par trade"))
    fig.update_layout(
        title={"text": "Rendement moyen par trade: score composite vs baselines<br>"
                        "<span style='font-size: 16px; font-weight: normal;'>"
                        "Source: backtest cohorte | horizon fixe, capital dedie par entree</span>",
               "y": 0.95},
        showlegend=False,
        margin=dict(t=120, l=70, r=40, b=60),
        height=480,
    )
    fig.update_xaxes(title_text="Methode")
    fig.update_yaxes(title_text="Rendement moy.", tickformat=".2%")
    fig.update_traces(cliponaxis=False)
    return fig


# ======================================================
# 3) ASSEMBLAGE HTML
# ======================================================

def fig_contribution(contrib: pd.Series, title_label: str) -> go.Figure:
    s = contrib.sort_values(ascending=False)
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in s.values]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=s.index.tolist(), y=s.values, marker_color=colors, name="Contribution"))
    fig.update_layout(
        title={"text": f"Contribution de chaque actif au rendement ({title_label})<br>"
                        "<span style='font-size: 15px; font-weight: normal;'>"
                        "Source: attribution de performance | vert = contribue positivement</span>",
               "y": 0.95},
        showlegend=False,
        margin=dict(t=120, l=70, r=40, b=60),
        height=480,
    )
    fig.update_xaxes(title_text="Actif")
    fig.update_yaxes(title_text="Contribution", tickformat=".2%")
    fig.update_traces(cliponaxis=False)
    return fig


def fig_horizon_sweep(horizon_sweep_df: pd.DataFrame, current_horizon: int | None = None) -> go.Figure:
    df = horizon_sweep_df.sort_values("horizon_steps")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["horizon_steps"], y=df["final_multiple_cohort"],
                              mode="lines+markers", name="Strategie cohorte"))
    fig.add_trace(go.Scatter(x=df["horizon_steps"], y=df["final_multiple_capital_unique"],
                              mode="lines+markers", name="Strategie capital unique"))
    if "final_multiple_risk_aware" in df.columns:
        fig.add_trace(go.Scatter(x=df["horizon_steps"], y=df["final_multiple_risk_aware"],
                                  mode="lines+markers", name="Strategie risk-aware"))
    fig.add_trace(go.Scatter(x=df["horizon_steps"], y=df["final_multiple_buy_hold_mcap"],
                              mode="lines+markers", name="Buy&Hold market cap"))
    fig.add_trace(go.Scatter(x=df["horizon_steps"], y=df["final_multiple_buy_hold_equal"],
                              mode="lines+markers", name="Buy&Hold equal-weight"))
    if current_horizon is not None:
        fig.add_vline(x=current_horizon, line_dash="dash", line_color="gray",
                       annotation_text=f"H actuel = {current_horizon}", annotation_position="top")

    fig.update_layout(
        title={"text": "Multiple d'equity final selon l'horizon de hold (H jours)<br>"
                        "<span style='font-size: 15px; font-weight: normal;'>"
                        "Source: pipeline reconstruit par horizon | ligne pointillee = horizon du graphe equity ci-dessus</span>",
               "y": 0.95},
        legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="center", x=0.5, font=dict(size=11)),
        margin=dict(t=140, l=70, r=40, b=60),
        height=520,
    )
    fig.update_xaxes(title_text="Horizon H (jours)")
    fig.update_yaxes(title_text="Multiple final")
    fig.update_traces(cliponaxis=False)
    return fig


def build_html_report(results: dict, out_path: str = "synthetic_report.html"):
    figs = [
        fig_trajectories(results["close_panel"]),
        fig_zones(results["signal_df"], results["tickers"]),
        fig_score(results["signal_df"], results["tickers"]),
        fig_weights(results["weights_daily"], results["tickers"]),
        fig_avg_weights(results["avg_weights"]),
        fig_equity_comparison(results["equity_strategy"], results["equity_mcap"], results["equity_equal"], results["equity_cohort"], results["equity_meanvar"]),
        fig_contribution(results["contrib_capital_unique"], "strategie capital unique"),
        fig_contribution(results["contrib_cohort"], "strategie cohorte, en PnL$"),
        fig_horizon_sweep(results["horizon_sweep_df"], current_horizon=results["horizon_steps"]),
        fig_cohort_summary(results["cohort_summary"]),
        fig_cohort_summary_by_ticker(results["cohort_summary_by_ticker"]),
        fig_all_strategies_by_ticker(results["all_strategies_summary"]),
    ]

    style = """
    <style>
        body { font-family: -apple-system, Arial, sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; background: #fafafa; }
        h1 { text-align: center; }
        .chart-block { background: #fff; border-radius: 10px; padding: 10px; margin-bottom: 40px;
                       box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
    </style>
    """
    note = """
    <div style="background:#fff3cd; border-left:4px solid #e0a800; padding:12px 16px;
                border-radius:6px; margin-bottom:24px; font-size:14px;">
    <b>Note methodologique - lecture du graphe Equity :</b> la courbe "Strategie cohorte"
    represente une <b>borne theorique sans contrainte de capital</b> (chaque signal valide
    ouvre une position dediee, sans jamais rationner le cash entre trades simultanes, et
    le seuil de score est desormais calcule en point-in-time / expanding pour eviter tout
    look-ahead). La courbe "Strategie capital unique" reflete la performance realiste avec
    un portefeuille fini (arbitrage entre positions, cash limite). Un ecart important entre
    les deux courbes mesure donc le cout de la contrainte de capital, pas une erreur de
    calcul.
    </div>
    """
    html_parts = [f"<html><head><meta charset='utf-8'>"
                  f"<title>Rapport univers synthetique</title>{style}</head><body>"
                  f"<h1>Rapport - Univers synthetique multi-actifs</h1>{note}"]
    for i, fig in enumerate(figs):
        include_js = "cdn" if i == 0 else False
        html_parts.append("<div class='chart-block'>")
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs=include_js))
        html_parts.append("</div>")
    html_parts.append("</body></html>")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    print(f"Rapport ecrit: {out_path}")


if __name__ == "__main__":
    # Univers complet (8 actifs synthetiques).
    results = run_pipeline(n_days=1500, seed=42, max_weight=1.0)
    build_html_report(results, out_path="synthetic_report.html")

    # Exemple: pour restreindre l'univers a un sous-ensemble d'actifs (ex: juste
    # HIGHVOL et RANGE), decommenter les 2 lignes suivantes:
    # results_sub = run_pipeline(n_days=1500, seed=42, tickers_filter=["HIGHVOL", "RANGE"])
    # build_html_report(results_sub, out_path="synthetic_report_highvol_range.html")
