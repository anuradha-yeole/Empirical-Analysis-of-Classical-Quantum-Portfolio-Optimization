"""
run_experiments.py

End-to-end experiment script for classical vs quantum portfolio optimization.

Steps:
1. Load NSE equity data via yfinance
2. Compute log returns, expected returns (mu), and covariance (sigma)
3. Run classical optimizers: MVO, GA, Simulated Annealing
4. Run quantum optimizers: QAOA, VQE, Exact MES (on a small asset set)
5. Backtest each portfolio on the historical data
6. Print a summary table and show Plotly charts
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from typing import List, Dict

import numpy as np
import pandas as pd

from src.data_loader import load_market_data, mean_and_covariance
from src.classical_optimizers import (
    solve_mvo,
    solve_ga,
    solve_simulated_annealing,
)
from src.quantum_optimizers import (
    solve_qaoa,
    solve_vqe,
    solve_exact_mes,
)
from src.risk_metrics import (
    PortfolioResult,
    backtest_portfolio,
    sharpe_ratio,
)
from src.visualization import (
    plot_risk_return,
    plot_equity_curves,
)


def select_universe(tickers: List[str], n_assets: int = 4) -> List[str]:
    """
    For quantum experiments we need a small asset universe.
    This picks the first n tickers.
    """
    if len(tickers) < n_assets:
        raise ValueError(f"Need at least {n_assets} tickers, got {len(tickers)}.")
    return tickers[:n_assets]


def main():
    # ---------------------------------------------------------
    # 1. Configuration
    # ---------------------------------------------------------
    # Example NSE tickers (user can change these)
    full_tickers = [
        "RELIANCE.NS",
        "TCS.NS",
        "INFY.NS",
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "KOTAKBANK.NS",
    ]

    start_date = "2021-01-01"
    end_date = "2024-01-01"
    risk_aversion = 0.5
    quantum_universe_size = 4      # use only 4 assets for QAOA/VQE/MES
    budget = 2                     # how many assets to select in quantum models

    print("=== Loading market data ===")
    market = load_market_data(full_tickers, start=start_date, end=end_date, interval="1d")
    print(f"Loaded prices with shape: {market.prices.shape}")

    # Restrict to a small universe for both classical & quantum so results are comparable
    small_universe = select_universe(market.tickers, n_assets=quantum_universe_size)
    prices = market.prices[small_universe].dropna()
    returns = market.returns[small_universe].dropna()

    print(f"Using asset universe: {small_universe}")
    print(f"Returns shape: {returns.shape}")

    mu, sigma = mean_and_covariance(returns)

    # ---------------------------------------------------------
    # 2. Run classical optimizers
    # ---------------------------------------------------------
    print("\n=== Running classical optimizers ===")
    classical_results: List[PortfolioResult] = []

    mvo_res = solve_mvo(mu, sigma, risk_aversion=risk_aversion)
    classical_results.append(mvo_res)
    print(f"[MVO]  Return={mvo_res.expected_return:.4f}, Risk={mvo_res.risk:.4f}")

    ga_res = solve_ga(mu, sigma, risk_aversion=risk_aversion)
    classical_results.append(ga_res)
    print(f"[GA]   Return={ga_res.expected_return:.4f}, Risk={ga_res.risk:.4f}")

    sa_res = solve_simulated_annealing(mu, sigma, risk_aversion=risk_aversion)
    classical_results.append(sa_res)
    print(f"[SA]   Return={sa_res.expected_return:.4f}, Risk={sa_res.risk:.4f}")

    # ---------------------------------------------------------
    # 3. Run quantum optimizers
    # ---------------------------------------------------------
    print("\n=== Running quantum optimizers (small asset universe) ===")
    quantum_results: List[PortfolioResult] = []

    try:
        exact_res = solve_exact_mes(mu, sigma, risk_aversion=risk_aversion, budget=budget)
        quantum_results.append(exact_res)
        print(f"[Exact MES]  Return={exact_res.expected_return:.4f}, Risk={exact_res.risk:.4f}")
    except Exception as e:
        print(f"Exact MES failed: {e}")

    try:
        qaoa_res = solve_qaoa(mu, sigma, risk_aversion=risk_aversion, budget=budget, reps=1)
        quantum_results.append(qaoa_res)
        print(f"[QAOA]       Return={qaoa_res.expected_return:.4f}, Risk={qaoa_res.risk:.4f}")
    except Exception as e:
        print(f"QAOA failed: {e}")

    try:
        vqe_res = solve_vqe(mu, sigma, risk_aversion=risk_aversion, budget=budget)
        quantum_results.append(vqe_res)
        print(f"[VQE]        Return={vqe_res.expected_return:.4f}, Risk={vqe_res.risk:.4f}")
    except Exception as e:
        print(f"VQE failed: {e}")

    all_results: List[PortfolioResult] = classical_results + quantum_results

    # ---------------------------------------------------------
    # 4. Backtest each portfolio
    # ---------------------------------------------------------
    print("\n=== Backtesting portfolios ===")
    equity_curves: Dict[str, pd.Series] = {}
    sharpe_scores: Dict[str, float] = {}

    for res in all_results:
        eq = backtest_portfolio(res.weights, returns, initial_capital=1.0)
        equity_curves[res.name] = eq
        sr = sharpe_ratio(returns @ res.weights)
        sharpe_scores[res.name] = sr
        print(f"[{res.name}] Final Equity={eq.iloc[-1]:.3f}, Sharpe={sr:.3f}")

    # ---------------------------------------------------------
    # 5. Summarize as a table
    # ---------------------------------------------------------
    print("\n=== Summary Table ===")
    rows = []
    for res in all_results:
        rows.append({
            "Name": res.name,
            "Expected Return": res.expected_return,
            "Risk (Var)": res.risk,
            "Objective": res.objective,
            "Sharpe": sharpe_scores.get(res.name, np.nan),
        })
    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---------------------------------------------------------
    # 6. Plot results
    # ---------------------------------------------------------
    print("\nShowing Plotly figures (close the windows to finish)...")

    # Risk vs return scatter
    fig_rr = plot_risk_return(all_results, title="Risk vs Return: Classical vs Quantum")
    fig_rr.show()

    # Equity curves
    fig_eq = plot_equity_curves(equity_curves, title="Backtest Equity Curves")
    fig_eq.show()


if __name__ == "__main__":
    main()
