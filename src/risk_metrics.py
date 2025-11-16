"""
risk_metrics.py

Common portfolio risk/return metrics and a simple result container.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class PortfolioResult:
    name: str
    weights: np.ndarray
    expected_return: float
    risk: float
    objective: float
    extra: Dict[str, float] = field(default_factory=dict)


def portfolio_return(weights: np.ndarray, mu: np.ndarray) -> float:
    """Expected portfolio return E[R_p] = w^T mu."""
    return float(weights @ mu)


def portfolio_variance(weights: np.ndarray, sigma: np.ndarray) -> float:
    """Portfolio variance Var(R_p) = w^T Σ w."""
    return float(weights @ sigma @ weights)


def sharpe_ratio(
    portfolio_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Annualized Sharpe ratio.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Periodic portfolio returns (e.g., daily log returns).
    risk_free_rate : float
        Risk-free rate per year.
    periods_per_year : int
        Number of periods per year (252 trading days).

    Returns
    -------
    float
    """
    excess = portfolio_returns - risk_free_rate / periods_per_year
    mean_excess = excess.mean()
    std_excess = excess.std()
    if std_excess == 0:
        return 0.0
    sharpe = (mean_excess / std_excess) * np.sqrt(periods_per_year)
    return float(sharpe)


def mean_squared_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Simple MSE helper."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(((y_true - y_pred) ** 2).mean())


def backtest_portfolio(
    weights: np.ndarray,
    returns: pd.DataFrame,
    initial_capital: float = 1.0,
) -> pd.Series:
    """
    Compute portfolio equity curve given asset returns and weights.

    Parameters
    ----------
    weights : np.ndarray
        Asset weights (sum to 1).
    returns : DataFrame
        Asset returns aligned in time (e.g., log returns).
    initial_capital : float
        Starting value.

    Returns
    -------
    equity_curve : pd.Series
        Portfolio value through time.
    """
    w = np.asarray(weights)
    port_ret = (returns * w).sum(axis=1)
    equity = initial_capital * (1 + port_ret).cumprod()
    equity.name = "equity"
    return equity
