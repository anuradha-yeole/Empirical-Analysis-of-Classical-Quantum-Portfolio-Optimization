"""
visualization.py

Plotly-based visualizations for:
- Efficient frontier / risk-return scatter
- Backtest equity curves
- Convergence traces (if you log them)
"""

from __future__ import annotations

from typing import List, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .risk_metrics import PortfolioResult


def plot_risk_return(
    portfolios: List[PortfolioResult],
    title: str = "Risk vs Return",
) -> go.Figure:
    """
    Scatter plot of portfolio risk (x) vs expected return (y).
    """
    fig = go.Figure()

    for p in portfolios:
        fig.add_trace(
            go.Scatter(
                x=[p.risk],
                y=[p.expected_return],
                mode="markers+text",
                name=p.name,
                text=[p.name],
                textposition="top center",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Variance (Risk)",
        yaxis_title="Expected Return",
        template="plotly_white",
    )
    return fig


def plot_equity_curves(
    curves: Dict[str, pd.Series],
    title: str = "Portfolio Backtest",
) -> go.Figure:
    """
    Plot multiple equity curves on the same figure.

    Parameters
    ----------
    curves : dict
        Mapping name -> equity pandas Series.
    """
    fig = go.Figure()
    for name, series in curves.items():
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=name,
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Equity",
        template="plotly_white",
    )
    return fig


def plot_convergence(
    traces: Dict[str, List[float]],
    title: str = "Optimizer Convergence",
    yaxis_title: str = "Objective value",
) -> go.Figure:
    """
    Simple line plot to show objective value per iteration for multiple optimizers.

    Parameters
    ----------
    traces : dict
        Mapping name -> list of objective values.
    """
    fig = go.Figure()
    for name, vals in traces.items():
        fig.add_trace(
            go.Scatter(
                x=list(range(len(vals))),
                y=vals,
                mode="lines",
                name=name,
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Iteration",
        yaxis_title=yaxis_title,
        template="plotly_white",
    )
    return fig
