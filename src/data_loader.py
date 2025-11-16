"""
data_loader.py

Utility functions to download and prepare NSE equity data for portfolio
optimization experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class MarketData:
    tickers: List[str]
    prices: pd.DataFrame          # wide: dates x tickers (Adj Close)
    returns: pd.DataFrame         # wide: dates x tickers (log returns)


def download_price_data(
    tickers: List[str],
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download adjusted close prices using yfinance.

    Parameters
    ----------
    tickers : list of str
        NSE symbols (e.g., 'RELIANCE.NS')
    start, end : str
        ISO date strings (YYYY-MM-DD)
    interval : str
        e.g. '1d', '1wk', '1mo'

    Returns
    -------
    DataFrame with columns=tickers, index=DatetimeIndex
    """
    data = yf.download(tickers, start=start, end=end, interval=interval)["Adj Close"]
    # If single ticker, yfinance returns a Series
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
    data = data.dropna(how="all")
    return data


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns from price data."""
    returns = np.log(prices / prices.shift(1)).dropna(how="all")
    return returns


def load_market_data(
    tickers: List[str],
    start: str,
    end: str,
    interval: str = "1d",
) -> MarketData:
    """Convenience wrapper to get prices + log-returns."""
    prices = download_price_data(tickers, start=start, end=end, interval=interval)
    returns = compute_log_returns(prices)
    return MarketData(tickers=tickers, prices=prices, returns=returns)


def mean_and_covariance(
    returns: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute expected returns and covariance matrix.

    Returns
    -------
    mu : np.ndarray, shape (N,)
        Mean log return per asset.
    sigma : np.ndarray, shape (N, N)
        Covariance matrix of log returns.
    """
    mu = returns.mean().values
    sigma = returns.cov().values
    return mu, sigma
