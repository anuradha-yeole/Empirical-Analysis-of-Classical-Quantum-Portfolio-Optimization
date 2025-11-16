"""
quantum_optimizers.py

Quantum portfolio optimizers using Qiskit:
- QAOA
- VQE
- Exact (NumPyMinimumEigensolver)

We use a binary formulation:
    x_i in {0,1}  -> whether asset i is included.
Weights are equal among selected assets.

Cost:
    J(x) = risk_aversion * x^T Σ x - (1-risk_aversion) * μ^T x
with a soft constraint Sum x_i = budget (penalty term).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from qiskit import QuantumCircuit
from qiskit.primitives import Sampler, Estimator
from qiskit.algorithms import QAOA, VQE
from qiskit.algorithms.optimizers import COBYLA, SPSA
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuboConverter
from qiskit_optimization.algorithms import MinimumEigenOptimizer

from .risk_metrics import PortfolioResult, portfolio_return, portfolio_variance


def build_qubo(
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_aversion: float = 0.5,
    budget: int | None = None,
    penalty: float = 10.0,
) -> QuadraticProgram:
    """
    Build a QUBO for binary portfolio selection.

    Parameters
    ----------
    mu, sigma : np.ndarray
        Expected returns and covariance.
    risk_aversion : float
        Tradeoff between risk and return.
    budget : int or None
        Number of assets to select (if None, no cardinality constraint).
    penalty : float
        Penalty factor for budget violation.

    Returns
    -------
    QuadraticProgram
    """
    n = len(mu)
    qp = QuadraticProgram("portfolio_selection")

    # Binary decision variables x_i
    for i in range(n):
        qp.binary_var(f"x_{i}")

    # Base quadratic/linear coefficients for risk and return
    Q = risk_aversion * sigma
    c = -(1.0 - risk_aversion) * mu

    # objective: x^T Q x + c^T x
    quadratic = {(i, j): float(Q[i, j]) for i in range(n) for j in range(n) if Q[i, j] != 0}
    linear = {i: float(c[i]) for i in range(n) if c[i] != 0.0}

    qp.minimize(quadratic=quadratic, linear=linear)

    # Budget constraint turned into penalty term: (sum x - budget)^2
    if budget is not None:
        # Ad
