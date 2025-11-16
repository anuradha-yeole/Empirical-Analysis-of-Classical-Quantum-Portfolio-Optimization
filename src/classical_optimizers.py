"""
classical_optimizers.py

Classical portfolio optimizers:
- Mean-Variance Optimization (MVO) via convex programming
- Genetic Algorithm (GA)
- Simulated Annealing (SA)

All optimizers share the same objective:
    J(w) = risk_aversion * w^T Σ w - (1 - risk_aversion) * w^T μ

Weights are long-only and lie on the simplex: w_i >= 0, sum w_i = 1.
"""

from __future__ import annotations

from typing import Tuple, Callable

import numpy as np
import cvxpy as cp

from .risk_metrics import (
    PortfolioResult,
    portfolio_return,
    portfolio_variance,
)


ObjectiveFn = Callable[[np.ndarray], float]


def _objective(
    w: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_aversion: float,
) -> float:
    var = portfolio_variance(w, sigma)
    ret = portfolio_return(w, mu)
    return float(risk_aversion * var - (1.0 - risk_aversion) * ret)


# ---------------------------
# Mean-Variance Optimization
# ---------------------------

def solve_mvo(
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_aversion: float = 0.5,
) -> PortfolioResult:
    """
    Basic convex Markowitz optimization.

    Minimize:  risk_aversion * w^T Σ w - (1-risk_aversion) * μ^T w
    s.t.      w_i >= 0, sum w_i = 1
    """
    n = len(mu)
    w = cp.Variable(n)

    risk = cp.quad_form(w, sigma)
    ret = mu @ w

    objective = cp.Minimize(risk_aversion * risk - (1 - risk_aversion) * ret)
    constraints = [cp.sum(w) == 1, w >= 0]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    weights = w.value
    weights = np.maximum(weights, 0)
    weights /= weights.sum()

    exp_ret = portfolio_return(weights, mu)
    var = portfolio_variance(weights, sigma)
    obj = _objective(weights, mu, sigma, risk_aversion)

    return PortfolioResult(
        name="MVO",
        weights=weights,
        expected_return=exp_ret,
        risk=var,
        objective=obj,
        extra={"status": prob.status},
    )


# ---------------------------
# Genetic Algorithm
# ---------------------------

def _random_simplex(n: int) -> np.ndarray:
    """Sample random point on probability simplex (Dirichlet)."""
    x = np.random.gamma(1.0, 1.0, size=n)
    return x / x.sum()


def solve_ga(
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_aversion: float = 0.5,
    population_size: int = 80,
    n_generations: int = 200,
    crossover_rate: float = 0.7,
    mutation_rate: float = 0.2,
) -> PortfolioResult:
    """
    Very lightweight GA for portfolio weights on the simplex.
    """

    n = len(mu)
    obj = lambda w: _objective(w, mu, sigma, risk_aversion)

    # Initialize population
    population = np.array([_random_simplex(n) for _ in range(population_size)])

    def fitness(w: np.ndarray) -> float:
        # GA maximizes fitness; we minimize objective
        return -obj(w)

    for _ in range(n_generations):
        # Evaluate
        fit_vals = np.array([fitness(ind) for ind in population])

        # Selection (tournament)
        new_pop = []
        for _ in range(population_size):
            i, j = np.random.randint(0, population_size, size=2)
            winner = population[i] if fit_vals[i] > fit_vals[j] else population[j]
            new_pop.append(winner.copy())
        population = np.array(new_pop)

        # Crossover
        for i in range(0, population_size, 2):
            if np.random.rand() < crossover_rate and i + 1 < population_size:
                alpha = np.random.rand()
                child1 = alpha * population[i] + (1 - alpha) * population[i + 1]
                child2 = alpha * population[i + 1] + (1 - alpha) * population[i]
                population[i], population[i + 1] = child1, child2

        # Mutation
        for i in range(population_size):
            if np.random.rand() < mutation_rate:
                noise = np.random.normal(0, 0.05, size=n)
                mutant = population[i] + noise
                mutant = np.maximum(mutant, 0)
                if mutant.sum() == 0:
                    mutant = _random_simplex(n)
                else:
                    mutant /= mutant.sum()
                population[i] = mutant

    # Choose best
    best_idx = np.argmax([fitness(ind) for ind in population])
    best = population[best_idx]

    exp_ret = portfolio_return(best, mu)
    var = portfolio_variance(best, sigma)
    obj_val = obj(best)

    return PortfolioResult(
        name="Genetic Algorithm",
        weights=best,
        expected_return=exp_ret,
        risk=var,
        objective=obj_val,
        extra={
            "population_size": population_size,
            "n_generations": n_generations,
        },
    )


# ---------------------------
# Simulated Annealing
# ---------------------------

def solve_simulated_annealing(
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_aversion: float = 0.5,
    n_steps: int = 5000,
    initial_temperature: float = 1.0,
    cooling_rate: float = 0.999,
) -> PortfolioResult:
    """
    Simple simulated annealing search on the simplex.
    """
    n = len(mu)
    obj = lambda w: _objective(w, mu, sigma, risk_aversion)

    current = _random_simplex(n)
    current_obj = obj(current)
    best = current.copy()
    best_obj = current_obj

    T = initial_temperature

    for _ in range(n_steps):
        # Propose a new point around current
        proposal = current + np.random.normal(0, 0.05, size=n)
        proposal = np.maximum(proposal, 0)
        if proposal.sum() == 0:
            proposal = _random_simplex(n)
        else:
            proposal /= proposal.sum()

        proposal_obj = obj(proposal)
        delta = proposal_obj - current_obj

        if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-6)):
            current, current_obj = proposal, proposal_obj

        if proposal_obj < best_obj:
            best, best_obj = proposal, proposal_obj

        T *= cooling_rate

    exp_ret = portfolio_return(best, mu)
    var = portfolio_variance(best, sigma)

    return PortfolioResult(
        name="Simulated Annealing",
        weights=best,
        expected_return=exp_ret,
