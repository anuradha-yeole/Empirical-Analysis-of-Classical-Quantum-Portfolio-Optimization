
# 🧮 Quantum vs Classical Portfolio Optimization

### *QAOA, VQE, Exact MES vs MVO, Genetic Algorithm, Simulated Annealing on Real NSE Data (2021–2024)*

*A comparative study of modern quantum algorithms and classical optimization workflows in quantitative finance.*

---

## 🌟 Overview

This repository implements a complete **quantum–classical benchmarking framework** for portfolio optimization, using **real NSE equity time-series data**.
We compare **Quantum Approximate Optimization Algorithm (QAOA)**, **Variational Quantum Eigensolver (VQE)**, and **Exact Minimum Eigensolver (MES)** against classical optimizers:

* **Mean-Variance Optimization (MVO)**
* **Genetic Algorithm (GA)**
* **Simulated Annealing (SA)**

The goal is to evaluate whether **near-term quantum algorithms (NISQ era)** can offer measurable advantages in high-volatility portfolio selection and risk-return tradeoffs.

---

## 🚀 Key Result

**QAOA achieved ~50% lower Mean Squared Error (MSE) compared to GA**
→ **0.005 vs 0.0105**, making it the best performer for noisy return forecasting.

This demonstrates the potential of quantum optimization for financial decision-making under uncertainty.

---

## 📊 Why Quantum Finance?

Portfolio selection can be framed as a **Quadratic Unconstrained Binary Optimization (QUBO)** problem:

[
\min_x ; \lambda , x^T \Sigma x - (1-\lambda) , \mu^T x
]

Mapping this to an **Ising Hamiltonian** allows quantum algorithms to search the solution landscape using energy minimization.

Quantum optimizers (QAOA, VQE) explore this space differently than classical gradient or meta-heuristic methods, which makes them attractive for:

* non-convex risk models
* discrete investment decisions
* noisy or highly correlated markets

---

## 🗂 Repository Structure

```
Quantum-Portfolio-Optimization/
│
├── notebooks/
│   └── Quantumvsclassical.ipynb        # main experimentation notebook
│
├── src/
│   ├── data_loader.py                  # yfinance ingestion, return computation
│   ├── classical_optimizers.py         # MVO, GA, Simulated Annealing
│   ├── quantum_optimizers.py           # QAOA, VQE, Exact MES (Qiskit)
│   ├── risk_metrics.py                 # sharpe, variance, backtesting
│   └── visualization.py                # Plotly-based charts
│
├── data/                               # downloaded price data (ignored in git)
├── results/                            # saved plots, metrics, experiments
│
├── run_experiments.py                  # one-command full pipeline runner
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

### **Python Libraries**

* **Qiskit** (quantum circuits & optimizers)
* **qiskit-optimization**
* **NumPy & Pandas**
* **cvxpy** (convex optimization)
* **Plotly** (risk-return charts & equity curves)
* **yfinance** (NSE data ingestion)

### **Classical Models**

* Markowitz Mean–Variance Optimization
* Genetic Algorithm (Dirichlet sampling + crossover/mutation)
* Simulated Annealing (stochastic search on simplex)

### **Quantum Models**

* QAOA with SPSA optimizer
* VQE with COBYLA + customizable ansatz
* Exact Minimum Eigensolver (reference optimum)

---

## ⚙️ How to Run the Entire Experiment

### **1️⃣ Install dependencies**

```bash
pip install -r requirements.txt
```

### **2️⃣ Run full pipeline**

This script will:

* download NSE data
* compute log-returns and covariance
* run classical + quantum optimizers
* backtest resulting portfolios
* print a risk/return summary
* show Plotly visualizations

```bash
python run_experiments.py
```

### **3️⃣ Output includes**

* risk vs return scatter plot
* equity curves for each optimizer
* Sharpe ratios & final portfolio values
* binary selection → normalized portfolio weights
* quantum energy-based allocations

---

## 📈 Sample Outputs

### **Risk vs Return (Classical vs Quantum)**

Plot shows variance on x-axis and expected return on y-axis.

### **Backtest Equity Curves**

Compares cumulative returns over 3 years for QAOA, VQE, MES, MVO, GA, and SA.

### **Optimizer Summary Table**

| Method    | Expected Return           | Risk                | Sharpe                | Notes                  |
| --------- | ------------------------- | ------------------- | --------------------- | ---------------------- |
| QAOA      | ⭐ Best                    | Low                 | Stable                | ~50% lower MSE than GA |
| VQE       | Good                      | Moderate            | Sensitive to noise    |                        |
| Exact MES | Optimal                   | N/A                 | Ground truth baseline |                        |
| GA        | Decent                    | High                | Unstable              |                        |
| SA        | Strong classical baseline | Lower variance      |                       |                        |
| MVO       | Simple                    | Overfits covariance |                       |                        |

---

## 🔍 How the QUBO is Built

Weights are modeled using binary variables:

* `x_i = 1` → include asset i
* `x_i = 0` → exclude asset i
* final weights = normalized selected assets

Budget constraint:

[
\((\sum_{i=1}^{n} x_i - B)^2\)
]

added as a penalty to enforce selection of exactly B assets.

---

## 🎯 Application Areas

This framework mirrors workflows used in hedge funds and quant firms:

* stochastic asset selection
* optimal hedging under uncertainty
* analyzing energy landscapes for portfolio construction
* validating quantum advantage using real market chaos

---

## 🧩 Future Extensions

* Implement **QAOA parameter sweeps**
* Explore **Tensor Network simulations**
* Add **CVaR-QAOA** for downside risk control
* Run VQE with **UCC-like or layered ansatz circuits**
* Use **larger universes (10–20 assets)** with truncation heuristics

---

## 🤝 Contributing

Feel free to open issues or submit PRs — especially for extending quantum backends or adding new classical heuristics.

---

## ⭐ If You Found This Useful

