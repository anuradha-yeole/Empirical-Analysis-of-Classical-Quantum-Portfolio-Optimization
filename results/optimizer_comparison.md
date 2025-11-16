# Optimizer Comparison

This file compares the **quantum and classical optimization algorithms** investigated in the study.  
Source: Research paper on Quantum & Classical Optimization :contentReference[oaicite:3]{index=3}

## Algorithms Evaluated
### Quantum
- **Exact Minimum Eigen Solver (MES)**
- **Variational Quantum Eigensolver (VQE)**
- **Quantum Approximate Optimization Algorithm (QAOA)**

### Classical
- **Mean–Variance Optimization (MVO)**
- **Genetic Algorithm (GA)**
- **Simulated Annealing (SA)**

---

## Summary of Behavior

### 🔷 Quantum Optimizers
| Algorithm | Behavior | Notes |
|----------|----------|-------|
| **QAOA** | Best overall performance | Lowest MSE (0.005), excellent risk-return |
| **Exact MES** | Ground-truth optimal | Slightly higher MSE than QAOA but deterministic |
| **VQE** | Good accuracy | Performs well in moderate-noise covariance regimes |

Quantum advantage emerges due to **parallel solution-space exploration** and **energy minimization** approaches inherent in quantum mechanics.

---

### 🔶 Classical Optimizers
| Algorithm | Behavior | Notes |
|----------|----------|-------|
| **GA** | Strongest classical model | MSE = 0.0105 (still ~2x worse than QAOA) |
| **SA** | Moderate | Suffers from local minima |
| **MVO** | Worst performance | Highly sensitive to noise, MSE = 0.946 |

---

## Conclusion
Quantum algorithms—especially QAOA—demonstrate **superior precision, robustness**, and **risk-return tradeoff** compared to classical methods.
