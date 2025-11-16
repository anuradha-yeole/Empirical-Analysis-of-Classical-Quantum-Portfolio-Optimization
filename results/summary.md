# Results Summary: Quantum vs Classical Portfolio Optimization
Based on the research study "Empirical Analysis of Classical and Quantum Algorithms for Portfolio Optimization" :contentReference[oaicite:1]{index=1}

This folder contains the summarized outputs of applying six optimization methods to 3-year NSE data across 20 equities.

Algorithms evaluated:
- **Quantum:** Exact Minimum Eigen Solver (Exact MES), QAOA, VQE  
- **Classical:** Mean–Variance Optimization (MVO), Genetic Algorithm (GA), Simulated Annealing (SA)

## Key Findings
- **QAOA achieved the best accuracy** among all algorithms  
- Quantum methods (QAOA, Exact MES, VQE) **significantly outperformed classical algorithms**  
- Exact MES and QAOA showed **MSE = 0.006 and 0.005 respectively**  
- Best classical baseline (GA) had **MSE = 0.0105**, nearly double that of QAOA  
- MVO and SA showed weak performance due to noisy covariance structure

## Ranked Performance (Best → Worst)
1. **QAOA (MSE 0.005)**
2. **Exact MES (0.006)**
3. **VQE (0.007)**
4. **Genetic Algorithm (0.0105)**
5. **Simulated Annealing (0.65)**
6. **MVO (0.946)**

These results highlight **quantum advantage** in accuracy and solution quality for portfolio optimization.

Refer to:
- `optimizer_comparison.md` (in-depth algorithm performance)
- `error_metrics.md` (numerical metrics)
- `portfolio_weights.md` (optimal weights per algorithm)
