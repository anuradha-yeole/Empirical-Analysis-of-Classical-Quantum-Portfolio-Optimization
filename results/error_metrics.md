# Error Metrics Comparison

This file presents numerical accuracy metrics (MSE, RMSE, MAE) from all algorithms tested in the study.  
Reference: Table I of the paper :contentReference[oaicite:4]{index=4}

## Summary Table

| Algorithm      | MSE     | RMSE    | MAE    |
|----------------|---------|---------|--------|
| Exact MES      | 0.006   | 0.245   | 0.196  |
| Sampling VQE   | 0.007   | 0.264   | 0.211  |
| QAOA           | 0.005   | 0.224   | 0.179  |
| MVO            | 0.946   | 0.973   | 0.778  |
| GA             | 0.0105  | 0.1025  | 0.082  |
| SA             | 0.65    | 0.806   | 0.645  |

## Observations
- QAOA achieves **best overall accuracy** (lowest MSE).
- Exact MES closely matches QAOA due to its deterministic nature.
- VQE performs well but is slightly less precise.
- Classical methods show **higher error**, especially MVO and SA.
