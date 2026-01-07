# Penalized Spline Benchmark: mgcv vs flexsurv vs MultistateModels.jl

## Overview

This benchmark compares penalized spline fitting for multistate survival models across three packages:

1. **mgcv** (R) - Gold standard for GAMs with REML/GCV/NCV smoothing selection
2. **flexsurv** (R) - Flexible parametric survival models with splines
3. **MultistateModels.jl** (Julia) - Our implementation with PIJCV smoothing selection

## Model Structure

**Illness-Death Model:**
```
State 1 (Healthy) ──┬──→ State 2 (Illness) ──→ State 3 (Death)
                    └──────────────────────→ State 3 (Death)
```

**True Hazard Functions:**
- h₁₂(t) = 0.3√t  (increasing, concave - mimics aging)
- h₁₃(t) = 0.1 + 0.02t  (linear increase)
- h₂₃(t) = 0.4 exp(-0.1t)  (decreasing exponential - acute illness)

## Comparison Dimensions

1. **Accuracy**: RMSE of fitted hazards vs true hazards
2. **Smoothing parameters**: λ values (accounting for parameterization differences)
3. **Computation time**: Wall-clock time (excluding JIT compilation)
4. **Effective degrees of freedom**: tr(F) estimates

## Files

| File | Description |
|------|-------------|
| `generate_benchmark_data.jl` | Simulates data and fits with MultistateModels.jl |
| `fit_mgcv_flexsurv.R` | Fits models with mgcv and flexsurv |
| `visualize_comparison.jl` | Creates comparison plots |
| `benchmark_data.csv` | Generated illness-death data |
| `benchmark_metadata.json` | True hazards and evaluation grid |
| `julia_results.json` | MultistateModels.jl fit results |
| `r_results.json` | mgcv and flexsurv fit results |

## Running the Benchmark

### Step 1: Generate Data and Fit Julia Model

```bash
cd MultistateModelsTests/benchmarks/spline_comparison
julia --project=../../.. generate_benchmark_data.jl
```

### Step 2: Fit R Models

```bash
cd MultistateModelsTests/benchmarks/spline_comparison
Rscript fit_mgcv_flexsurv.R
```

### Step 3: Visualize Comparison

```bash
cd MultistateModelsTests/benchmarks/spline_comparison
julia --project=../../.. visualize_comparison.jl
```

## Implementation Details

### MultistateModels.jl
- **Spline basis**: Cubic B-splines (degree=3) with 4 interior knots
- **Penalty**: Second-order difference penalty on spline coefficients
- **Smoothing selection**: PIJCV (Predictive Infinitesimal Jackknife CV)
- **Algorithm**: mgcv-style performance iteration (alternate β and λ updates)

### mgcv
- **Model**: Piecewise exponential model (PAM) with Poisson response
- **Spline basis**: Cubic regression splines (bs="cr", k=8)
- **Smoothing selection**: NCV (Neighbourhood Cross-Validation, Wood 2024)

### flexsurv
- **Model**: Royston-Parmar flexible parametric survival model
- **Spline basis**: Natural cubic splines on log cumulative hazard
- **Smoothing**: Fixed knots (no automatic selection)

## Key Differences to Note

1. **mgcv vs MultistateModels.jl**: mgcv uses PAM approximation (discretized time), 
   while MultistateModels.jl uses exact continuous-time likelihood.

2. **Smoothing parameter scaling**: mgcv's `sp` and our `λ` may differ by a constant 
   factor due to different penalty matrix normalization conventions.

3. **flexsurv limitations**: flexsurv doesn't have automatic smoothing parameter 
   selection, so we use fixed knots for a fair comparison.

## References

- Wood, S.N. (2024). "Neighbourhood Cross-Validation" arXiv:2404.16490
- Wood, S.N. (2017). "Generalized Additive Models: An Introduction with R" (2nd ed.)
- Royston, P. & Parmar, M.K.B. (2002). "Flexible parametric proportional-hazards 
  and proportional-odds models for censored survival data"
- Jackson, C. (2016). "flexsurv: A Platform for Parametric Survival Modeling in R"
