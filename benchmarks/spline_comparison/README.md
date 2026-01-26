# Penalized Spline Benchmark: mgcv vs flexsurv vs MultistateModels.jl

**Last Run**: 2026-01-26  
**Julia Version**: 1.11  
**MultistateModels.jl Branch**: `penalized_splines`

## Overview

This benchmark compares penalized spline fitting for multistate survival models across three packages:

1. **mgcv** (R) - Gold standard for GAMs with REML/GCV/NCV smoothing selection
2. **flexsurv** (R) - Flexible parametric survival models with splines
3. **MultistateModels.jl** (Julia) - Our implementation with PIJCV smoothing selection

## Latest Results

### RMSE vs True Hazard (lower is better)

| Method | h12 | h13 | h23 |
|--------|-----|-----|-----|
| **Julia (PIJCV)** | **0.130** | 0.088 | **0.355** |
| mgcv (NCV) | 0.305 | **0.049** | 0.699 |
| flexsurv | 0.296 | **0.044** | 0.759 |

### Computation Time

| Method | Time (seconds) |
|--------|---------------|
| Julia | 28.57 |
| mgcv | 0.22 |
| flexsurv | 0.17 |

*Note: Julia time includes JIT compilation. After warmup, fits are ~2-5 seconds.*

### Smoothing Parameters

| Method | λ_h12 | λ_h13 | λ_h23 |
|--------|-------|-------|-------|
| Julia | 0.239 | 38.4 | 0.304 |
| mgcv (sp) | 1.5M | 3.5M | 6005 |

*Note: mgcv uses a different parameterization with much larger penalty scale.*

### Key Findings

1. **MultistateModels.jl achieves significantly better accuracy** for h12 (2.3x) and h23 (2x) compared to mgcv/flexsurv
2. Julia uses continuous-time exact likelihood vs PAM discretization in mgcv
3. Per-hazard λ selection: λ_h13 ≈ 38 (more smoothing for sparse h13 transition)
4. EDF ≈ 14.7 (Julia) vs EDF ≈ 6.2 (mgcv) - Julia allows more flexible fits where data supports it

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
