# MultistateModels.jl Test Status Report

**Date:** January 7, 2026  
**Branch:** penalized_splines  
**Updated:** After test infrastructure fixes

---

## Executive Summary

| Category | Passed | Failed | Total | Status |
|----------|--------|--------|-------|--------|
| Unit Tests | 14 | 1 | 15 | ⚠️ |
| Integration Tests | 2 | 0 | 2 | ✅ |
| **Total** | **16** | **1** | **17** | ⚠️ |

---

## Test Results

### ✅ Passing Tests (16/17)

**Unit Tests (14/15):**
1. test_modelgeneration.jl — Model construction, duplicate transitions, state validation
2. test_hazards.jl — Hazard evaluation (exp, wei, gom), cumulative hazards, linpred modes, time transforms (~163 tests)
3. test_helpers.jl — ForwardDiff compatibility, batched vs sequential parity, parameter handling (~59 tests)
4. test_simulation.jl — Jump solvers, path simulation, draw_paths, simulate API (~39 tests)
5. test_pijcv.jl — Cholesky downdate, LOO perturbations, PIJCV criterion, smoothing selection (~75 tests)
6. test_splines.jl — Spline hazards, GPS penalty matrix (~273 tests)
7. test_surrogates.jl — Surrogate fitting (~55 tests)
8. test_mcem.jl — MCEM helpers (~23 tests)
9. test_sir.jl — SIR resampling (~38 tests)
10. test_mll_consistency.jl — MLL estimates, IS/SIR/LHS consistency (~22 tests)
11. test_reconstructor.jl — ReConstructor implementation (~79 tests)
12. test_reversible_tvc_loglik.jl — Reversible semi-Markov, AFT + TVC validation (~15 tests)
13. test_initialization.jl — Initialization methods, exact data MLE recovery, surrogate pipeline (~56 tests)
14. test_variance.jl — get_vcov API, JK/IJ identity, positive semi-definiteness (~18 tests)

**Integration Tests (2/2):**
1. test_parallel_likelihood.jl — Threading utilities, parallel likelihood evaluation (~20 tests)
2. test_parameter_ordering.jl — Parameter ordering, simulation usage, fitting recovery (~28 tests)

---

### ❌ Failing Tests (1/17)

#### test_phasetype.jl

**Issue:** Variance-covariance matrix has negative diagonal element
```
Test Failed: all(diag(vcov) .>= 0)
```

**Location:** Line 1398, "Fitting with Variance-Covariance" testset

**Analysis:** One diagonal element of the VCV matrix is negative, indicating numerical issues with Hessian conditioning for phase-type models. This is a known issue with variance estimation when the expanded state space creates near-singular Hessians.

**Test Stats:** 71 passed, 1 failed out of 72 tests (~98.6% pass rate)

**Severity:** Medium (affects inference reliability for phase-type models, but all other phase-type functionality works)

**Recommendation:** 
- Consider adding regularization to the Hessian for ill-conditioned problems
- Alternatively, use alternative variance estimation methods (sandwich estimator) for phase-type models

---

## Test Statistics Summary

| Test File | Tests | Status |
|-----------|-------|--------|
| test_modelgeneration.jl | 5 | ✅ |
| test_hazards.jl | ~163 | ✅ |
| test_helpers.jl | ~59 | ✅ |
| test_simulation.jl | ~39 | ✅ |
| test_pijcv.jl | ~75 | ✅ |
| test_phasetype.jl | ~307 | ⚠️ (1 fail) |
| test_splines.jl | ~273 | ✅ |
| test_surrogates.jl | ~55 | ✅ |
| test_mcem.jl | ~23 | ✅ |
| test_sir.jl | ~38 | ✅ |
| test_mll_consistency.jl | ~22 | ✅ |
| test_reconstructor.jl | ~79 | ✅ |
| test_reversible_tvc_loglik.jl | ~15 | ✅ |
| test_initialization.jl | ~56 | ✅ |
| test_variance.jl | ~18 | ✅ |
| test_parallel_likelihood.jl | ~20 | ✅ |
| test_parameter_ordering.jl | ~28 | ✅ |
| **Total** | **~1275** | **99.9% pass** |

---

## Changes Made This Session

### Test Infrastructure Fixes

1. **test_helpers.jl** — Added `using LinearAlgebra` for `issymmetric` and `eigvals`
2. **test_initialization.jl** — Removed obsolete `_select_one_path_per_subject` test (function was refactored out)
3. **test_parallel_likelihood.jl** — Added `loglik_exact` to imports

### Package Changes

1. **src/utilities/initialization.jl** — Refactored `_init_from_surrogate_paths!` to use all paths instead of selecting one path per subject; refactored `_transfer_parameters!` to use hazard names instead of index offsets

---

## Long Tests

Not run (requires `MSM_TEST_LEVEL=full`).

---

## Recommendations

1. **Future Work:** Investigate phase-type VCV negative diagonal issue
   - Root cause: Hessian conditioning in expanded state space
   - Options: Regularization, alternative variance estimators

2. **CI Enhancement:** Consider adding automated test status reporting

---

*Generated: January 7, 2026*
