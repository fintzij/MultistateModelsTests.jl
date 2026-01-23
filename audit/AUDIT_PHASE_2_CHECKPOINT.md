# Phase 2 Checkpoint: Module Coverage Mapping and Gap Analysis

**Date**: 2026-01-22
**Subagent**: B (Coverage Mapping)
**Branch**: penalized_splines

---

## Executive Summary

Analyzed 11 source directories containing ~37,000 lines of code across ~70 source files. Cross-referenced against Phase 1 test inventory (72 test files, 41,991 lines).

**Key Finding**: Likelihood module has excellent analytical test coverage via test_loglik_analytical.jl. Critical gaps exist in fitting dispatch (_fit_markov_panel has no unit tests), simulation edge cases, and error path validation.

---

## Module Coverage Table

| Module Path | Risk Tier | LOC | Key Functions | Direct Unit Tests | Coverage |
|-------------|-----------|-----|---------------|-------------------|----------|
| likelihood/ | CRITICAL | 2,466 | 6 files, ~15 functions | test_loglik_analytical.jl (40 tests) | HIGH |
| inference/ | CRITICAL | 8,573 | 10 files, ~40 functions | test_mcem.jl, test_sir.jl | MEDIUM |
| simulation/ | CRITICAL | 2,248 | 2 files, ~20 functions | test_simulation.jl (476 lines) | MEDIUM |
| utilities/parameters.jl | CRITICAL | 752 | ~15 functions | test_reconstructor.jl (partial) | LOW |
| output/variance.jl | HIGH | 2,871 | ~25 functions | test_variance.jl (533 lines) | MEDIUM |
| hazard/spline.jl | HIGH | 1,498 | ~20 functions | test_splines.jl (1504 lines) | HIGH |
| hazard/tpm.jl | HIGH | 365 | ~8 functions | test_phasetype*.jl (indirect) | MEDIUM |
| phasetype/ | HIGH | 3,421 | 8 files, ~35 functions | 6,174 lines across 7 files | HIGH |
| construction/ | MEDIUM | 1,736 | 5 files, ~25 functions | test_modelgeneration.jl (weak) | LOW |
| hazard/ (other) | MEDIUM | 2,976 | 8 files, ~30 functions | test_hazards.jl, test_compute_hazard.jl | MEDIUM |
| types/ | LOW | 2,482 | 7 files (type defs) | Implicit via integration | N/A |
| utilities/ (other) | MEDIUM | 4,212 | 14 files, ~50 functions | Mixed coverage | LOW |

---

## Risk Tier Definitions

| Tier | Description | Modules |
|------|-------------|---------|
| CRITICAL | Bugs cause silently wrong results | likelihood, inference/fit_*, simulation sampling, parameters |
| HIGH | Bugs cause detectable wrong results | variance, spline evaluation, phase-type TPM |
| MEDIUM | Bugs cause degraded functionality | construction, hazard evaluation, accessors |
| LOW | Bugs cause cosmetic/performance issues | types, misc utilities |

---

## Critical Gaps List

### 1. Fitting Dispatch: _fit_markov_panel (CRITICAL - NO UNIT TESTS)

**Location**: src/inference/fit_markov.jl

**Functions lacking direct tests**:
- _fit_markov_panel() - Main entry point for Markov panel fitting
- _compute_vcov_markov() - Variance computation for Markov models

**Missing edge cases**:
- Empty dataset (0 subjects)
- Single subject with single observation
- Panel data with all observations at same time
- Degenerate transition matrix (all staying in absorbing state)
- Parameters at box constraint boundaries
- Singular Hessian / non-invertible Fisher information

**Risk**: Silent incorrect variance estimates, optimization convergence issues undetected.

### 2. Fitting Dispatch: _fit_mcem Partial Coverage (CRITICAL)

**Location**: src/inference/fit_mcem.jl

**Functions with partial coverage**:
- _fit_mcem() - Main MCEM entry point
- DrawSamplePaths!() - Path sampling (tested indirectly via longtests)
- _compute_vcov_mcem() - MCEM variance computation

**Missing edge cases**:
- Importance weight collapse (all weights to 1 subject)
- Path sampling when all subjects in absorbing state
- Convergence assessment with flat likelihood
- Monte Carlo variance with small sample sizes (npaths < 10)

### 3. Simulation Module Edge Cases (CRITICAL)

**Location**: src/simulation/simulate.jl

**Functions with incomplete edge case coverage**:
- simulate_path() - Main path simulation
- _exponential_jump_time() - Closed-form jump time
- _find_jump_time() - Root-finding for jump times

**Missing edge cases**:
- Zero hazard rate (rate = 0)
- Infinite hazard rate (rate approaches infinity)
- Very short observation intervals (dt near 1e-15)
- Very long observation intervals (dt near 1e10)
- Covariate values at extreme boundaries
- Simulation from absorbing states (should error informatively)
- Time-varying covariates with high-frequency changes

### 4. Parameter Transformation Functions (CRITICAL)

**Location**: src/utilities/parameters.jl

**Functions lacking direct unit tests**:
- unflatten_parameters() - Critical for AD compatibility
- update_pars_cache!() - In-place parameter update
- rebuild_parameters() - Parameter reconstruction
- get_parameters_natural() - Natural scale extraction

**Missing edge cases**:
- Empty parameter vector
- Single parameter models
- Parameters containing NaN or Inf
- Dual number (ForwardDiff) handling edge cases
- Parameter vector length mismatch

### 5. Variance Estimation: compute_subject_hessians (HIGH)

**Location**: src/output/variance.jl

**Functions with incomplete coverage**:
- compute_subject_hessians() - AD for per-subject Hessians
- compute_subject_gradients() - AD for per-subject gradients
- _compute_vcov_tolerance() - Adaptive pseudo-inverse tolerance

**Missing edge cases**:
- Singular subject Hessians
- Subjects with zero-length observations
- Subjects with identical observations (no information)
- Very small eigenvalues in Fisher information
- Negative eigenvalues (non-PSD Hessian)

### 6. Spline Penalty Infrastructure (HIGH)

**Location**: src/hazard/spline.jl

**Functions with incomplete edge case coverage**:
- calibrate_splines() - Knot placement via CDF inversion
- rectify_coefs!() - Monotone spline coefficient cleanup
- build_penalty_matrix_gps() - GPS penalty construction

**Missing edge cases**:
- Zero penalty matrix (lambda = 0)
- Infinite penalty (lambda approaches infinity)
- Non-uniform knot spacing edge cases
- Spline basis with single knot
- Degree-0 splines (step functions)

### 7. Phase-Type TPM Computation (HIGH)

**Location**: src/phasetype/expansion_loglik.jl

**Functions with edge case gaps**:
- compute_tmat_batched!() - Schur-based TPM
- compute_tpm_from_schur() - Cached Schur TPM

**Missing edge cases**:
- dt = 0 (instantaneous transition)
- dt very small (near 1e-15)
- dt very large (near 1e10)
- Defective Q matrix (repeated eigenvalues)
- Q with zero off-diagonal rates

---

## Codebase Weaknesses Discovered

### W6 (HIGH): Missing Input Validation for Negative Times

**Location**: src/utilities/validation.jl lines 67-74

**Description**: The check_data! function validates that tstart <= tstop but does NOT validate that times are non-negative. Negative start times are silently accepted.

**Impact**: Silently wrong results when users accidentally pass negative times. Hazard functions may behave unexpectedly for t < 0.

**Recommendation**: Add explicit validation for negative times.

### W7 (MEDIUM): BoundsError Instead of Descriptive Error for Missing Columns

**Location**: Documented in test_error_messages.jl lines 44-57

**Description**: When required data columns are missing, user gets a cryptic BoundsError instead of a helpful message listing the missing columns.

**Impact**: Poor user experience, harder debugging.

### W8 (MEDIUM): No Validation of Covariate Values

**Location**: Multiple files in hazard/covariates.jl, likelihood/loglik_*.jl

**Description**: Covariate values are extracted and used without validation. NaN, Inf, or extreme values are silently propagated.

**Impact**: Silent NaN propagation leading to wrong likelihood values.

### W9 (MEDIUM): Deeply Nested Conditionals in fit_mcem.jl

**Location**: src/inference/fit_mcem.jl lines 400-600

**Description**: The _fit_mcem function has multiple levels of nested conditionals handling different proposal types, surrogate states, and convergence modes. This makes isolated unit testing difficult.

**Recommendation**: Extract nested logic into helper functions.

### W10 (LOW): Magic Numbers in Numerical Tolerances

**Location**: src/utilities/constants.jl

**Description**: Constants file exists but some tolerances are still hardcoded in source files.

**Examples found**:
- variance.jl line 238: atol=1e-14 hardcoded
- fit_mcem.jl line 783: 1e-8 buffer for clamping
- spline.jl line 95: 1e-14 for near-zero check

**Recommendation**: Move all numerical tolerances to constants.jl with descriptive names.

### W11 (LOW): Inconsistent Error Handling Between Modules

**Location**: Various

**Patterns observed**:
1. Some functions throw ArgumentError for invalid inputs
2. Some throw generic Exception
3. Some use @assert (which can be disabled)
4. Some use @warn and continue (silent degradation)

### W12 (LOW): Copy-Pasted Code in Variance Computation

**Location**: src/output/variance.jl lines 100-200

**Description**: compute_subject_gradients has 3 nearly identical method signatures with similar but slightly different implementations for ExactData, MPanelData, and SMPanelData. This could be DRYd with a single generic implementation.

---

## Handoff Statement

Phase 2 complete. Critical gaps identified in:
1. _fit_markov_panel (no unit tests)
2. Simulation edge cases (boundary values)
3. Parameter transformation edge cases
4. Input validation gaps (negative times, missing columns)

For Phase 3: Prioritize unit tests for _fit_markov_panel and input validation. The 40 analytical likelihood tests provide strong foundation - extend this pattern to fitting functions.

**Checkpoint**: 2026-01-22
