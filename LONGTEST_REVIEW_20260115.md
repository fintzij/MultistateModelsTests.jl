# Adversarial Statistical Review: Long Test Suite
**Date**: 2026-01-15  
**Reviewer**: julia-statistician agent  
**Branch**: penalized_splines  
**Status**: Issues identified, remediation required

---

## Executive Summary

A systematic adversarial review of all 21 longtest files in `MultistateModelsTests/longtests/` identified **2 critical issues**, **6 warnings**, and **3 minor issues**. The most significant finding is a **selection bias pattern** in panel data construction functions that excludes subjects who reach the absorbing state before the first observation time, potentially causing tests to pass even when estimators have systematic bias.

---

## Critical Issues

### CRITICAL-1: Selection Bias in Panel Data Creation Functions

**Severity**: 🔴 CRITICAL  
**Files Affected**:
- `longtest_helpers.jl` (lines 186, 240, 294)
- `longtest_aft_suite.jl` (lines 428-429)

**Description**:  
The `create_panel_data()`, `create_panel_data_with_covariate()`, `create_panel_data_with_tvc()`, and `make_panel_data()` functions contain logic that excludes observation intervals where subjects have already transitioned to the absorbing state:

```julia
# Only include if not already absorbed at start
if state_start < n_states
```

**Statistical Impact**:
1. Subjects who transition to the absorbing state before the first panel observation time are **completely dropped** from the dataset
2. This creates **informative censoring** — missingness depends on the outcome being measured
3. Bias direction: Excludes fast progressors → **underestimates hazard rates**
4. The subsequent ID re-indexing (`id_map`) makes the number of dropped subjects invisible

**Evidence Required**:  
Verify whether this is:
- (A) Correct behavior: Panel data should only include intervals where subjects are at risk
- (B) A bug: All subjects should contribute data, with appropriate likelihood handling

**Current Assessment**:  
Option (A) is likely correct for standard survival analysis (subjects only contribute while at risk). However:
- Tests do not document this assumption
- Tests do not verify the expected vs actual sample size
- A test could pass with biased estimation if the bias magnitude happens to fall within tolerances

---

### CRITICAL-2: Survivorship Bias in collect_event_durations()

**Severity**: 🔴 CRITICAL  
**File**: `longtest_simulation_distribution.jl` (lines 278-292)

**Description**:  
The function only collects durations from subjects who experienced an event:

```julia
if path.states[end] != path.states[1]  # Only include if transitioned
    collected += 1
    durations[collected] = path.times[end] - path.times[1]
end
```

**Mitigation Present**:  
The code correctly compares against a `Truncated()` distribution, which accounts for this selection. However:
- If too many subjects are right-censored, the test errors out rather than documenting the proportion
- No diagnostic output shows what fraction of simulations resulted in events

---

## Warnings

### WARNING-1: Very Loose Parameter Recovery Tolerances

**Files Affected**: Multiple MCEM tests  
**Tolerances in Use**:
- `PARAM_REL_TOL = 0.35` (35% relative error) — longtest_parametric_suite.jl
- `BETA_ABS_TOL = 0.40` — longtest_parametric_suite.jl
- `MCEM_TVC_BETA_ABS_TOL = 0.65` — longtest_config.jl
- `PROPOSAL_COMPARISON_TOL = 0.55` (55%) — longtest_mcem.jl

**Risk**: A systematically biased estimator with 20-30% bias could pass these tests.

---

### WARNING-2: Variance Ratio Tolerance Too Loose

**File**: `longtest_variance_validation.jl`  
**Tolerance**: `VAR_RATIO_TOL = 0.5` (50%)

**Risk**: Under correct model specification, IJ variance should equal model-based variance (ratio = 1.0). A 50% tolerance could miss variance estimation bugs.

---

### WARNING-3: Inconsistent Sample Sizes Across Tests

**Observation**:
- Most tests: `N_SUBJECTS = 1000`
- longtest_mcem_tvc.jl: `N_SUBJECTS = 2000`
- longtest_robust_parametric.jl: `N_SUBJECTS = 10000-20000`

**Risk**: Inconsistent power across tests; harder to compare expected precision.

---

### WARNING-4: Boundary Time Exclusion in Spline Tests

**File**: `longtest_spline_exact.jl`  
**Code**:
```julia
# Note: Avoid very early times (< 1.0) where spline boundary effects cause larger errors
test_times = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0]
```

**Risk**: Explicitly avoids the region where bugs are most likely to manifest.

---

### WARNING-5: Loose Smooth Effect Recovery Tolerances

**File**: `longtest_smooth_covariate_recovery.jl`  
**Tolerances**:
- `MAX_ABS_ERROR = 0.85` (for amplitude ~1 function)
- `RMSE_TOLERANCE = 0.50`

**Risk**: Allows substantial function misspecification to pass.

---

### WARNING-6: Silent Parameter Padding

**File**: `longtest_phasetype_exact.jl` (lines 157-161)  
**Code**:
```julia
if length(true_rates) < n_hazards
    true_rates = vcat(true_rates, fill(0.25, n_hazards - length(true_rates)))
end
```

**Risk**: Dimension mismatches are silently fixed rather than causing test failures.

---

## Minor Issues

### MINOR-1: Re-indexing Hides Dropped Subjects

**Files**: `longtest_helpers.jl`, `longtest_aft_suite.jl`

After filtering, IDs are re-indexed to 1..N, making it impossible to determine how many subjects were dropped without additional logging.

---

### MINOR-2: Mixed Observation Type Pattern Undocumented

**File**: `longtest_parametric_suite.jl`

Uses `obstype_by_transition = Dict(1 => 2, 2 => 1)` (transition 1 panel, transition 2 exact) but this specific censoring pattern is not documented in the test description.

---

### MINOR-3: No "Smoke Test" for Test Sensitivity

**All Files**

No test deliberately introduces a biased estimator to verify that tests can actually fail. This would validate test sensitivity.

---

## Action Items

### ACTION-1: Add Sample Size Validation to Panel Data Functions
**Priority**: HIGH  
**Files**: `longtest_helpers.jl`, `longtest_aft_suite.jl`

**Task**: Add diagnostic output and optional assertion to verify expected vs actual sample sizes after panel data creation.

**Success Criteria**:
- [ ] Each `create_panel_data*` function returns or logs `n_subjects_simulated` and `n_subjects_retained`
- [ ] Warning is emitted if more than 5% of subjects are dropped
- [ ] Tests document the expected proportion of subjects contributing data

**Implementation**:
```julia
# At end of create_panel_data functions:
n_simulated = length(paths)
n_retained = length(unique(df.id))
n_dropped = n_simulated - n_retained
if n_dropped > 0
    @info "Panel data: $n_dropped/$n_simulated subjects dropped (reached absorbing state before first panel time)"
end
```

---

### ACTION-2: Document Censoring Assumptions in Test Docstrings
**Priority**: HIGH  
**Files**: All files using panel data

**Task**: Add explicit documentation of the censoring mechanism to each test's docstring.

**Success Criteria**:
- [ ] Each panel data test has a docstring section explaining the observation pattern
- [ ] The `state_start < n_states` filtering logic is explicitly mentioned
- [ ] Any known bias implications are documented

---

### ACTION-3: Add Event Rate Verification
**Priority**: MEDIUM  
**Files**: Tests generating simulated data

**Task**: Add checks that the observed event rate matches theoretical expectations.

**Success Criteria**:
- [ ] For exponential hazards: verify `n_events ≈ n_subjects * (1 - exp(-rate * max_time))`
- [ ] Allow 20% tolerance for MC variability
- [ ] Emit warning if event rate is unexpectedly low (suggests censoring issue)

**Implementation**:
```julia
function verify_event_rate(data::DataFrame, expected_rate::Float64, max_time::Float64; rtol=0.2)
    n_subjects = length(unique(data.id))
    n_events = count(data.statefrom .!= data.stateto)
    expected_events = n_subjects * (1 - exp(-expected_rate * max_time))
    actual_prop = n_events / n_subjects
    expected_prop = expected_events / n_subjects
    if !isapprox(actual_prop, expected_prop; rtol=rtol)
        @warn "Event rate mismatch" actual=actual_prop expected=expected_prop
    end
end
```

---

### ACTION-4: Tighten Variance Ratio Tolerance
**Priority**: MEDIUM  
**File**: `longtest_variance_validation.jl`

**Task**: Reduce `VAR_RATIO_TOL` from 0.50 to 0.25 or provide justification for current value.

**Success Criteria**:
- [ ] Either tolerance is reduced to ≤0.25
- [ ] OR documentation explains why 50% tolerance is statistically justified
- [ ] Test continues to pass with tightened tolerance

---

### ACTION-5: Replace Silent Padding with Assertion
**Priority**: MEDIUM  
**File**: `longtest_phasetype_exact.jl`

**Task**: Replace parameter padding with explicit assertion.

**Success Criteria**:
- [ ] If `length(true_rates) != n_hazards`, test fails with descriptive error
- [ ] No silent padding or truncation of parameter vectors

**Implementation**:
```julia
@assert length(true_rates) == n_hazards "Parameter count mismatch: got $(length(true_rates)), expected $n_hazards"
```

---

### ACTION-6: Add Boundary Time Tests for Splines
**Priority**: LOW  
**File**: `longtest_spline_exact.jl`

**Task**: Add separate test specifically for boundary behavior with relaxed tolerances.

**Success Criteria**:
- [ ] New test evaluates spline at t ∈ [0.1, 0.5, 1.0]
- [ ] Uses separate (documented) tolerance for boundary region
- [ ] Failure triggers warning, not test failure (informational)

---

### ACTION-7: Document Tolerance Rationale
**Priority**: LOW  
**Files**: `longtest_config.jl`, individual test files

**Task**: Add comments explaining why each tolerance value was chosen.

**Success Criteria**:
- [ ] Each tolerance constant has a comment explaining:
  - Why this value (not tighter or looser)
  - What statistical phenomenon it accounts for
  - Reference to any empirical calibration

---

### ACTION-8: Add Test Sensitivity Validation
**Priority**: LOW  
**Files**: New file `longtest_sensitivity_check.jl`

**Task**: Create a test that deliberately introduces bias to verify tests can fail.

**Success Criteria**:
- [ ] Test generates data from parameters A
- [ ] Fits with deliberately wrong starting values or constraints
- [ ] Verifies that parameter recovery tests correctly fail
- [ ] Documents expected failure modes

---

## Verification Checklist for Remediation

After fixes are implemented, verify:

1. [ ] `create_panel_data()` functions emit diagnostic info about dropped subjects
2. [ ] Running tests shows reasonable dropout rates (document expected values)
3. [ ] Event rates in simulated data match theoretical expectations
4. [ ] All tolerance constants have documented rationale
5. [ ] No silent padding/truncation of parameter vectors
6. [ ] Test docstrings describe censoring assumptions
7. [ ] Variance validation test passes with tighter tolerance OR justification added

---

## Files Requiring Modification

| File | Actions Required |
|------|-----------------|
| `longtest_helpers.jl` | ACTION-1, ACTION-2 |
| `longtest_aft_suite.jl` | ACTION-1, ACTION-2 |
| `longtest_config.jl` | ACTION-7 |
| `longtest_parametric_suite.jl` | ACTION-2, ACTION-3 |
| `longtest_variance_validation.jl` | ACTION-4 |
| `longtest_phasetype_exact.jl` | ACTION-5 |
| `longtest_spline_exact.jl` | ACTION-6 |
| `longtest_simulation_distribution.jl` | ACTION-2 |
| NEW: `longtest_sensitivity_check.jl` | ACTION-8 |

---

## Appendix: Test Files Reviewed

1. longtest_aft_suite.jl
2. longtest_config.jl
3. longtest_exact_markov.jl
4. longtest_helpers.jl
5. longtest_mcem.jl
6. longtest_mcem_splines.jl
7. longtest_mcem_tvc.jl
8. longtest_parametric_suite.jl
9. longtest_phasetype.jl
10. longtest_phasetype_exact.jl
11. longtest_phasetype_panel.jl
12. longtest_pijcv_loocv.jl
13. longtest_robust_markov_phasetype.jl
14. longtest_robust_parametric.jl
15. longtest_simulation_distribution.jl
16. longtest_simulation_tvc.jl
17. longtest_sir.jl
18. longtest_smooth_covariate_recovery.jl
19. longtest_spline_exact.jl
20. longtest_tensor_product_recovery.jl
21. longtest_variance_validation.jl
22. phasetype_longtest_helpers.jl

---

## Sign-off

This review identifies issues that should be addressed to ensure long tests provide robust validation of statistical correctness. The critical issues do not necessarily indicate bugs in the main package—they indicate that tests may not catch certain classes of bugs.

**Next Step**: Hand off to implementation agent to address ACTION-1 through ACTION-8.
