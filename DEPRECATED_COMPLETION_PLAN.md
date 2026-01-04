# MultistateModelsTests Completion Plan

## Current Phase: 5 (Run Full Test Suite)

## Status Summary

| Component | Status | Notes |
|-----------|--------|-------|
| TVC bug fix | ✅ Complete | `_prepare_simulation_data()` now preserves TVC structure |
| TVC tests enabled | ✅ Complete | `TVC_TESTS_BROKEN = false` |
| Orphan cache cleanup | ✅ Complete | 6 files removed |
| Cache function generalized | ✅ Complete | `capture_longtest_result!()` now accepts `n_states` and `transitions` |
| Phase-type cache integration | ✅ Complete | Added `capture_simple_longtest_result!()` to pt tests |
| Spline cache integration | ✅ Complete | Added to `longtest_mcem_splines.jl` (covariate test only) |
| MCEM TVC cache integration | ✅ Complete | Added to `longtest_mcem_tvc.jl` (10 tests) |

## Completed Work

### Phase 1: Critical Infrastructure Fixes ✅

#### 1.1 TVC Simulation Bug Fix (src/simulation/simulate.jl)

**Root Cause:** `_prepare_simulation_data()` called `_collapse_to_single_interval()` 
which took only the first row's covariates, discarding TVC structure.

**Solution:**
- Added `_has_tvc_structure(data)` (lines 305-333): Detects multiple rows per subject with differing covariates
- Added `_extend_tvc_to_tmax(data, tmax)` (lines 335-354): Preserves TVC intervals
- Modified `_prepare_simulation_data()` to route appropriately

**Verification:** End-to-end test with 500 subjects, max relative error 10.7% < 15% threshold

#### 1.2 Orphan Cache Files Cleaned

Removed 6 files with incorrect naming convention:
- `exact_markov_{exp,gom,wei}_nocov.json`
- `mcem_{exp,gom,wei}_panel_nocov.json`

### Phase 2: Enable TVC Tests ✅

#### 2.1 TVC Tests Enabled (longtests/longtest_parametric_suite.jl)

Changed `TVC_TESTS_BROKEN = false` (was `true`)

### Phase 2.5: Generalize Cache Infrastructure ✅

#### 2.5.1 Fix Hardcoded State/Transition Assumptions

**Problem:** `capture_longtest_result!()` in `longtest_helpers.jl` had:
- `n_states = 3` hardcoded
- `for s in 1:3` for prevalence loop
- `[(1, 2), (2, 3)]` hardcoded for cumulative incidence

**Solution:** Added parameters with backward-compatible defaults:
```julia
function capture_longtest_result!(
    ...;
    n_states::Int=3,
    transitions::Vector{Tuple{Int,Int}}=[(1, 2), (2, 3)]
)
```

**Files Changed:** `longtests/longtest_helpers.jl` (lines 714-716, 725, 822, 834)

#### 2.5.2 Add Simple Cache Function for Phase-Type Tests

**Problem:** Phase-type models have different state space structure (expanded phases)
and don't fit the standard 3-state progressive model assumed by `capture_longtest_result!()`.

**Solution:** Added `capture_simple_longtest_result!()` function that captures:
- Parameter recovery metrics only (no prevalence/CI simulation validation)
- Suitable for phase-type and other specialized tests

**Files Changed:** `longtests/longtest_helpers.jl` (lines ~770-820)

### Phase 3: Phase-Type Cache Integration ✅

#### 3.1 longtest_phasetype_exact.jl

Added cache capture to 3 tests:
- `pt_exact_nocov` (2-Phase Illness-Death)
- `pt_exact_fixed` (2-Phase with Fixed Covariate)
- `pt_exact_tvc` (2-Phase with TVC)

**Files Changed:** `longtests/longtest_phasetype_exact.jl`
- Added includes for `longtest_config.jl` and `longtest_helpers.jl`
- Added `capture_simple_longtest_result!()` calls after each parameter recovery test

#### 3.2 longtest_phasetype_panel.jl

Added cache capture to 6 tests:
- `pt_panel_simple` (Simple 2-State Model)
- `pt_panel_id` (Illness-Death Model)
- `pt_mixed_simple` (Mixed Exact + Panel)
- `pt_mixed_structured` (Structured Mixed Observation)
- `pt_panel_fixed` (Fixed Covariate - Panel)
- `pt_panel_tvc` (TVC - Panel)

**Files Changed:** `longtests/longtest_phasetype_panel.jl`
- Added includes for `longtest_config.jl` and `longtest_helpers.jl`
- Added `capture_simple_longtest_result!()` calls after each parameter recovery test

### Phase 4: Spline & MCEM TVC Cache Integration ✅

#### 4.1 longtest_mcem_tvc.jl

Added cache capture to 10 tests:
- `mcem_exp_tvc` (Markov Panel Exponential + TVC)
- `mcem_exp_binary_tvc` (Binary TVC)
- `mcem_exp_cont_tvc` (Continuous TVC)
- `mcem_wei_tvc` (Weibull + TVC)
- `mcem_gom_tvc` (Gompertz + TVC)
- `mcem_prog_tvc` (Progressive with TVC)
- `mcem_multi_tvc` (Multiple TVC Change Points)
- `mcem_aft_tvc` (AFT Effect with TVC)
- `mcem_wei_tvc_pt` (Weibull + TVC - PhaseType)
- `mcem_gom_tvc_pt` (Gompertz + TVC - PhaseType)

**Files Changed:** `longtests/longtest_mcem_tvc.jl`
- Added includes for `longtest_config.jl` and `longtest_helpers.jl`
- Added `capture_simple_longtest_result!()` calls

#### 4.2 longtest_mcem_splines.jl

Added cache capture to 1 test:
- `sp_mcem_fixed` (Spline with Covariates)

**Files Changed:** `longtests/longtest_mcem_splines.jl`
- Added includes for `longtest_config.jl` and `longtest_helpers.jl`
- Added `capture_simple_longtest_result!()` call

## Remaining Work

### Phase 5: Run Full Test Suite

```bash
cd MultistateModelsTests
julia --project=. -e 'using MultistateModelsTests; MultistateModelsTests.run_longtests()'
```

### Phase 6: Render Reports

```bash
cd MultistateModelsTests/reports
quarto render
```

---

## Current Cache State

After running full test suite, expected files:

**Parametric (existing):**
```
exp_{exact,panel}_{nocov,fixed,tvc}.json
wei_{exact,mcem}_{nocov,fixed,tvc}.json
gom_{exact,mcem}_{nocov,fixed,tvc}.json
```

**Phase-Type (new):**
```
pt_exact_{nocov,fixed,tvc}.json
pt_panel_{simple,id,fixed,tvc}.json
pt_mixed_{simple,structured}.json
```

**MCEM TVC (new):**
```
mcem_{exp,wei,gom}_tvc.json
mcem_exp_{binary,cont}_tvc.json
mcem_{prog,multi,aft}_tvc.json
mcem_{wei,gom}_tvc_pt.json
```

**Spline (new):**
```
sp_mcem_fixed.json
```

---

## Original Plan Details
- `cache/longtest_results/exact_markov_exp_nocov.json`
- `cache/longtest_results/exact_markov_gom_nocov.json`
- `cache/longtest_results/exact_markov_wei_nocov.json`
- `cache/longtest_results/mcem_exp_panel_nocov.json`
- `cache/longtest_results/mcem_gom_panel_nocov.json`
- `cache/longtest_results/mcem_wei_panel_nocov.json`

---

## Phase 2: Complete Parametric Test Suite

### Task 2.1: ✅ DONE - Enable TVC Tests

**Status:** COMPLETED  
**File:** `longtests/longtest_parametric_suite.jl`

**Change Applied:** Line ~450
```julia
const TVC_TESTS_BROKEN = false  # Was: true
```

**Acceptance Criteria:**
- [ ] All 6 TVC tests run without skipping
- [ ] All 6 TVC tests pass
- [ ] Cache files created: `*_tvc.json` (6 files)

---

## Phase 3: Integrate Phase-Type Tests with Cache

### Task 3.1: ⏸️ DEFERRED - Phase-Type Cache Integration

**Status:** DEFERRED  
**Reason:** The `capture_longtest_result!()` function is tightly coupled to 3-state 
illness-death model structure (hardcoded `n_states=3`, transitions 1→2 and 2→3 for
cumulative incidence). Phase-type models have expanded state spaces (e.g., 5 states
for 2-phase illness-death).

**Options for future:**
1. Generalize `capture_longtest_result!()` to handle arbitrary state spaces
2. Create simpler `save_phasetype_result!()` for parameter recovery only
3. Skip cache integration - phase-type tests already validate correctness via @test

**Current State:**
- `longtest_phasetype_exact.jl` works and passes tests
- `longtest_phasetype_panel.jl` works and passes tests  
- Tests verify parameter recovery via local `check_parameter_recovery()` function

---

## Phase 4: Integrate Spline Tests with Cache

### Task 4.1: ⏸️ DEFERRED - Spline Cache Integration

**Status:** DEFERRED  
**Reason:** Spline tests verify hazard curve shape recovery, not traditional parameter
values. Would need custom cache format and visualization.

---

## Phase 5: Run Tests and Populate Cache

### Task 5.1: Execute Parametric Test Suite

The parametric suite (exp, wei, gom) has cache integration. Run it to populate cache:

```bash
cd MultistateModelsTests
julia --project=. -e '
    using Test, MultistateModels
    include("longtests/longtest_helpers.jl")
    include("longtests/longtest_parametric_suite.jl")
'
```

**Acceptance Criteria:**
- [ ] 18 JSON files in `cache/longtest_results/`
- [ ] Files: `{exp,wei,gom}_{exact,panel}_{nocov,fixed,tvc}.json`

---

### Task 5.2: Run Benchmarks

```bash
cd MultistateModelsTests
julia --project=. benchmarks/run_benchmarks.jl
```

---

## Phase 6: Render and Validate Reports

### Task 6.1: Render Full Report Suite

```bash
cd MultistateModelsTests/reports
quarto render
```

**Acceptance Criteria:**
- [ ] All reports render without error
- [ ] Parametric test results display correctly

---

## Revised Test Coverage Target

| Family | Exact | Panel/MCEM | Cache | Notes |
|--------|-------|------------|-------|-------|
| exp | ✅ 3 tests | ✅ 3 tests | ✅ Yes | Full coverage |
| wei | ✅ 3 tests | ✅ 3 tests | ✅ Yes | Full coverage |
| gom | ✅ 3 tests | ✅ 3 tests | ✅ Yes | Full coverage |
| pt | ✅ 4 tests | ✅ tests | ❌ Deferred | Tests work, no cache |
| sp | ✅ tests | ✅ tests | ❌ Deferred | Tests work, no cache |

**Parametric coverage:** 18 tests with cache integration  
**Phase-type/Spline:** Tests work but cache deferred

---

## Implementation Progress

### Phase 1
- [x] Task 1.1: Fix TVC bug in `_prepare_simulation_data()`
- [ ] Task 1.2: Clean orphan cache files

### Phase 2
- [x] Task 2.1: Enable TVC tests (TVC_TESTS_BROKEN = false)

### Phase 3
- [~] Task 3.1-3.2: Phase-type cache integration (DEFERRED)

### Phase 4
- [~] Task 4.1: Spline cache integration (DEFERRED)

### Phase 5
- [ ] Task 5.1: Run parametric test suite (18 tests)
- [ ] Task 5.2: Run benchmarks

### Phase 6
- [ ] Task 6.1: Render reports
- [ ] Task 6.2: Visual validation
