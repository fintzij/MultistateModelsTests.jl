# Long Test Infrastructure Overhaul Plan

## Goal
Implement a complete, systematic long test suite for MultistateModels.jl that validates parameter recovery across all supported model configurations. Update the long test reporting infrastructure to display results clearly.

## Implementation Status

| Task | Status | Notes |
|------|--------|-------|
| LongTestResults.jl infrastructure | ✅ Complete | Added hazard_family, data_type, covariate_type, passed, param_passed, data_summary fields |
| ReportHelpers.jl updates | ✅ Complete | Added plot_data_summary(), plot_validation_panel(), format_test_name(), create_test_inventory_df() |
| longtest_helpers.jl capture function | ✅ Complete | Added capture_longtest_result!() with prevalence/CI for 1→2 and 2→3 only |
| Parametric test suite (exp/wei/gom) | ✅ Complete | longtest_parametric_suite.jl: 18 tests (3 families × 2 data types × 3 cov types) |
| Phase-type tests | 🔄 Existing | Covered by longtest_phasetype.jl |
| Spline tests | 🔄 Existing | Covered by longtest_mcem_splines.jl |
| long_tests.qmd report | ⏳ Pending | Needs rewrite to use new infrastructure |
| Test execution & verification | ⏳ Pending | Run all tests and verify cache population |

## Test Coverage Matrix

| Family | Exact Data | Panel (Markov/MCEM) | Covariates |
|--------|------------|---------------------|------------|
| `exp`  | ✓ | ✓ (Markov) | none, fixed, tvc |
| `wei`  | ✓ | ✓ (MCEM) | none, fixed, tvc |
| `gom`  | ✓ | ✓ (MCEM) | none, fixed, tvc |
| `pt`   | ✓ | ✓ (Markov) | none, fixed, tvc |
| `sp`   | ✓ | ✓ (MCEM) | none, fixed, tvc |

**Parametric suite (this implementation): 3 families × 2 data types × 3 covariate types = 18 tests**
**Total with pt/sp: 5 families × 2 data types × 3 covariate types = 30 long tests**

### Notes
- Exponential and phase-type are Markov → use matrix exponential for panel data
- Semi-Markov families (wei, gom, sp) require MCEM for panel data
- Use progressive 3-state model (1→2→3), NOT illness-death
- No 1→3 transitions anywhere

## Test Configuration

Each test must use:
- **N = 1000 subjects**
- **Standardized evaluation times** for prevalence/CI computation
- **Consistent RNG seeding** for reproducibility
- **Parameter recovery tolerance**: 20% relative error for main params, 0.3 absolute for betas

## Result Capture Requirements

For each test, capture and save to JSON:
1. **Test metadata**: name, family, data type, covariate type, n_subjects
2. **Parameters**: true values, estimates, SEs, 95% CIs (all on estimation scale)
3. **Simulated data summary**: for display in report (state counts at observation times)
4. **Prevalence curves**: true vs fitted at evaluation times (states 1, 2, 3)
5. **Cumulative incidence**: true vs fitted for transitions 1→2 and 2→3 only (NOT 1→3)

## Long Test Report Structure (`long_tests.qmd`)

```
1. Test Inventory (TOP of document)
   - Table showing all 30 tests with pass/fail status
   - Summary statistics (X/Y passed, last run date)
   - Organized by family

2. For each test (grouped by family, then data type, then covariate):
   2.1 Test name and configuration header
   2.2 Parameter table (true, estimated, SE, CI, pass/fail per param)
   2.3 Data visualization:
       - State occupancy counts at observation times
       - Transition counts (1→2, 2→3)
   2.4 Validation plots (2x2 panel):
       - Prevalence: State 1 (true vs fitted with CI bands)
       - Prevalence: State 2
       - Prevalence: State 3  
       - Cumulative incidence: 1→2 and 2→3 overlaid
```

## Implementation Tasks

### Task 1: Update `LongTestResult` struct in `LongTestResults.jl`
- [ ] Add `data_summary` field for storing observation counts
- [ ] Add `test_category` (family), `data_type`, `covariate_type` fields
- [ ] Remove 1→3 cumulative incidence fields
- [ ] Add `passed::Bool` field for overall test status

### Task 2: Create/update long test files
- [ ] `longtest_exact_markov.jl` → exp, pt with exact data (6 tests)
- [ ] `longtest_exact_semimarkov.jl` → wei, gom, sp with exact data (9 tests)
- [ ] `longtest_panel_markov.jl` → exp, pt with panel data using matrix exp (6 tests)
- [ ] `longtest_mcem.jl` → wei, gom, sp with panel data using MCEM (9 tests)
- [ ] Each file covers: no covariates, fixed covariates, tvc

### Task 3: Update `capture_*_result!` functions
- [ ] Include data summary (state counts at each observation time)
- [ ] Only compute CI for 1→2 and 2→3
- [ ] Use N=1000 subjects
- [ ] Compute and store pass/fail status

### Task 4: Rewrite `long_tests.qmd`
- [ ] Test inventory table at TOP
- [ ] Loop over cached results by category
- [ ] For each: parameter table, data summary, 4-panel plot
- [ ] Use `ReportHelpers.jl` plotting functions

### Task 5: Update `ReportHelpers.jl`
- [ ] Add `plot_data_summary()` function
- [ ] Update `plot_cumulative_incidence()` to only show 1→2, 2→3
- [ ] Add `create_test_inventory_table()` function
- [ ] Add `create_parameter_table()` function

## File Organization

```
MultistateModelsTests/
├── longtests/
│   ├── longtest_exact_markov.jl      # exp, pt × exact × (none, fixed, tvc)
│   ├── longtest_exact_semimarkov.jl  # wei, gom, sp × exact × (none, fixed, tvc)
│   ├── longtest_panel_markov.jl      # exp, pt × panel × (none, fixed, tvc)
│   ├── longtest_mcem.jl              # wei, gom, sp × panel × (none, fixed, tvc)
│   └── longtest_helpers.jl           # Shared helper functions
├── src/
│   ├── LongTestResults.jl            # Result struct and save/load
│   └── ReportHelpers.jl              # Plotting and table utilities
├── cache/
│   └── longtest_results/             # 30 JSON files, one per test
└── reports/
    └── long_tests.qmd                # Report displaying all results
```

## Naming Convention for Cached Results

Each test saves to: `{family}_{datatype}_{covtype}.json`

Examples:
- `exp_exact_nocov.json`
- `exp_exact_fixed.json`
- `exp_exact_tvc.json`
- `exp_panel_nocov.json`
- `wei_exact_nocov.json`
- `wei_mcem_fixed.json`
- `pt_panel_tvc.json`
- `sp_mcem_nocov.json`

## Acceptance Criteria

- [ ] All 30 tests run without errors
- [ ] All tests pass parameter recovery checks
- [ ] Report renders with no missing data
- [ ] Each test shows: inventory entry, param table, data viz, validation plots
- [ ] No 1→3 transitions appear anywhere
- [ ] N=1000 subjects in all tests
- [ ] Test inventory at TOP of report

## Run Order

1. Update infrastructure (`LongTestResults.jl`, `ReportHelpers.jl`)
2. Update/create long test files one category at a time
3. Run each long test file to generate cached results
4. Update `long_tests.qmd` to display results
5. Render and verify

## Progress Tracking

### Infrastructure
- [ ] LongTestResults.jl updated
- [ ] ReportHelpers.jl updated
- [ ] longtest_helpers.jl created/updated

### Exact Data Tests (15 tests)
- [ ] exp_exact_nocov
- [ ] exp_exact_fixed
- [ ] exp_exact_tvc
- [ ] wei_exact_nocov
- [ ] wei_exact_fixed
- [ ] wei_exact_tvc
- [ ] gom_exact_nocov
- [ ] gom_exact_fixed
- [ ] gom_exact_tvc
- [ ] pt_exact_nocov
- [ ] pt_exact_fixed
- [ ] pt_exact_tvc
- [ ] sp_exact_nocov
- [ ] sp_exact_fixed
- [ ] sp_exact_tvc

### Panel Data Tests (15 tests)
- [ ] exp_panel_nocov
- [ ] exp_panel_fixed
- [ ] exp_panel_tvc
- [ ] wei_mcem_nocov
- [ ] wei_mcem_fixed
- [ ] wei_mcem_tvc
- [ ] gom_mcem_nocov
- [ ] gom_mcem_fixed
- [ ] gom_mcem_tvc
- [ ] pt_panel_nocov
- [ ] pt_panel_fixed
- [ ] pt_panel_tvc
- [ ] sp_mcem_nocov
- [ ] sp_mcem_fixed
- [ ] sp_mcem_tvc

### Report
- [ ] long_tests.qmd rewritten
- [ ] All 30 results displayed
- [ ] Report renders successfully
