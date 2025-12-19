# MultistateModelsTests Reporting Infrastructure Plan

This document outlines the plan to reorganize `MultistateModelsTests` into a comprehensive validation and documentation hub using Quarto reports.

## Status: ✅ IMPLEMENTED

All reports have been created and are ready for rendering.

## Goals

1. Replace static Markdown files with dynamic **Quarto (`.qmd`)** documents
2. Ensure all reports—including plots, tables, and benchmarks—are computationally reproducible
3. Keep reports in sync with the `MultistateModels` codebase
4. Provide comprehensive documentation for the package

## Directory Structure (Current)

```text
MultistateModelsTests/
├── Project.toml              # Dependencies (Makie, Quarto, etc.)
├── src/                      # Helper modules
│   └── MultistateModelsTests.jl
├── unit/                     # Unit tests (unchanged)
├── longtests/                # Long-running statistical tests (unchanged)
├── reports/                  # Quarto reports
│   ├── _quarto.yml           # Quarto project config
│   ├── index.qmd             # Report index/landing page
│   ├── architecture.qmd      # ✅ Package architecture
│   ├── unit_tests.qmd        # ✅ Unit test coverage
│   ├── long_tests.qmd        # ✅ Long test status
│   ├── simulation_diagnostics.qmd  # ✅ Simulation validation
│   └── benchmarks.qmd        # ✅ Performance benchmarks
├── diagnostics/              # Legacy diagnostic scripts
└── REPORTS_PLAN.md           # This file
```

## Rendering Reports

```bash
cd MultistateModelsTests/reports
quarto render
```

To view locally:
```bash
quarto preview
```

---

## Report 1: Package Architecture & Workflow

**File:** `reports/architecture.qmd`

**Goal:** Document the internal mechanics of `MultistateModels` - how models are created, how simulation works, how inference is performed, and how different options trigger different behavior.

### Table of Contents

1. **Introduction**
   - Design philosophy
   - Package overview

2. **Model Construction**
   - 2.1 The `MultistateProcess` struct
   - 2.2 Hazard specification
     - Exponential
     - Weibull
     - Gompertz
     - B-Splines
   - 2.3 Parameterization conventions (flexsurv alignment)
   - 2.4 Covariate effects (PH vs AFT)
   - 2.5 Time-varying covariates
   - 2.6 The `multistatemodel()` constructor
   - 2.7 Data requirements and validation

3. **Simulation Engine**
   - 3.1 `simulate()` vs `simulate_path()`
   - 3.2 Transform strategies (`CachedTransformStrategy`, `DirectTransformStrategy`)
   - 3.3 Jump solvers (`OptimJumpSolver`, etc.)
   - 3.4 Phase-type expansion for panel data

4. **Inference**
   - 4.1 Likelihood calculations
     - Exactly observed paths
     - Panel/interval-censored data
   - 4.2 MCEM algorithm
   - 4.3 Optimization wrappers
   - 4.4 Parameter constraints and transformations

5. **Key Internal Functions**
   - 5.1 `eval_hazard` / `eval_cumhaz`
   - 5.2 `survprob` / `total_cumulhaz`
   - 5.3 `get_hazard_params`
   - 5.4 Time transform functions

6. **Option Reference**
   - Table mapping user options to internal behavior

---

## Report 2: Unit Test Coverage

**File:** `reports/02_unit_tests.qmd`

**Goal:** Connect unit tests to the architecture, summarize coverage, and explain how key functions are validated.

### Table of Contents

1. **Coverage Summary**
   - Overall statistics
   - Coverage by module

2. **Hazard Functions**
   - Table: Function | Test File | Condition | Status
   - Tests for each family (Exp, Wei, Gom, Splines)

3. **Likelihood Functions**
   - Validation of log-likelihood components
   - Exact vs panel likelihood tests

4. **Simulation**
   - Path generation tests
   - Sampling correctness

5. **Model Construction**
   - Data validation tests
   - Parameter initialization

6. **Splines & Penalization**
   - B-spline basis tests
   - Penalty matrix tests

7. **Phase-Type Models**
   - Expansion/collapse tests
   - FFBS tests

8. **Helpers & Utilities**
   - Miscellaneous utility tests

---

## Report 3: Long Tests Status

**File:** `reports/03_long_tests.qmd`

**Goal:** Validate statistical recovery of parameters with comprehensive diagnostics.

### Table of Contents

1. **Summary Dashboard**
   - Pass/Fail status table for all long tests
   - Last run timestamps

2. **Parametric Models**
   - 2.1 Exponential (exact observation)
   - 2.2 Weibull (exact observation)
   - 2.3 Gompertz (exact observation)
   - 2.4 Markov panel data
   
   *For each test:*
   - Setup description
   - True parameter values
   - Table: Parameter | True | Estimated | SE | Coverage
   - Plots:
     - Observed cumulative incidence vs Expected (true params) vs Estimated (fitted params)
     - State prevalence over time

3. **Spline Models**
   - 3.1 B-splines with various knot configurations
   - Recovery of hazard curves (Hazard vs Time plots)

4. **Semi-Markov Models**
   - Duration-dependent hazards

5. **Time-Varying Covariates**
   - Recovery of TVC effects
   - Step-function covariates

6. **MCEM Convergence**
   - Convergence diagnostics
   - Trace plots

---

## Report 4: Simulation Diagnostics

**File:** `reports/04_simulation_diagnostics.qmd`

**Goal:** Visual validation of the simulation engine - confirm that simulated event times match theoretical distributions.

### Table of Contents

1. **Methodology**
   - Explanation of the validation approach
   - ECDF vs CDF comparison
   - Time-transform parity testing

2. **Parametric Families**
   - 2.1 Exponential
     - PH baseline-only
     - PH with covariate
     - AFT baseline-only
     - AFT with covariate
   - 2.2 Weibull
     - PH baseline-only
     - PH with covariate
     - AFT baseline-only
     - AFT with covariate
   - 2.3 Gompertz
     - PH baseline-only
     - PH with covariate
     - AFT baseline-only
     - AFT with covariate

3. **B-Splines**
   - 3.1 Cubic splines (4 interior knots)
   - 3.2 Quadratic splines (3 interior knots)
   - 3.3 Natural splines
   - 3.4 Splines with covariates (PH)

4. **Time-Varying Covariates**
   - 4.1 Step-function covariates (Exponential PH)
   - 4.2 Step-function covariates (Weibull PH)
   - 4.3 Multi-interval TVC scenarios

5. **Guarantees Validated**
   - Call-stack accuracy
   - Distributional fidelity
   - Time-transform parity

---

## Report 5: Benchmarks

**File:** `reports/05_benchmarks.qmd`

**Goal:** Track performance of key algorithms for inference.

### Table of Contents

1. **Sampling Methods Comparison**
   - 1.1 SIR (Sampling Importance Resampling)
   - 1.2 LHS (Latin Hypercube Sampling)
   - 1.3 Importance Sampling (IS)
   - Table: Method | ESS | Runtime | ESS/second

2. **MCEM Acceleration**
   - 2.1 Standard EM
   - 2.2 SQUAREM
   - Convergence plots

3. **Scalability**
   - 3.1 Runtime vs Number of Subjects
   - 3.2 Runtime vs Number of States
   - 3.3 Runtime vs Number of Transitions

4. **Memory Usage**
   - Peak memory by problem size

---

## Implementation Phases

### Phase 1: Infrastructure Setup
- [x] Create this plan document
- [x] Create `reports/` directory
- [x] Create `_quarto.yml` configuration
- [ ] Create `src/ReportHelpers.jl` for shared plotting/table functions
- [x] Create `reports/index.qmd` landing page

### Phase 2: Report 4 (Simulation Diagnostics)
- [x] Port `generate_model_diagnostics.jl` logic to Quarto
- [ ] Add B-spline scenarios
- [ ] Add TVC scenarios (partial - Exp/Wei done)
- [ ] Deprecate `diagnostics/` folder

### Phase 3: Report 3 (Long Tests)
- [ ] Refactor long tests to return structured results
- [ ] Create standardized plotting functions
- [ ] Build report with all long test results

### Phase 4: Report 2 (Unit Tests)
- [ ] Create test metadata mapping
- [ ] Build coverage summary tables
- [ ] Link tests to architecture documentation

### Phase 5: Report 1 (Architecture)
- [ ] Document model construction
- [ ] Document simulation engine
- [ ] Document inference pipeline
- [ ] Create option reference tables

### Phase 6: Report 5 (Benchmarks)
- [ ] Formalize SIR/LHS/IS comparison
- [ ] Add SQUAREM benchmarks
- [ ] Add scalability tests

---

## Technical Notes

### Caching Strategy
- Reports 3 and 5 will use Quarto's `cache: true` option
- Heavy computations will be cached to avoid re-running unless code changes

### Dependencies
The `Project.toml` needs:
- `CairoMakie` for plotting
- `DataFrames` for table handling
- `StatsBase` for statistical summaries
- `PrettyTables` for formatted output

### Execution
Reports can be rendered with:
```bash
cd MultistateModelsTests/reports
quarto render
```

Or individual reports:
```bash
quarto render 04_simulation_diagnostics.qmd
```

---

## Status

| Report | Status | Last Updated |
|--------|--------|--------------|
| 01_architecture | Placeholder | 2024-12-19 |
| 02_unit_tests | Placeholder | 2024-12-19 |
| 03_long_tests | Placeholder | 2024-12-19 |
| 04_simulation_diagnostics | **Functional** | 2024-12-19 |
| 05_benchmarks | Placeholder | 2024-12-19 |
