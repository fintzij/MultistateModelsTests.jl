# MultistateModelsTests

Comprehensive test suite for [MultistateModels.jl](https://github.com/fintzij/MultistateModels.jl).

[**View Full Test Reports & Benchmarks**](https://fintzij.github.io/MultistateModelsTests.jl/)

This repository contains unit tests, integration tests, and long-running statistical validation tests for the MultistateModels.jl package.

## Test Coverage Summary

| Category | Tests | Status | Description |
|----------|-------|--------|-------------|
| **Unit Tests** | 1,246 | ✅ | Fast tests (~2 min) |
| **Exact Data** | 45 | ✅ | Exact Markov inference validation |
| **MCEM Parametric** | 51 | ✅ | MCEM with Exp/Weibull/Gompertz (incl. PhaseType proposal) |
| **MCEM Splines** | 48 | ✅ | MCEM with M-spline hazards (incl. PhaseType proposal) |
| **MCEM TVC** | 50 | ✅ | Time-varying covariates (incl. PhaseType proposal) |
| **SIR/LHS** | 54 | ✅ | Resampling methods with PhaseType proposal |
| **Simulation Distribution** | 65 | ✅ | Event time distribution correctness |
| **Simulation TVC** | 9,702 | ✅ | TVC simulation validation |
| **Phase-Type Hazards** | 40 | ✅ | Coxian PT hazard models with covariates |
| **Total** | **11,300+** | ✅ | All tests passing |

### What is Tested

- **Hazard Functions**: Analytic validation of Exponential, Weibull, Gompertz hazards (PH/AFT)
- **Model Generation**: State space parsing, transition matrices, formula handling
- **Simulation**: Event time sampling, competing risks, path reconstruction
- **MCEM Inference**: Convergence to true parameters across all hazard families
- **MCEM Proposals**: Markov proposal and PhaseType proposal for semi-Markov models
- **Phase-Type Hazard Models**: Coxian phase-type sojourn distributions with covariates
- **SIR/LHS Resampling**: Importance resampling methods for MCEM acceleration
- **Variance Estimation**: IJ/JK sandwich estimators, coverage validation
- **Spline Hazards**: M-spline construction, boundary conditions, monotonicity
- **Time-Varying Covariates**: Piecewise hazard computation, interval handling

## Installation

```julia
using Pkg

# Clone this repo, then:
Pkg.activate(".")

# Add MultistateModels.jl as a development dependency
Pkg.develop(path="../MultistateModels.jl")  # Or wherever your local copy is
# Or from GitHub:
# Pkg.add(url="https://github.com/fintzij/MultistateModels.jl.git")

Pkg.instantiate()
```

## Structure

```
MultistateModelsTests/
├── Project.toml                    # Package manifest
├── README.md                       # This file
│
├── src/                            # Package module code
│   ├── MultistateModelsTests.jl    # Main module with runtests()
│   ├── LongTestResults.jl          # Result storage structures
│   └── ReportHelpers.jl            # Quarto report utilities
│
├── fixtures/                       # Shared test data generators
│   └── TestFixtures.jl
│
├── unit/                           # Quick tests (~2 min)
│   ├── test_hazards.jl             # Hazard function validation
│   ├── test_mcem.jl                # MCEM infrastructure tests
│   ├── test_phasetype.jl           # Phase-type model tests
│   ├── test_simulation.jl          # Simulation correctness
│   ├── test_splines.jl             # Spline hazard tests
│   ├── test_variance.jl            # Variance estimation tests
│   └── ...                         # Additional unit tests
│
├── integration/                    # Integration tests
│   ├── test_parallel_likelihood.jl
│   └── test_parameter_ordering.jl
│
├── longtests/                      # Statistical validation (~30+ min)
│   ├── longtest_config.jl          # Shared configuration
│   ├── longtest_helpers.jl         # Common test utilities
│   ├── longtest_exact_markov.jl    # Exact data MLE validation
│   ├── longtest_mcem.jl            # MCEM convergence tests
│   ├── longtest_mcem_splines.jl    # Spline MCEM validation
│   ├── longtest_mcem_tvc.jl        # TVC MCEM validation
│   ├── longtest_parametric_suite.jl # Full parametric test matrix
│   ├── longtest_phasetype*.jl      # Phase-type proposal tests
│   ├── longtest_sir.jl             # SIR/LHS resampling tests
│   ├── longtest_variance_validation.jl
│   └── ...                         # Additional long tests
│
├── cache/                          # Runtime outputs (gitignored)
│   ├── longtest_*_YYYYMMDD.txt     # Console output from test runs
│   ├── unit_output_YYYYMMDD.txt    # Unit test console output
│   └── longtest_results/           # Structured JSON results
│       └── *.json                  # Per-scenario results for Quarto
│
├── reports/                        # ★ MASTER TEST REPORTS ★
│   ├── _quarto.yml                 # Quarto project config
│   ├── index.qmd                   # Report landing page
│   ├── 02_unit_tests.qmd           # Unit test report
│   ├── 03_long_tests.qmd           # Long test report (uses cache/longtest_results/)
│   ├── 04_simulation_diagnostics.qmd
│   ├── 05_benchmarks.qmd
│   └── _site/                      # ★ RENDERED HTML REPORTS ★
│       ├── index.html              # View this for test status
│       ├── 02_unit_tests.html
│       ├── 03_long_tests.html
│       └── ...
│
├── diagnostics/                    # Development diagnostics
│   ├── generate_model_diagnostics.jl  # Diagnostic generator script
│   ├── *.md / *.html               # Rendered diagnostic reports
│   └── assets/                     # Generated diagnostic plots
│
├── benchmarks/                     # Performance benchmarks
│   └── spline_comparison/          # Julia vs R spline comparison
│
├── scripts/                        # Utility scripts
│   ├── run_longtests.jl            # Long test runner
│   └── run_all_tests.jl            # Full test runner
│
└── scratch/                        # Temporary development files
```

### Key Directories

| Directory | Purpose | Gitignored |
|-----------|---------|------------|
| `unit/` | Fast unit tests (~2 min) | No |
| `integration/` | Integration tests | No |
| `longtests/` | Statistical validation (~30+ min) | No |
| `cache/` | Runtime outputs (txt, json) | Yes |
| `reports/` | Quarto source + rendered HTML | Partial (_site/) |
| `diagnostics/` | Development diagnostic reports | Partial (assets/) |
| `scratch/` | Temporary development files | Yes |

### Master Test Reports

**The canonical test reports are in `reports/_site/`**. These HTML files are rendered from Quarto:

```bash
# Re-render reports after running tests
cd MultistateModelsTests/reports
quarto render
```

The reports pull data from:
- `cache/longtest_results/*.json` - Structured test results  
- Unit test output captured during `Pkg.test()`

## Running Tests

### Quick Start

```julia
using MultistateModelsTests
MultistateModelsTests.runtests()  # Run unit tests only
```

### Full Test Suite (Unit + Long Tests)

```bash
MSM_TEST_LEVEL=full julia --project=. -e 'using MultistateModelsTests; MultistateModelsTests.runtests()'
```

### Selective Long Tests

Run a specific long test suite by setting `MSM_LONGTEST_ONLY`:

```bash
# Run only phasetype long tests
MSM_LONGTEST_ONLY=phasetype MSM_TEST_LEVEL=full julia --project=. -e 'using MultistateModelsTests; MultistateModelsTests.runtests()'

# Run only MCEM parametric long tests  
MSM_LONGTEST_ONLY=mcem_parametric MSM_TEST_LEVEL=full julia --project=. -e 'using MultistateModelsTests; MultistateModelsTests.runtests()'
```

Available test keys:
- `exact_data` - Exact data direct MLE
- `mcem_parametric` - MCEM with parametric hazards
- `mcem_splines` - MCEM with spline hazards
- `mcem_tvc` - MCEM with time-varying covariates
- `sim_dist` - Simulation distribution fidelity
- `sim_tvc` - Simulation with TVCs
- `robust_exact` - Robust variance (parametric)
- `markov_phasetype_validation` - Markov/PhaseType consistency
- `phasetype` - Phase-type proposals (exact + panel)
- `variance_validation` - Variance estimation validation

### Long Tests Only (via scripts)

```bash
julia --project=. scripts/run_longtests.jl
```

## Test Categories

### Unit Tests
Fast tests (~2 min total) covering:
- Hazard function correctness (analytic validation)
- Model generation and parsing
- Simulation mechanics
- Phase-type approximations
- Spline construction
- Helper utilities

### Integration Tests
Medium-duration tests covering:
- Parallel likelihood computation
- Parameter ordering consistency

### Long Tests
Statistical validation tests (~30+ min) covering:
- MCEM convergence to true parameters
- Simulation distribution correctness
- Phase-type importance sampling accuracy
- Robust variance estimation
- **Variance-covariance validation** (IJ/JK estimators)

## Variance Validation Tests

The `longtest_variance_validation.jl` test validates the variance-covariance estimation infrastructure:

### What is Tested

1. **IJ vs Model-based Variance**: Under correct model specification, the infinitesimal jackknife (sandwich) variance should approximately equal the model-based (inverse Hessian) variance.

2. **JK = ((n-1)/n) × IJ Relationship**: The jackknife variance is algebraically related to IJ variance by this factor. This is an exact identity, not a statistical property.

3. **Estimated vs Empirical Variance**: Variance estimates are compared against empirical variance computed from 1000 simulation replicates.

4. **95% CI Coverage**: Wald confidence intervals should achieve approximately 95% coverage.

5. **Positive Definiteness**: All variance matrices must have non-negative eigenvalues.

### Variance Estimator Formulas

- **Model-based**: `Var(θ̂) = H⁻¹` (inverse observed Fisher information)
- **IJ (sandwich)**: `Var_{IJ}(θ̂) = H⁻¹ K H⁻¹` where `K = Σᵢ gᵢgᵢᵀ`
- **JK**: `Var_{JK}(θ̂) = ((n-1)/n) × Var_{IJ}(θ̂)`

### Validation Results (1000 replicates)

| Test | Result | Details |
|------|--------|---------|
| IJ vs Model-based (Exponential) | ✅ | Ratio: 0.978 |
| IJ vs Model-based (Weibull) | ✅ | Ratios: [0.982, 0.954] |
| IJ vs Model-based (Markov Panel) | ✅ | Ratios: [1.04, 1.001, 1.007] |
| Model vs Empirical (Exponential) | ✅ | Ratio: 0.965 (SE: 0.0234 vs 0.0238) |
| Model vs Empirical (Weibull) | ✅ | Ratios: [0.991, 0.973] |
| IJ vs Empirical | ✅ | Ratio: 0.922 |
| 95% CI Coverage | ✅ | 94.0% |
| Positive Definiteness | ✅ | All eigenvalues ≥ 0 |

### Running Variance Validation

```bash
# Run only variance validation long test
MSM_LONGTEST_ONLY=variance_validation MSM_TEST_LEVEL=full julia --project=. -e 'using Pkg; Pkg.test()'

# Or directly:
julia --project=. -e 'include("MultistateModelsTests/longtests/longtest_variance_validation.jl")'
```

## Diagnostic Reports

The `diagnostics/` directory contains report generation tools for visual validation of simulation and inference. These generate PNG plots comparing:

- Analytic vs computed hazard/cumulative hazard/survival functions
- Simulated distributions vs theoretical distributions
- Phase-type approximation accuracy

### Regenerating Reports

From Julia:

```julia
include("MultistateModelsTests/src/MultistateModelsTests.jl")
using .MultistateModelsTests
MultistateModelsTests.generate_simulation_diagnostics()
```

Or from command line:

```bash
julia --project=MultistateModelsTests MultistateModelsTests/diagnostics/generate_model_diagnostics.jl
```

Output is saved to `MultistateModelsTests/diagnostics/assets/`. See `diagnostics/README.md` for details.