# ADVERSARIAL CODE REVIEW: MultistateModels.jl Report & Test Infrastructure

## 🚨 ALL ERRORS ARE CRITICAL

**Every discrepancy, every failed test, every mismatched plot is a potential bug in statistical software used for research.** There are no "minor" issues. A 1% error in a hazard function can cascade into completely wrong survival estimates.

**Unit tests and simulation diagnostics are FAST** — they complete in seconds. There is **no excuse for loose tolerances**. These tests should use:
- `rtol=1e-10` for analytical formula verification
- `rtol=1e-6` for numerical integration comparisons
- `atol=1e-12` for values that should be exactly zero

If a test needs loose tolerances to pass, **the code is wrong**.

---

## Red Flags Identified — ALL CRITICAL

| Issue | Location | Status |
|-------|----------|--------|
| **Plot discrepancies** | simulation_diagnostics.qmd | 🚨 CRITICAL - Empirical vs theoretical CDFs don't match |
| **Unit test failures** | unit_tests.qmd / test cache | 🚨 CRITICAL - Tests are failing |
| **MCEM convergence failures** | longtest results | 🚨 CRITICAL - Parameter estimates wildly off |

---

## ADVERSARIAL REVIEW PROTOCOL

### 1. Trust Nothing, Verify Everything

Before debugging, establish ground truth independently:

```julia
# DO NOT trust the package's eval_hazard/eval_cumhaz
# Compute expected values from scratch using textbook formulas

# Exponential: h(t) = λ, H(t) = λt, F(t) = 1 - exp(-λt)
# Weibull: h(t) = κλt^{κ-1}, H(t) = λt^κ, F(t) = 1 - exp(-λt^κ)
# Gompertz: h(t) = b*exp(at), H(t) = (b/a)(exp(at)-1), F(t) = 1 - exp(-H(t))
```

### 2. Question Every Formula

The `expected_cdf()` function in simulation_diagnostics.qmd (lines 108-143) is **suspect**. Verify:

- [ ] Does the Weibull PH formula match the package's internal parameterization?
- [ ] Does the Gompertz AFT formula correctly implement time-scaling?
- [ ] Are log-transforms applied consistently between test and package?

### 3. Trace Data Flow End-to-End

For each failing scenario:
1. What parameters are set?
2. What does the package compute internally?
3. What does the test expect?
4. Where do they diverge?

---

## Issue 1: Simulation Diagnostic Plot Discrepancies 🚨

**File**: `reports/simulation_diagnostics.qmd`

**Symptom**: Empirical CDFs from simulated data don't match theoretical CDFs

**This indicates one of**:
1. ❌ `expected_cdf()` formula is wrong (test bug)
2. ❌ Package's `simulate_path()` uses wrong distribution (package bug)
3. ❌ Parameter transformation mismatch (integration bug)
4. ❌ Covariate effect (PH/AFT) implemented differently than expected

### Required Tolerances for Simulation Diagnostics

With 40,000 samples, the Dvoretzky–Kiefer–Wolfowitz inequality gives:
$$P(\sup_t |F_n(t) - F(t)| > \epsilon) \leq 2e^{-2n\epsilon^2}$$

For n=40,000 and 99.9% confidence: **max CDF difference should be < 0.015**

Current threshold of 0.02 is **too loose**. Use:
```julia
const MAX_CDF_DIFF_THRESHOLD = 0.01  # Tighten from 0.02
```

Any scenario with MaxCDFDiff > 0.01 indicates a **bug, not Monte Carlo noise**.

### Verification Protocol

**Step 1: Isolate the simplest failing case**
```julia
# Test exponential baseline (no covariates) first
# If this fails, the bug is fundamental
model, cfg = build_test_model("exp", :ph, false)
durations = simulate_event_times(model, 10000)

# Compute empirical mean - should match theoretical EXACTLY (within MC error)
empirical_mean = mean(durations)
theoretical_mean = 1 / cfg.rate  # E[Exp(λ)] = 1/λ

# With n=10000, SE of mean ≈ theoretical_mean/100
# So rtol=0.03 is 3 standard errors - very conservative
@test isapprox(empirical_mean, theoretical_mean, rtol=0.03)
```

**Step 2: Check parameter extraction**
```julia
# What parameters does the model actually have?
using MultistateModels: get_parameters
params = get_parameters(model)
@info "Model parameters" params

# Are these what expected_cdf() assumes?
```

**Step 3: Cross-check cumulative hazard with TIGHT tolerances**
```julia
using QuadGK

# Package's cumulative hazard
h = model.hazards[1]
H_pkg = eval_cumhaz(h, 0.0, t, params, covars)

# Numerical integration of hazard (ground truth)
H_numerical = quadgk(s -> eval_hazard(h, s, params, covars), 0.0, t)[1]

# Analytical formula from expected_cdf
H_expected = ...  # From test code

# TIGHT tolerances - these are fast analytical computations
@test isapprox(H_pkg, H_numerical, rtol=1e-10) "Package cumhaz inconsistent with its own hazard!"
@test isapprox(H_pkg, H_expected, rtol=1e-10) "Package cumhaz differs from expected formula!"
```

### Suspect Code Blocks

**1. Gompertz parameter setting (line ~95-97)**:
```julia
elseif family == "gom"
    base = [cfg.shape, log(cfg.rate)]  # shape unconstrained, rate log-transformed
```
**VERIFY**: Does MultistateModels.jl expect `[shape, log(rate)]` or `[log(shape), log(rate)]`?

**2. Gompertz AFT expected CDF (lines ~134-140)**:
```julia
else  # AFT
    time_scale = exp(-linpred)
    scaled_shape = shape * time_scale
    scaled_rate = rate * time_scale
    (scaled_rate / scaled_shape) * (exp(scaled_shape * t) - 1)
end
```
**VERIFY**: Is this the correct AFT transformation for Gompertz? Compare against:
- Package source code in `src/hazard/`
- flexsurv R package documentation
- Published statistical literature

**3. Weibull covariate effect (lines ~126-128)**:
```julia
elseif family == "wei"
    shape, scale = cfg.shape, cfg.scale
    mult = effect == :ph ? exp(beta * xval) : exp(-shape * beta * xval)
```
**VERIFY**: The AFT multiplier `exp(-shape * beta * xval)` - is this correct? Some parameterizations use `exp(-beta * xval)` on the time scale.

---

## Issue 2: Unit Test Failures 🚨

**File**: `reports/unit_tests.qmd`

**Symptom**: Unit tests are failing (check `cache/test_cache.json`)

### Required Tolerances for Unit Tests

Unit tests verify **analytical formulas**. They should use **machine precision tolerances**:

```julia
# Hazard function evaluation (pure math)
@test eval_hazard(...) ≈ expected rtol=1e-12

# Cumulative hazard vs numerical integration  
@test H_analytical ≈ H_numerical rtol=1e-10

# Survival probability
@test S ≈ exp(-H) rtol=1e-12

# Transition probability matrix
@test sum(P[i,:]) ≈ 1.0 atol=1e-14
```

**Any unit test using `rtol > 1e-6` for analytical computations is SUSPECT.**

### Verification Protocol

**Step 1: Identify which tests fail**
```bash
# Check the cache
cat cache/test_cache.json | jq '.test_results | to_entries[] | select(.value.failed > 0 or .value.errors > 0)'
```

**Step 2: Run failing tests in isolation**
```bash
cd MultistateModelsTests
julia --project=. -e 'include("unit/test_hazards.jl")'
julia --project=. -e 'include("unit/test_simulation.jl")'
```

**Step 3: Hunt for loose tolerances**
```bash
# Find any test with tolerance > 1e-6
grep -rn "rtol=0\.[0-9]" unit/
grep -rn "rtol=1e-[1-5]" unit/
grep -rn "atol=0\.[0-9]" unit/
```

**Every loose tolerance found must be justified or tightened.**

### Common Unit Test Bugs

1. **Tolerance masking errors**: Tests pass with `rtol=0.1` but fail with `rtol=1e-10`
   ```julia
   # This is a BUG, not a "passing test"
   @test 1.05 ≈ 1.0 rtol=0.1  # passes but WRONG
   @test 1.05 ≈ 1.0 rtol=1e-10  # fails - CORRECT behavior
   ```

2. **Random seed dependence**: Tests pass/fail based on RNG state
   ```julia
   # Run same test 10 times - must pass ALL
   for i in 1:10
       @testset "Iteration $i" begin
           include("unit/test_simulation.jl")
       end
   end
   ```

3. **Silent exception swallowing**:
   ```julia
   # BAD - hides bugs
   try
       result = dangerous_function()
   catch
       result = fallback  # Bug is now invisible
   end
   ```

---

## Issue 3: MCEM Parameter Recovery Failures 🚨

**Evidence from prior session**:
- `wei_mcem_nocov`: log_shape estimated as 3.14, true value 0.26 → **12x error**
- `gom_mcem_nocov`: shape estimated as 0.50, true value 0.05 → **10x error**

**This is catastrophic failure, not statistical noise.**

### Possible Causes

1. **MCEM not converging**: Check iteration count and convergence diagnostics
2. **Wrong likelihood**: Panel data likelihood computed incorrectly
3. **Initialization failure**: Starting values so bad MCEM finds local minimum
4. **Parameter transformation bug**: Estimating on wrong scale

### Verification

```julia
# Load a failing result
result = load_longtest_result("wei_mcem_nocov")

# Check convergence info
@info "MCEM diagnostics" iterations=result[:mcem_iterations] converged=result[:converged]

# Compare log-likelihood at true vs estimated
ll_true = result[:loglik_at_true]
ll_estimated = result[:loglik_at_estimated]
@info "Log-likelihoods" ll_true ll_estimated

# If ll_estimated < ll_true, optimization failed to find MLE
# If ll_estimated > ll_true significantly, likelihood function is WRONG
```

---

## Tolerance Standards Summary

| Test Type | Computation | Required Tolerance |
|-----------|-------------|-------------------|
| Unit: hazard evaluation | Analytical | `rtol=1e-12` |
| Unit: cumhaz vs quadgk | Numerical integration | `rtol=1e-10` |
| Unit: survival probability | Analytical | `rtol=1e-12` |
| Unit: TPM row sums | Must equal 1 | `atol=1e-14` |
| Simulation: CDF match | Monte Carlo (n=40k) | max diff < 0.01 |
| Simulation: mean match | Monte Carlo (n=10k) | `rtol=0.03` (3 SE) |
| Long test: param recovery | Statistical | `rtol=0.20` (with variance) |

**Unit tests and simulations are FAST. Tight tolerances are mandatory.**

---

## Adversarial Checklist

Before declaring any issue "fixed", verify:

- [ ] **All unit tests pass with `rtol ≤ 1e-10`** for analytical computations
- [ ] **No tolerance > 1e-6** in unit tests without documented justification
- [ ] **Simulation CDF differences < 0.01** for all 12 scenarios
- [ ] **Analytical formulas match package internals** - read the source code
- [ ] **Parameter transformations are consistent** - log-scale vs natural scale
- [ ] **Covariate effects match documentation** - PH multiplies hazard, AFT scales time
- [ ] **Cache reflects current code** - regenerate after any changes
- [ ] **Plots regenerate without errors** - `quarto render` completes successfully
- [ ] **No silent exception handling** - errors must propagate

---

## Files Requiring Scrutiny

| File | What to Check |
|------|---------------|
| `simulation_diagnostics.qmd` lines 108-143 | `expected_cdf()` formulas against textbook |
| `simulation_diagnostics.qmd` lines 89-100 | Parameter vector construction |
| `src/hazard/*.jl` in main package | Actual hazard implementations |
| `unit/test_hazards.jl` | What tolerances are used? MUST be ≤ 1e-10 |
| `unit/test_simulation.jl` | Statistical test tolerances |
| `longtests/longtest_helpers.jl` | How is param recovery checked? |
| `cache/test_cache.json` | Which tests actually failed? |

---

## Expected Outcome

After this review, you should have:

1. **Root cause identified** for plot discrepancies (formula bug, parameterization mismatch, or package bug)
2. **All unit tests passing** with tolerances ≤ 1e-10 for analytical computations
3. **All 12 simulation scenarios** with CDF difference < 0.01
4. **Verified fixes** — not just "tests pass" but "code is mathematically correct"
5. **Regenerated reports** showing perfect overlay of empirical and theoretical CDFs

**Do not declare success until:**
- Every unit test uses appropriate tight tolerances
- Every simulation plot shows empirical CDF overlaying theoretical CDF exactly
- You can explain WHY each formula is correct, not just that "it passes"
