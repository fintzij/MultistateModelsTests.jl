# Simulation Diagnostics (test-owned)

This note lives with the plotting scripts so the figures, assets, and statistical guarantees stay versioned alongside the tests. It documents how to regenerate the diagnostic suite and provides an inline gallery of every panel produced by `generate_model_diagnostics.jl`.

- **Generator:** `MultistateModelsTests/diagnostics/generate_model_diagnostics.jl`
- **Environment:** `MultistateModelsTests/diagnostics/Project.toml`
- **Assets:** `MultistateModelsTests/diagnostics/assets/*.png`

## Reproducing the figures

```bash
cd MultistateModelsTests/diagnostics
julia --project generate_model_diagnostics.jl 2>&1 | tee generation_log.txt
```

The script iterates over all **24 combinations** of:
- **Family**: Exponential (`:exp`), Weibull (`:wei`), Gompertz (`:gom`), Spline (`:sp`)
- **Effect**: Proportional hazards (`:ph`), Accelerated failure time (`:aft`)
- **Covariate mode**: Baseline-only, fixed covariate, time-varying covariate (TVC)

Each scenario emits three output files:

1. **Function panel** — Analytic hazard/cumulative hazard/survival overlaid with `eval_hazard`, `eval_cumhaz`, and `survprob` (with/without Tang time transforms)
2. **Simulation panel** — ECDF vs. analytic CDF, KS statistic vs. sample size, time-transform parity curve `ΔF(t)`, plus histogram/PDF overlay
3. **KS convergence plot** — KS statistic vs. 1/√n with DKW bound and slope analysis

The script outputs a summary showing pass/fail status for each quality check.

---

## Parameterization Reference

All hazard families follow **flexsurv** conventions. Parameters are stored on the **natural scale** (v0.3.0+).

### Exponential Family

| Parameter | Symbol | Constraint | Storage |
|-----------|--------|------------|---------|
| Rate | $\lambda$ | $\lambda > 0$ | Natural scale with box constraint |
| Covariate effect | $\beta$ | $\beta \in \mathbb{R}$ | Natural scale |

| Quantity | Formula |
|----------|---------|
| Baseline hazard | $h_0(t) = \lambda$ |
| Cumulative hazard | $H_0(t) = \lambda t$ |
| PH effect | $h(t\|x) = \lambda e^{\beta x}$, $H(t\|x) = \lambda e^{\beta x} t$ |
| AFT effect | $h(t\|x) = \lambda e^{-\beta x}$, $H(t\|x) = \lambda e^{-\beta x} t$ |

**Note**: For exponential, PH and AFT are equivalent up to sign of β.

### Weibull Family

| Parameter | Symbol | Constraint | Storage |
|-----------|--------|------------|---------|
| Shape | $\kappa$ | $\kappa > 0$ | Natural scale with box constraint |
| Scale | $\sigma$ | $\sigma > 0$ | Natural scale with box constraint |
| Covariate effect | $\beta$ | $\beta \in \mathbb{R}$ | Natural scale |

| Quantity | Formula |
|----------|---------|
| Baseline hazard | $h_0(t) = \kappa \sigma t^{\kappa-1}$ |
| Cumulative hazard | $H_0(t) = \sigma t^\kappa$ |
| PH effect | $h(t\|x) = \kappa \sigma t^{\kappa-1} e^{\beta x}$, $H(t\|x) = \sigma e^{\beta x} t^\kappa$ |
| AFT effect | $h(t\|x) = \kappa \sigma (t e^{-\beta x})^{\kappa-1} e^{-\beta x}$, $H(t\|x) = \sigma (t e^{-\beta x})^\kappa$ |

### Gompertz Family

| Parameter | Symbol | Constraint | Storage |
|-----------|--------|------------|---------|
| Shape | $a$ | $a \in \mathbb{R}$ | Natural scale (unconstrained) |
| Rate | $b$ | $b > 0$ | Natural scale with box constraint |
| Covariate effect | $\beta$ | $\beta \in \mathbb{R}$ | Natural scale |

| Quantity | Formula |
|----------|---------|
| Baseline hazard | $h_0(t) = b e^{at}$ |
| Cumulative hazard | $H_0(t) = \frac{b}{a}(e^{at} - 1)$ |
| PH effect | $h(t\|x) = b e^{at + \beta x}$, $H(t\|x) = \frac{b e^{\beta x}}{a}(e^{at} - 1)$ |
| AFT effect | $h(t\|x) = b e^{a \cdot t e^{-\beta x}} e^{-\beta x}$, $H(t\|x) = \frac{b e^{-\beta x}}{a e^{-\beta x}}(e^{a t e^{-\beta x}} - 1)$ |

**Note**: When $a = 0$, Gompertz reduces to exponential with rate $b$.

### Spline Family

| Parameter | Symbol | Constraint | Storage |
|-----------|--------|------------|---------|
| Coefficients | $\beta_i$ | $\beta_i > 0$ | Natural scale with box constraint |
| Covariate effect | $\gamma$ | $\gamma \in \mathbb{R}$ | Natural scale |

| Quantity | Formula |
|----------|---------|
| Baseline hazard | $h_0(t) = \sum_i \beta_i B_i(t)$ |
| Cumulative hazard | $H_0(t) = \sum_i \beta_i \int_0^t B_i(s) \, ds$ |
| PH effect | $h(t\|x) = h_0(t) e^{\gamma x}$, $H(t\|x) = H_0(t) e^{\gamma x}$ |
| AFT effect | $h(t\|x) = h_0(t e^{-\gamma x}) e^{-\gamma x}$, $H(t\|x) = H_0(t e^{-\gamma x})$ |

### Time-Varying Covariates

For TVC scenarios with piecewise-constant covariates at change points $t_1, t_2, \ldots, t_k$:

**PH effect**: 
$$H(t) = \sum_{i=1}^{k} e^{\beta x_i} \cdot [H_0(\min(t, t_i)) - H_0(t_{i-1})]$$

**AFT effect** (scaled time):
$$\tau(t) = \int_0^t e^{-\beta x(s)} \, ds = \sum_{i=1}^{k} e^{-\beta x_i} \cdot (\min(t, t_i) - t_{i-1})$$

Then $H(t) = H_0(\tau(t))$ and $h(t) = h_0(\tau(t)) \cdot e^{-\beta x(t)}$.

---

## Pass/Fail Criteria

| Diagnostic | Metric | Pass Threshold |
|------------|--------|----------------|
| Function accuracy | max \|Analytic - Computed\| | < 1e-8 |
| ECDF residual | max \|ECDF - F\| | < 0.015 (3× DKW at n=40k) |
| KS convergence slope | Slope of log(KS) vs log(n) | ∈ [-0.6, -0.4] |
| Time-transform parity | max \|F_tang - F_fallback\| | < 1e-4 |

---

## Figure Gallery

Each table row links the panels emitted for that scenario. Images render directly in Markdown-aware environments.

### Exponential Family

| Scenario | Function Panel | Simulation Panel | KS Convergence |
|----------|----------------|------------------|----------------|
| PH, baseline | ![](assets/function_panel_exp_ph_baseline.png) | ![](assets/simulation_panel_exp_ph_baseline.png) | ![](assets/ks_convergence_exp_ph_baseline.png) |
| PH, covariate | ![](assets/function_panel_exp_ph_covariate.png) | ![](assets/simulation_panel_exp_ph_covariate.png) | ![](assets/ks_convergence_exp_ph_covariate.png) |
| PH, TVC | ![](assets/function_panel_exp_ph_tvc.png) | ![](assets/simulation_panel_exp_ph_tvc.png) | ![](assets/ks_convergence_exp_ph_tvc.png) |
| AFT, baseline | ![](assets/function_panel_exp_aft_baseline.png) | ![](assets/simulation_panel_exp_aft_baseline.png) | ![](assets/ks_convergence_exp_aft_baseline.png) |
| AFT, covariate | ![](assets/function_panel_exp_aft_covariate.png) | ![](assets/simulation_panel_exp_aft_covariate.png) | ![](assets/ks_convergence_exp_aft_covariate.png) |
| AFT, TVC | ![](assets/function_panel_exp_aft_tvc.png) | ![](assets/simulation_panel_exp_aft_tvc.png) | ![](assets/ks_convergence_exp_aft_tvc.png) |

### Weibull Family

| Scenario | Function Panel | Simulation Panel | KS Convergence |
|----------|----------------|------------------|----------------|
| PH, baseline | ![](assets/function_panel_wei_ph_baseline.png) | ![](assets/simulation_panel_wei_ph_baseline.png) | ![](assets/ks_convergence_wei_ph_baseline.png) |
| PH, covariate | ![](assets/function_panel_wei_ph_covariate.png) | ![](assets/simulation_panel_wei_ph_covariate.png) | ![](assets/ks_convergence_wei_ph_covariate.png) |
| PH, TVC | ![](assets/function_panel_wei_ph_tvc.png) | ![](assets/simulation_panel_wei_ph_tvc.png) | ![](assets/ks_convergence_wei_ph_tvc.png) |
| AFT, baseline | ![](assets/function_panel_wei_aft_baseline.png) | ![](assets/simulation_panel_wei_aft_baseline.png) | ![](assets/ks_convergence_wei_aft_baseline.png) |
| AFT, covariate | ![](assets/function_panel_wei_aft_covariate.png) | ![](assets/simulation_panel_wei_aft_covariate.png) | ![](assets/ks_convergence_wei_aft_covariate.png) |
| AFT, TVC | ![](assets/function_panel_wei_aft_tvc.png) | ![](assets/simulation_panel_wei_aft_tvc.png) | ![](assets/ks_convergence_wei_aft_tvc.png) |

### Gompertz Family

| Scenario | Function Panel | Simulation Panel | KS Convergence |
|----------|----------------|------------------|----------------|
| PH, baseline | ![](assets/function_panel_gom_ph_baseline.png) | ![](assets/simulation_panel_gom_ph_baseline.png) | ![](assets/ks_convergence_gom_ph_baseline.png) |
| PH, covariate | ![](assets/function_panel_gom_ph_covariate.png) | ![](assets/simulation_panel_gom_ph_covariate.png) | ![](assets/ks_convergence_gom_ph_covariate.png) |
| PH, TVC | ![](assets/function_panel_gom_ph_tvc.png) | ![](assets/simulation_panel_gom_ph_tvc.png) | ![](assets/ks_convergence_gom_ph_tvc.png) |
| AFT, baseline | ![](assets/function_panel_gom_aft_baseline.png) | ![](assets/simulation_panel_gom_aft_baseline.png) | ![](assets/ks_convergence_gom_aft_baseline.png) |
| AFT, covariate | ![](assets/function_panel_gom_aft_covariate.png) | ![](assets/simulation_panel_gom_aft_covariate.png) | ![](assets/ks_convergence_gom_aft_covariate.png) |
| AFT, TVC | ![](assets/function_panel_gom_aft_tvc.png) | ![](assets/simulation_panel_gom_aft_tvc.png) | ![](assets/ks_convergence_gom_aft_tvc.png) |

### Spline Family

| Scenario | Function Panel | Simulation Panel | KS Convergence |
|----------|----------------|------------------|----------------|
| PH, baseline | ![](assets/function_panel_sp_ph_baseline.png) | ![](assets/simulation_panel_sp_ph_baseline.png) | ![](assets/ks_convergence_sp_ph_baseline.png) |
| PH, covariate | ![](assets/function_panel_sp_ph_covariate.png) | ![](assets/simulation_panel_sp_ph_covariate.png) | ![](assets/ks_convergence_sp_ph_covariate.png) |
| PH, TVC | ![](assets/function_panel_sp_ph_tvc.png) | ![](assets/simulation_panel_sp_ph_tvc.png) | ![](assets/ks_convergence_sp_ph_tvc.png) |
| AFT, baseline | ![](assets/function_panel_sp_aft_baseline.png) | ![](assets/simulation_panel_sp_aft_baseline.png) | ![](assets/ks_convergence_sp_aft_baseline.png) |
| AFT, covariate | ![](assets/function_panel_sp_aft_covariate.png) | ![](assets/simulation_panel_sp_aft_covariate.png) | ![](assets/ks_convergence_sp_aft_covariate.png) |
| AFT, TVC | ![](assets/function_panel_sp_aft_tvc.png) | ![](assets/simulation_panel_sp_aft_tvc.png) | ![](assets/ks_convergence_sp_aft_tvc.png) |

---

## Guarantees Checked by the Gallery

- **Call-stack accuracy**: Blue/orange solver traces overlay black analytic curves in every function panel, proving the PH/AFT plumbing (and Tang caches) agree with closed-form hazards.
- **Distributional fidelity**: ECDF residuals stay within ~3×10⁻³, matching `test/longtest_simulation_distribution.jl` tolerances.
- **KS convergence**: KS statistic decreases as $n^{-0.5}$ (slope ≈ -0.5 in log-log space), confirming correct distribution.
- **Time-transform parity**: Tang-enabled simulations match the fallback sampler with max |ΔF| < 1e-4.
- **Family coverage**: All 24 combinations of `{exp, wei, gom, sp} × {ph, aft} × {baseline, covariate, tvc}` are tested.

Keep this document in sync with the assets whenever simulator or hazard changes land so reviewers can diff both code and visuals in one place.
