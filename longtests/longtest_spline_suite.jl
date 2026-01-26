# =============================================================================
# Unpenalized Spline Long Test Suite
# =============================================================================
#
# Comprehensive test suite covering spline hazard families with exact data,
# panel+Markov, and panel+PhaseType inference methods.
#
# Tests: 3 inference methods × 2 effect types × 3 covariate configs = 18 tests
#
# Inference Methods:
#   - exact: Exact transition times (direct MLE)
#   - panel_markov: Panel data with Markov (matrix exponential) inference
#   - panel_phasetype: Panel data with PhaseType FFBS inference
#
# Effect Types:
#   - PH: Proportional hazards (covariate multiplies hazard)
#   - AFT: Accelerated failure time (covariate accelerates time)
#
# Covariate Configurations:
#   - nocov: No covariates (baseline hazards only)
#   - fixed: Time-fixed binary covariate
#   - tvc: Time-varying binary covariate (changes at t=5)
#
# Model Structure:
#   2-state progressive model: 1 → 2
#   All subjects start in state 1 at time 0
#
# Validation Strategy:
#   Round-trip test: Generate data from spline DGP with known coefficients,
#   fit spline model, compare estimated hazard curve to true hazard curve
#   at multiple time points (not parameter values, since B-spline basis differs).
#
# Naming Convention for Tests:
#   sp_{effect}_{datatype}_{covtype}
#   e.g., sp_ph_exact_nocov, sp_aft_panel_markov_fixed
#
# =============================================================================

using Test
using MultistateModels
using DataFrames
using Random
using LinearAlgebra
using Statistics
using Printf

import MultistateModels: Hazard, @formula, multistatemodel, fit, set_parameters!, 
    simulate, get_parameters_flat, get_parameters, cumulative_hazard, eval_hazard

# Longtest config and helpers are loaded by MultistateModelsTests module.
# For standalone runs, include from src/ (canonical location).
if !@isdefined(PARAM_REL_TOL)
    include(joinpath(@__DIR__, "..", "src", "longtest_config.jl"))
    include(joinpath(@__DIR__, "..", "src", "longtest_helpers.jl"))
end

# LongTestResults (include for standalone runs)
if !isdefined(Main, :LongTestResult) && !@isdefined(LongTestResult)
    include(joinpath(@__DIR__, "..", "src", "LongTestResults.jl"))
end

# =============================================================================
# Test Configuration
# =============================================================================

const RNG_SEED_SPLINE_SUITE = 0x5E11AE02  # Different seed from longtest_spline_exact.jl

# Spline configuration (simple, identifiable setup)
const SPLINE_DEGREE_SUITE = 3              # Cubic splines
const N_INTERIOR_KNOTS_SUITE = 2           # 2 interior knots
const BOUNDARY_KNOTS_SUITE = [0.0, 15.0]   # Match MAX_TIME
const EXTRAPOLATION_SUITE = "constant"     # Constant extrapolation outside boundary

# True spline coefficients for DGP
# These define the baseline hazard via B-spline basis on NATURAL scale
# (coefficients must be non-negative, as enforced by box constraints)
# With degree=3 and 2 interior knots, we have 2+3+1 = 6 coefficients
# Actually: n_basis = n_interior_knots + degree + 1 = 2 + 3 + 1 = 6
# But intercept absorbed, so spline coefs = 5 for the basis
# These values produce hazards in the reasonable range [0.05, 0.25]
const TRUE_SPLINE_COEFFS = [0.08, 0.12, 0.15, 0.18, 0.22]  # 5 coefficients, positive

# Covariate effect (on log-hazard scale for PH, on log-time scale for AFT)
const TRUE_BETA_SUITE = 0.5

# Panel observation times
const PANEL_TIMES_SUITE = collect(0.0:1.0:15.0)  # Dense panel for reliable inference

# Tolerances for hazard comparison
# Note: We compare hazard CURVES, not parameters (B-spline basis differs between DGP and fitted)
const HAZARD_RTOL_SUITE = 0.35       # 35% relative tolerance for pointwise h(t)
const BETA_ATOL_SUITE = 0.40         # Absolute tolerance for log hazard ratio

# Option to save results to cache for reports
const SAVE_RESULTS_TO_CACHE_SUITE = get(ENV, "LONGTEST_SAVE_RESULTS", "true") == "true"

# =============================================================================
# Helper Functions
# =============================================================================

"""
    get_interior_knots_suite(max_time::Float64)

Compute interior knots for spline (evenly spaced).
"""
function get_interior_knots_suite(max_time::Float64)
    n = N_INTERIOR_KNOTS_SUITE
    return collect(range(max_time / (n + 1), max_time * n / (n + 1), length=n))
end

"""
    create_spline_hazard(covariate_type::String, effect::Symbol; max_time::Float64=15.0)

Create a SplineHazard with the standard configuration.
effect is :ph or :aft
"""
function create_spline_hazard(covariate_type::String, effect::Symbol; max_time::Float64=15.0)
    formula = covariate_type == "nocov" ? @formula(0 ~ 1) : @formula(0 ~ x)
    knots = get_interior_knots_suite(max_time)
    
    return Hazard(formula, "sp", 1, 2;
        degree=SPLINE_DEGREE_SUITE,
        knots=knots,
        boundaryknots=BOUNDARY_KNOTS_SUITE,
        extrapolation=EXTRAPOLATION_SUITE,
        linpred_effect=effect)
end

"""
    get_true_params_suite(covariate_type::String)

Get true parameter values for the spline DGP.
Returns NamedTuple with h12 key.
"""
function get_true_params_suite(covariate_type::String)
    if covariate_type == "nocov"
        return (h12 = copy(TRUE_SPLINE_COEFFS),)
    else
        return (h12 = vcat(copy(TRUE_SPLINE_COEFFS), TRUE_BETA_SUITE),)
    end
end

"""
    evaluate_hazard_at_times_suite(fitted, test_times::Vector{Float64}, covars=NamedTuple())

Evaluate fitted hazard at multiple time points.
"""
function evaluate_hazard_at_times_suite(fitted, test_times::Vector{Float64}, covars=NamedTuple())
    pars = get_parameters(fitted, 1, scale=:log)
    haz = fitted.hazards[1]
    return [haz(t, pars, covars) for t in test_times]
end

"""
    evaluate_true_hazard_at_times(model_dgp, test_times::Vector{Float64}, covars=NamedTuple())

Evaluate DGP model hazard at multiple time points.
"""
function evaluate_true_hazard_at_times(model_dgp, test_times::Vector{Float64}, covars=NamedTuple())
    pars = get_parameters(model_dgp, 1, scale=:log)
    haz = model_dgp.hazards[1]
    return [haz(t, pars, covars) for t in test_times]
end

"""
    compare_hazard_curves(h_true, h_fitted, tolerance; label="Hazard")

Compare hazard curves and return whether all passed.
"""
function compare_hazard_curves(h_true::Vector{Float64}, h_fitted::Vector{Float64}, 
                               tolerance::Float64; label::String="Hazard")
    all_passed = true
    for i in eachindex(h_true)
        rel_diff = abs(h_fitted[i] - h_true[i]) / max(h_true[i], 0.001)
        passed = rel_diff <= tolerance
        all_passed = all_passed && passed
    end
    return all_passed
end

"""
    create_spline_result_suite(test_name, passed, fitted, data; kwargs...)

Create a LongTestResult for spline suite tests.
"""
function create_spline_result_suite(
    test_name::String,
    passed::Bool,
    fitted,
    data::DataFrame;
    covariate_type::String="nocov",
    inference_method::String="exact"
)
    result = LongTestResult(
        test_name = test_name,
        test_description = "Spline $(covariate_type) - $(inference_method) inference",
        hazard_family = "sp",
        data_type = inference_method,
        covariate_type = covariate_type,
        n_subjects = length(unique(data.id)),
        n_simulations = 500,
        passed = passed
    )
    
    # Data summary
    result.data_summary = Dict{String, Any}(
        "n_subjects" => length(unique(data.id)),
        "n_transitions" => sum(data.statefrom .!= data.stateto),
        "max_time" => maximum(data.tstop)
    )
    
    # Mark as curve validation (not parameter recovery)
    result.true_params["validation_type"] = 1.0
    result.estimated_params["validation_type"] = 1.0
    result.param_passed["validation_type"] = passed
    
    return result
end

# =============================================================================
# Exact Data Test Functions
# =============================================================================

"""
    run_spline_exact_test(effect::Symbol, covariate_type::String) -> LongTestResult

Run exact data test for given effect type and covariate configuration.
effect is :ph or :aft
"""
function run_spline_exact_test(effect::Symbol, covariate_type::String)
    effect_str = effect == :ph ? "ph" : "aft"
    test_name = "sp_$(effect_str)_exact_$(covariate_type)"
    
    VERBOSE_LONGTESTS && println("  ▶ Running: $test_name")
    
    # Set reproducible seed
    test_seed = hash((effect_str, "exact", covariate_type, RNG_SEED_SPLINE_SUITE))
    Random.seed!(test_seed)
    
    # Create DGP model
    h12_dgp = create_spline_hazard(covariate_type, effect)
    true_params = get_true_params_suite(covariate_type)
    
    # Create template based on covariate type
    template = if covariate_type == "nocov"
        create_baseline_template(N_SUBJECTS; max_time=MAX_TIME)
    elseif covariate_type == "fixed"
        create_tfc_template(N_SUBJECTS; max_time=MAX_TIME)
    else  # tvc
        create_tvc_template(N_SUBJECTS; max_time=MAX_TIME, changepoint=TVC_CHANGEPOINT)
    end
    
    model_dgp = multistatemodel(h12_dgp; data=template)
    set_parameters!(model_dgp, true_params)
    
    # Simulate exact data
    sim_result = simulate(model_dgp; paths=false, data=true, nsim=1)
    exact_data = sim_result[1]
    
    n_transitions = sum(exact_data.statefrom .!= exact_data.stateto)
    max_obs_time = maximum(exact_data.tstop)
    VERBOSE_LONGTESTS && println("    Data: $(length(unique(exact_data.id))) subjects, $n_transitions transitions")
    
    # Fit model with same spline config
    h12_fit = create_spline_hazard(covariate_type, effect; max_time=max_obs_time)
    model_fit = multistatemodel(h12_fit; data=exact_data)
    
    # Fit (use vcov_type=:ij for PH, :none for AFT with covariates due to Hessian issues)
    # Also disable lambda selection for AFT + covariates (known stability issues)
    # This mirrors the approach in longtest_spline_exact.jl
    if effect == :aft && covariate_type != "nocov"
        fitted = fit(model_fit; verbose=false, vcov_type=:none, 
                     select_lambda=:none, lambda_init=0.1)
    else
        fitted = fit(model_fit; verbose=false, vcov_type=:ij)
    end
    
    # Validate: compare hazard curves at test times
    test_times = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    test_times = filter(t -> t <= max_obs_time, test_times)
    
    # For covariate tests, evaluate at x=0 and x=1
    if covariate_type == "nocov"
        h_true = evaluate_true_hazard_at_times(model_dgp, test_times)
        h_fitted = evaluate_hazard_at_times_suite(fitted, test_times)
        
        all_passed = compare_hazard_curves(h_true, h_fitted, HAZARD_RTOL_SUITE)
        
        if VERBOSE_LONGTESTS
            println("    Hazard comparison (baseline):")
            for (i, t) in enumerate(test_times)
                rel_diff = abs(h_fitted[i] - h_true[i]) / max(h_true[i], 0.001)
                status = rel_diff <= HAZARD_RTOL_SUITE ? "✓" : "✗"
                @printf("      t=%.1f: true=%.4f, fitted=%.4f, rel_diff=%.3f %s\n", 
                        t, h_true[i], h_fitted[i], rel_diff, status)
            end
        end
    else
        covars_x0 = (x = 0.0,)
        covars_x1 = (x = 1.0,)
        
        h_true_x0 = evaluate_true_hazard_at_times(model_dgp, test_times, covars_x0)
        h_true_x1 = evaluate_true_hazard_at_times(model_dgp, test_times, covars_x1)
        h_fitted_x0 = evaluate_hazard_at_times_suite(fitted, test_times, covars_x0)
        h_fitted_x1 = evaluate_hazard_at_times_suite(fitted, test_times, covars_x1)
        
        all_h0_passed = compare_hazard_curves(h_true_x0, h_fitted_x0, HAZARD_RTOL_SUITE)
        all_h1_passed = compare_hazard_curves(h_true_x1, h_fitted_x1, HAZARD_RTOL_SUITE)
        
        # Also check log hazard ratio (beta) at each time
        all_beta_passed = true
        for (i, t) in enumerate(test_times)
            log_hr_fitted = log(h_fitted_x1[i]) - log(h_fitted_x0[i])
            log_hr_true = log(h_true_x1[i]) - log(h_true_x0[i])
            abs_diff = abs(log_hr_fitted - log_hr_true)
            passed = abs_diff <= BETA_ATOL_SUITE
            all_beta_passed = all_beta_passed && passed
        end
        
        all_passed = all_h0_passed && all_h1_passed && all_beta_passed
        
        if VERBOSE_LONGTESTS
            println("    Hazard comparison (x=0 and x=1):")
            println("      x=0: $(all_h0_passed ? "PASS" : "FAIL")")
            println("      x=1: $(all_h1_passed ? "PASS" : "FAIL")")
            println("      beta: $(all_beta_passed ? "PASS" : "FAIL")")
        end
    end
    
    VERBOSE_LONGTESTS && println("    Result: $(all_passed ? "PASS" : "FAIL")")
    
    # Create and save result
    result = create_spline_result_suite(test_name, all_passed, fitted, exact_data;
                                        covariate_type=covariate_type,
                                        inference_method="exact")
    
    SAVE_RESULTS_TO_CACHE_SUITE && save_longtest_result(result; force=true)
    
    return result
end

# =============================================================================
# Panel + Markov Test Functions
# =============================================================================

"""
    run_spline_panel_markov_test(effect::Symbol, covariate_type::String) -> LongTestResult

Run panel + Markov inference test.
Note: Splines with panel data use matrix exponential (Markov approximation).
"""
function run_spline_panel_markov_test(effect::Symbol, covariate_type::String)
    effect_str = effect == :ph ? "ph" : "aft"
    test_name = "sp_$(effect_str)_panel_markov_$(covariate_type)"
    
    VERBOSE_LONGTESTS && println("  ▶ Running: $test_name")
    
    # Set reproducible seed
    test_seed = hash((effect_str, "panel_markov", covariate_type, RNG_SEED_SPLINE_SUITE))
    Random.seed!(test_seed)
    
    # Create DGP model (exact data generation, then convert to panel)
    h12_dgp = create_spline_hazard(covariate_type, effect)
    true_params = get_true_params_suite(covariate_type)
    
    # Create template based on covariate type
    template = if covariate_type == "nocov"
        create_baseline_template(N_SUBJECTS; max_time=MAX_TIME)
    elseif covariate_type == "fixed"
        create_tfc_template(N_SUBJECTS; max_time=MAX_TIME)
    else  # tvc
        create_tvc_template(N_SUBJECTS; max_time=MAX_TIME, changepoint=TVC_CHANGEPOINT)
    end
    
    model_dgp = multistatemodel(h12_dgp; data=template)
    set_parameters!(model_dgp, true_params)
    
    # Simulate paths and convert to panel data
    Random.seed!(test_seed)
    paths = simulate(model_dgp; paths=true, data=false, nsim=1)[1]
    
    # Convert to panel observations
    panel_data = if covariate_type == "nocov"
        create_panel_data(paths, PANEL_TIMES_SUITE, 2; verbose=VERBOSE_LONGTESTS)
    elseif covariate_type == "fixed"
        x_vals = template.x[1:N_SUBJECTS]  # One value per subject
        create_panel_data_with_covariate(paths, PANEL_TIMES_SUITE, 2, x_vals; verbose=VERBOSE_LONGTESTS)
    else  # tvc
        create_panel_data_with_tvc(paths, PANEL_TIMES_SUITE, 2; 
                                   changepoint=TVC_CHANGEPOINT, verbose=VERBOSE_LONGTESTS)
    end
    
    if isempty(panel_data) || nrow(panel_data) == 0
        VERBOSE_LONGTESTS && println("    WARNING: No panel data generated, skipping test")
        result = create_spline_result_suite(test_name, false, nothing, DataFrame(id=[1], tstart=[0.0], tstop=[1.0], statefrom=[1], stateto=[1], obstype=[1]);
                                            covariate_type=covariate_type, inference_method="panel")
        SAVE_RESULTS_TO_CACHE_SUITE && save_longtest_result(result; force=true)
        return result
    end
    
    n_subjects_retained = length(unique(panel_data.id))
    VERBOSE_LONGTESTS && println("    Data: $n_subjects_retained subjects retained for panel data")
    
    # Fit model with Markov surrogate (matrix exponential)
    max_obs_time = maximum(panel_data.tstop)
    h12_fit = create_spline_hazard(covariate_type, effect; max_time=max_obs_time)
    model_fit = multistatemodel(h12_fit; data=panel_data, surrogate=:markov)
    
    # Fit
    fitted = fit(model_fit; verbose=false, vcov_type=:ij)
    
    # Validate: compare hazard curves at test times
    test_times = [2.0, 4.0, 6.0, 8.0, 10.0]
    test_times = filter(t -> t <= max_obs_time, test_times)
    
    # For covariate tests, evaluate at x=0 and x=1
    if covariate_type == "nocov"
        h_true = evaluate_true_hazard_at_times(model_dgp, test_times)
        h_fitted = evaluate_hazard_at_times_suite(fitted, test_times)
        
        all_passed = compare_hazard_curves(h_true, h_fitted, HAZARD_RTOL_SUITE)
    else
        covars_x0 = (x = 0.0,)
        covars_x1 = (x = 1.0,)
        
        h_true_x0 = evaluate_true_hazard_at_times(model_dgp, test_times, covars_x0)
        h_true_x1 = evaluate_true_hazard_at_times(model_dgp, test_times, covars_x1)
        h_fitted_x0 = evaluate_hazard_at_times_suite(fitted, test_times, covars_x0)
        h_fitted_x1 = evaluate_hazard_at_times_suite(fitted, test_times, covars_x1)
        
        all_h0_passed = compare_hazard_curves(h_true_x0, h_fitted_x0, HAZARD_RTOL_SUITE)
        all_h1_passed = compare_hazard_curves(h_true_x1, h_fitted_x1, HAZARD_RTOL_SUITE)
        
        # Check beta
        all_beta_passed = true
        for (i, t) in enumerate(test_times)
            log_hr_fitted = log(h_fitted_x1[i]) - log(h_fitted_x0[i])
            log_hr_true = log(h_true_x1[i]) - log(h_true_x0[i])
            abs_diff = abs(log_hr_fitted - log_hr_true)
            passed = abs_diff <= BETA_ATOL_SUITE
            all_beta_passed = all_beta_passed && passed
        end
        
        all_passed = all_h0_passed && all_h1_passed && all_beta_passed
    end
    
    VERBOSE_LONGTESTS && println("    Result: $(all_passed ? "PASS" : "FAIL")")
    
    # Create and save result
    result = create_spline_result_suite(test_name, all_passed, fitted, panel_data;
                                        covariate_type=covariate_type,
                                        inference_method="panel")
    
    SAVE_RESULTS_TO_CACHE_SUITE && save_longtest_result(result; force=true)
    
    return result
end

# =============================================================================
# Panel + PhaseType Test Functions
# =============================================================================

"""
    run_spline_panel_phasetype_test(effect::Symbol, covariate_type::String) -> LongTestResult

Run panel + PhaseType FFBS inference test.
Note: This uses PhaseType surrogate for inference, not spline-specific MCEM.
"""
function run_spline_panel_phasetype_test(effect::Symbol, covariate_type::String)
    effect_str = effect == :ph ? "ph" : "aft"
    test_name = "sp_$(effect_str)_panel_phasetype_$(covariate_type)"
    
    VERBOSE_LONGTESTS && println("  ▶ Running: $test_name")
    
    # Set reproducible seed
    test_seed = hash((effect_str, "panel_phasetype", covariate_type, RNG_SEED_SPLINE_SUITE))
    Random.seed!(test_seed)
    
    # Create DGP model
    h12_dgp = create_spline_hazard(covariate_type, effect)
    true_params = get_true_params_suite(covariate_type)
    
    # Create template based on covariate type
    template = if covariate_type == "nocov"
        create_baseline_template(N_SUBJECTS; max_time=MAX_TIME)
    elseif covariate_type == "fixed"
        create_tfc_template(N_SUBJECTS; max_time=MAX_TIME)
    else  # tvc
        create_tvc_template(N_SUBJECTS; max_time=MAX_TIME, changepoint=TVC_CHANGEPOINT)
    end
    
    model_dgp = multistatemodel(h12_dgp; data=template)
    set_parameters!(model_dgp, true_params)
    
    # Simulate paths and convert to panel data
    Random.seed!(test_seed)
    paths = simulate(model_dgp; paths=true, data=false, nsim=1)[1]
    
    # Convert to panel observations
    panel_data = if covariate_type == "nocov"
        create_panel_data(paths, PANEL_TIMES_SUITE, 2; verbose=VERBOSE_LONGTESTS)
    elseif covariate_type == "fixed"
        x_vals = template.x[1:N_SUBJECTS]
        create_panel_data_with_covariate(paths, PANEL_TIMES_SUITE, 2, x_vals; verbose=VERBOSE_LONGTESTS)
    else  # tvc
        create_panel_data_with_tvc(paths, PANEL_TIMES_SUITE, 2; 
                                   changepoint=TVC_CHANGEPOINT, verbose=VERBOSE_LONGTESTS)
    end
    
    if isempty(panel_data) || nrow(panel_data) == 0
        VERBOSE_LONGTESTS && println("    WARNING: No panel data generated, skipping test")
        result = create_spline_result_suite(test_name, false, nothing, DataFrame(id=[1], tstart=[0.0], tstop=[1.0], statefrom=[1], stateto=[1], obstype=[1]);
                                            covariate_type=covariate_type, inference_method="mcem")
        SAVE_RESULTS_TO_CACHE_SUITE && save_longtest_result(result; force=true)
        return result
    end
    
    n_subjects_retained = length(unique(panel_data.id))
    VERBOSE_LONGTESTS && println("    Data: $n_subjects_retained subjects retained for panel data")
    
    # Fit model with PhaseType surrogate
    max_obs_time = maximum(panel_data.tstop)
    h12_fit = create_spline_hazard(covariate_type, effect; max_time=max_obs_time)
    model_fit = multistatemodel(h12_fit; data=panel_data, surrogate=:phasetype)
    
    # Fit with MCEM using PhaseType proposal
    fitted = fit(model_fit; 
        verbose=false, 
        vcov_type=:ij,
        method=:MCEM,
        tol=MCEM_TOL,
        ess_target_initial=MCEM_ESS_INITIAL,
        max_ess=MCEM_ESS_MAX,
        maxiter=MCEM_MAX_ITER
    )
    
    # Validate: compare hazard curves at test times
    test_times = [2.0, 4.0, 6.0, 8.0, 10.0]
    test_times = filter(t -> t <= max_obs_time, test_times)
    
    # For covariate tests, evaluate at x=0 and x=1
    if covariate_type == "nocov"
        h_true = evaluate_true_hazard_at_times(model_dgp, test_times)
        h_fitted = evaluate_hazard_at_times_suite(fitted, test_times)
        
        all_passed = compare_hazard_curves(h_true, h_fitted, HAZARD_RTOL_SUITE)
    else
        covars_x0 = (x = 0.0,)
        covars_x1 = (x = 1.0,)
        
        h_true_x0 = evaluate_true_hazard_at_times(model_dgp, test_times, covars_x0)
        h_true_x1 = evaluate_true_hazard_at_times(model_dgp, test_times, covars_x1)
        h_fitted_x0 = evaluate_hazard_at_times_suite(fitted, test_times, covars_x0)
        h_fitted_x1 = evaluate_hazard_at_times_suite(fitted, test_times, covars_x1)
        
        all_h0_passed = compare_hazard_curves(h_true_x0, h_fitted_x0, HAZARD_RTOL_SUITE)
        all_h1_passed = compare_hazard_curves(h_true_x1, h_fitted_x1, HAZARD_RTOL_SUITE)
        
        # Check beta (use relaxed tolerance for MCEM)
        all_beta_passed = true
        for (i, t) in enumerate(test_times)
            log_hr_fitted = log(h_fitted_x1[i]) - log(h_fitted_x0[i])
            log_hr_true = log(h_true_x1[i]) - log(h_true_x0[i])
            abs_diff = abs(log_hr_fitted - log_hr_true)
            # Use relaxed tolerance for MCEM
            passed = abs_diff <= MCEM_TVC_BETA_ABS_TOL
            all_beta_passed = all_beta_passed && passed
        end
        
        all_passed = all_h0_passed && all_h1_passed && all_beta_passed
    end
    
    VERBOSE_LONGTESTS && println("    Result: $(all_passed ? "PASS" : "FAIL")")
    
    # Create and save result
    result = create_spline_result_suite(test_name, all_passed, fitted, panel_data;
                                        covariate_type=covariate_type,
                                        inference_method="mcem")
    
    SAVE_RESULTS_TO_CACHE_SUITE && save_longtest_result(result; force=true)
    
    return result
end

# =============================================================================
# Test Suite Execution
# =============================================================================

const EFFECT_TYPES = [:ph, :aft]
const COV_TYPES_SUITE = ["nocov", "fixed", "tvc"]

@testset "Unpenalized Spline Suite" begin
    
    @testset "Exact Data Tests" begin
        for effect in EFFECT_TYPES
            effect_str = effect == :ph ? "PH" : "AFT"
            @testset "$effect_str" begin
                for cov_type in COV_TYPES_SUITE
                    @testset "$cov_type" begin
                        result = run_spline_exact_test(effect, cov_type)
                        @test result.passed
                    end
                end
            end
        end
    end
    
    @testset "Panel + Markov Tests" begin
        for effect in EFFECT_TYPES
            effect_str = effect == :ph ? "PH" : "AFT"
            @testset "$effect_str" begin
                for cov_type in COV_TYPES_SUITE
                    @testset "$cov_type" begin
                        result = run_spline_panel_markov_test(effect, cov_type)
                        @test result.passed
                    end
                end
            end
        end
    end
    
    @testset "Panel + PhaseType Tests" begin
        for effect in EFFECT_TYPES
            effect_str = effect == :ph ? "PH" : "AFT"
            @testset "$effect_str" begin
                for cov_type in COV_TYPES_SUITE
                    @testset "$cov_type" begin
                        result = run_spline_panel_phasetype_test(effect, cov_type)
                        @test result.passed
                    end
                end
            end
        end
    end
    
end
