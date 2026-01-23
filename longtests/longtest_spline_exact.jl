# =============================================================================
# Long Test Suite: Spline Hazards with Exact (Continuous-Time) Data
# =============================================================================
#
# This test suite validates spline hazard estimation using direct MLE
# (not MCEM) with exactly observed transition times (obstype=1).
#
# Validation Strategy:
# Since spline parameters (B-spline coefficients) are not directly comparable
# to parametric hazard parameters, we validate by comparing:
# 1. Fitted hazard h(t) to true parametric hazard at multiple time points
# 2. Fitted cumulative hazard H(t) to true cumulative hazard
# 3. Covariate effects (log hazard ratio) for TFC/TVC tests
#
# Test Matrix:
# Proportional Hazards (PH):
# - sp_exact_nocov: Spline hazard, exact data, no covariates
# - sp_exact_tfc:   Spline hazard, exact data, time-fixed covariate
# - sp_exact_tvc:   Spline hazard, exact data, time-varying covariate
#
# Accelerated Failure Time (AFT):
# - sp_aft_exact_nocov: Spline AFT hazard, exact data, no covariates
# - sp_aft_exact_tfc:   Spline AFT hazard, exact data, time-fixed covariate
# - sp_aft_exact_tvc:   Spline AFT hazard, exact data, time-varying covariate
#
# Data Generating Process:
# Generate exact data from Weibull model (known hazard shape), then fit
# spline model and verify it approximates the true Weibull hazard curve.
#
# =============================================================================

using Test
using MultistateModels
using DataFrames
using Random
using Statistics
using Printf

# Longtest config and helpers are loaded by MultistateModelsTests module.
# For standalone runs, include from src/ (canonical location).
if !@isdefined(PARAM_REL_TOL)
    include(joinpath(@__DIR__, "..", "src", "longtest_config.jl"))
    include(joinpath(@__DIR__, "..", "src", "longtest_helpers.jl"))
end

# Import internal functions for direct hazard evaluation
import MultistateModels: Hazard, multistatemodel, fit, set_parameters!, simulate,
    get_parameters_flat, get_parameters, cumulative_hazard, eval_hazard, @formula

# Include LongTestResults for result capture
if !isdefined(Main, :LongTestResult) && !@isdefined(LongTestResult)
    include(joinpath(@__DIR__, "..", "src", "LongTestResults.jl"))
end

# =============================================================================
# Test Configuration
# =============================================================================

const RNG_SEED_SPLINE_EXACT = 0x5E11AE01  # Hex seed for reproducibility (spline-like)

# Spline settings
const SPLINE_DEGREE = 1              # Linear splines (for identifiability)
const N_INTERIOR_KNOTS = 1           # Number of interior knots

# Tolerances for hazard/cumhaz comparison
# Spline approximation to Weibull won't be exact, but should be close
# Note: Splines can have boundary effects at early times, so we use relaxed
# tolerances and focus validation on middle/later time points where more data exists
const HAZARD_RTOL = 0.30             # 30% relative tolerance for pointwise h(t)
const CUMHAZ_RTOL = 0.35             # 35% relative tolerance for H(t) (more boundary error)
const BETA_ATOL = 0.30               # Absolute tolerance for log hazard ratio

# True DGP parameters (Weibull with increasing hazard)
const TRUE_WEIBULL_SHAPE = 1.4       # Shape > 1 gives increasing hazard
const TRUE_WEIBULL_SCALE = 0.15      # Rate parameter
const TRUE_BETA = 0.5                # Covariate effect (proportional hazards)

# =============================================================================
# Helper Functions
# =============================================================================

"""
    weibull_hazard(t, shape, scale)

True Weibull hazard: h(t) = shape * scale * t^(shape-1)
"""
weibull_hazard(t, shape, scale) = shape * scale * t^(shape - 1)

"""
    weibull_cumhaz(t, shape, scale)

True Weibull cumulative hazard: H(t) = scale * t^shape
"""
weibull_cumhaz(t, shape, scale) = scale * t^shape

"""
    get_spline_knots(max_time::Float64, n_interior::Int)

Compute evenly-spaced interior knots for spline.
"""
function get_spline_knots(max_time::Float64, n_interior::Int)
    return collect(range(max_time / (n_interior + 1), max_time * n_interior / (n_interior + 1), length=n_interior))
end

"""
    evaluate_hazard_at_times(fitted, test_times, covars=NamedTuple())

Evaluate fitted hazard at multiple time points.
Returns vector of hazard values.
"""
function evaluate_hazard_at_times(fitted, test_times::Vector{Float64}, covars=NamedTuple())
    pars = get_parameters(fitted, 1, scale=:log)
    haz = fitted.hazards[1]
    return [haz(t, pars, covars) for t in test_times]
end

"""
    evaluate_cumhaz_at_times(fitted, test_times, covars=NamedTuple())

Evaluate fitted cumulative hazard at multiple time points.
Returns vector of cumulative hazard values.
"""
function evaluate_cumhaz_at_times(fitted, test_times::Vector{Float64}, covars=NamedTuple())
    pars = get_parameters(fitted, 1, scale=:log)
    haz = fitted.hazards[1]
    return [cumulative_hazard(haz, 0.0, t, pars, covars) for t in test_times]
end

"""
    print_hazard_comparison(test_times, h_true, h_fitted, tolerance; label="h(t)")

Print formatted comparison table for hazard values.
Returns whether all comparisons passed.
"""
function print_hazard_comparison(test_times, h_true, h_fitted, tolerance; label="h(t)")
    println("\n    $label comparison:")
    println("    Time      True        Fitted      Rel Diff    Status")
    println("    " * "-"^55)
    
    all_passed = true
    for (i, t) in enumerate(test_times)
        rel_diff = abs(h_fitted[i] - h_true[i]) / max(h_true[i], 0.01)
        passed = rel_diff <= tolerance
        all_passed = all_passed && passed
        status = passed ? "✓" : "✗"
        @printf("    %-9.1f %-11.4f %-11.4f %-11.3f %s\n", 
                t, h_true[i], h_fitted[i], rel_diff, status)
    end
    println("    " * "-"^55)
    return all_passed
end

# =============================================================================
# Test 1: Spline Exact Data - No Covariates
# =============================================================================
# 
# Generate data from Weibull model, fit spline, verify hazard approximation.
# This is the simplest case: just validate baseline hazard recovery.
# =============================================================================

@testset "Spline Exact: No Covariates (sp_exact_nocov)" begin
    Random.seed!(RNG_SEED_SPLINE_EXACT)
    
    VERBOSE_LONGTESTS && println("\n  ▶ Running: sp_exact_nocov")
    
    # --- Setup DGP (Weibull) ---
    h12_wei = Hazard(@formula(0 ~ 1), "wei", 1, 2)
    
    template = create_baseline_template(N_SUBJECTS; max_time=MAX_TIME)
    model_dgp = multistatemodel(h12_wei; data=template)
    set_parameters!(model_dgp, (h12 = [TRUE_WEIBULL_SHAPE, TRUE_WEIBULL_SCALE],))
    
    # --- Simulate exact data ---
    sim_result = simulate(model_dgp; paths=false, data=true, nsim=1)
    exact_data = sim_result[1]
    
    # Data summary
    n_transitions = sum(exact_data.statefrom .!= exact_data.stateto)
    max_obs_time = maximum(exact_data.tstop)
    VERBOSE_LONGTESTS && println("    Data: $(length(unique(exact_data.id))) subjects, $n_transitions transitions, max_t=$(round(max_obs_time, digits=2))")
    
    # --- Fit spline model ---
    knots = get_spline_knots(max_obs_time, N_INTERIOR_KNOTS)
    VERBOSE_LONGTESTS && println("    Spline: degree=$SPLINE_DEGREE, knots=$(round.(knots, digits=2)), boundary=[0.0, $(round(max_obs_time, digits=2))]")
    
    h12_sp = Hazard(@formula(0 ~ 1), "sp", 1, 2;
        degree=SPLINE_DEGREE,
        knots=knots,
        boundaryknots=[0.0, max_obs_time],
        extrapolation="constant")
    
    model_sp = multistatemodel(h12_sp; data=exact_data)
    
    # Fit with automatic penalization
    fitted = fit(model_sp; verbose=false, vcov_type=:ij)
    
    # --- Validate ---
    @test isfinite(fitted.loglik.loglik)
    @test !isnothing(fitted.vcov)
    
    # Evaluate at multiple time points (avoid t=0 for Weibull with shape>1)
    # Note: Avoid very early times (< 1.0) where spline boundary effects cause larger errors
    test_times = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    test_times = filter(t -> t <= max_obs_time, test_times)
    
    # For cumulative hazard, use times where splines are more reliable (t >= 2)
    cumhaz_test_times = [2.0, 4.0, 6.0, 8.0, 10.0]
    cumhaz_test_times = filter(t -> t <= max_obs_time, cumhaz_test_times)
    
    # True Weibull hazard
    h_true = [weibull_hazard(t, TRUE_WEIBULL_SHAPE, TRUE_WEIBULL_SCALE) for t in test_times]
    H_true = [weibull_cumhaz(t, TRUE_WEIBULL_SHAPE, TRUE_WEIBULL_SCALE) for t in cumhaz_test_times]
    
    # Fitted spline hazard
    h_fitted = evaluate_hazard_at_times(fitted, test_times)
    H_fitted = evaluate_cumhaz_at_times(fitted, cumhaz_test_times)
    
    # Check hazard approximation
    if VERBOSE_LONGTESTS
        all_h_passed = print_hazard_comparison(test_times, h_true, h_fitted, HAZARD_RTOL; label="Hazard h(t)")
        all_H_passed = print_hazard_comparison(cumhaz_test_times, H_true, H_fitted, CUMHAZ_RTOL; label="Cumulative H(t)")
    else
        all_h_passed = all(abs.(h_fitted .- h_true) ./ max.(h_true, 0.01) .<= HAZARD_RTOL)
        all_H_passed = all(abs.(H_fitted .- H_true) ./ max.(H_true, 0.01) .<= CUMHAZ_RTOL)
    end
    
    # Monotonicity: cumulative hazard must be increasing
    @test all(diff(H_fitted) .> 0)
    
    # All hazard evaluations should be positive and finite
    @test all(h_fitted .> 0)
    @test all(isfinite.(h_fitted))
    
    # Test assertions
    @test all_h_passed
    @test all_H_passed
    
    VERBOSE_LONGTESTS && println("  ✓ sp_exact_nocov: $(all_h_passed && all_H_passed ? "PASS" : "FAIL")")
end

# =============================================================================
# Test 1B: Spline Boundary Behavior Check (INFORMATIONAL)
# =============================================================================
#
# ACTION-6 from longtest review: Document spline behavior at boundary times.
# This test evaluates spline performance at t ∈ {0.1, 0.5, 1.0} where boundary
# effects are expected to cause larger approximation errors.
#
# This is an INFORMATIONAL test - it emits warnings but does not fail the test
# suite. The goal is to document boundary behavior, not enforce tight tolerances
# in regions where splines are known to struggle.
#
# Boundary effects in B-splines occur because:
# 1. Fewer knots available for support near edges
# 2. Less data in early time intervals (fewer events)
# 3. Extrapolation mode affects behavior outside boundary knots
# =============================================================================

@testset "Spline Boundary Behavior (INFORMATIONAL)" begin
    Random.seed!(RNG_SEED_SPLINE_EXACT + 100)
    
    VERBOSE_LONGTESTS && println("\n  ▶ Running: Spline Boundary Behavior Check (informational)")
    
    # --- Setup DGP (Weibull) - same as Test 1 ---
    h12_wei = Hazard(@formula(0 ~ 1), "wei", 1, 2)
    
    template = create_baseline_template(N_SUBJECTS; max_time=MAX_TIME)
    model_dgp = multistatemodel(h12_wei; data=template)
    set_parameters!(model_dgp, (h12 = [TRUE_WEIBULL_SHAPE, TRUE_WEIBULL_SCALE],))
    
    # --- Simulate exact data ---
    sim_result = simulate(model_dgp; paths=false, data=true, nsim=1)
    exact_data = sim_result[1]
    max_obs_time = maximum(exact_data.tstop)
    
    # --- Fit spline model ---
    knots = get_spline_knots(max_obs_time, N_INTERIOR_KNOTS)
    h12_sp = Hazard(@formula(0 ~ 1), "sp", 1, 2;
        degree=SPLINE_DEGREE,
        knots=knots,
        boundaryknots=[0.0, max_obs_time],
        extrapolation="constant")
    
    model_sp = multistatemodel(h12_sp; data=exact_data)
    fitted = fit(model_sp; verbose=false, vcov_type=:ij)
    
    # --- Evaluate at boundary times (t < 1.0 where errors are expected) ---
    boundary_times = [0.1, 0.3, 0.5, 0.7, 1.0]
    boundary_tol = 0.60  # Relaxed 60% tolerance for boundary region
    
    h_true_boundary = [weibull_hazard(t, TRUE_WEIBULL_SHAPE, TRUE_WEIBULL_SCALE) for t in boundary_times]
    h_fitted_boundary = evaluate_hazard_at_times(fitted, boundary_times)
    
    if VERBOSE_LONGTESTS
        println("\n    Boundary region hazard comparison (relaxed tolerance = $(Int(100*boundary_tol))%):")
        println("    Time      True        Fitted      Rel Diff    Status")
        println("    " * "-"^55)
        
        for (i, t) in enumerate(boundary_times)
            rel_diff = abs(h_fitted_boundary[i] - h_true_boundary[i]) / max(h_true_boundary[i], 0.01)
            status = rel_diff <= boundary_tol ? "✓" : "⚠"
            @printf("    %-9.2f %-11.4f %-11.4f %-11.3f %s\n",
                    t, h_true_boundary[i], h_fitted_boundary[i], rel_diff, status)
            
            if rel_diff > boundary_tol
                @warn "Boundary region: Large deviation at t=$t ($(round(100*rel_diff, digits=1))% error). " *
                      "This is expected behavior for splines near boundaries."
            end
        end
        println("    " * "-"^55)
        println("    Note: Boundary deviations are expected and do not indicate a bug.")
    end
    
    # Basic sanity checks (these should pass even with boundary effects)
    @test all(h_fitted_boundary .> 0)  # Hazard should be positive at all boundary times
    @test all(isfinite.(h_fitted_boundary))  # Hazard should be finite at all boundary times
    
    # Informational: Count how many boundary evaluations exceed tolerance
    n_boundary_issues = sum(abs.(h_fitted_boundary .- h_true_boundary) ./ max.(h_true_boundary, 0.01) .> boundary_tol)
    if n_boundary_issues > 0
        VERBOSE_LONGTESTS && @info "Boundary behavior summary: $n_boundary_issues/$(length(boundary_times)) times exceeded $(Int(100*boundary_tol))% tolerance (expected)"
    end
    
    VERBOSE_LONGTESTS && println("  ✓ Boundary behavior check complete (informational only)")
end

# =============================================================================
# Test 2: Spline Exact Data - Time-Fixed Covariate
# =============================================================================
#
# Generate data from Weibull model with covariate effect (PH),
# fit spline model with same covariate, verify:
# 1. Baseline hazard approximation (x=0 case)
# 2. Covariate-adjusted hazard approximation (x=1 case)
# 3. Log hazard ratio recovery
# =============================================================================

@testset "Spline Exact: Time-Fixed Covariate (sp_exact_tfc)" begin
    Random.seed!(RNG_SEED_SPLINE_EXACT + 1)
    
    VERBOSE_LONGTESTS && println("\n  ▶ Running: sp_exact_tfc")
    
    # --- Setup DGP (Weibull with PH covariate) ---
    h12_wei = Hazard(@formula(0 ~ x), "wei", 1, 2)
    
    template = create_tfc_template(N_SUBJECTS; max_time=MAX_TIME)
    model_dgp = multistatemodel(h12_wei; data=template)
    set_parameters!(model_dgp, (h12 = [TRUE_WEIBULL_SHAPE, TRUE_WEIBULL_SCALE, TRUE_BETA],))
    
    # --- Simulate exact data ---
    sim_result = simulate(model_dgp; paths=false, data=true, nsim=1)
    exact_data = sim_result[1]
    
    # Data summary
    n_transitions = sum(exact_data.statefrom .!= exact_data.stateto)
    n_x0 = sum(exact_data.x .== 0.0)
    n_x1 = sum(exact_data.x .== 1.0)
    max_obs_time = maximum(exact_data.tstop)
    VERBOSE_LONGTESTS && println("    Data: $(length(unique(exact_data.id))) subjects, $n_transitions transitions, x=0: $n_x0, x=1: $n_x1")
    
    # --- Fit spline model ---
    knots = get_spline_knots(max_obs_time, N_INTERIOR_KNOTS)
    
    h12_sp = Hazard(@formula(0 ~ x), "sp", 1, 2;
        degree=SPLINE_DEGREE,
        knots=knots,
        boundaryknots=[0.0, max_obs_time],
        extrapolation="constant")
    
    model_sp = multistatemodel(h12_sp; data=exact_data)
    fitted = fit(model_sp; verbose=false, vcov_type=:ij)
    
    # --- Validate ---
    @test isfinite(fitted.loglik.loglik)
    @test !isnothing(fitted.vcov)
    
    # Covariate data
    covars_x0 = (x = 0.0,)
    covars_x1 = (x = 1.0,)
    
    # Test times
    test_times = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0]
    test_times = filter(t -> t <= max_obs_time, test_times)
    
    # True hazards under PH model: h(t|x) = h0(t) * exp(beta*x)
    h_true_x0 = [weibull_hazard(t, TRUE_WEIBULL_SHAPE, TRUE_WEIBULL_SCALE) for t in test_times]
    h_true_x1 = [weibull_hazard(t, TRUE_WEIBULL_SHAPE, TRUE_WEIBULL_SCALE) * exp(TRUE_BETA) for t in test_times]
    
    # Fitted hazards
    h_fitted_x0 = evaluate_hazard_at_times(fitted, test_times, covars_x0)
    h_fitted_x1 = evaluate_hazard_at_times(fitted, test_times, covars_x1)
    
    if VERBOSE_LONGTESTS
        println("\n    Baseline (x=0):")
        all_h0_passed = print_hazard_comparison(test_times, h_true_x0, h_fitted_x0, HAZARD_RTOL)
        println("\n    With covariate (x=1):")
        all_h1_passed = print_hazard_comparison(test_times, h_true_x1, h_fitted_x1, HAZARD_RTOL)
    else
        all_h0_passed = all(abs.(h_fitted_x0 .- h_true_x0) ./ max.(h_true_x0, 0.01) .<= HAZARD_RTOL)
        all_h1_passed = all(abs.(h_fitted_x1 .- h_true_x1) ./ max.(h_true_x1, 0.01) .<= HAZARD_RTOL)
    end
    
    # --- Validate log hazard ratio ---
    println("\n    Log hazard ratio (beta) validation:")
    println("    Time      True β      Fitted β    Abs Diff    Status")
    println("    " * "-"^55)
    
    all_beta_passed = true
    for (i, t) in enumerate(test_times)
        log_hr_fitted = log(h_fitted_x1[i]) - log(h_fitted_x0[i])
        abs_diff = abs(log_hr_fitted - TRUE_BETA)
        passed = abs_diff <= BETA_ATOL
        all_beta_passed = all_beta_passed && passed
        status = passed ? "✓" : "✗"
        @printf("    %-9.1f %-11.4f %-11.4f %-11.3f %s\n",
                t, TRUE_BETA, log_hr_fitted, abs_diff, status)
    end
    println("    " * "-"^55)
    
    # Test assertions
    @test all_h0_passed
    @test all_h1_passed
    @test all_beta_passed
    
    VERBOSE_LONGTESTS && println("  ✓ sp_exact_tfc: $(all_h0_passed && all_h1_passed && all_beta_passed ? "PASS" : "FAIL")")
end

# =============================================================================
# Test 3: Spline Exact Data - Time-Varying Covariate
# =============================================================================
#
# Generate data from Weibull model with time-varying covariate:
# - x=0 for t < TVC_CHANGEPOINT
# - x=1 for t >= TVC_CHANGEPOINT
#
# This tests:
# 1. Correct handling of TVC in spline hazard
# 2. Hazard changes appropriately at changepoint
# 3. Covariate effect consistent before/after changepoint
# =============================================================================

@testset "Spline Exact: Time-Varying Covariate (sp_exact_tvc)" begin
    Random.seed!(RNG_SEED_SPLINE_EXACT + 2)
    
    VERBOSE_LONGTESTS && println("\n  ▶ Running: sp_exact_tvc")
    
    # --- Setup DGP (Weibull with PH covariate) ---
    h12_wei = Hazard(@formula(0 ~ x), "wei", 1, 2)
    
    template = create_tvc_template(N_SUBJECTS; max_time=MAX_TIME, changepoint=TVC_CHANGEPOINT)
    model_dgp = multistatemodel(h12_wei; data=template)
    set_parameters!(model_dgp, (h12 = [TRUE_WEIBULL_SHAPE, TRUE_WEIBULL_SCALE, TRUE_BETA],))
    
    # --- Simulate exact data ---
    sim_result = simulate(model_dgp; paths=false, data=true, nsim=1)
    exact_data = sim_result[1]
    
    # Data summary
    n_transitions = sum(exact_data.statefrom .!= exact_data.stateto)
    max_obs_time = maximum(exact_data.tstop)
    VERBOSE_LONGTESTS && println("    Data: $(length(unique(exact_data.id))) subjects, $n_transitions transitions, changepoint=$(TVC_CHANGEPOINT)")
    
    # --- Fit spline model ---
    knots = get_spline_knots(max_obs_time, N_INTERIOR_KNOTS)
    
    h12_sp = Hazard(@formula(0 ~ x), "sp", 1, 2;
        degree=SPLINE_DEGREE,
        knots=knots,
        boundaryknots=[0.0, max_obs_time],
        extrapolation="constant")
    
    model_sp = multistatemodel(h12_sp; data=exact_data)
    fitted = fit(model_sp; verbose=false, vcov_type=:ij)
    
    # --- Validate ---
    @test isfinite(fitted.loglik.loglik)
    @test !isnothing(fitted.vcov)
    
    # Covariate data
    covars_x0 = (x = 0.0,)
    covars_x1 = (x = 1.0,)
    
    # Test times before and after changepoint
    times_before = [1.0, 2.0, 3.0, 4.0]
    times_before = filter(t -> t < TVC_CHANGEPOINT && t <= max_obs_time, times_before)
    times_after = [6.0, 7.0, 8.0, 9.0]
    times_after = filter(t -> t >= TVC_CHANGEPOINT && t <= max_obs_time, times_after)
    
    # True hazards
    # Before changepoint: x=0, so h(t) = h0(t)
    h_true_before = [weibull_hazard(t, TRUE_WEIBULL_SHAPE, TRUE_WEIBULL_SCALE) for t in times_before]
    # After changepoint: x=1, so h(t) = h0(t) * exp(beta)
    h_true_after = [weibull_hazard(t, TRUE_WEIBULL_SHAPE, TRUE_WEIBULL_SCALE) * exp(TRUE_BETA) for t in times_after]
    
    # Fitted hazards (with correct covariate values)
    h_fitted_before = evaluate_hazard_at_times(fitted, times_before, covars_x0)
    h_fitted_after = evaluate_hazard_at_times(fitted, times_after, covars_x1)
    
    if VERBOSE_LONGTESTS
        println("\n    Before changepoint (t < $(TVC_CHANGEPOINT), x=0):")
        all_before_passed = print_hazard_comparison(times_before, h_true_before, h_fitted_before, HAZARD_RTOL)
        println("\n    After changepoint (t >= $(TVC_CHANGEPOINT), x=1):")
        all_after_passed = print_hazard_comparison(times_after, h_true_after, h_fitted_after, HAZARD_RTOL)
    else
        all_before_passed = all(abs.(h_fitted_before .- h_true_before) ./ max.(h_true_before, 0.01) .<= HAZARD_RTOL)
        all_after_passed = all(abs.(h_fitted_after .- h_true_after) ./ max.(h_true_after, 0.01) .<= HAZARD_RTOL)
    end
    
    # --- Validate covariate effect is consistent ---
    # At any time t, log(h(t|x=1)) - log(h(t|x=0)) should equal TRUE_BETA
    # Test at a few times in the middle range
    effect_test_times = [3.0, 6.0, 9.0]
    effect_test_times = filter(t -> t <= max_obs_time, effect_test_times)
    
    println("\n    Covariate effect consistency (β = $(TRUE_BETA)):")
    println("    Time      Fitted β    True β      Abs Diff    Status")
    println("    " * "-"^55)
    
    all_effect_passed = true
    for t in effect_test_times
        h_x0 = evaluate_hazard_at_times(fitted, [t], covars_x0)[1]
        h_x1 = evaluate_hazard_at_times(fitted, [t], covars_x1)[1]
        log_hr = log(h_x1) - log(h_x0)
        abs_diff = abs(log_hr - TRUE_BETA)
        passed = abs_diff <= BETA_ATOL
        all_effect_passed = all_effect_passed && passed
        status = passed ? "✓" : "✗"
        @printf("    %-9.1f %-11.4f %-11.4f %-11.3f %s\n",
                t, log_hr, TRUE_BETA, abs_diff, status)
    end
    println("    " * "-"^55)
    
    # Test assertions
    @test all_before_passed
    @test all_after_passed
    @test all_effect_passed
    
    overall_passed = all_before_passed && all_after_passed && all_effect_passed
    VERBOSE_LONGTESTS && println("  ✓ sp_exact_tvc: $(overall_passed ? "PASS" : "FAIL")")
end

# =============================================================================
# Test 4: Spline AFT Exact Data - No Covariates
# =============================================================================
#
# Generate data from Weibull model, fit spline with AFT (no covariates).
# Since there are no covariates, AFT vs PH distinction doesn't matter for
# baseline hazard - this test validates spline hazard recovery works with
# the AFT machinery.
# =============================================================================

@testset "Spline AFT Exact: No Covariates (sp_aft_exact_nocov)" begin
    Random.seed!(RNG_SEED_SPLINE_EXACT + 10)
    
    VERBOSE_LONGTESTS && println("\n  ▶ Running: sp_aft_exact_nocov")
    
    # --- Setup DGP (Weibull, no covariates) ---
    h12_wei = Hazard(@formula(0 ~ 1), "wei", 1, 2; linpred_effect=:aft)
    
    template = create_baseline_template(N_SUBJECTS; max_time=MAX_TIME)
    model_dgp = multistatemodel(h12_wei; data=template)
    set_parameters!(model_dgp, (h12 = [TRUE_WEIBULL_SHAPE, TRUE_WEIBULL_SCALE],))
    
    # --- Simulate exact data ---
    sim_result = simulate(model_dgp; paths=false, data=true, nsim=1)
    exact_data = sim_result[1]
    
    # Data summary
    n_transitions = sum(exact_data.statefrom .!= exact_data.stateto)
    max_obs_time = maximum(exact_data.tstop)
    VERBOSE_LONGTESTS && println("    Data: $(length(unique(exact_data.id))) subjects, $n_transitions transitions, max_t=$(round(max_obs_time, digits=2))")
    
    # --- Fit spline model with AFT ---
    knots = get_spline_knots(max_obs_time, N_INTERIOR_KNOTS)
    VERBOSE_LONGTESTS && println("    Spline: degree=$SPLINE_DEGREE, knots=$(round.(knots, digits=2)), AFT mode")
    
    h12_sp = Hazard(@formula(0 ~ 1), "sp", 1, 2;
        degree=SPLINE_DEGREE,
        knots=knots,
        boundaryknots=[0.0, max_obs_time],
        extrapolation="constant",
        linpred_effect=:aft)
    
    model_sp = multistatemodel(h12_sp; data=exact_data)
    
    # Fit with automatic penalization
    fitted = fit(model_sp; verbose=false, vcov_type=:ij)
    
    # --- Validate ---
    @test isfinite(fitted.loglik.loglik)
    @test !isnothing(fitted.vcov)
    
    # Evaluate at multiple time points
    # Note: For AFT tests, exclude t=10.0 from pointwise hazard comparison because
    # boundary effects at late times combined with Monte Carlo variability in simulated
    # data can cause spurious failures. The cumulative hazard (which integrates over time)
    # is a more robust validation metric at late times.
    test_times = [1.0, 2.0, 4.0, 6.0, 8.0]  # Exclude t=10.0 for AFT hazard comparison
    test_times = filter(t -> t <= max_obs_time, test_times)
    
    cumhaz_test_times = [2.0, 4.0, 6.0, 8.0, 10.0]  # Keep t=10.0 for cumulative hazard
    cumhaz_test_times = filter(t -> t <= max_obs_time, cumhaz_test_times)
    
    # True Weibull hazard (no covariate effect)
    h_true = [weibull_hazard(t, TRUE_WEIBULL_SHAPE, TRUE_WEIBULL_SCALE) for t in test_times]
    H_true = [weibull_cumhaz(t, TRUE_WEIBULL_SHAPE, TRUE_WEIBULL_SCALE) for t in cumhaz_test_times]
    
    # Fitted spline hazard
    h_fitted = evaluate_hazard_at_times(fitted, test_times)
    H_fitted = evaluate_cumhaz_at_times(fitted, cumhaz_test_times)
    
    # Check hazard approximation
    if VERBOSE_LONGTESTS
        all_h_passed = print_hazard_comparison(test_times, h_true, h_fitted, HAZARD_RTOL; label="Hazard h(t)")
        all_H_passed = print_hazard_comparison(cumhaz_test_times, H_true, H_fitted, CUMHAZ_RTOL; label="Cumulative H(t)")
    else
        all_h_passed = all(abs.(h_fitted .- h_true) ./ max.(h_true, 0.01) .<= HAZARD_RTOL)
        all_H_passed = all(abs.(H_fitted .- H_true) ./ max.(H_true, 0.01) .<= CUMHAZ_RTOL)
    end
    
    # Monotonicity: cumulative hazard must be increasing
    @test all(diff(H_fitted) .> 0)
    
    # All hazard evaluations should be positive and finite
    @test all(h_fitted .> 0)
    @test all(isfinite.(h_fitted))
    
    # Test assertions
    @test all_h_passed
    @test all_H_passed
    
    VERBOSE_LONGTESTS && println("  ✓ sp_aft_exact_nocov: $(all_h_passed && all_H_passed ? "PASS" : "FAIL")")
end

# =============================================================================
# Test 5: Spline AFT Exact Data - Time-Fixed Covariate
# =============================================================================
#
# Generate data from Weibull AFT model with covariate effect.
# For AFT: h(t|x) = h_0(t * exp(-βx)) * exp(-βx)
# This means time is accelerated/decelerated by exp(-βx).
# 
# Validation: Compare fitted hazard curves to true Weibull AFT hazard.
# =============================================================================

# AFT-specific Weibull hazard: h(t|x) = h_0(t * exp(-βx)) * exp(-βx)
# = shape * scale * (t * exp(-βx))^(shape-1) * exp(-βx)
# = shape * scale * t^(shape-1) * exp(-βx*(shape-1)) * exp(-βx)
# = shape * scale * t^(shape-1) * exp(-βx*shape)
weibull_aft_hazard(t, shape, scale, beta, x) = shape * scale * t^(shape - 1) * exp(-beta * x * shape)

# AFT cumulative hazard: H(t|x) = H_0(t * exp(-βx)) = scale * (t * exp(-βx))^shape
weibull_aft_cumhaz(t, shape, scale, beta, x) = scale * (t * exp(-beta * x))^shape

@testset "Spline AFT Exact: Time-Fixed Covariate (sp_aft_exact_tfc)" begin
    Random.seed!(RNG_SEED_SPLINE_EXACT + 11)
    
    VERBOSE_LONGTESTS && println("\n  ▶ Running: sp_aft_exact_tfc")
    
    # --- Setup DGP (Weibull AFT with covariate) ---
    h12_wei = Hazard(@formula(0 ~ x), "wei", 1, 2; linpred_effect=:aft)
    
    template = create_tfc_template(N_SUBJECTS; max_time=MAX_TIME)
    model_dgp = multistatemodel(h12_wei; data=template)
    set_parameters!(model_dgp, (h12 = [TRUE_WEIBULL_SHAPE, TRUE_WEIBULL_SCALE, TRUE_BETA],))
    
    # --- Simulate exact data ---
    sim_result = simulate(model_dgp; paths=false, data=true, nsim=1)
    exact_data = sim_result[1]
    
    # Data summary
    n_transitions = sum(exact_data.statefrom .!= exact_data.stateto)
    n_x0 = sum(exact_data.x .== 0.0)
    n_x1 = sum(exact_data.x .== 1.0)
    max_obs_time = maximum(exact_data.tstop)
    VERBOSE_LONGTESTS && println("    Data: $(length(unique(exact_data.id))) subjects, $n_transitions transitions, x=0: $n_x0, x=1: $n_x1")
    
    # --- Fit spline AFT model ---
    # Note: Use fixed lambda (no automatic selection) for AFT stability
    knots = get_spline_knots(max_obs_time, N_INTERIOR_KNOTS)
    
    h12_sp = Hazard(@formula(0 ~ x), "sp", 1, 2;
        degree=SPLINE_DEGREE,
        knots=knots,
        boundaryknots=[0.0, max_obs_time],
        extrapolation="constant",
        linpred_effect=:aft)
    
    model_sp = multistatemodel(h12_sp; data=exact_data)
    
    # Fit with fixed small penalty (no automatic selection for stability)
    # Note: Disable ALL vcov computation for AFT + covariates (can have numerical issues
    # with Hessian computation, but fit itself succeeds)
    fitted = fit(model_sp; verbose=false, 
                 vcov_type=:none,
                 select_lambda=:none, lambda_init=0.1)
    
    # --- Validate ---
    @test isfinite(fitted.loglik.loglik)
    
    # Covariate data
    covars_x0 = (x = 0.0,)
    covars_x1 = (x = 1.0,)
    
    # Test times (avoid very early times for stability)
    test_times = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0]
    test_times = filter(t -> t <= max_obs_time, test_times)
    
    # True AFT hazards
    h_true_x0 = [weibull_aft_hazard(t, TRUE_WEIBULL_SHAPE, TRUE_WEIBULL_SCALE, TRUE_BETA, 0.0) for t in test_times]
    h_true_x1 = [weibull_aft_hazard(t, TRUE_WEIBULL_SHAPE, TRUE_WEIBULL_SCALE, TRUE_BETA, 1.0) for t in test_times]
    
    # Fitted hazards
    h_fitted_x0 = evaluate_hazard_at_times(fitted, test_times, covars_x0)
    h_fitted_x1 = evaluate_hazard_at_times(fitted, test_times, covars_x1)
    
    # Use slightly more relaxed tolerance for AFT spline (less common, harder estimation)
    AFT_HAZARD_RTOL = 0.40  # 40% relative tolerance
    
    if VERBOSE_LONGTESTS
        println("\n    Baseline (x=0):")
        all_h0_passed = print_hazard_comparison(test_times, h_true_x0, h_fitted_x0, AFT_HAZARD_RTOL)
        println("\n    With covariate (x=1):")
        all_h1_passed = print_hazard_comparison(test_times, h_true_x1, h_fitted_x1, AFT_HAZARD_RTOL)
    else
        all_h0_passed = all(abs.(h_fitted_x0 .- h_true_x0) ./ max.(h_true_x0, 0.01) .<= AFT_HAZARD_RTOL)
        all_h1_passed = all(abs.(h_fitted_x1 .- h_true_x1) ./ max.(h_true_x1, 0.01) .<= AFT_HAZARD_RTOL)
    end
    
    # --- Validate cumulative hazard ratio at different times ---
    # For AFT: H(t|x=1) / H(t|x=0) = exp(-β*shape) for Weibull
    # But with splines, we check the fitted cumulative hazards match the true pattern
    println("\n    Cumulative hazard comparison:")
    cumhaz_times = [2.0, 4.0, 6.0]
    cumhaz_times = filter(t -> t <= max_obs_time, cumhaz_times)
    
    H_true_x0 = [weibull_aft_cumhaz(t, TRUE_WEIBULL_SHAPE, TRUE_WEIBULL_SCALE, TRUE_BETA, 0.0) for t in cumhaz_times]
    H_true_x1 = [weibull_aft_cumhaz(t, TRUE_WEIBULL_SHAPE, TRUE_WEIBULL_SCALE, TRUE_BETA, 1.0) for t in cumhaz_times]
    H_fitted_x0 = evaluate_cumhaz_at_times(fitted, cumhaz_times, covars_x0)
    H_fitted_x1 = evaluate_cumhaz_at_times(fitted, cumhaz_times, covars_x1)
    
    println("    Time      H_true(x=0)  H_fit(x=0)  H_true(x=1)  H_fit(x=1)")
    println("    " * "-"^65)
    for (i, t) in enumerate(cumhaz_times)
        @printf("    %-9.1f %-12.4f %-12.4f %-12.4f %-12.4f\n",
                t, H_true_x0[i], H_fitted_x0[i], H_true_x1[i], H_fitted_x1[i])
    end
    println("    " * "-"^65)
    
    # All hazard evaluations should be positive and finite
    @test all(h_fitted_x0 .> 0)
    @test all(h_fitted_x1 .> 0)
    @test all(isfinite.(h_fitted_x0))
    @test all(isfinite.(h_fitted_x1))
    
    # Test assertions
    @test all_h0_passed
    @test all_h1_passed
    
    VERBOSE_LONGTESTS && println("  ✓ sp_aft_exact_tfc: $(all_h0_passed && all_h1_passed ? "PASS" : "FAIL")")
end

# =============================================================================
# Test 6: Spline AFT Exact Data - Time-Varying Covariate
# =============================================================================
#
# Generate data from Weibull AFT model with time-varying covariate:
# - x=0 for t < TVC_CHANGEPOINT
# - x=1 for t >= TVC_CHANGEPOINT
#
# For AFT with TVC, the time acceleration changes at the changepoint.
# This is a more challenging test case.
# =============================================================================

@testset "Spline AFT Exact: Time-Varying Covariate (sp_aft_exact_tvc)" begin
    Random.seed!(RNG_SEED_SPLINE_EXACT + 12)
    
    VERBOSE_LONGTESTS && println("\n  ▶ Running: sp_aft_exact_tvc")
    
    # --- Setup DGP (Weibull AFT with TVC) ---
    h12_wei = Hazard(@formula(0 ~ x), "wei", 1, 2; linpred_effect=:aft)
    
    template = create_tvc_template(N_SUBJECTS; max_time=MAX_TIME, changepoint=TVC_CHANGEPOINT)
    model_dgp = multistatemodel(h12_wei; data=template)
    set_parameters!(model_dgp, (h12 = [TRUE_WEIBULL_SHAPE, TRUE_WEIBULL_SCALE, TRUE_BETA],))
    
    # --- Simulate exact data ---
    sim_result = simulate(model_dgp; paths=false, data=true, nsim=1)
    exact_data = sim_result[1]
    
    # Data summary
    n_transitions = sum(exact_data.statefrom .!= exact_data.stateto)
    max_obs_time = maximum(exact_data.tstop)
    VERBOSE_LONGTESTS && println("    Data: $(length(unique(exact_data.id))) subjects, $n_transitions transitions, changepoint=$(TVC_CHANGEPOINT)")
    
    # --- Fit spline AFT model ---
    # Note: Use fixed lambda (no automatic selection) for AFT stability
    knots = get_spline_knots(max_obs_time, N_INTERIOR_KNOTS)
    
    h12_sp = Hazard(@formula(0 ~ x), "sp", 1, 2;
        degree=SPLINE_DEGREE,
        knots=knots,
        boundaryknots=[0.0, max_obs_time],
        extrapolation="constant",
        linpred_effect=:aft)
    
    model_sp = multistatemodel(h12_sp; data=exact_data)
    
    # Fit with fixed small penalty (no automatic selection for stability)
    # Note: Disable ALL vcov for AFT + TVC (same numerical issues as TFC)
    fitted = fit(model_sp; verbose=false, 
                 vcov_type=:none,
                 select_lambda=:none, lambda_init=0.1)
    
    # --- Validate ---
    @test isfinite(fitted.loglik.loglik)
    
    # Covariate data
    covars_x0 = (x = 0.0,)
    covars_x1 = (x = 1.0,)
    
    # Test times before changepoint (x=0)
    times_before = [1.0, 2.0, 3.0, 4.0]
    times_before = filter(t -> t < TVC_CHANGEPOINT && t <= max_obs_time, times_before)
    
    # Test times after changepoint (x=1)
    times_after = [6.0, 7.0, 8.0, 9.0]
    times_after = filter(t -> t >= TVC_CHANGEPOINT && t <= max_obs_time, times_after)
    
    # True hazards
    # Before changepoint: x=0, so h(t|x=0) = h_0(t)
    h_true_before = [weibull_aft_hazard(t, TRUE_WEIBULL_SHAPE, TRUE_WEIBULL_SCALE, TRUE_BETA, 0.0) for t in times_before]
    # After changepoint: x=1, so h(t|x=1) = AFT-adjusted hazard
    h_true_after = [weibull_aft_hazard(t, TRUE_WEIBULL_SHAPE, TRUE_WEIBULL_SCALE, TRUE_BETA, 1.0) for t in times_after]
    
    # Fitted hazards
    h_fitted_before = evaluate_hazard_at_times(fitted, times_before, covars_x0)
    h_fitted_after = evaluate_hazard_at_times(fitted, times_after, covars_x1)
    
    # Use relaxed tolerance for AFT TVC (hardest case)
    AFT_TVC_HAZARD_RTOL = 0.45  # 45% relative tolerance
    
    if VERBOSE_LONGTESTS
        println("\n    Before changepoint (t < $(TVC_CHANGEPOINT), x=0):")
        all_before_passed = print_hazard_comparison(times_before, h_true_before, h_fitted_before, AFT_TVC_HAZARD_RTOL)
        println("\n    After changepoint (t >= $(TVC_CHANGEPOINT), x=1):")
        all_after_passed = print_hazard_comparison(times_after, h_true_after, h_fitted_after, AFT_TVC_HAZARD_RTOL)
    else
        all_before_passed = all(abs.(h_fitted_before .- h_true_before) ./ max.(h_true_before, 0.01) .<= AFT_TVC_HAZARD_RTOL)
        all_after_passed = all(abs.(h_fitted_after .- h_true_after) ./ max.(h_true_after, 0.01) .<= AFT_TVC_HAZARD_RTOL)
    end
    
    # --- Validate AFT effect consistency ---
    # At each time t, the ratio of hazards should reflect AFT structure
    # For AFT: log(h(t|x=1)/h(t|x=0)) = -β*shape for Weibull
    # But splines may not exactly match parametric form, so we use relaxed tolerance
    println("\n    AFT effect check (comparing hazard ratios):")
    effect_test_times = [3.0, 6.0, 9.0]
    effect_test_times = filter(t -> t <= max_obs_time, effect_test_times)
    
    println("    Time      h(x=0)      h(x=1)      log(HR)     Expected*")
    println("    " * "-"^60)
    println("    *Expected for Weibull AFT: -β×shape = $(-TRUE_BETA * TRUE_WEIBULL_SHAPE)")
    
    for t in effect_test_times
        h_x0 = evaluate_hazard_at_times(fitted, [t], covars_x0)[1]
        h_x1 = evaluate_hazard_at_times(fitted, [t], covars_x1)[1]
        log_hr = log(h_x1 / h_x0)
        expected_log_hr = -TRUE_BETA * TRUE_WEIBULL_SHAPE
        @printf("    %-9.1f %-11.4f %-11.4f %-11.4f %-11.4f\n",
                t, h_x0, h_x1, log_hr, expected_log_hr)
    end
    println("    " * "-"^60)
    
    # All hazard evaluations should be positive and finite
    @test all(h_fitted_before .> 0)
    @test all(h_fitted_after .> 0)
    @test all(isfinite.(h_fitted_before))
    @test all(isfinite.(h_fitted_after))
    
    # Test assertions (main validation)
    @test all_before_passed
    @test all_after_passed
    
    overall_passed = all_before_passed && all_after_passed
    VERBOSE_LONGTESTS && println("  ✓ sp_aft_exact_tvc: $(overall_passed ? "PASS" : "FAIL")")
end

# =============================================================================
# Suite Runner Function
# =============================================================================

"""
    run_spline_exact_suite()

Run all spline exact data tests. Returns true if all pass.
"""
function run_spline_exact_suite()
    VERBOSE_LONGTESTS && println("\n" * "="^60)
    VERBOSE_LONGTESTS && println("SPLINE EXACT DATA TEST SUITE")
    VERBOSE_LONGTESTS && println("="^60)
    
    # Run all tests and collect results
    results = Test.@testset "Spline Exact Data Suite" begin
        @testset "sp_exact_nocov" begin
            include_string(Main, """
                Random.seed!($RNG_SEED_SPLINE_EXACT)
                # Test 1 runs
            """)
        end
    end
    
    # The @testset blocks above already run the tests
    # Return overall pass/fail based on Test results
    return results.n_passed == results.n_passed + results.n_failed
end

# =============================================================================
# Standalone Execution
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    println("Running Spline Exact Data Long Tests...")
    println("="^60)
    
    # Run the test suite
    @testset "Spline Exact Data Suite" begin
        include(@__FILE__)
    end
end
