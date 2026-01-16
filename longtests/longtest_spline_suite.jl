# =============================================================================
# Unpenalized Spline Long Test Suite
# =============================================================================
#
# Comprehensive test suite covering unpenalized spline hazard inference.
# Tests: (exact × panel) × (PH × AFT) × (Markov × PhaseType surrogate) × (nocov × fixed × tvc)
#
# Test Matrix:
#   Exact data: 2 effect types × 3 covariate types = 6 tests (no surrogate needed)
#   Panel data: 2 effect types × 2 surrogates × 3 covariate types = 12 tests
#   Total: 18 tests
#
# Naming Convention: sp_{effect}{data}{surrogate}_{covariate}
#   - Effect: ph or aft
#   - Data: exact or panel
#   - Surrogate (panel only): markov or phasetype
#   - Covariate: nocov, fixed, tvc
#
# Data Generating Process:
#   Generate data from spline models with known B-spline coefficients.
#   The true spline coefficients produce hazard curves with realistic shapes
#   (gradually increasing hazard). We use the same spline model for both DGP
#   and fitting so we can directly compare estimated coefficients to true.
#
# Model Structure:
#   3-state progressive model: 1 → 2 → 3 (NO 1→3 transitions)
#   All subjects start in state 1 at time 0
#
# Spline Configuration:
#   degree=3 (cubic)
#   n_interior_knots=2 (giving 6 basis functions = degree + n_interior_knots + 1)
#   boundaryknots=[0.0, MAX_TIME]
#   extrapolation="constant"
#   No penalization (λ=0)
#
# =============================================================================

using Test
using MultistateModels
using DataFrames
using Random
using LinearAlgebra
using Statistics

import MultistateModels: Hazard, @formula, multistatemodel, fit, set_parameters!, 
    simulate, get_parameters_flat, get_parameters

# Configuration - include for standalone runs
if !isdefined(Main, :VERBOSE_LONGTESTS) && !@isdefined(VERBOSE_LONGTESTS)
    include(joinpath(@__DIR__, "longtest_config.jl"))
end

# Helpers - check for a function defined in longtest_helpers.jl
if !isdefined(Main, :create_baseline_template) && !@isdefined(create_baseline_template)
    include(joinpath(@__DIR__, "longtest_helpers.jl"))
end

# LongTestResults (include for standalone runs)
if !isdefined(Main, :LongTestResult) && !@isdefined(LongTestResult)
    include(joinpath(@__DIR__, "..", "src", "LongTestResults.jl"))
end

# =============================================================================
# Spline-Specific Configuration
# =============================================================================

# Spline structure
const SPLINE_DEGREE_SUITE = 3                  # Cubic splines
const N_INTERIOR_KNOTS_SUITE = 2               # Number of interior knots
const SPLINE_BOUNDARYKNOTS = [0.0, MAX_TIME]   # [0.0, 15.0]

# True covariate effect (log hazard ratio)
const TRUE_SPLINE_BETA = 0.5  # Same as parametric tests

# True spline coefficients (NATURAL SCALE - must be non-negative!)
# With degree=3 and 2 interior knots: MultistateModels uses 4 basis functions
# (n_interior_knots + 2, which includes boundary constraints)
# These coefficients represent the HAZARD directly (not log-hazard)
# Values chosen to give hazard rates in reasonable range ~0.05-0.20
const TRUE_SPLINE_COEFS_H12 = [0.08, 0.10, 0.14, 0.18]  # Gradually increasing hazard
const TRUE_SPLINE_COEFS_H23 = [0.06, 0.08, 0.11, 0.14]  # Gradually increasing hazard

# Tolerances for spline coefficient recovery
# Spline coefficients can have more variability than parametric models
const SPLINE_COEF_RTOL = 0.35        # 35% relative tolerance for coefficients

# MCEM settings for panel tests
const SPLINE_MCEM_TOL = 0.05         # MCEM convergence tolerance
const SPLINE_MCEM_MAX_ITER = 25      # Maximum MCEM iterations

# Surrogate comparison tolerances
const PROPOSAL_COMPARISON_TOL = 0.35     # Tolerance for surrogate comparison
const BETA_COMPARISON_TOL_REL = 0.20     # Tighter tolerance for beta comparison

# Panel observation times for spline tests
const SPLINE_PANEL_TIMES = [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0]

# =============================================================================
# Helper Functions for Spline Tests
# =============================================================================

"""
    get_spline_knots(n_interior::Int, boundary_knots::Vector{Float64})

Compute evenly-spaced interior knots for spline.
"""
function get_spline_knots(n_interior::Int, boundary_knots::Vector{Float64})
    t_min, t_max = boundary_knots
    return collect(range(t_min + (t_max - t_min) / (n_interior + 1), 
                         t_max - (t_max - t_min) / (n_interior + 1), 
                         length=n_interior))
end

"""
    get_spline_hazard_specs(effect_type::String, covariate_type::String)

Get spline hazard specifications for the given effect type and covariate type.
Returns tuple of (h12, h23) Hazard objects.
"""
function get_spline_hazard_specs(effect_type::String, covariate_type::String)
    formula = if covariate_type == "nocov"
        @formula(0 ~ 1)
    else
        @formula(0 ~ x)
    end
    
    knots = get_spline_knots(N_INTERIOR_KNOTS_SUITE, SPLINE_BOUNDARYKNOTS)
    
    h12 = Hazard(formula, "sp", 1, 2;
        degree=SPLINE_DEGREE_SUITE,
        knots=knots,
        boundaryknots=SPLINE_BOUNDARYKNOTS,
        extrapolation="constant",
        linpred_effect = effect_type == "aft" ? :aft : :ph)
    
    h23 = Hazard(formula, "sp", 2, 3;
        degree=SPLINE_DEGREE_SUITE,
        knots=knots,
        boundaryknots=SPLINE_BOUNDARYKNOTS,
        extrapolation="constant",
        linpred_effect = effect_type == "aft" ? :aft : :ph)
    
    return (h12, h23)
end

"""
    get_spline_true_params(covariate_type::String)

Get true parameter values for spline model.
Returns NamedTuple with h12 and h23 coefficient vectors.
"""
function get_spline_true_params(covariate_type::String)
    has_covariate = covariate_type != "nocov"
    
    h12 = has_covariate ? vcat(TRUE_SPLINE_COEFS_H12, TRUE_SPLINE_BETA) : TRUE_SPLINE_COEFS_H12
    h23 = has_covariate ? vcat(TRUE_SPLINE_COEFS_H23, TRUE_SPLINE_BETA) : TRUE_SPLINE_COEFS_H23
    
    return (h12 = h12, h23 = h23)
end

"""
    get_spline_param_names(covariate_type::String)

Get parameter names for display/comparison.
"""
function get_spline_param_names(covariate_type::String)
    has_covariate = covariate_type != "nocov"
    n_coefs = length(TRUE_SPLINE_COEFS_H12)
    
    names = String[]
    
    # h12 spline coefficients
    for i in 1:n_coefs
        push!(names, "h12_coef_$i")
    end
    has_covariate && push!(names, "h12_beta")
    
    # h23 spline coefficients  
    for i in 1:n_coefs
        push!(names, "h23_coef_$i")
    end
    has_covariate && push!(names, "h23_beta")
    
    return names
end

"""
    get_spline_beta_param_names(covariate_type::String)

Get names of beta (covariate) parameters for tolerance checking.
"""
function get_spline_beta_param_names(covariate_type::String)
    if covariate_type == "nocov"
        return String[]
    end
    return ["h12_beta", "h23_beta"]
end

"""
    generate_spline_data(hazard_specs, true_params, covariate_type::String)

Generate exact observation data from spline model.
"""
function generate_spline_data(hazard_specs, true_params, covariate_type::String)
    # Create template based on covariate type
    template = if covariate_type == "nocov"
        create_baseline_template(N_SUBJECTS; max_time=MAX_TIME)
    elseif covariate_type == "fixed"
        create_tfc_template(N_SUBJECTS; max_time=MAX_TIME)
    else  # tvc
        create_tvc_template(N_SUBJECTS; max_time=MAX_TIME)
    end
    
    # Build model and simulate
    model = multistatemodel(hazard_specs...; data=template)
    set_parameters!(model, true_params)
    
    sim_data = simulate(model; data=true, paths=false, nsim=1)[1]
    
    return sim_data
end

"""
    generate_spline_panel_data(hazard_specs, true_params, covariate_type::String, panel_times::Vector{Float64})

Generate panel observation data from spline model using simulate() with obstype_by_transition.
"""
function generate_spline_panel_data(hazard_specs, true_params, covariate_type::String, panel_times::Vector{Float64})
    nobs = length(panel_times) - 1
    
    # Create template with panel observation structure
    if covariate_type == "nocov"
        template = DataFrame(
            id = repeat(1:N_SUBJECTS, inner=nobs),
            tstart = repeat(panel_times[1:end-1], N_SUBJECTS),
            tstop = repeat(panel_times[2:end], N_SUBJECTS),
            statefrom = ones(Int, N_SUBJECTS * nobs),
            stateto = ones(Int, N_SUBJECTS * nobs),
            obstype = fill(2, N_SUBJECTS * nobs)
        )
    elseif covariate_type == "fixed"
        x_vals = rand([0.0, 1.0], N_SUBJECTS)
        template = DataFrame(
            id = repeat(1:N_SUBJECTS, inner=nobs),
            tstart = repeat(panel_times[1:end-1], N_SUBJECTS),
            tstop = repeat(panel_times[2:end], N_SUBJECTS),
            statefrom = ones(Int, N_SUBJECTS * nobs),
            stateto = ones(Int, N_SUBJECTS * nobs),
            obstype = fill(2, N_SUBJECTS * nobs),
            x = repeat(x_vals, inner=nobs)
        )
    else  # tvc
        cp = TVC_CHANGEPOINT
        rows = []
        for subj in 1:N_SUBJECTS
            for i in 1:(length(panel_times) - 1)
                t_start = panel_times[i]
                t_stop = panel_times[i + 1]
                
                if t_start < cp && t_stop > cp
                    push!(rows, (id=subj, tstart=t_start, tstop=cp, statefrom=1, stateto=1, obstype=2, x=0.0))
                    push!(rows, (id=subj, tstart=cp, tstop=t_stop, statefrom=1, stateto=1, obstype=2, x=1.0))
                else
                    x = t_start >= cp ? 1.0 : 0.0
                    push!(rows, (id=subj, tstart=t_start, tstop=t_stop, statefrom=1, stateto=1, obstype=2, x=x))
                end
            end
        end
        template = DataFrame(rows)
    end
    
    # Build model
    model = multistatemodel(hazard_specs...; data=template)
    set_parameters!(model, true_params)
    
    # Simulate with mixed observation types
    obstype_map = Dict(1 => 2, 2 => 1)  # h12 panel, h23 exact
    sim_result = simulate(model; paths=false, data=true, nsim=1, autotmax=false,
                         obstype_by_transition=obstype_map)
    
    return sim_result[1, 1]
end

# =============================================================================
# Test Functions
# =============================================================================

"""
    run_spline_exact_test(effect_type::String, covariate_type::String) -> LongTestResult

Run exact data test for spline model with given effect and covariate type.
"""
function run_spline_exact_test(effect_type::String, covariate_type::String)
    test_name = "sp_$(effect_type)_exact_$(covariate_type)"
    
    VERBOSE_LONGTESTS && @info "  Running: $test_name"
    
    # Get specifications
    hazard_specs = get_spline_hazard_specs(effect_type, covariate_type)
    true_params = get_spline_true_params(covariate_type)
    param_names = get_spline_param_names(covariate_type)
    beta_names = get_spline_beta_param_names(covariate_type)
    
    # Generate data with test-specific seed
    test_seed = hash(("sp", effect_type, "exact", covariate_type, RNG_SEED))
    Random.seed!(test_seed)
    data = generate_spline_data(hazard_specs, true_params, covariate_type)
    
    # Verify event rate
    if covariate_type == "nocov"
        VERBOSE_LONGTESTS && verify_event_rate_simple(data; verbose=true)
    end
    
    # Fit model (no penalty for unpenalized splines)
    model = multistatemodel(hazard_specs...; data=data)
    fitted = fit(model; verbose=false, compute_vcov=true)
    
    # Capture results
    result = capture_longtest_result!(
        test_name,
        fitted,
        true_params,
        param_names,
        hazard_specs;
        hazard_family = "sp",
        data_type = "exact",
        covariate_type = covariate_type,
        data = data,
        beta_param_names = beta_names,
        shape_param_names = String[]  # Spline coefficients, not shapes
    )
    
    VERBOSE_LONGTESTS && @info "    Result: $(result.passed ? "PASS" : "FAIL")"
    
    return result
end

"""
    run_spline_panel_test(effect_type::String, covariate_type::String, surrogate_type::String) -> LongTestResult

Run panel data test for spline model with given effect, covariate, and surrogate type.
"""
function run_spline_panel_test(effect_type::String, covariate_type::String, surrogate_type::String)
    test_name = "sp_$(effect_type)_panel_$(surrogate_type)_$(covariate_type)"
    
    VERBOSE_LONGTESTS && @info "  Running: $test_name"
    
    # Get specifications
    hazard_specs = get_spline_hazard_specs(effect_type, covariate_type)
    true_params = get_spline_true_params(covariate_type)
    param_names = get_spline_param_names(covariate_type)
    beta_names = get_spline_beta_param_names(covariate_type)
    
    # Generate panel data with test-specific seed
    test_seed = hash(("sp", effect_type, "panel", surrogate_type, covariate_type, RNG_SEED))
    Random.seed!(test_seed)
    panel_data = generate_spline_panel_data(hazard_specs, true_params, covariate_type, SPLINE_PANEL_TIMES)
    
    # Determine surrogate
    surrogate = surrogate_type == "markov" ? :markov : :phasetype
    
    # Fit model with MCEM
    model = multistatemodel(hazard_specs...; data=panel_data, surrogate=surrogate)
    fitted = fit(model;
        verbose=false,
        compute_vcov=true,
        method=:MCEM,
        tol=SPLINE_MCEM_TOL,
        ess_target_initial=MCEM_ESS_INITIAL,
        max_ess=MCEM_ESS_MAX,
        maxiter=SPLINE_MCEM_MAX_ITER
    )
    
    # Use relaxed tolerance for MCEM panel tests
    is_mcem_with_covariates = covariate_type != "nocov"
    beta_tol = is_mcem_with_covariates ? MCEM_TVC_BETA_ABS_TOL : BETA_ABS_TOL
    
    # Capture results
    result = capture_longtest_result!(
        test_name,
        fitted,
        true_params,
        param_names,
        hazard_specs;
        hazard_family = "sp",
        data_type = "mcem",
        covariate_type = covariate_type,
        data = panel_data,
        beta_param_names = beta_names,
        shape_param_names = String[],
        beta_abs_tol = beta_tol
    )
    
    VERBOSE_LONGTESTS && @info "    Result: $(result.passed ? "PASS" : "FAIL")"
    
    return result
end

"""
    compare_surrogate_results(result_markov::LongTestResult, result_phasetype::LongTestResult) -> Bool

Compare results from Markov and PhaseType surrogates to verify consistency.
Returns true if estimates are within tolerance of each other.
"""
function compare_surrogate_results(result_markov::LongTestResult, result_phasetype::LongTestResult)
    all_close = true
    
    for param_name in keys(result_markov.estimated_params)
        est_markov = result_markov.estimated_params[param_name]
        est_phasetype = result_phasetype.estimated_params[param_name]
        
        # Use relative tolerance for comparison
        if abs(est_markov) > 0.01
            rel_diff = abs(est_markov - est_phasetype) / abs(est_markov)
        else
            rel_diff = abs(est_markov - est_phasetype)
        end
        
        # Use tighter tolerance for beta parameters
        tol = contains(param_name, "beta") ? BETA_COMPARISON_TOL_REL : PROPOSAL_COMPARISON_TOL
        
        if rel_diff > tol
            VERBOSE_LONGTESTS && @warn "Surrogate comparison: $param_name differs by $(round(100*rel_diff, digits=1))%"
            all_close = false
        end
    end
    
    return all_close
end

# =============================================================================
# Test Suite Execution
# =============================================================================

const EFFECT_TYPES = ["ph", "aft"]
const SURROGATE_TYPES = ["markov", "phasetype"]
const COV_TYPES_SPLINE = ["nocov", "fixed", "tvc"]

# Option to save results to cache for reports
const SAVE_RESULTS_TO_CACHE_SPLINE = get(ENV, "LONGTEST_SAVE_RESULTS", "true") == "true"

@testset "Unpenalized Spline Long Test Suite" begin
    
    # =========================================================================
    # Exact Data Tests (6 tests)
    # =========================================================================
    @testset "Exact Data Tests" begin
        for effect_type in EFFECT_TYPES
            @testset "$(uppercase(effect_type))" begin
                for cov_type in COV_TYPES_SPLINE
                    @testset "$cov_type" begin
                        result = run_spline_exact_test(effect_type, cov_type)
                        SAVE_RESULTS_TO_CACHE_SPLINE && save_longtest_result(result; force=true)
                        @test result.passed
                    end
                end
            end
        end
    end
    
    # =========================================================================
    # Panel Data Tests (12 tests)
    # =========================================================================
    @testset "Panel Data Tests (MCEM)" begin
        for effect_type in EFFECT_TYPES
            @testset "$(uppercase(effect_type))" begin
                for surrogate_type in SURROGATE_TYPES
                    @testset "$surrogate_type surrogate" begin
                        for cov_type in COV_TYPES_SPLINE
                            @testset "$cov_type" begin
                                result = run_spline_panel_test(effect_type, cov_type, surrogate_type)
                                SAVE_RESULTS_TO_CACHE_SPLINE && save_longtest_result(result; force=true)
                                @test result.passed
                            end
                        end
                    end
                end
            end
        end
    end
    
    # =========================================================================
    # Surrogate Comparison Tests
    # =========================================================================
    @testset "Surrogate Comparison" begin
        # Compare Markov vs PhaseType surrogates for same test cases
        for effect_type in EFFECT_TYPES
            @testset "$(uppercase(effect_type))" begin
                for cov_type in COV_TYPES_SPLINE
                    @testset "$cov_type" begin
                        # Load results from panel tests
                        result_markov = load_longtest_result("sp_$(effect_type)_panel_markov_$(cov_type)")
                        result_phasetype = load_longtest_result("sp_$(effect_type)_panel_phasetype_$(cov_type)")
                        
                        if !isnothing(result_markov) && !isnothing(result_phasetype)
                            @test compare_surrogate_results(result_markov, result_phasetype)
                        else
                            @warn "Skipping surrogate comparison for sp_$(effect_type)_panel_*_$(cov_type): results not available"
                        end
                    end
                end
            end
        end
    end
    
end
