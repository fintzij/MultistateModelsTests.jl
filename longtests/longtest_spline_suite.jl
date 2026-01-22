# =============================================================================
# Unpenalized Spline Long Test Suite
# =============================================================================
#
# Comprehensive test suite covering unpenalized spline hazard inference.
#
# Test Matrix:
#   degree=1 splines × 2 interior knots
#   × (exact, panel)
#   × (ph, aft)
#   × (nocov, fixed, tvc)
#   × (markov, phasetype) - panel only
#   × (non-monotone, monotone increasing)
#
# Counts:
#   Exact data: 2 effects × 3 covariates × 2 monotonicity = 12 tests
#   Panel data: 2 effects × 3 covariates × 2 surrogates × 2 monotonicity = 24 tests
#   Total: 36 tests
#
# Naming Convention: sp_{effect}_{data}_{surrogate}_{covariate}_{monotone}
#   - Effect: ph or aft
#   - Data: exact or panel
#   - Surrogate (panel only): markov or phasetype
#   - Covariate: nocov, fixed, tvc
#   - Monotone: free or mono
#
# DGP Workflow (ALWAYS):
#   1. Simulate exact data from Weibull model
#   2. Fit spline to exact Weibull data → calibrated "true" coefficients
#   3. Simulate NEW data (exact or panel) from calibrated spline
#   4. Fit spline with SAME specification to new data
#   5. Verify PARAMETER RECOVERY (calibrated vs fitted coefficients)
#
# Spline Configuration:
#   degree=1 (linear splines)
#   n_interior_knots=2
#   boundaryknots=[0.0, MAX_TIME]
#   extrapolation="flat" (for degree=1)
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

# Spline structure for fitting
const SPLINE_DEGREE_SUITE = 1                  # Linear splines (for identifiability)
const N_INTERIOR_KNOTS_SUITE = 2               # Two interior knots
const SPLINE_MAX_TIME = 5.0                    # Observation window
const SPLINE_BOUNDARYKNOTS = [0.0, SPLINE_MAX_TIME]

# Number of spline coefficients: degree + n_interior_knots + 1 = 1 + 2 + 1 = 4
const N_SPLINE_COEFS = SPLINE_DEGREE_SUITE + N_INTERIOR_KNOTS_SUITE + 1  # = 4

# True covariate effect (log hazard ratio) - used by Weibull DGP
const TRUE_SPLINE_BETA = 0.5

# =============================================================================
# Weibull DGP Parameters (for calibration)
# =============================================================================

const TRUE_WEIBULL_SHAPE_H12 = 1.3   # Slightly increasing hazard
const TRUE_WEIBULL_SCALE_H12 = 0.15  # Rate parameter
const TRUE_WEIBULL_SHAPE_H23 = 1.2   # Slightly increasing hazard
const TRUE_WEIBULL_SCALE_H23 = 0.10  # Rate parameter

# Parameter recovery tolerances
const PARAM_RTOL_EXACT = 0.25        # 25% relative tolerance for exact data
const PARAM_RTOL_MCEM = 0.35         # 35% relative tolerance for MCEM
const BETA_ABS_TOL_SUITE = 0.20      # Absolute tolerance for covariate effects

# MCEM settings for panel tests
const SPLINE_MCEM_TOL = 0.05         # MCEM convergence tolerance
const SPLINE_MCEM_MAX_ITER = 25      # Maximum MCEM iterations

# Panel observation times
const SPLINE_PANEL_TIMES = collect(0.0:0.5:SPLINE_MAX_TIME)  # 10 intervals

# TVC changepoint
const SPLINE_TVC_CHANGEPOINT = 2.5   # Covariate changes at t=2.5

# =============================================================================
# Helper Functions
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
    get_spline_hazard_specs(effect_type::String, covariate_type::String, is_monotone::Bool)

Get spline hazard specifications for the given configuration.
Returns tuple of (h12, h23) Hazard objects.
"""
function get_spline_hazard_specs(effect_type::String, covariate_type::String, is_monotone::Bool)
    formula = if covariate_type == "nocov"
        @formula(0 ~ 1)
    else
        @formula(0 ~ x)
    end
    
    knots = get_spline_knots(N_INTERIOR_KNOTS_SUITE, SPLINE_BOUNDARYKNOTS)
    monotone_val = is_monotone ? 1 : 0  # 1 = increasing (matching Weibull with shape > 1)
    
    h12 = Hazard(formula, "sp", 1, 2;
        degree=SPLINE_DEGREE_SUITE,
        knots=knots,
        boundaryknots=SPLINE_BOUNDARYKNOTS,
        extrapolation="flat",
        monotone=monotone_val,
        linpred_effect = effect_type == "aft" ? :aft : :ph)
    
    h23 = Hazard(formula, "sp", 2, 3;
        degree=SPLINE_DEGREE_SUITE,
        knots=knots,
        boundaryknots=SPLINE_BOUNDARYKNOTS,
        extrapolation="flat",
        monotone=monotone_val,
        linpred_effect = effect_type == "aft" ? :aft : :ph)
    
    return (h12, h23)
end

"""
    get_weibull_hazard_specs(effect_type::String, covariate_type::String)

Get Weibull hazard specifications for calibration DGP.
"""
function get_weibull_hazard_specs(effect_type::String, covariate_type::String)
    formula = if covariate_type == "nocov"
        @formula(0 ~ 1)
    else
        @formula(0 ~ x)
    end
    
    h12 = Hazard(formula, "wei", 1, 2;
        linpred_effect = effect_type == "aft" ? :aft : :ph)
    
    h23 = Hazard(formula, "wei", 2, 3;
        linpred_effect = effect_type == "aft" ? :aft : :ph)
    
    return (h12, h23)
end

"""
    get_weibull_true_params(covariate_type::String)

Get true Weibull parameters for calibration DGP.
"""
function get_weibull_true_params(covariate_type::String)
    has_covariate = covariate_type != "nocov"
    
    h12 = has_covariate ? [TRUE_WEIBULL_SHAPE_H12, TRUE_WEIBULL_SCALE_H12, TRUE_SPLINE_BETA] : 
                          [TRUE_WEIBULL_SHAPE_H12, TRUE_WEIBULL_SCALE_H12]
    h23 = has_covariate ? [TRUE_WEIBULL_SHAPE_H23, TRUE_WEIBULL_SCALE_H23, TRUE_SPLINE_BETA] :
                          [TRUE_WEIBULL_SHAPE_H23, TRUE_WEIBULL_SCALE_H23]
    
    return (h12 = h12, h23 = h23)
end

"""
    calibrate_spline_from_weibull(effect_type, covariate_type, is_monotone; verbose=false)

Calibrate spline coefficients by fitting spline to exact data from Weibull DGP.
Returns calibrated_params and exact_data used for calibration.
"""
function calibrate_spline_from_weibull(effect_type::String, covariate_type::String, is_monotone::Bool;
                                         verbose::Bool=false)
    weibull_specs = get_weibull_hazard_specs(effect_type, covariate_type)
    weibull_params = get_weibull_true_params(covariate_type)
    
    template = if covariate_type == "nocov"
        create_baseline_template(N_SUBJECTS; max_time=SPLINE_MAX_TIME)
    elseif covariate_type == "fixed"
        create_tfc_template(N_SUBJECTS; max_time=SPLINE_MAX_TIME)
    else
        create_tvc_template(N_SUBJECTS; max_time=SPLINE_MAX_TIME,
                            changepoint=SPLINE_TVC_CHANGEPOINT)
    end
    
    weibull_model = multistatemodel(weibull_specs...; data=template)
    set_parameters!(weibull_model, weibull_params)
    exact_data = simulate(weibull_model; data=true, paths=false, nsim=1)[1]
    
    verbose && @info "    Calibration: generated $(nrow(exact_data)) exact observations from Weibull"
    
    spline_specs = get_spline_hazard_specs(effect_type, covariate_type, is_monotone)
    spline_model = multistatemodel(spline_specs...; data=exact_data)
    fitted_spline = fit(spline_model; verbose=false, vcov_type=:none, penalty=:none)
    
    calibrated_h12 = get_parameters(fitted_spline, 1, scale=:natural)
    calibrated_h23 = get_parameters(fitted_spline, 2, scale=:natural)
    calibrated_params = (h12 = calibrated_h12, h23 = calibrated_h23)
    
    if verbose
        @info "    Calibration complete:"
        @info "      h12: $(round.(calibrated_h12, digits=4))"
        @info "      h23: $(round.(calibrated_h23, digits=4))"
    end
    
    return calibrated_params, exact_data
end

"""
    generate_spline_exact_data(hazard_specs, params, covariate_type)

Generate exact observation data from spline model.
"""
function generate_spline_exact_data(hazard_specs, params, covariate_type::String)
    template = if covariate_type == "nocov"
        create_baseline_template(N_SUBJECTS; max_time=SPLINE_MAX_TIME)
    elseif covariate_type == "fixed"
        create_tfc_template(N_SUBJECTS; max_time=SPLINE_MAX_TIME)
    else
        create_tvc_template(N_SUBJECTS; max_time=SPLINE_MAX_TIME, 
                            changepoint=SPLINE_TVC_CHANGEPOINT)
    end
    
    model = multistatemodel(hazard_specs...; data=template)
    set_parameters!(model, params)
    
    sim_data = simulate(model; data=true, paths=false, nsim=1)[1]
    
    return sim_data
end

"""
    generate_spline_panel_data(hazard_specs, params, covariate_type)

Generate panel observation data from spline model.
"""
function generate_spline_panel_data(hazard_specs, params, covariate_type::String)
    panel_times = SPLINE_PANEL_TIMES
    nobs = length(panel_times) - 1
    
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
    else
        cp = SPLINE_TVC_CHANGEPOINT
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
    
    model = multistatemodel(hazard_specs...; data=template)
    set_parameters!(model, params)
    
    obstype_map = Dict(1 => 2, 2 => 1)
    sim_result = simulate(model; paths=false, data=true, nsim=1, autotmax=false,
                         obstype_by_transition=obstype_map)
    
    return sim_result[1, 1]
end

"""
    check_parameter_recovery(fitted, calibrated_params, hazard_idx; rtol=0.25, verbose=false)

Check if fitted spline coefficients match calibrated coefficients.
"""
function check_parameter_recovery(fitted, calibrated_params::NamedTuple, hazard_idx::Int;
                                   rtol::Float64=0.25, verbose::Bool=false)
    hazard = fitted.hazards[hazard_idx]
    hazname = hazard.hazname
    
    fitted_params = get_parameters(fitted, hazard_idx, scale=:natural)
    calib_params = calibrated_params[hazname]
    
    n_baseline = hazard.npar_baseline
    
    details = Tuple[]
    max_rel_err = 0.0
    all_passed = true
    
    for i in 1:n_baseline
        true_val = calib_params[i]
        fit_val = fitted_params[i]
        
        rel_err = abs(true_val) > 1e-6 ? abs(fit_val - true_val) / abs(true_val) : abs(fit_val - true_val)
        max_rel_err = max(max_rel_err, rel_err)
        
        passed = rel_err <= rtol
        all_passed = all_passed && passed
        push!(details, ("coef_$i", true_val, fit_val, rel_err, passed))
    end
    
    if verbose
        println("    Parameter recovery for $hazname:")
        println("    " * "-"^65)
        println("    Param       Calibrated  Fitted      Rel Error   Status")
        println("    " * "-"^65)
        for (name, true_val, fit_val, rel_err, passed) in details
            status = passed ? "PASS" : "FAIL"
            println("    $(rpad(name, 11)) $(rpad(round(true_val, digits=4), 11)) $(rpad(round(fit_val, digits=4), 11)) $(rpad(round(rel_err*100, digits=1), 9))% $status")
        end
        println("    " * "-"^65)
        println("    Max relative error: $(round(max_rel_err*100, digits=1))%, tolerance: $(round(rtol*100, digits=0))%")
    end
    
    return (passed=all_passed, max_rel_err=max_rel_err, details=details)
end

"""
    check_beta_recovery(fitted, calibrated_params, hazard_idx; abs_tol=0.20, verbose=false)

Check if covariate effect (beta) is recovered within tolerance.
"""
function check_beta_recovery(fitted, calibrated_params::NamedTuple, hazard_idx::Int;
                              abs_tol::Float64=BETA_ABS_TOL_SUITE, verbose::Bool=false)
    hazard = fitted.hazards[hazard_idx]
    hazname = hazard.hazname
    
    if !hazard.has_covariates
        return (passed=true, abs_err=0.0, true_val=NaN, fit_val=NaN)
    end
    
    fitted_params = get_parameters(fitted, hazard_idx, scale=:natural)
    calib_params = calibrated_params[hazname]
    
    true_val = calib_params[end]
    fit_val = fitted_params[end]
    abs_err = abs(fit_val - true_val)
    passed = abs_err <= abs_tol
    
    if verbose
        status = passed ? "PASS" : "FAIL"
        println("    Beta recovery for $hazname: true=$(round(true_val, digits=3)), fitted=$(round(fit_val, digits=3)), err=$(round(abs_err, digits=3)) $status")
    end
    
    return (passed=passed, abs_err=abs_err, true_val=true_val, fit_val=fit_val)
end

# =============================================================================
# Test Functions
# =============================================================================

"""
    run_spline_exact_test(effect_type, covariate_type, is_monotone)

Run exact data test for spline model.
"""
function run_spline_exact_test(effect_type::String, covariate_type::String, is_monotone::Bool)
    mono_str = is_monotone ? "mono" : "free"
    test_name = "sp_$(effect_type)_exact_$(covariate_type)_$(mono_str)"
    
    VERBOSE_LONGTESTS && @info "  Running: $test_name"
    
    test_seed = hash(("sp", effect_type, "exact", covariate_type, is_monotone, RNG_SEED))
    Random.seed!(test_seed)
    
    calibrated_params, _ = calibrate_spline_from_weibull(
        effect_type, covariate_type, is_monotone; verbose=VERBOSE_LONGTESTS
    )
    
    spline_specs = get_spline_hazard_specs(effect_type, covariate_type, is_monotone)
    data = generate_spline_exact_data(spline_specs, calibrated_params, covariate_type)
    
    model = multistatemodel(spline_specs...; data=data)
    fitted = fit(model; verbose=false, vcov_type=:ij, penalty=:none)
    
    h12_result = check_parameter_recovery(
        fitted, calibrated_params, 1;
        rtol=PARAM_RTOL_EXACT, verbose=VERBOSE_LONGTESTS
    )
    
    h23_result = check_parameter_recovery(
        fitted, calibrated_params, 2;
        rtol=PARAM_RTOL_EXACT, verbose=VERBOSE_LONGTESTS
    )
    
    beta_h12 = check_beta_recovery(fitted, calibrated_params, 1; verbose=VERBOSE_LONGTESTS)
    beta_h23 = check_beta_recovery(fitted, calibrated_params, 2; verbose=VERBOSE_LONGTESTS)
    
    passed = h12_result.passed && h23_result.passed && beta_h12.passed && beta_h23.passed
    
    VERBOSE_LONGTESTS && @info "    Result: $(passed ? "PASS" : "FAIL")"
    
    return (
        test_name = test_name,
        passed = passed,
        h12_passed = h12_result.passed,
        h23_passed = h23_result.passed,
        beta_passed = beta_h12.passed && beta_h23.passed,
        h12_max_err = h12_result.max_rel_err,
        h23_max_err = h23_result.max_rel_err
    )
end

"""
    run_spline_panel_test(effect_type, covariate_type, surrogate_type, is_monotone)

Run panel data test for spline model with MCEM.
"""
function run_spline_panel_test(effect_type::String, covariate_type::String, 
                                surrogate_type::String, is_monotone::Bool)
    mono_str = is_monotone ? "mono" : "free"
    test_name = "sp_$(effect_type)_panel_$(surrogate_type)_$(covariate_type)_$(mono_str)"
    
    VERBOSE_LONGTESTS && @info "  Running: $test_name"
    
    test_seed = hash(("sp", effect_type, "panel", surrogate_type, covariate_type, is_monotone, RNG_SEED))
    Random.seed!(test_seed)
    
    calibrated_params, _ = calibrate_spline_from_weibull(
        effect_type, covariate_type, is_monotone; verbose=VERBOSE_LONGTESTS
    )
    
    spline_specs = get_spline_hazard_specs(effect_type, covariate_type, is_monotone)
    panel_data = generate_spline_panel_data(spline_specs, calibrated_params, covariate_type)
    
    surrogate = surrogate_type == "markov" ? :markov : :phasetype
    
    model = multistatemodel(spline_specs...; data=panel_data, surrogate=surrogate)
    fitted = fit(model;
        verbose=false,
        vcov_type=:ij,
        method=:MCEM,
        penalty=:none,
        tol=SPLINE_MCEM_TOL,
        ess_target_initial=MCEM_ESS_INITIAL,
        max_ess=MCEM_ESS_MAX,
        maxiter=SPLINE_MCEM_MAX_ITER
    )
    
    h12_result = check_parameter_recovery(
        fitted, calibrated_params, 1;
        rtol=PARAM_RTOL_MCEM, verbose=VERBOSE_LONGTESTS
    )
    
    h23_result = check_parameter_recovery(
        fitted, calibrated_params, 2;
        rtol=PARAM_RTOL_MCEM, verbose=VERBOSE_LONGTESTS
    )
    
    beta_tol = covariate_type != "nocov" ? MCEM_TVC_BETA_ABS_TOL : BETA_ABS_TOL_SUITE
    
    beta_h12 = check_beta_recovery(fitted, calibrated_params, 1; abs_tol=beta_tol, verbose=VERBOSE_LONGTESTS)
    beta_h23 = check_beta_recovery(fitted, calibrated_params, 2; abs_tol=beta_tol, verbose=VERBOSE_LONGTESTS)
    
    passed = h12_result.passed && h23_result.passed && beta_h12.passed && beta_h23.passed
    
    VERBOSE_LONGTESTS && @info "    Result: $(passed ? "PASS" : "FAIL")"
    
    return (
        test_name = test_name,
        passed = passed,
        h12_passed = h12_result.passed,
        h23_passed = h23_result.passed,
        beta_passed = beta_h12.passed && beta_h23.passed,
        h12_max_err = h12_result.max_rel_err,
        h23_max_err = h23_result.max_rel_err
    )
end

# =============================================================================
# Test Suite Execution
# =============================================================================

const EFFECT_TYPES = ["ph", "aft"]
const COV_TYPES = ["nocov", "fixed", "tvc"]
const SURROGATE_TYPES = ["markov", "phasetype"]
const MONOTONE_TYPES = [false, true]

@testset "Unpenalized Spline Long Test Suite" begin
    
    @testset "Exact Data Tests" begin
        for effect_type in EFFECT_TYPES
            @testset "$(uppercase(effect_type))" begin
                for is_monotone in MONOTONE_TYPES
                    mono_label = is_monotone ? "monotone" : "free"
                    @testset "$mono_label" begin
                        for cov_type in COV_TYPES
                            @testset "$cov_type" begin
                                result = run_spline_exact_test(effect_type, cov_type, is_monotone)
                                @test result.passed
                            end
                        end
                    end
                end
            end
        end
    end
    
    @testset "Panel Data Tests (MCEM)" begin
        for effect_type in EFFECT_TYPES
            @testset "$(uppercase(effect_type))" begin
                for is_monotone in MONOTONE_TYPES
                    mono_label = is_monotone ? "monotone" : "free"
                    @testset "$mono_label" begin
                        for surrogate_type in SURROGATE_TYPES
                            @testset "$surrogate_type" begin
                                for cov_type in COV_TYPES
                                    @testset "$cov_type" begin
                                        result = run_spline_panel_test(
                                            effect_type, cov_type, surrogate_type, is_monotone
                                        )
                                        @test result.passed
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    
end
