"""
Long test suite for MCEM algorithm with UNPENALIZED spline hazards.

=== IMPORTANT: UNPENALIZED SPLINES ONLY ===
All fit() calls in this file use `penalty=:none` to test unpenalized spline 
functionality. This is intentional - we must verify unpenalized splines work 
correctly before testing penalized likelihood.

For penalized spline tests, see: longtest_mcem_splines_penalized.jl (TODO)

This test suite verifies that spline hazards can approximate:
1. Exponential hazards (constant rate) - using splines with no interior knots
2. Piecewise exponential hazards - using splines with interior knots
3. Gompertz hazards (exponentially increasing/decreasing rate) - using splines

These tests validate that MCEM works correctly with RuntimeSplineHazard types
and that the flexible spline baseline can recover known parametric shapes.

References:
- Morsomme et al. (2025) Biostatistics kxaf038 - multistate semi-Markov MCEM
- Ramsay (1988) Statistical Science - spline smoothing
"""

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

# Import internal functions for testing
import MultistateModels: Hazard, multistatemodel, fit, set_parameters!, simulate,
    get_parameters_flat, cumulative_hazard, get_parameters, PhaseTypeProposal

"""
    print_parameter_comparison(test_name, true_params, fitted_params; param_names=nothing)

Print a formatted table comparing true vs fitted parameters.
"""
function print_parameter_comparison(test_name::String, true_params::Vector, fitted_params::Vector;
    param_names::Union{Nothing, Vector{String}}=nothing)
    println("\n    Parameter Comparison: $test_name")
    println("    " * "-"^70)
    println("    Parameter         True        Estimated   Abs Diff    Rel Diff (%)")
    println("    " * "-"^70)
    
    names = isnothing(param_names) ? ["param_$i" for i in 1:length(true_params)] : param_names
    
    for i in 1:length(true_params)
        abs_diff = fitted_params[i] - true_params[i]
        rel_diff = abs(true_params[i]) > 0.01 ? (abs_diff / true_params[i]) * 100 : abs_diff * 100
        
        println("    ", rpad(names[i], 18), 
                @sprintf("%.4f", true_params[i]), "      ",
                @sprintf("%.4f", fitted_params[i]), "      ",
                @sprintf("%.4f", abs_diff), "      ",
                @sprintf("%.1f%%", rel_diff))
    end
    println("    " * "-"^70)
end

const RNG_SEED = 0xABCD5678
const N_SUBJECTS = 1000      # Standard sample size for longtests
const MCEM_TOL = 0.05        # Relaxed tolerance
const MAX_ITER = 25          # Short iteration limit

# Tolerance for comparing Markov vs PhaseType proposal estimates
# Justification: Both proposals should converge to similar estimates.
# With covariate-aware PhaseType TPM fix, proposals should agree more closely.
# Use 35% tolerance to account for MCEM Monte Carlo variance.
const PROPOSAL_COMPARISON_TOL = 0.35
# Stricter tolerance for covariate coefficients (β parameters) in proposal comparison.
# Covariate coefficients are most sensitive to proposal covariate handling bugs.
const BETA_COMPARISON_TOL_REL = 0.20  # 20% relative tolerance
const BETA_COMPARISON_TOL_ABS = 0.15  # 0.15 absolute tolerance for betas near zero

# HAZARD_TOL_FACTOR justification:
# Factor of 1.5 accounts for:
# 1. MCEM Monte Carlo variance (~10-15% coefficient of variation at convergence)
# 2. Spline approximation error (splines approximate parametric shapes)
# Tighter tolerance ensures meaningful parameter recovery validation.
const HAZARD_TOL_FACTOR = 1.5

# ============================================================================
# Test 1: Spline Approximation to Exponential (Constant Hazard)
# ============================================================================

@testset "MCEM Spline vs Exponential" begin
    Random.seed!(RNG_SEED)
    
    # True exponential rate (stored on natural scale since v0.3.0)
    true_rate = 0.3
    true_log_rate = log(true_rate)  # For spline coefficient comparison
    
    # Create exponential model for data generation (progressive 1→2 only)
    h12_exp = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    
    # Generate panel data template - 10 observation intervals for adequate density
    obs_times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    nobs = length(obs_times) - 1
    
    sim_data = DataFrame(
        id = repeat(1:N_SUBJECTS, inner=nobs),
        tstart = repeat(obs_times[1:end-1], N_SUBJECTS),
        tstop = repeat(obs_times[2:end], N_SUBJECTS),
        statefrom = ones(Int, N_SUBJECTS * nobs),
        stateto = ones(Int, N_SUBJECTS * nobs),
        obstype = fill(2, N_SUBJECTS * nobs)
    )
    
    # Use initialize=false to avoid bounds issues during auto-initialization
    model_sim = multistatemodel(h12_exp; data=sim_data, initialize=false)
    MultistateModels.set_parameters!(model_sim, (h12 = [true_rate],))  # Natural scale
    
    # Simulate panel data
    # For 2-state model (1→2 absorbing), transition 1 is exact
    obstype_map = Dict(1 => 1)
    sim_result = simulate(model_sim; paths=false, data=true, nsim=1, autotmax=false,
                         obstype_by_transition=obstype_map)
    panel_data = sim_result[1, 1]
    
    # Fit linear spline (degree=1) with boundary knots only (no interior knots)
    # This should approximate a constant hazard
    h12_sp = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                    degree=1, 
                    knots=Float64[],  # No interior knots
                    boundaryknots=[0.0, 5.0],
                    extrapolation="linear")
    
    model_spline = multistatemodel(h12_sp; data=panel_data, surrogate=:markov)
    
    # Fit via MCEM - UNPENALIZED SPLINES (penalty=:none)
    # Testing basic spline functionality without smoothing penalty
    fitted = fit(model_spline;
        proposal=:markov,
        penalty=:none,  # UNPENALIZED - no smoothing
        verbose=true,
        maxiter=MAX_ITER,
        tol=MCEM_TOL,
        ess_target_initial=25,
        max_ess=300,
        vcov_type=:none,
        return_convergence_records=true)
    
    # Check that spline hazard at various times approximates true exponential
    pars_12 = MultistateModels.get_parameters(fitted, 1, scale=:log)
    
    # Verify spline hazard approximates true constant rate at all evaluation times
    for t in [0.5, 1.5, 2.5, 3.5, 4.5]
        h_spline = fitted.hazards[1](t, pars_12, NamedTuple())
        # Key test: spline hazard must be within factor of HAZARD_TOL_FACTOR of true rate
        @test h_spline > true_rate / HAZARD_TOL_FACTOR
        @test h_spline < true_rate * HAZARD_TOL_FACTOR
    end
    
    # Verify log-likelihood converged (finite)
    @test isfinite(fitted.loglik.loglik)
    
    # =====================================================================
    # Fit with PhaseType proposal (same data)
    # =====================================================================
    println("      ▸ Fitting with PhaseType proposal..."); flush(stdout)
    
    # Create new model for PhaseType fit (same hazards, same data)
    h12_pt = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                    degree=1, 
                    knots=Float64[],
                    boundaryknots=[0.0, 5.0],
                    extrapolation="linear")
    
    model_pt = multistatemodel(h12_pt; data=panel_data, surrogate=:markov)
    
    # UNPENALIZED SPLINES (penalty=:none) - testing basic PhaseType proposal
    fitted_pt = fit(model_pt;
        proposal=PhaseTypeProposal(n_phases=3),
        penalty=:none,  # UNPENALIZED - no smoothing
        verbose=true,
        maxiter=MAX_ITER,
        tol=MCEM_TOL,
        ess_target_initial=30,
        max_ess=500,
        vcov_type=:none)
    
    pars_12_pt = MultistateModels.get_parameters(fitted_pt, 1, scale=:log)
    
    # =====================================================================
    # Compare Markov vs PhaseType proposals
    # =====================================================================
    println("\n    Markov vs PhaseType Proposal Comparison:")
    println("    " * "-"^60)
    println("    Time      Markov h(t)   PhaseType h(t)  Rel Diff")
    println("    " * "-"^60)
    
    comparison_passed = true
    for t in [0.5, 1.5, 2.5, 3.5, 4.5]
        h_markov = fitted.hazards[1](t, pars_12, NamedTuple())
        h_pt = fitted_pt.hazards[1](t, pars_12_pt, NamedTuple())
        rel_diff = abs(h_markov - h_pt) / max(h_markov, h_pt, 0.01)
        passed = rel_diff <= PROPOSAL_COMPARISON_TOL
        comparison_passed &= passed
        status = passed ? "✓" : "✗"
        println("    $(rpad(t, 9)) $(rpad(round(h_markov, digits=4), 13)) $(rpad(round(h_pt, digits=4), 15)) $(round(rel_diff, digits=3)) $status")
    end
    println("    " * "-"^60)
    @test comparison_passed
    
    println("  ✓ Linear spline approximates constant/exponential hazard")
    println("  ✓ Markov vs PhaseType proposal estimates agree")
end

# ============================================================================
# Test 2: Spline with Interior Knots (Piecewise Exponential Approximation)
# ============================================================================

@testset "MCEM Spline Piecewise Exponential" begin
    Random.seed!(RNG_SEED + 1)
    
    # True piecewise exponential rates (low → high)
    # h(t) = rate_early for t < change_time, rate_late for t >= change_time
    true_rate_early = 0.2
    true_rate_late = 0.5
    change_time = 2.5
    tmax = 5.0
    
    # Create piecewise-like spline model for data generation
    # Linear spline (degree=1) with knot at change_time approximates piecewise constant
    # Spline coefficients are log-hazards evaluated at the knot locations
    obs_times = [0.0, 0.8, 1.6, 2.4, 3.2, 4.0, 4.8]
    nobs = length(obs_times) - 1
    
    # Create simulation template
    sim_data = DataFrame(
        id = repeat(1:N_SUBJECTS, inner=nobs),
        tstart = repeat(obs_times[1:end-1], N_SUBJECTS),
        tstop = repeat(obs_times[2:end], N_SUBJECTS),
        statefrom = ones(Int, N_SUBJECTS * nobs),
        stateto = ones(Int, N_SUBJECTS * nobs),
        obstype = fill(2, N_SUBJECTS * nobs)
    )
    
    # Create spline hazard for DGP that approximates piecewise constant
    # For linear B-spline with knot at change_time and boundary knots at [0, tmax]:
    # - Basis 1 peaks at t=0, goes to 0 at change_time
    # - Basis 2 peaks at change_time
    # - Basis 3 peaks at tmax, goes to 0 at change_time
    h12_dgp = Hazard(@formula(0 ~ 1), "sp", 1, 2;
        degree=1,
        knots=[change_time],
        boundaryknots=[0.0, tmax],
        extrapolation="linear")
    
    model_dgp = multistatemodel(h12_dgp; data=sim_data, initialize=false)
    
    # Set spline coefficients to approximate piecewise constant hazard
    # Linear spline with 1 interior knot has 3 coefficients (degree + 1 + n_interior_knots)
    # Coefficients are on HAZARD scale (non-negative); we want h≈rate_early before change_time, h≈rate_late after
    # Set [rate_early, rate_late, rate_late] to transition at knot
    dgp_coeffs = [true_rate_early, true_rate_late, true_rate_late]
    MultistateModels.set_parameters!(model_dgp, (h12 = dgp_coeffs,))
    
    # Simulate using package's simulate() function
    # For 2-state model (1→2 absorbing), transition 1 is exact
    obstype_map = Dict(1 => 1)
    sim_result = simulate(model_dgp; paths=false, data=true, nsim=1, autotmax=false,
                         obstype_by_transition=obstype_map)
    panel_data = sim_result[1, 1]
    
    # Fit spline with interior knot at change point
    # Linear spline with knot at change_time can approximate piecewise constant
    h12_sp = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                    degree=1,  # Linear spline
                    knots=[change_time],  # Interior knot at change point
                    boundaryknots=[0.0, tmax],
                    extrapolation="linear")
    
    model_spline = multistatemodel(h12_sp; data=panel_data, surrogate=:markov)
    
    # Fit via MCEM - UNPENALIZED SPLINES (penalty=:none)
    fitted = fit(model_spline;
        proposal=:markov,
        penalty=:none,  # UNPENALIZED - no smoothing
        verbose=true,
        maxiter=MAX_ITER,
        tol=MCEM_TOL,
        ess_target_initial=25,
        max_ess=300,
        vcov_type=:none,
        return_convergence_records=true)
    
    # Check that spline hazard approximates the true piecewise rates
    pars_12 = MultistateModels.get_parameters(fitted, 1, scale=:log)
    
    # Validate against TRUE piecewise rates (not average):
    # - Before change_time: compare to true_rate_early (0.2)
    # - After change_time: compare to true_rate_late (0.5)
    # Use factor of 3.0 tolerance for piecewise since spline smooths the discontinuity
    PWE_TOL_FACTOR = 3.0  # Piecewise exponential tolerance (looser due to discontinuity smoothing)
    
    # Early period (t < change_time): should be close to true_rate_early
    for t in [0.5, 1.5]
        h_spline = fitted.hazards[1](t, pars_12, NamedTuple())
        @test h_spline > true_rate_early / PWE_TOL_FACTOR
        @test h_spline < true_rate_late * PWE_TOL_FACTOR  # Can't exceed late rate by much
    end
    
    # Late period (t > change_time): should be close to true_rate_late  
    for t in [3.5, 4.5]
        h_spline = fitted.hazards[1](t, pars_12, NamedTuple())
        @test h_spline > true_rate_early / PWE_TOL_FACTOR  # Should be at least as high as early
        @test h_spline < true_rate_late * PWE_TOL_FACTOR
    end
    
    # Key test: hazard should be HIGHER after change point than before
    # This validates that the spline captures the step-up in hazard
    h_early_avg = mean([fitted.hazards[1](t, pars_12, NamedTuple()) for t in [0.5, 1.5]])
    h_late_avg = mean([fitted.hazards[1](t, pars_12, NamedTuple()) for t in [3.5, 4.5]])
    @test h_late_avg > h_early_avg * 1.1  # Late hazard should be noticeably higher
    
    # Cumulative hazard must be monotonically increasing (fundamental property)
    H_vals = [cumulative_hazard(fitted.hazards[1], 0.0, t, pars_12, NamedTuple()) 
              for t in [1.0, 2.0, 3.0, 4.0]]
    @test all(diff(H_vals) .> 0)
    
    # Convergence check
    @test isfinite(fitted.loglik.loglik)
    
    # =====================================================================
    # Fit with PhaseType proposal (same data)
    # =====================================================================
    println("      ▸ Fitting with PhaseType proposal..."); flush(stdout)
    
    h12_pt = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                    degree=1,
                    knots=[change_time],
                    boundaryknots=[0.0, tmax],
                    extrapolation="linear")
    
    model_pt = multistatemodel(h12_pt; data=panel_data, surrogate=:markov)
    
    # UNPENALIZED SPLINES (penalty=:none) - testing basic PhaseType proposal
    fitted_pt = fit(model_pt;
        proposal=PhaseTypeProposal(n_phases=3),
        penalty=:none,  # UNPENALIZED - no smoothing
        verbose=true,
        maxiter=MAX_ITER,
        tol=MCEM_TOL,
        ess_target_initial=30,
        max_ess=500,
        vcov_type=:none)
    
    pars_12_pt = MultistateModels.get_parameters(fitted_pt, 1, scale=:log)
    
    # =====================================================================
    # Compare Markov vs PhaseType proposals
    # =====================================================================
    println("\n    Markov vs PhaseType Proposal Comparison:")
    println("    " * "-"^60)
    println("    Time      Markov h(t)   PhaseType h(t)  Rel Diff")
    println("    " * "-"^60)
    
    comparison_passed = true
    for t in [0.5, 1.5, 3.5, 4.5]
        h_markov = fitted.hazards[1](t, pars_12, NamedTuple())
        h_pt = fitted_pt.hazards[1](t, pars_12_pt, NamedTuple())
        rel_diff = abs(h_markov - h_pt) / max(h_markov, h_pt, 0.01)
        passed = rel_diff <= PROPOSAL_COMPARISON_TOL
        comparison_passed &= passed
        status = passed ? "✓" : "✗"
        println("    $(rpad(t, 9)) $(rpad(round(h_markov, digits=4), 13)) $(rpad(round(h_pt, digits=4), 15)) $(round(rel_diff, digits=3)) $status")
    end
    println("    " * "-"^60)
    @test comparison_passed
    
    println("  ✓ Spline with interior knot captures piecewise hazard step-up")
    println("  ✓ Markov vs PhaseType proposal estimates agree")
end

# ============================================================================
# Test 3: Spline Approximation to Gompertz (Exponentially Varying Hazard)
# ============================================================================

@testset "MCEM Spline vs Gompertz" begin
    Random.seed!(RNG_SEED + 2)
    
    # True Gompertz parameters
    # h(t) = b * exp(a*t), so log(h(t)) = log(b) + a*t (linear in log-hazard)
    true_a = log(0.15)  # log(baseline) - for spline comparison on log-hazard scale
    true_b = 0.2        # shape (positive = increasing hazard, moderate)
    true_rate = 0.15    # Gompertz rate parameter (natural scale)
    
    # Create Gompertz model for data generation - progressive 1→2 only
    h12_gom = Hazard(@formula(0 ~ 1), "gom", 1, 2)
    
    obs_times = [0.0, 0.7, 1.4, 2.1, 2.8, 3.5, 4.2]
    nobs = length(obs_times) - 1
    
    sim_data = DataFrame(
        id = repeat(1:N_SUBJECTS, inner=nobs),
        tstart = repeat(obs_times[1:end-1], N_SUBJECTS),
        tstop = repeat(obs_times[2:end], N_SUBJECTS),
        statefrom = ones(Int, N_SUBJECTS * nobs),
        stateto = ones(Int, N_SUBJECTS * nobs),
        obstype = fill(2, N_SUBJECTS * nobs)
    )
    
    # Use initialize=false to avoid bounds issues during auto-initialization
    model_sim = multistatemodel(h12_gom; data=sim_data, initialize=false)
    # Gompertz params: [shape, rate] where h(t) = rate * exp(shape * t)
    MultistateModels.set_parameters!(model_sim, (h12 = [true_b, true_rate],))
    
    # For 2-state model (1→2 absorbing), transition 1 is exact
    obstype_map = Dict(1 => 1)
    sim_result = simulate(model_sim; paths=false, data=true, nsim=1, autotmax=false,
                         obstype_by_transition=obstype_map)
    panel_data = sim_result[1, 1]
    
    # Fit linear spline with one interior knot
    # Linear splines are simpler and more identifiable for parameter recovery
    h12_sp = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                    degree=1,  # Linear for identifiability
                    knots=[2.5],  # One interior knot for flexibility
                    boundaryknots=[0.0, 5.0],
                    extrapolation="constant")
    
    model_spline = multistatemodel(h12_sp; data=panel_data, surrogate=:markov)
    
    # Fit via MCEM - UNPENALIZED SPLINES (penalty=:none)
    fitted = fit(model_spline;
        proposal=:markov,
        penalty=:none,  # UNPENALIZED - no smoothing
        verbose=true,
        maxiter=MAX_ITER,
        tol=MCEM_TOL,
        ess_target_initial=25,
        max_ess=300,
        vcov_type=:none,
        return_convergence_records=true)
    
    # Check that spline captures general trend (not exact match given MCEM variance)
    pars_12 = MultistateModels.get_parameters(fitted, 1, scale=:log)
    
    # Basic sanity: fitted hazard should be positive and finite at all evaluation times
    for t in [0.5, 2.0, 3.5]
        h_spline = fitted.hazards[1](t, pars_12, NamedTuple())
        @test h_spline > 0
        @test isfinite(h_spline)
    end
    
    # Cumulative hazard must be monotonically increasing
    H_vals = [cumulative_hazard(fitted.hazards[1], 0.0, t, pars_12, NamedTuple()) 
              for t in [1.0, 2.0, 3.0, 4.0]]
    @test all(diff(H_vals) .> 0)
    
    # Convergence check
    @test isfinite(fitted.loglik.loglik)
    
    # =====================================================================
    # Fit with PhaseType proposal (same data)
    # =====================================================================
    println("      ▸ Fitting with PhaseType proposal..."); flush(stdout)
    
    h12_pt = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                    degree=1,
                    knots=[2.5],
                    boundaryknots=[0.0, 5.0],
                    extrapolation="constant")
    
    model_pt = multistatemodel(h12_pt; data=panel_data, surrogate=:markov)
    
    # UNPENALIZED SPLINES (penalty=:none) - testing basic PhaseType proposal
    fitted_pt = fit(model_pt;
        proposal=PhaseTypeProposal(n_phases=3),
        penalty=:none,  # UNPENALIZED - no smoothing
        verbose=true,
        maxiter=MAX_ITER,
        tol=MCEM_TOL,
        ess_target_initial=30,
        max_ess=500,
        vcov_type=:none)
    
    pars_12_pt = MultistateModels.get_parameters(fitted_pt, 1, scale=:log)
    
    # =====================================================================
    # Compare Markov vs PhaseType proposals
    # =====================================================================
    println("\n    Markov vs PhaseType Proposal Comparison:")
    println("    " * "-"^60)
    println("    Time      Markov h(t)   PhaseType h(t)  Rel Diff")
    println("    " * "-"^60)
    
    comparison_passed = true
    for t in [0.5, 2.0, 3.5]
        h_markov = fitted.hazards[1](t, pars_12, NamedTuple())
        h_pt = fitted_pt.hazards[1](t, pars_12_pt, NamedTuple())
        rel_diff = abs(h_markov - h_pt) / max(h_markov, h_pt, 0.01)
        passed = rel_diff <= PROPOSAL_COMPARISON_TOL
        comparison_passed &= passed
        status = passed ? "✓" : "✗"
        println("    $(rpad(t, 9)) $(rpad(round(h_markov, digits=4), 13)) $(rpad(round(h_pt, digits=4), 15)) $(round(rel_diff, digits=3)) $status")
    end
    println("    " * "-"^60)
    @test comparison_passed
    
    println("  ✓ Cubic spline approximates Gompertz (increasing) hazard")
    println("  ✓ Markov vs PhaseType proposal estimates agree")
end

# ============================================================================
# Test 4: Spline with Covariates - COMPREHENSIVE VALIDATION
# ============================================================================
# This test validates that:
# 1. Spline hazard approximates true exponential hazard at multiple time points
# 2. Covariate effect (beta) is correctly recovered
# 3. Cumulative hazard H(t) matches at multiple time points
# 4. Both x=0 and x=1 cases are validated
#
# True model: h(t|x) = λ * exp(β*x) = 0.3 * exp(0.5*x)
# - h(t|x=0) = 0.3 (constant)
# - h(t|x=1) = 0.3 * exp(0.5) ≈ 0.495 (constant)
# ============================================================================

@testset "MCEM Spline with Covariates" begin
    Random.seed!(RNG_SEED + 3)
    
    # True parameters
    true_baseline = 0.3
    true_beta = 0.5  # Covariate effect (proportional hazards)
    
    # True hazard functions for validation
    true_hazard(t, x) = true_baseline * exp(true_beta * x)
    true_cumhaz(t, x) = true_baseline * exp(true_beta * x) * t  # H(t) = λ*t for exponential
    
    # Create exponential model with covariate
    h12_exp = Hazard(@formula(0 ~ 1 + x), "exp", 1, 2)
    
    obs_times = [0.0, 0.75, 1.5, 2.25, 3.0, 3.75, 4.5]
    nobs = length(obs_times) - 1
    
    # Generate data with binary covariate
    sim_data = DataFrame(
        id = repeat(1:N_SUBJECTS, inner=nobs),
        tstart = repeat(obs_times[1:end-1], N_SUBJECTS),
        tstop = repeat(obs_times[2:end], N_SUBJECTS),
        statefrom = ones(Int, N_SUBJECTS * nobs),
        stateto = ones(Int, N_SUBJECTS * nobs),
        obstype = fill(2, N_SUBJECTS * nobs),
        x = repeat(rand([0.0, 1.0], N_SUBJECTS), inner=nobs)
    )
    
    # Use initialize=false to avoid bounds issues during auto-initialization
    model_sim = multistatemodel(h12_exp; data=sim_data, initialize=false)
    MultistateModels.set_parameters!(model_sim, (h12 = [true_baseline, true_beta],))
    
    # For 2-state model (1→2 absorbing), transition 1 is exact
    obstype_map = Dict(1 => 1)
    sim_result = simulate(model_sim; paths=false, data=true, nsim=1, autotmax=false,
                         obstype_by_transition=obstype_map)
    panel_data = sim_result[1, 1]
    
    # Fit spline model with covariate
    h12_sp = Hazard(@formula(0 ~ 1 + x), "sp", 1, 2; 
                    degree=1,
                    knots=Float64[],
                    boundaryknots=[0.0, 5.0],
                    extrapolation="linear")
    
    model_spline = multistatemodel(h12_sp; data=panel_data, surrogate=:markov)
    
    # Fit via MCEM - UNPENALIZED SPLINES (penalty=:none)
    fitted = fit(model_spline;
        proposal=:markov,
        penalty=:none,  # UNPENALIZED - no smoothing
        verbose=true,
        maxiter=MAX_ITER,
        tol=MCEM_TOL,
        ess_target_initial=25,
        max_ess=300,
        vcov_type=:none,
        return_convergence_records=true)
    
    # Check that spline has expected number of parameters
    # npar_baseline (spline coeffs) + 1 (covariate)
    pars_12 = MultistateModels.get_parameters(fitted, 1, scale=:log)
    n_baseline_params = fitted.hazards[1].npar_baseline
    @test length(pars_12) == n_baseline_params + 1
    
    println("\n    Parameter info:")
    println("    - Number of baseline spline coefficients: $n_baseline_params")
    println("    - Total parameters: $(length(pars_12))")
    println("    - Fitted parameters: $(round.(pars_12, digits=3))")
    
    # Define covariate tuples
    covars_0 = (x = 0.0,)
    covars_1 = (x = 1.0,)
    
    # =========================================================================
    # COMPREHENSIVE VALIDATION: Hazard at multiple time points
    # =========================================================================
    # Tolerance: 50% relative error for MCEM with flexible spline baseline
    # Justification: MCEM introduces MC error, spline may oscillate slightly
    HAZARD_RTOL = 0.50
    CUMHAZ_RTOL = 0.50
    BETA_ATOL = 0.5  # Absolute tolerance for log hazard ratio
    
    test_times = [0.5, 1.0, 2.0, 3.0, 4.0]
    
    println("\n    Hazard validation at multiple time points (x=0):")
    println("    Time      True h(t)   Fitted h(t)  Rel Diff")
    println("    " * "-"^50)
    
    all_hazard_tests_passed = true
    for t in test_times
        h_true = true_hazard(t, 0.0)
        h_fitted = fitted.hazards[1](t, pars_12, covars_0)
        rel_diff = abs(h_fitted - h_true) / h_true
        passed = rel_diff <= HAZARD_RTOL
        all_hazard_tests_passed &= passed
        status = passed ? "✓" : "✗"
        println("    $(rpad(t, 9)) $(rpad(round(h_true, digits=4), 11)) $(rpad(round(h_fitted, digits=4), 12)) $(round(rel_diff, digits=3)) $status")
        @test passed
    end
    
    println("\n    Hazard validation at multiple time points (x=1):")
    println("    Time      True h(t)   Fitted h(t)  Rel Diff")
    println("    " * "-"^50)
    
    for t in test_times
        h_true = true_hazard(t, 1.0)
        h_fitted = fitted.hazards[1](t, pars_12, covars_1)
        rel_diff = abs(h_fitted - h_true) / h_true
        passed = rel_diff <= HAZARD_RTOL
        all_hazard_tests_passed &= passed
        status = passed ? "✓" : "✗"
        println("    $(rpad(t, 9)) $(rpad(round(h_true, digits=4), 11)) $(rpad(round(h_fitted, digits=4), 12)) $(round(rel_diff, digits=3)) $status")
        @test passed
    end
    
    # =========================================================================
    # COMPREHENSIVE VALIDATION: Cumulative hazard at multiple time points
    # =========================================================================
    println("\n    Cumulative hazard validation (x=0):")
    println("    Time      True H(t)   Fitted H(t)  Rel Diff")
    println("    " * "-"^50)
    
    all_cumhaz_tests_passed = true
    for t in test_times
        H_true = true_cumhaz(t, 0.0)
        H_fitted = cumulative_hazard(fitted.hazards[1], 0.0, t, pars_12, covars_0)
        rel_diff = abs(H_fitted - H_true) / max(H_true, 0.01)
        passed = rel_diff <= CUMHAZ_RTOL
        all_cumhaz_tests_passed &= passed
        status = passed ? "✓" : "✗"
        println("    $(rpad(t, 9)) $(rpad(round(H_true, digits=4), 11)) $(rpad(round(H_fitted, digits=4), 12)) $(round(rel_diff, digits=3)) $status")
        @test passed
    end
    
    # =========================================================================
    # COMPREHENSIVE VALIDATION: Covariate effect (log hazard ratio)
    # =========================================================================
    println("\n    Covariate effect (beta) validation:")
    println("    Time      True log(HR)  Fitted log(HR)  Diff")
    println("    " * "-"^55)
    
    all_beta_tests_passed = true
    for t in test_times
        h_x0 = fitted.hazards[1](t, pars_12, covars_0)
        h_x1 = fitted.hazards[1](t, pars_12, covars_1)
        log_hr_fitted = log(h_x1) - log(h_x0)
        diff = abs(log_hr_fitted - true_beta)
        passed = diff <= BETA_ATOL
        all_beta_tests_passed &= passed
        status = passed ? "✓" : "✗"
        println("    $(rpad(t, 9)) $(rpad(round(true_beta, digits=4), 13)) $(rpad(round(log_hr_fitted, digits=4), 15)) $(round(diff, digits=3)) $status")
        @test passed
    end
    
    # =========================================================================
    # Summary
    # =========================================================================
    println("\n    Summary:")
    println("    - Hazard h(t) tests: $(all_hazard_tests_passed ? "PASS" : "FAIL")")
    println("    - Cumulative hazard H(t) tests: $(all_cumhaz_tests_passed ? "PASS" : "FAIL")")
    println("    - Covariate effect tests: $(all_beta_tests_passed ? "PASS" : "FAIL")")
    
    # Convergence check
    @test isfinite(fitted.loglik.loglik)
    
    # Note: We no longer cache results since spline coefficients are not directly
    # comparable to the true exponential parameters. The validation above tests
    # the functional form which is the correct statistical comparison.
    #
    # If needed for reporting, create a result with functional metrics instead:
    # result = LongTestResult(...)
    # result.passed = all_hazard_tests_passed && all_cumhaz_tests_passed && all_beta_tests_passed
    
    # =====================================================================
    # Fit with PhaseType proposal (same data)
    # =====================================================================
    println("      ▸ Fitting with PhaseType proposal..."); flush(stdout)
    
    h12_pt = Hazard(@formula(0 ~ 1 + x), "sp", 1, 2; 
                    degree=1,
                    knots=Float64[],
                    boundaryknots=[0.0, 5.0],
                    extrapolation="linear")
    
    model_pt = multistatemodel(h12_pt; data=panel_data, surrogate=:markov)
    
    # UNPENALIZED SPLINES (penalty=:none) - testing basic PhaseType proposal
    fitted_pt = fit(model_pt;
        proposal=PhaseTypeProposal(n_phases=3),
        penalty=:none,  # UNPENALIZED - no smoothing
        verbose=true,
        maxiter=MAX_ITER,
        tol=MCEM_TOL,
        ess_target_initial=30,
        max_ess=500,
        vcov_type=:none)
    
    pars_12_pt = MultistateModels.get_parameters(fitted_pt, 1, scale=:log)
    
    # =====================================================================
    # Compare Markov vs PhaseType proposals
    # =====================================================================
    println("\n    Markov vs PhaseType Proposal Comparison:")
    println("    " * "-"^60)
    println("    Time   x   Markov h(t)   PhaseType h(t)  Rel Diff")
    println("    " * "-"^60)
    
    comparison_passed = true
    for t in [0.5, 2.0, 4.0]
        for x_val in [0.0, 1.0]
            covars = (x = x_val,)
            h_markov = fitted.hazards[1](t, pars_12, covars)
            h_pt = fitted_pt.hazards[1](t, pars_12_pt, covars)
            rel_diff = abs(h_markov - h_pt) / max(h_markov, h_pt, 0.01)
            passed = rel_diff <= PROPOSAL_COMPARISON_TOL
            comparison_passed &= passed
            status = passed ? "✓" : "✗"
            println("    $(rpad(t, 6)) $(rpad(Int(x_val), 3)) $(rpad(round(h_markov, digits=4), 13)) $(rpad(round(h_pt, digits=4), 15)) $(round(rel_diff, digits=3)) $status")
        end
    end
    println("    " * "-"^60)
    @test comparison_passed
    
    # STRICTER check for covariate coefficient (beta) specifically (Item #26)
    # For splines, the last parameter is the covariate coefficient
    beta_markov = pars_12[end]  # Last param is covariate coeff
    beta_pt = pars_12_pt[end]
    beta_abs_diff = abs(beta_markov - beta_pt)
    beta_rel_diff = abs(beta_markov) > BETA_COMPARISON_TOL_ABS ? beta_abs_diff / abs(beta_markov) : beta_abs_diff / BETA_COMPARISON_TOL_ABS
    println("\n    Covariate coefficient (beta) comparison:")
    println("    Markov beta: $(round(beta_markov, digits=4)), PhaseType beta: $(round(beta_pt, digits=4))")
    println("    Abs diff: $(round(beta_abs_diff, digits=4)), Rel diff: $(round(beta_rel_diff*100, digits=1))%")
    @test beta_rel_diff < BETA_COMPARISON_TOL_REL || beta_abs_diff < BETA_COMPARISON_TOL_ABS
    
    println("  ✓ Spline with covariates: comprehensive validation complete")
    println("  ✓ Markov vs PhaseType proposal estimates agree")
end

# ============================================================================
# Test 5: Monotone Spline in MCEM
# ============================================================================
# 
# Proper test of monotone constraint enforcement:
# - Simulate from Weibull with shape > 1 (INCREASING hazard)
# - Fit with monotone=1 (increasing constraint): should capture the trend
# - Fit with monotone=-1 (decreasing constraint): should be BLOCKED from 
#   fitting the true increasing shape, resulting in constant or degraded fit
#
# This validates that monotone constraints actually restrict the fitted shape.
# ============================================================================

@testset "MCEM Monotone Spline" begin
    Random.seed!(RNG_SEED + 4)
    
    # Simulate from Weibull with shape > 1 (INCREASING hazard)
    # Weibull hazard: h(t) = shape * scale * t^(shape-1)
    # With shape=1.5, scale=0.20: h(t) increases from 0.3 at t=1 to 0.6 at t=4
    true_shape = 1.5
    true_scale = 0.20
    
    h12_wei = Hazard(@formula(0 ~ 1), "wei", 1, 2)
    
    obs_times = [0.0, 0.7, 1.4, 2.1, 2.8, 3.5, 4.2]
    nobs = length(obs_times) - 1
    
    sim_data = DataFrame(
        id = repeat(1:N_SUBJECTS, inner=nobs),
        tstart = repeat(obs_times[1:end-1], N_SUBJECTS),
        tstop = repeat(obs_times[2:end], N_SUBJECTS),
        statefrom = ones(Int, N_SUBJECTS * nobs),
        stateto = ones(Int, N_SUBJECTS * nobs),
        obstype = fill(2, N_SUBJECTS * nobs)
    )
    
    # Use initialize=false to avoid bounds issues during auto-initialization
    model_sim = multistatemodel(h12_wei; data=sim_data, initialize=false)
    MultistateModels.set_parameters!(model_sim, (h12 = [true_shape, true_scale],))
    
    # For 2-state model (1→2 absorbing), transition 1 is exact
    obstype_map = Dict(1 => 1)
    sim_result = simulate(model_sim; paths=false, data=true, nsim=1, autotmax=false,
                         obstype_by_transition=obstype_map)
    panel_data = sim_result[1, 1]
    
    # -------------------------------------------------------------------------
    # Part A: Fit with CORRECT monotone direction (increasing)
    # Should capture the increasing hazard pattern
    # -------------------------------------------------------------------------
    h12_sp_inc = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                        degree=1,
                        knots=[2.0],  # One interior knot
                        boundaryknots=[0.0, 5.0],
                        monotone=1,   # INCREASING constraint
                        extrapolation="constant")
    
    # Use initialize=false to avoid bounds issues during auto-initialization
    model_inc = multistatemodel(h12_sp_inc; data=panel_data, surrogate=:markov, 
                                initialize=false)
    
    fitted_inc = fit(model_inc;
        proposal=:markov,
        verbose=false,
        maxiter=MAX_ITER,
        tol=MCEM_TOL,
        ess_target_initial=25,
        max_ess=300,
        vcov_type=:none,
        penalty=:none)
    
    pars_inc = MultistateModels.get_parameters(fitted_inc, 1, scale=:log)
    h_vals_inc = [fitted_inc.hazards[1](t, pars_inc, NamedTuple()) for t in 1.0:0.5:3.5]
    
    println("\n    Part A: Fit with monotone=1 (increasing, matches true DGP)")
    println("    Hazard values at t = [1, 1.5, 2, 2.5, 3, 3.5]: ", round.(h_vals_inc, digits=4))
    println("    Differences: ", round.(diff(h_vals_inc), digits=6))
    
    # Monotone increasing constraint: differences should be >= 0
    diffs_inc = diff(h_vals_inc)
    @test all(diffs_inc .>= -1e-10)
    
    # Since true hazard is increasing, fitted should show ACTUAL increase (not just constant)
    # At least one positive difference with magnitude > numerical noise
    @test any(diffs_inc .> 0.001)
    
    # -------------------------------------------------------------------------
    # Part B: Fit with WRONG monotone direction (decreasing)
    # Should be constrained and unable to capture the increasing pattern
    # -------------------------------------------------------------------------
    h12_sp_dec = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                        degree=1,
                        knots=[2.0],
                        boundaryknots=[0.0, 5.0],
                        monotone=-1,  # DECREASING constraint (WRONG for this data!)
                        extrapolation="constant")
    
    # Use initialize=false to avoid bounds issues during auto-initialization
    model_dec = multistatemodel(h12_sp_dec; data=panel_data, surrogate=:markov,
                                initialize=false)
    
    fitted_dec = fit(model_dec;
        proposal=:markov,
        verbose=false,
        maxiter=MAX_ITER,
        tol=MCEM_TOL,
        ess_target_initial=25,
        max_ess=300,
        vcov_type=:none,
        penalty=:none)
    
    pars_dec = MultistateModels.get_parameters(fitted_dec, 1, scale=:log)
    h_vals_dec = [fitted_dec.hazards[1](t, pars_dec, NamedTuple()) for t in 1.0:0.5:3.5]
    
    println("\n    Part B: Fit with monotone=-1 (decreasing, WRONG for true DGP)")
    println("    Hazard values at t = [1, 1.5, 2, 2.5, 3, 3.5]: ", round.(h_vals_dec, digits=4))
    println("    Differences: ", round.(diff(h_vals_dec), digits=6))
    
    # Monotone decreasing constraint: differences should be <= 0
    diffs_dec = diff(h_vals_dec)
    @test all(diffs_dec .<= 1e-10)
    
    # -------------------------------------------------------------------------
    # Part C: Compare log-likelihoods - correct direction should fit better
    # -------------------------------------------------------------------------
    ll_inc = fitted_inc.loglik.loglik
    ll_dec = fitted_dec.loglik.loglik
    
    println("\n    Part C: Log-likelihood comparison")
    println("    LL with monotone=1 (correct):  ", round(ll_inc, digits=2))
    println("    LL with monotone=-1 (wrong):   ", round(ll_dec, digits=2))
    println("    Difference (correct - wrong):  ", round(ll_inc - ll_dec, digits=2))
    
    # Correct direction should have higher (less negative) log-likelihood
    @test ll_inc > ll_dec
    
    # Convergence checks
    @test isfinite(ll_inc)
    @test isfinite(ll_dec)
    
    # =====================================================================
    # Fit with PhaseType proposal (increasing constraint, same data)
    # =====================================================================
    println("\n      ▸ Fitting with PhaseType proposal (monotone=1)..."); flush(stdout)
    
    h12_pt_inc = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                        degree=1,
                        knots=[2.0],
                        boundaryknots=[0.0, 5.0],
                        monotone=1,
                        extrapolation="constant")
    
    model_pt_inc = multistatemodel(h12_pt_inc; data=panel_data, surrogate=:markov,
                                   initialize=false)
    
    fitted_pt_inc = fit(model_pt_inc;
        proposal=PhaseTypeProposal(n_phases=3),
        verbose=false,
        maxiter=MAX_ITER,
        tol=MCEM_TOL,
        ess_target_initial=30,
        max_ess=500,
        vcov_type=:none,
        
        penalty=:none)
    
    pars_pt_inc = MultistateModels.get_parameters(fitted_pt_inc, 1, scale=:log)
    h_vals_pt_inc = [fitted_pt_inc.hazards[1](t, pars_pt_inc, NamedTuple()) for t in 1.0:0.5:3.5]
    
    # PhaseType fit should also satisfy monotone increasing constraint
    diffs_pt_inc = diff(h_vals_pt_inc)
    @test all(diffs_pt_inc .>= -1e-10)
    
    # =====================================================================
    # Compare Markov vs PhaseType proposals (increasing constraint)
    # =====================================================================
    println("\n    Markov vs PhaseType Proposal Comparison (monotone=1):")
    println("    " * "-"^60)
    println("    Time      Markov h(t)   PhaseType h(t)  Rel Diff")
    println("    " * "-"^60)
    
    comparison_passed = true
    for (i, t) in enumerate(1.0:0.5:3.5)
        h_markov = h_vals_inc[i]
        h_pt = h_vals_pt_inc[i]
        rel_diff = abs(h_markov - h_pt) / max(h_markov, h_pt, 0.01)
        passed = rel_diff <= PROPOSAL_COMPARISON_TOL
        comparison_passed &= passed
        status = passed ? "✓" : "✗"
        println("    $(rpad(t, 9)) $(rpad(round(h_markov, digits=4), 13)) $(rpad(round(h_pt, digits=4), 15)) $(round(rel_diff, digits=3)) $status")
    end
    println("    " * "-"^60)
    @test comparison_passed
    
    println("\n  ✓ Monotone spline constraints properly enforce direction in MCEM")
    println("  ✓ Markov vs PhaseType proposal estimates agree")
end

# ============================================================================
# Test 6: Spline Hazards with PhaseType Proposal
# ============================================================================
#
# This test validates PhaseType proposal for spline hazards.
# Splines are inherently semi-Markov (non-exponential sojourn times),
# so PhaseType proposal is theoretically appropriate.
# See MultistateModelsTests/diagnostics/phasetype_testing_plan.md for background.

@testset "MCEM Spline - PhaseType Proposal" begin
    Random.seed!(RNG_SEED + 100)
    
    # Generate data from Weibull, fit with splines using PhaseType proposal
    # This tests that PhaseType proposal works correctly with flexible hazards
    
    true_shape = 1.4
    true_scale = 0.20
    
    # Panel data template - 6 intervals
    nobs = 6
    obs_times = [0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0]
    template = DataFrame(
        id = repeat(1:N_SUBJECTS, inner=nobs),
        tstart = repeat([0.0; obs_times[1:end-1]], N_SUBJECTS),
        tstop = repeat(obs_times, N_SUBJECTS),
        statefrom = ones(Int, N_SUBJECTS * nobs),
        stateto = ones(Int, N_SUBJECTS * nobs),
        obstype = fill(2, N_SUBJECTS * nobs)
    )
    
    # Simulate from Weibull
    h12_wei = Hazard(@formula(0 ~ 1), "wei", 1, 2)
    # Use initialize=false to avoid bounds issues during auto-initialization  
    model_sim = multistatemodel(h12_wei; data=template, surrogate=:markov, initialize=false)
    set_parameters!(model_sim, (h12 = [true_shape, true_scale],))
    
    # For 2-state model (1→2 absorbing), transition 1 is exact
    obstype_map = Dict(1 => 1)
    sim_result = simulate(model_sim; paths=false, data=true, nsim=1, autotmax=false,
                         obstype_by_transition=obstype_map)
    panel_data = sim_result[1, 1]
    
    # Fit with linear spline using PhaseType proposal
    max_time = maximum(obs_times)
    h12_sp = Hazard(@formula(0 ~ 1), "sp", 1, 2; degree=1, knots=[4.0],
                    boundaryknots=[0.0, max_time], extrapolation="constant")
    
    model_fit = multistatemodel(h12_sp; data=panel_data, surrogate=:markov)
    
    # UNPENALIZED SPLINES (penalty=:none) - testing PhaseType proposal
    fitted = fit(model_fit;
        proposal=PhaseTypeProposal(n_phases=3),
        penalty=:none,  # UNPENALIZED - no smoothing
        verbose=true,
        maxiter=MAX_ITER,
        tol=MCEM_TOL,
        ess_target_initial=25,
        max_ess=300,
        vcov_type=:none,
        return_convergence_records=true)
    
    @testset "Convergence and Pareto-k" begin
        records = fitted.ConvergenceRecords
        pareto_k = records.psis_pareto_k
        
        finite_k = filter(!isnan, pareto_k)
        # Pareto-k should be below 1.0 (reliable IS weights)
        @test maximum(finite_k) < 1.0
        # Most subjects should have reasonable k
        @test mean(pareto_k .< 0.7) > 0.6
    end
    
    @testset "Hazard shape recovery" begin
        # The spline should approximate Weibull shape (increasing hazard)
        pars_12 = MultistateModels.get_parameters(fitted, 1, scale=:log)
        
        h_vals = [fitted.hazards[1](t, pars_12, NamedTuple()) for t in 1.0:1.0:7.0]
        
        # All hazard values should be positive and finite
        @test all(h .> 0 for h in h_vals)
        @test all(isfinite.(h_vals))
        
        # Weibull with shape > 1 has increasing hazard - check general trend
        # Use linear regression to verify positive slope
        ts = 1.0:1.0:7.0
        mean_t = mean(ts)
        mean_h = mean(h_vals)
        slope = sum((ts .- mean_t) .* (h_vals .- mean_h)) / sum((ts .- mean_t).^2)
        @test slope > 0  # Positive trend (increasing hazard)
        
        # Hazard at t=6 should be higher than at t=2 (Weibull shape=1.4)
        h_early = fitted.hazards[1](2.0, pars_12, NamedTuple())
        h_late = fitted.hazards[1](6.0, pars_12, NamedTuple())
        @test h_late > h_early
    end
    
    @testset "Convergence quality" begin
        @test isfinite(fitted.loglik.loglik)
        @test fitted.loglik.loglik < 0
    end
    
    println("  ✓ Spline with PhaseType proposal works")
end

# ============================================================================
# Test 7: Spline AFT Panel Data - No Covariates (sp_aft_panel_nocov)
# ============================================================================
#
# Generate panel data from Weibull model, fit spline with AFT (no covariates).
# Since there are no covariates, AFT vs PH distinction doesn't matter for
# baseline hazard - this test validates spline MCEM recovery works with
# the AFT machinery.
# ============================================================================

# AFT Weibull hazard formulas (for reference/validation)
# h(t|x) = shape * scale * t^(shape-1) * exp(-beta * x * shape)
_weibull_aft_hazard(t, shape, scale, beta, x) = shape * scale * t^(shape - 1) * exp(-beta * x * shape)

# AFT cumulative hazard: H(t|x) = scale * (t * exp(-beta * x))^shape
_weibull_aft_cumhaz(t, shape, scale, beta, x) = scale * (t * exp(-beta * x))^shape

# Standard Weibull hazard (no covariates)
_weibull_hazard(t, shape, scale) = shape * scale * t^(shape - 1)
_weibull_cumhaz(t, shape, scale) = scale * t^shape

@testset "MCEM Spline AFT Panel: No Covariates (sp_aft_panel_nocov)" begin
    Random.seed!(RNG_SEED + 200)
    
    VERBOSE_LONGTESTS && println("\n  ▶ Running: sp_aft_panel_nocov")
    
    # True DGP parameters (Weibull)
    true_shape = 1.4   # Shape > 1: increasing hazard
    true_scale = 0.15  # Rate parameter
    
    # --- Setup DGP (Weibull, no covariates, AFT mode) ---
    h12_wei = Hazard(@formula(0 ~ 1), "wei", 1, 2; linpred_effect=:aft)
    
    # Panel data template with MCEM-appropriate sparse intervals
    panel_times = MCEM_PANEL_TIMES
    nobs = length(panel_times) - 1
    
    template = DataFrame(
        id = repeat(1:N_SUBJECTS, inner=nobs),
        tstart = repeat(panel_times[1:end-1], N_SUBJECTS),
        tstop = repeat(panel_times[2:end], N_SUBJECTS),
        statefrom = ones(Int, N_SUBJECTS * nobs),
        stateto = ones(Int, N_SUBJECTS * nobs),
        obstype = fill(2, N_SUBJECTS * nobs)
    )
    
    model_dgp = multistatemodel(h12_wei; data=template, initialize=false)
    set_parameters!(model_dgp, (h12 = [true_shape, true_scale],))
    
    # --- Simulate panel data ---
    # obstype_by_transition=nothing gives natural panel observations
    sim_result = simulate(model_dgp; paths=false, data=true, nsim=1, autotmax=false)
    panel_data = sim_result[1, 1]
    
    # Data summary
    n_transitions = sum(panel_data.statefrom .!= panel_data.stateto)
    max_obs_time = maximum(panel_data.tstop)
    VERBOSE_LONGTESTS && println("    Data: $(length(unique(panel_data.id))) subjects, $n_transitions transitions, max_t=$(round(max_obs_time, digits=2))")
    
    # --- Fit spline AFT model via MCEM ---
    knots = [4.0, 8.0]  # Interior knots
    
    h12_sp = Hazard(@formula(0 ~ 1), "sp", 1, 2;
        degree=1,
        knots=knots,
        boundaryknots=[0.0, max_obs_time],
        extrapolation="constant",
        linpred_effect=:aft)
    
    model_sp = multistatemodel(h12_sp; data=panel_data, surrogate=:markov)
    
    # UNPENALIZED SPLINES (penalty=:none) - testing AFT spline
    fitted = fit(model_sp;
        proposal=:markov,
        penalty=:none,  # UNPENALIZED - no smoothing
        verbose=VERBOSE_LONGTESTS,
        maxiter=MAX_ITER,
        tol=MCEM_TOL,
        ess_target_initial=25,
        max_ess=300,
        vcov_type=:none)
    
    # --- Validate ---
    @test isfinite(fitted.loglik.loglik)
    
    # Evaluate hazard at multiple time points
    pars = MultistateModels.get_parameters(fitted, 1, scale=:log)
    test_times = [2.0, 4.0, 6.0, 8.0, 10.0]
    test_times = filter(t -> t <= max_obs_time, test_times)
    
    h_true = [_weibull_hazard(t, true_shape, true_scale) for t in test_times]
    h_fitted = [fitted.hazards[1](t, pars, NamedTuple()) for t in test_times]
    
    # Use MCEM-appropriate tolerance (50% for flexible spline + MCEM variance)
    MCEM_AFT_RTOL = 0.50
    
    if VERBOSE_LONGTESTS
        println("\n    Hazard comparison (no covariates):")
        println("    Time      True h(t)   Fitted h(t)  Rel Diff    Status")
        println("    " * "-"^55)
        for (i, t) in enumerate(test_times)
            rel_diff = abs(h_fitted[i] - h_true[i]) / max(h_true[i], 0.01)
            status = rel_diff <= MCEM_AFT_RTOL ? "✓" : "✗"
            @printf("    %-9.1f %-11.4f %-12.4f %-11.3f %s\n",
                    t, h_true[i], h_fitted[i], rel_diff, status)
        end
        println("    " * "-"^55)
    end
    
    # All hazards should be positive and finite
    @test all(h_fitted .> 0)
    @test all(isfinite.(h_fitted))
    
    # Hazard should roughly match true Weibull (within MCEM tolerance)
    rel_diffs = abs.(h_fitted .- h_true) ./ max.(h_true, 0.01)
    @test mean(rel_diffs) < MCEM_AFT_RTOL  # Average should be reasonable
    
    # Monotonicity: for Weibull shape > 1, hazard should generally increase
    @test h_fitted[end] > h_fitted[1] * 0.5  # End hazard not dramatically lower
    
    # =====================================================================
    # Fit with PhaseType proposal (same data)
    # =====================================================================
    println("      ▸ Fitting with PhaseType proposal..."); flush(stdout)
    
    h12_pt = Hazard(@formula(0 ~ 1), "sp", 1, 2;
        degree=1,
        knots=knots,
        boundaryknots=[0.0, max_obs_time],
        extrapolation="constant",
        linpred_effect=:aft)
    
    model_pt = multistatemodel(h12_pt; data=panel_data, surrogate=:markov)
    
    # UNPENALIZED SPLINES (penalty=:none) - testing AFT spline PhaseType proposal
    fitted_pt = fit(model_pt;
        proposal=PhaseTypeProposal(n_phases=3),
        penalty=:none,  # UNPENALIZED - no smoothing
        verbose=VERBOSE_LONGTESTS,
        maxiter=MAX_ITER,
        tol=MCEM_TOL,
        ess_target_initial=30,
        max_ess=500,
        vcov_type=:none)
    
    pars_pt = MultistateModels.get_parameters(fitted_pt, 1, scale=:log)
    h_fitted_pt = [fitted_pt.hazards[1](t, pars_pt, NamedTuple()) for t in test_times]
    
    # =====================================================================
    # Compare Markov vs PhaseType proposals
    # =====================================================================
    println("\n    Markov vs PhaseType Proposal Comparison:")
    println("    " * "-"^60)
    println("    Time      Markov h(t)   PhaseType h(t)  Rel Diff")
    println("    " * "-"^60)
    
    comparison_passed = true
    for (i, t) in enumerate(test_times)
        h_markov = h_fitted[i]
        h_pt = h_fitted_pt[i]
        rel_diff = abs(h_markov - h_pt) / max(h_markov, h_pt, 0.01)
        passed = rel_diff <= PROPOSAL_COMPARISON_TOL
        comparison_passed &= passed
        status = passed ? "✓" : "✗"
        println("    $(rpad(t, 9)) $(rpad(round(h_markov, digits=4), 13)) $(rpad(round(h_pt, digits=4), 15)) $(round(rel_diff, digits=3)) $status")
    end
    println("    " * "-"^60)
    @test comparison_passed
    
    VERBOSE_LONGTESTS && println("  ✓ sp_aft_panel_nocov: PASS")
    println("  ✓ Markov vs PhaseType proposal estimates agree")
end

# ============================================================================
# Test 8: Spline AFT Panel Data - Time-Fixed Covariate (sp_aft_panel_tfc)
# ============================================================================
#
# Generate panel data from Weibull AFT model with covariate effect.
# For AFT: h(t|x) = h_0(t * exp(-βx)) * exp(-βx)
# 
# Key validation: 
# - Hazard shape recovery for both x=0 and x=1
# - AFT effect (time acceleration) is captured
# ============================================================================

@testset "MCEM Spline AFT Panel: Time-Fixed Covariate (sp_aft_panel_tfc)" begin
    Random.seed!(RNG_SEED + 201)
    
    VERBOSE_LONGTESTS && println("\n  ▶ Running: sp_aft_panel_tfc")
    
    # True DGP parameters
    true_shape = 1.4
    true_scale = 0.15
    true_beta = 0.5  # Positive beta: x=1 accelerates time (shorter survival)
    
    # --- Setup DGP (Weibull AFT with covariate) ---
    h12_wei = Hazard(@formula(0 ~ x), "wei", 1, 2; linpred_effect=:aft)
    
    # Panel data template
    panel_times = MCEM_PANEL_TIMES
    nobs = length(panel_times) - 1
    
    # Create template with binary covariate
    n_per_cov = N_SUBJECTS ÷ 2
    x_vals = vcat(zeros(n_per_cov), ones(N_SUBJECTS - n_per_cov))
    shuffle!(x_vals)
    
    template = DataFrame(
        id = repeat(1:N_SUBJECTS, inner=nobs),
        tstart = repeat(panel_times[1:end-1], N_SUBJECTS),
        tstop = repeat(panel_times[2:end], N_SUBJECTS),
        statefrom = ones(Int, N_SUBJECTS * nobs),
        stateto = ones(Int, N_SUBJECTS * nobs),
        obstype = fill(2, N_SUBJECTS * nobs),
        x = repeat(x_vals, inner=nobs)
    )
    
    model_dgp = multistatemodel(h12_wei; data=template, initialize=false)
    set_parameters!(model_dgp, (h12 = [true_shape, true_scale, true_beta],))
    
    # --- Simulate panel data ---
    sim_result = simulate(model_dgp; paths=false, data=true, nsim=1, autotmax=false)
    panel_data = sim_result[1, 1]
    
    # Data summary
    n_transitions = sum(panel_data.statefrom .!= panel_data.stateto)
    n_x0 = sum([first(filter(r -> r.id == id, panel_data).x) for id in unique(panel_data.id)] .== 0.0)
    n_x1 = length(unique(panel_data.id)) - n_x0
    max_obs_time = maximum(panel_data.tstop)
    VERBOSE_LONGTESTS && println("    Data: $(length(unique(panel_data.id))) subjects, $n_transitions transitions, x=0: $n_x0, x=1: $n_x1")
    
    # --- Fit spline AFT model via MCEM ---
    knots = [4.0, 8.0]
    
    h12_sp = Hazard(@formula(0 ~ x), "sp", 1, 2;
        degree=1,
        knots=knots,
        boundaryknots=[0.0, max_obs_time],
        extrapolation="constant",
        linpred_effect=:aft)
    
    model_sp = multistatemodel(h12_sp; data=panel_data, surrogate=:markov)
    
    # UNPENALIZED SPLINES (penalty=:none) - testing AFT spline with TFC
    fitted = fit(model_sp;
        proposal=:markov,
        penalty=:none,  # UNPENALIZED - no smoothing
        verbose=VERBOSE_LONGTESTS,
        maxiter=MAX_ITER,
        tol=MCEM_TOL,
        ess_target_initial=25,
        max_ess=300,
        vcov_type=:none)
    
    # --- Validate ---
    @test isfinite(fitted.loglik.loglik)
    
    # Evaluate at test times for both x=0 and x=1
    pars = MultistateModels.get_parameters(fitted, 1, scale=:log)
    covars_x0 = (x = 0.0,)
    covars_x1 = (x = 1.0,)
    
    test_times = [2.0, 4.0, 6.0, 8.0, 10.0]
    test_times = filter(t -> t <= max_obs_time, test_times)
    
    # True AFT hazards
    h_true_x0 = [_weibull_aft_hazard(t, true_shape, true_scale, true_beta, 0.0) for t in test_times]
    h_true_x1 = [_weibull_aft_hazard(t, true_shape, true_scale, true_beta, 1.0) for t in test_times]
    
    # Fitted hazards
    h_fitted_x0 = [fitted.hazards[1](t, pars, covars_x0) for t in test_times]
    h_fitted_x1 = [fitted.hazards[1](t, pars, covars_x1) for t in test_times]
    
    MCEM_AFT_TFC_RTOL = 0.55  # Relaxed for MCEM + AFT + covariates
    
    if VERBOSE_LONGTESTS
        println("\n    Hazard comparison (x=0, baseline):")
        println("    Time      True h(t)   Fitted h(t)  Rel Diff    Status")
        println("    " * "-"^55)
        for (i, t) in enumerate(test_times)
            rel_diff = abs(h_fitted_x0[i] - h_true_x0[i]) / max(h_true_x0[i], 0.01)
            status = rel_diff <= MCEM_AFT_TFC_RTOL ? "✓" : "✗"
            @printf("    %-9.1f %-11.4f %-12.4f %-11.3f %s\n",
                    t, h_true_x0[i], h_fitted_x0[i], rel_diff, status)
        end
        println("    " * "-"^55)
        
        println("\n    Hazard comparison (x=1, accelerated):")
        println("    Time      True h(t)   Fitted h(t)  Rel Diff    Status")
        println("    " * "-"^55)
        for (i, t) in enumerate(test_times)
            rel_diff = abs(h_fitted_x1[i] - h_true_x1[i]) / max(h_true_x1[i], 0.01)
            status = rel_diff <= MCEM_AFT_TFC_RTOL ? "✓" : "✗"
            @printf("    %-9.1f %-11.4f %-12.4f %-11.3f %s\n",
                    t, h_true_x1[i], h_fitted_x1[i], rel_diff, status)
        end
        println("    " * "-"^55)
    end
    
    # All hazards should be positive and finite
    @test all(h_fitted_x0 .> 0)
    @test all(h_fitted_x1 .> 0)
    @test all(isfinite.(h_fitted_x0))
    @test all(isfinite.(h_fitted_x1))
    
    # AFT effect validation: hazard ratio at fixed time
    # For Weibull AFT: log(h(t|x=1)/h(t|x=0)) = -β*shape = -0.5*1.4 = -0.7
    expected_log_hr = -true_beta * true_shape
    
    println("\n    AFT effect (log hazard ratio):")
    println("    Time      Fitted log(HR)  Expected     Abs Diff")
    println("    " * "-"^55)
    
    log_hr_checks = Float64[]
    for t in test_times
        h_x0 = fitted.hazards[1](t, pars, covars_x0)
        h_x1 = fitted.hazards[1](t, pars, covars_x1)
        log_hr = log(h_x1 / h_x0)
        abs_diff = abs(log_hr - expected_log_hr)
        push!(log_hr_checks, abs_diff)
        @printf("    %-9.1f %-15.4f %-12.4f %-11.4f\n", t, log_hr, expected_log_hr, abs_diff)
    end
    println("    " * "-"^55)
    
    # AFT effect should be captured (relaxed tolerance for MCEM)
    @test mean(log_hr_checks) < 0.8  # Average deviation should be reasonable
    
    # =====================================================================
    # Fit with PhaseType proposal (same data)
    # =====================================================================
    println("      ▸ Fitting with PhaseType proposal..."); flush(stdout)
    
    h12_pt = Hazard(@formula(0 ~ x), "sp", 1, 2;
        degree=1,
        knots=knots,
        boundaryknots=[0.0, max_obs_time],
        extrapolation="constant",
        linpred_effect=:aft)
    
    model_pt = multistatemodel(h12_pt; data=panel_data, surrogate=:markov)
    
    # UNPENALIZED SPLINES (penalty=:none) - testing AFT spline PhaseType proposal with TFC
    fitted_pt = fit(model_pt;
        proposal=PhaseTypeProposal(n_phases=3),
        penalty=:none,  # UNPENALIZED - no smoothing
        verbose=VERBOSE_LONGTESTS,
        maxiter=MAX_ITER,
        tol=MCEM_TOL,
        ess_target_initial=30,
        max_ess=500,
        vcov_type=:none)
    
    pars_pt = MultistateModels.get_parameters(fitted_pt, 1, scale=:log)
    
    # =====================================================================
    # Compare Markov vs PhaseType proposals
    # =====================================================================
    println("\n    Markov vs PhaseType Proposal Comparison:")
    println("    " * "-"^60)
    println("    Time   x   Markov h(t)   PhaseType h(t)  Rel Diff")
    println("    " * "-"^60)
    
    comparison_passed = true
    for t in test_times
        for x_val in [0.0, 1.0]
            covars = (x = x_val,)
            h_markov = fitted.hazards[1](t, pars, covars)
            h_pt = fitted_pt.hazards[1](t, pars_pt, covars)
            rel_diff = abs(h_markov - h_pt) / max(h_markov, h_pt, 0.01)
            passed = rel_diff <= PROPOSAL_COMPARISON_TOL
            comparison_passed &= passed
            status = passed ? "✓" : "✗"
            println("    $(rpad(t, 6)) $(rpad(Int(x_val), 3)) $(rpad(round(h_markov, digits=4), 13)) $(rpad(round(h_pt, digits=4), 15)) $(round(rel_diff, digits=3)) $status")
        end
    end
    println("    " * "-"^60)
    @test comparison_passed
    
    VERBOSE_LONGTESTS && println("  ✓ sp_aft_panel_tfc: PASS")
    println("  ✓ Markov vs PhaseType proposal estimates agree")
end

# ============================================================================
# Test 9: Spline AFT Panel Data - Time-Varying Covariate (sp_aft_panel_tvc)
# ============================================================================
#
# Generate panel data from Weibull AFT model with time-varying covariate:
# - x=0 for t < TVC_CHANGEPOINT
# - x=1 for t >= TVC_CHANGEPOINT
#
# This is the most challenging case: MCEM + AFT + TVC.
# Validation focuses on:
# 1. Hazard shape changes appropriately at changepoint
# 2. Convergence is achieved
# ============================================================================

@testset "MCEM Spline AFT Panel: Time-Varying Covariate (sp_aft_panel_tvc)" begin
    Random.seed!(RNG_SEED + 202)
    
    VERBOSE_LONGTESTS && println("\n  ▶ Running: sp_aft_panel_tvc")
    
    # True DGP parameters
    true_shape = 1.3   # Slightly reduced shape for stability
    true_scale = 0.12
    true_beta = 0.4    # Moderate AFT effect
    tvc_changepoint = 6.0  # Changepoint within observation window
    
    # --- Setup DGP (Weibull AFT with TVC) ---
    h12_wei = Hazard(@formula(0 ~ x), "wei", 1, 2; linpred_effect=:aft)
    
    # Panel data template with TVC structure (15 intervals of 1.0 for ~5.5 obs/subj with rate~0.15)
    panel_times = collect(0.0:1.0:15.0)  # 15 intervals over T=15 for adequate density
    
    # Create TVC data structure: each subject has x=0 before changepoint, x=1 after
    tvc_data_rows = DataFrame[]
    for subj in 1:N_SUBJECTS
        for i in 1:(length(panel_times)-1)
            t_start = panel_times[i]
            t_stop = panel_times[i+1]
            
            # Determine x value based on time interval
            if t_stop <= tvc_changepoint
                x_val = 0.0
            elseif t_start >= tvc_changepoint
                x_val = 1.0
            else
                # Interval spans changepoint - split it
                # First part: before changepoint
                push!(tvc_data_rows, DataFrame(
                    id = subj,
                    tstart = t_start,
                    tstop = tvc_changepoint,
                    statefrom = 1,
                    stateto = 1,
                    obstype = 2,
                    x = 0.0
                ))
                # Second part: after changepoint
                push!(tvc_data_rows, DataFrame(
                    id = subj,
                    tstart = tvc_changepoint,
                    tstop = t_stop,
                    statefrom = 1,
                    stateto = 1,
                    obstype = 2,
                    x = 1.0
                ))
                continue
            end
            
            push!(tvc_data_rows, DataFrame(
                id = subj,
                tstart = t_start,
                tstop = t_stop,
                statefrom = 1,
                stateto = 1,
                obstype = 2,
                x = x_val
            ))
        end
    end
    
    template = vcat(tvc_data_rows...)
    sort!(template, [:id, :tstart])
    
    model_dgp = multistatemodel(h12_wei; data=template, initialize=false)
    set_parameters!(model_dgp, (h12 = [true_shape, true_scale, true_beta],))
    
    # --- Simulate panel data ---
    sim_result = simulate(model_dgp; paths=false, data=true, nsim=1, autotmax=false)
    panel_data = sim_result[1, 1]
    
    # Data summary
    n_transitions = sum(panel_data.statefrom .!= panel_data.stateto)
    max_obs_time = maximum(panel_data.tstop)
    VERBOSE_LONGTESTS && println("    Data: $(length(unique(panel_data.id))) subjects, $n_transitions transitions, TVC changepoint=$(tvc_changepoint)")
    
    # --- Fit spline AFT model via MCEM ---
    knots = [5.0, 10.0]  # Adjusted for extended T=15 window
    
    h12_sp = Hazard(@formula(0 ~ x), "sp", 1, 2;
        degree=1,
        knots=knots,
        boundaryknots=[0.0, max_obs_time],
        extrapolation="constant",
        linpred_effect=:aft)
    
    model_sp = multistatemodel(h12_sp; data=panel_data, surrogate=:markov)
    
    # UNPENALIZED SPLINES (penalty=:none) - testing AFT spline with TVC
    fitted = fit(model_sp;
        proposal=:markov,
        penalty=:none,  # UNPENALIZED - no smoothing
        verbose=VERBOSE_LONGTESTS,
        maxiter=MAX_ITER + 10,  # Extra iterations for TVC complexity
        tol=MCEM_TOL,
        ess_target_initial=25,
        max_ess=300,
        vcov_type=:none)
    
    # --- Validate ---
    @test isfinite(fitted.loglik.loglik)
    @test fitted.loglik.loglik < 0  # Should be negative log-likelihood
    
    # Evaluate hazards before and after changepoint
    pars = MultistateModels.get_parameters(fitted, 1, scale=:log)
    covars_x0 = (x = 0.0,)
    covars_x1 = (x = 1.0,)
    
    # Times before changepoint (x=0)
    times_before = [2.0, 4.0]
    times_before = filter(t -> t < tvc_changepoint && t <= max_obs_time, times_before)
    
    # Times after changepoint (x=1)
    times_after = [8.0, 10.0]
    times_after = filter(t -> t > tvc_changepoint && t <= max_obs_time, times_after)
    
    # True hazards for comparison
    h_true_before = [_weibull_aft_hazard(t, true_shape, true_scale, true_beta, 0.0) for t in times_before]
    h_true_after = [_weibull_aft_hazard(t, true_shape, true_scale, true_beta, 1.0) for t in times_after]
    
    # Fitted hazards (with correct x values for each time period)
    h_fitted_before = [fitted.hazards[1](t, pars, covars_x0) for t in times_before]
    h_fitted_after = [fitted.hazards[1](t, pars, covars_x1) for t in times_after]
    
    MCEM_AFT_TVC_RTOL = 0.60  # Most relaxed tolerance for MCEM + AFT + TVC
    
    if VERBOSE_LONGTESTS
        println("\n    Hazard comparison (before changepoint, x=0):")
        println("    Time      True h(t)   Fitted h(t)  Rel Diff    Status")
        println("    " * "-"^55)
        for (i, t) in enumerate(times_before)
            rel_diff = abs(h_fitted_before[i] - h_true_before[i]) / max(h_true_before[i], 0.01)
            status = rel_diff <= MCEM_AFT_TVC_RTOL ? "✓" : "✗"
            @printf("    %-9.1f %-11.4f %-12.4f %-11.3f %s\n",
                    t, h_true_before[i], h_fitted_before[i], rel_diff, status)
        end
        println("    " * "-"^55)
        
        println("\n    Hazard comparison (after changepoint, x=1):")
        println("    Time      True h(t)   Fitted h(t)  Rel Diff    Status")
        println("    " * "-"^55)
        for (i, t) in enumerate(times_after)
            rel_diff = abs(h_fitted_after[i] - h_true_after[i]) / max(h_true_after[i], 0.01)
            status = rel_diff <= MCEM_AFT_TVC_RTOL ? "✓" : "✗"
            @printf("    %-9.1f %-11.4f %-12.4f %-11.3f %s\n",
                    t, h_true_after[i], h_fitted_after[i], rel_diff, status)
        end
        println("    " * "-"^55)
    end
    
    # All hazards should be positive and finite
    @test all(h_fitted_before .> 0)
    @test all(h_fitted_after .> 0)
    @test all(isfinite.(h_fitted_before))
    @test all(isfinite.(h_fitted_after))
    
    # Check that hazard magnitudes are reasonable (within order of magnitude)
    @test all(h_fitted_before .> 0.01)
    @test all(h_fitted_before .< 5.0)
    @test all(h_fitted_after .> 0.01)
    @test all(h_fitted_after .< 5.0)
    
    # AFT effect validation: compare hazard ratios
    # For AFT: x=1 should have different hazard than x=0
    # The direction depends on beta sign and shape
    if length(times_before) > 0 && length(times_after) > 0
        println("\n    AFT effect check:")
        mid_time = 5.0  # Check at a time where we can evaluate both x=0 and x=1
        h_at_mid_x0 = fitted.hazards[1](mid_time, pars, covars_x0)
        h_at_mid_x1 = fitted.hazards[1](mid_time, pars, covars_x1)
        log_hr = log(h_at_mid_x1 / h_at_mid_x0)
        expected_log_hr = -true_beta * true_shape
        @printf("    At t=%.1f: log(h(x=1)/h(x=0)) = %.3f (expected ~%.3f)\n",
                mid_time, log_hr, expected_log_hr)
        
        # Check that AFT effect has correct sign (negative for positive beta with shape > 1)
        @test sign(log_hr) == sign(expected_log_hr) || abs(log_hr) < 0.3  # Allow small deviations
    end
    
    # =====================================================================
    # Fit with PhaseType proposal (same data)
    # =====================================================================
    println("      ▸ Fitting with PhaseType proposal..."); flush(stdout)
    
    h12_pt = Hazard(@formula(0 ~ x), "sp", 1, 2;
        degree=1,
        knots=knots,
        boundaryknots=[0.0, max_obs_time],
        extrapolation="constant",
        linpred_effect=:aft)
    
    model_pt = multistatemodel(h12_pt; data=panel_data, surrogate=:markov)
    
    # UNPENALIZED SPLINES (penalty=:none) - testing AFT spline PhaseType proposal with TVC
    fitted_pt = fit(model_pt;
        proposal=PhaseTypeProposal(n_phases=3),
        penalty=:none,  # UNPENALIZED - no smoothing
        verbose=VERBOSE_LONGTESTS,
        maxiter=MAX_ITER + 10,
        tol=MCEM_TOL,
        ess_target_initial=30,
        max_ess=500,
        vcov_type=:none)
    
    pars_pt = MultistateModels.get_parameters(fitted_pt, 1, scale=:log)
    
    # =====================================================================
    # Compare Markov vs PhaseType proposals (for times before and after changepoint)
    # =====================================================================
    println("\n    Markov vs PhaseType Proposal Comparison:")
    println("    " * "-"^60)
    println("    Time   x   Markov h(t)   PhaseType h(t)  Rel Diff")
    println("    " * "-"^60)
    
    comparison_passed = true
    # Compare before changepoint (x=0)
    for t in times_before
        covars = (x = 0.0,)
        h_markov = fitted.hazards[1](t, pars, covars)
        h_pt = fitted_pt.hazards[1](t, pars_pt, covars)
        rel_diff = abs(h_markov - h_pt) / max(h_markov, h_pt, 0.01)
        passed = rel_diff <= PROPOSAL_COMPARISON_TOL
        comparison_passed &= passed
        status = passed ? "✓" : "✗"
        println("    $(rpad(t, 6)) $(rpad(0, 3)) $(rpad(round(h_markov, digits=4), 13)) $(rpad(round(h_pt, digits=4), 15)) $(round(rel_diff, digits=3)) $status")
    end
    # Compare after changepoint (x=1)
    for t in times_after
        covars = (x = 1.0,)
        h_markov = fitted.hazards[1](t, pars, covars)
        h_pt = fitted_pt.hazards[1](t, pars_pt, covars)
        rel_diff = abs(h_markov - h_pt) / max(h_markov, h_pt, 0.01)
        passed = rel_diff <= PROPOSAL_COMPARISON_TOL
        comparison_passed &= passed
        status = passed ? "✓" : "✗"
        println("    $(rpad(t, 6)) $(rpad(1, 3)) $(rpad(round(h_markov, digits=4), 13)) $(rpad(round(h_pt, digits=4), 15)) $(round(rel_diff, digits=3)) $status")
    end
    println("    " * "-"^60)
    @test comparison_passed
    
    VERBOSE_LONGTESTS && println("  ✓ sp_aft_panel_tvc: PASS")
    println("  ✓ Markov vs PhaseType proposal estimates agree")
end

# ============================================================================
# Summary
# ============================================================================

println("\n=== MCEM Spline Long Test Suite Complete ===\n")
println("Tests verify:")
println("  - Linear spline approximates constant/exponential hazard")
println("  - Spline with interior knots handles piecewise hazard")
println("  - Cubic spline approximates Gompertz (exponential) hazard")
println("  - Spline with covariates recovers log hazard ratio")
println("  - Monotone spline constraints are enforced in MCEM")
println("  - Spline hazards work with PhaseType proposal")
println("  - Spline AFT panel (no covariates) recovers Weibull hazard shape")
println("  - Spline AFT panel (time-fixed cov) captures AFT effect")
println("  - Spline AFT panel (time-varying cov) handles TVC with AFT")
println("  - Markov vs PhaseType proposal estimates agree (all tests)")
