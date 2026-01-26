# =============================================================================
# Integration Tests: Basic MCEM with Unpenalized Spline Hazards
# =============================================================================
#
# Tests that MCEM fitting works correctly with spline hazards (no penalty).
# Penalized splines and λ selection are tested in test_mcem_lambda_selection.jl
#
# Run with: julia --project=MultistateModelsTests MultistateModelsTests/integration/test_mcem_splines_basic.jl
# =============================================================================

using Test
using MultistateModels
using DataFrames
using Random
using Statistics

import MultistateModels: Hazard, multistatemodel, fit, set_parameters!, simulate, get_parameters_flat, PhaseTypeProposal

const RNG_SEED = 0x12345678
const N_SUBJECTS = 150
const MCEM_TOL = 0.15
const MAX_ITER = 12

@testset "MCEM Basic Spline Tests (Unpenalized)" begin

    @testset "Spline approximates exponential" begin
        Random.seed!(RNG_SEED)
        true_rate = 0.3
        
        # Create simulation model with exponential hazard
        h12_exp = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        obs_times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        nobs = length(obs_times) - 1
        
        sim_data = DataFrame(
            id = repeat(1:N_SUBJECTS, inner=nobs),
            tstart = repeat(obs_times[1:end-1], N_SUBJECTS),
            tstop = repeat(obs_times[2:end], N_SUBJECTS),
            statefrom = ones(Int, N_SUBJECTS * nobs),
            stateto = ones(Int, N_SUBJECTS * nobs),
            obstype = fill(2, N_SUBJECTS * nobs)
        )
        
        model_sim = multistatemodel(h12_exp; data=sim_data, initialize=false)
        set_parameters!(model_sim, (h12 = [true_rate],))
        
        sim_result = simulate(model_sim; paths=false, data=true, nsim=1, autotmax=false, obstype_by_transition=Dict(1=>1))
        panel_data = sim_result[1, 1]
        
        # Fit with linear spline (should approximate constant)
        h12_sp = Hazard(@formula(0 ~ 1), "sp", 1, 2; degree=1, knots=[2.5], boundaryknots=[0.0, 5.0], extrapolation="linear")
        model_spline = multistatemodel(h12_sp; data=panel_data, surrogate=:markov)
        
        fitted = fit(model_spline; proposal=:markov, penalty=:none, verbose=true, maxiter=MAX_ITER, tol=MCEM_TOL, ess_target_initial=25, max_ess=200, vcov_type=:none)
        
        @test fitted isa MultistateModels.MultistateModelFitted
        @test isfinite(fitted.loglik.loglik)
        
        # Check hazard is approximately constant
        pars = MultistateModels.get_parameters(fitted, 1, scale=:log)
        h_vals = [fitted.hazards[1](t, pars, NamedTuple()) for t in [0.5, 1.5, 2.5, 3.5, 4.5]]
        h_min, h_max = extrema(h_vals)
        @test h_max / h_min < 2.0  # Ratio should be close to 1
        
        # Check mean hazard is near true rate
        h_mean = mean(h_vals)
        @test 0.5 * true_rate < h_mean < 2.0 * true_rate
        
        println("\n    ✓ Spline approximates constant hazard (h_mean=$(round(h_mean,digits=3)))")
    end

    @testset "Spline with interior knots" begin
        Random.seed!(RNG_SEED + 1)
        
        # Simulate from Weibull (non-constant hazard)
        h12_wei = Hazard(@formula(0 ~ 1), "wei", 1, 2)
        obs_times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        nobs = length(obs_times) - 1
        
        sim_data = DataFrame(
            id = repeat(1:N_SUBJECTS, inner=nobs),
            tstart = repeat(obs_times[1:end-1], N_SUBJECTS),
            tstop = repeat(obs_times[2:end], N_SUBJECTS),
            statefrom = ones(Int, N_SUBJECTS * nobs),
            stateto = ones(Int, N_SUBJECTS * nobs),
            obstype = fill(2, N_SUBJECTS * nobs)
        )
        
        model_sim = multistatemodel(h12_wei; data=sim_data, initialize=false)
        set_parameters!(model_sim, (h12 = [1.5, 0.2],))  # Increasing hazard
        
        sim_result = simulate(model_sim; paths=false, data=true, nsim=1, autotmax=false, obstype_by_transition=Dict(1=>1))
        panel_data = sim_result[1, 1]
        
        # Fit with quadratic spline and interior knots
        h12_sp = Hazard(@formula(0 ~ 1), "sp", 1, 2; degree=2, knots=[1.5, 3.0], boundaryknots=[0.0, 5.0], extrapolation="constant")
        model_spline = multistatemodel(h12_sp; data=panel_data, surrogate=:markov)
        
        fitted = fit(model_spline; proposal=:markov, penalty=:none, verbose=false, maxiter=MAX_ITER, tol=MCEM_TOL, ess_target_initial=25, max_ess=200, vcov_type=:none)
        
        @test isfinite(fitted.loglik.loglik)
        
        # Check cumulative hazard is monotonically increasing
        pars = MultistateModels.get_parameters(fitted, 1, scale=:log)
        H_vals = [MultistateModels.cumulative_hazard(fitted.hazards[1], 0.0, t, pars, NamedTuple()) for t in [1.0, 2.0, 3.0, 4.0]]
        @test all(diff(H_vals) .> 0)
        
        println("\n    ✓ Spline with interior knots works")
    end

    @testset "Spline with covariates" begin
        Random.seed!(RNG_SEED + 2)
        true_baseline = 0.3
        true_beta = 0.5
        
        # Simulation model with covariate effect
        h12_exp = Hazard(@formula(0 ~ 1 + x), "exp", 1, 2)
        obs_times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        nobs = length(obs_times) - 1
        
        sim_data = DataFrame(
            id = repeat(1:N_SUBJECTS, inner=nobs),
            tstart = repeat(obs_times[1:end-1], N_SUBJECTS),
            tstop = repeat(obs_times[2:end], N_SUBJECTS),
            statefrom = ones(Int, N_SUBJECTS * nobs),
            stateto = ones(Int, N_SUBJECTS * nobs),
            obstype = fill(2, N_SUBJECTS * nobs),
            x = repeat(rand([0.0, 1.0], N_SUBJECTS), inner=nobs)
        )
        
        model_sim = multistatemodel(h12_exp; data=sim_data, initialize=false)
        set_parameters!(model_sim, (h12 = [true_baseline, true_beta],))
        
        sim_result = simulate(model_sim; paths=false, data=true, nsim=1, autotmax=false, obstype_by_transition=Dict(1=>1))
        panel_data = sim_result[1, 1]
        
        # Fit spline with covariate
        h12_sp = Hazard(@formula(0 ~ 1 + x), "sp", 1, 2; degree=1, knots=[2.5], boundaryknots=[0.0, 5.0], extrapolation="linear")
        model_spline = multistatemodel(h12_sp; data=panel_data, surrogate=:markov)
        
        fitted = fit(model_spline; proposal=:markov, penalty=:none, verbose=false, maxiter=MAX_ITER, tol=MCEM_TOL, ess_target_initial=25, max_ess=200, vcov_type=:none)
        
        @test isfinite(fitted.loglik.loglik)
        
        # Check covariate effect is recovered
        pars = MultistateModels.get_parameters(fitted, 1, scale=:log)
        n_baseline = fitted.hazards[1].npar_baseline
        fitted_beta = pars[n_baseline + 1]
        @test fitted_beta > 0  # Should be positive like true_beta
        
        # Check hazard ratio
        h_x0 = fitted.hazards[1](2.0, pars, (x=0.0,))
        h_x1 = fitted.hazards[1](2.0, pars, (x=1.0,))
        fitted_hr = h_x1 / h_x0
        true_hr = exp(true_beta)
        @test 0.5 * true_hr < fitted_hr < 2.0 * true_hr
        
        println("\n    ✓ Spline with covariates works (β=$(round(fitted_beta,digits=3)))")
    end

    @testset "PhaseType proposal with splines" begin
        Random.seed!(RNG_SEED + 3)
        true_rate = 0.3
        
        # Create simulation data
        h12_exp = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        obs_times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        nobs = length(obs_times) - 1
        
        sim_data = DataFrame(
            id = repeat(1:N_SUBJECTS, inner=nobs),
            tstart = repeat(obs_times[1:end-1], N_SUBJECTS),
            tstop = repeat(obs_times[2:end], N_SUBJECTS),
            statefrom = ones(Int, N_SUBJECTS * nobs),
            stateto = ones(Int, N_SUBJECTS * nobs),
            obstype = fill(2, N_SUBJECTS * nobs)
        )
        
        model_sim = multistatemodel(h12_exp; data=sim_data, initialize=false)
        set_parameters!(model_sim, (h12 = [true_rate],))
        
        sim_result = simulate(model_sim; paths=false, data=true, nsim=1, autotmax=false, obstype_by_transition=Dict(1=>1))
        panel_data = sim_result[1, 1]
        
        # Fit with PhaseType proposal
        h12_sp = Hazard(@formula(0 ~ 1), "sp", 1, 2; degree=1, knots=[2.5], boundaryknots=[0.0, 5.0], extrapolation="linear")
        model_spline = multistatemodel(h12_sp; data=panel_data, surrogate=:markov)
        
        fitted = fit(model_spline; proposal=PhaseTypeProposal(n_phases=3), penalty=:none, verbose=false, maxiter=MAX_ITER, tol=MCEM_TOL, ess_target_initial=25, max_ess=200, vcov_type=:none)
        
        @test isfinite(fitted.loglik.loglik)
        println("\n    ✓ PhaseType proposal works with splines")
    end
end

println("\n" * "="^50)
println("All basic MCEM spline tests completed!")
println("="^50)
