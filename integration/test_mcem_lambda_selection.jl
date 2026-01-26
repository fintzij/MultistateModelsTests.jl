# =============================================================================
# Integration Tests: MCEM with Penalized Splines and Automatic λ Selection
# =============================================================================
#
# Tests that penalized spline fitting with automatic λ selection (PIJCV) works
# correctly in MCEM.
# 
# Basic unpenalized spline tests are in test_mcem_splines_basic.jl
#
# Run with: julia --project=MultistateModelsTests MultistateModelsTests/integration/test_mcem_lambda_selection.jl
# =============================================================================

using Test
using MultistateModels
using DataFrames
using Random
using Statistics

import MultistateModels: Hazard, multistatemodel, fit, set_parameters!, simulate, PhaseTypeProposal

const RNG_SEED = 0x12345678
const N_SUBJECTS = 150
const MCEM_TOL = 0.15
const MAX_ITER = 15

@testset "MCEM Penalized Spline Tests (λ Selection)" begin

    @testset "PIJCV λ selection runs without error" begin
        Random.seed!(RNG_SEED)
        true_rate = 0.3
        
        # Create simulation model
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
        
        # Fit with penalized spline and automatic λ selection
        h12_sp = Hazard(@formula(0 ~ 1), "sp", 1, 2; degree=2, knots=[1.0, 2.0, 3.0, 4.0], boundaryknots=[0.0, 5.0], extrapolation="linear")
        model_spline = multistatemodel(h12_sp; data=panel_data, surrogate=:markov)
        
        fitted = fit(model_spline; 
            proposal=:markov, 
            penalty=:auto,
            select_lambda=:pijcv,
            lambda_init=1.0,
            verbose=true, 
            maxiter=MAX_ITER, 
            tol=MCEM_TOL, 
            ess_target_initial=25, 
            max_ess=200, 
            vcov_type=:none
        )
        
        @test fitted isa MultistateModels.MultistateModelFitted
        @test isfinite(fitted.loglik.loglik)
        
        # Check that smoothing_parameters and edf exist
        @test hasfield(typeof(fitted), :smoothing_parameters)
        @test hasfield(typeof(fitted), :edf)
        
        if !isnothing(fitted.smoothing_parameters)
            # λ should be positive
            @test all(fitted.smoothing_parameters .> 0.0)
            println("\n    Selected λ = $(round.(fitted.smoothing_parameters, digits=4))")
            println("    EDF = $(fitted.edf)")
        end
        
        println("\n    ✓ PIJCV λ selection completed successfully")
    end

    @testset "Fixed λ vs automatic λ comparison" begin
        Random.seed!(RNG_SEED + 1)
        
        # Simulate from Weibull (smooth non-constant hazard)
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
        set_parameters!(model_sim, (h12 = [1.5, 0.3],))
        
        sim_result = simulate(model_sim; paths=false, data=true, nsim=1, autotmax=false, obstype_by_transition=Dict(1=>1))
        panel_data = sim_result[1, 1]
        
        # Fit with fixed λ
        h12_sp = Hazard(@formula(0 ~ 1), "sp", 1, 2; degree=2, knots=[1.0, 2.0, 3.0, 4.0], boundaryknots=[0.0, 5.0], extrapolation="linear")
        model_fixed = multistatemodel(h12_sp; data=panel_data, surrogate=:markov)
        
        fitted_fixed = fit(model_fixed; 
            proposal=:markov, 
            penalty=:auto,
            select_lambda=:none,
            lambda_init=10.0,  # Fixed large λ
            verbose=false, 
            maxiter=MAX_ITER, 
            tol=MCEM_TOL, 
            ess_target_initial=25, 
            max_ess=200, 
            vcov_type=:none
        )
        
        # Fit with automatic λ selection
        model_auto = multistatemodel(h12_sp; data=panel_data, surrogate=:markov)
        
        fitted_auto = fit(model_auto; 
            proposal=:markov, 
            penalty=:auto,
            select_lambda=:pijcv,
            lambda_init=1.0,
            verbose=false, 
            maxiter=MAX_ITER, 
            tol=MCEM_TOL, 
            ess_target_initial=25, 
            max_ess=200, 
            vcov_type=:none
        )
        
        @test isfinite(fitted_fixed.loglik.loglik)
        @test isfinite(fitted_auto.loglik.loglik)
        
        # Both should produce valid hazard functions
        pars_fixed = MultistateModels.get_parameters(fitted_fixed, 1, scale=:log)
        pars_auto = MultistateModels.get_parameters(fitted_auto, 1, scale=:log)
        
        h_fixed = fitted_fixed.hazards[1](2.5, pars_fixed, NamedTuple())
        h_auto = fitted_auto.hazards[1](2.5, pars_auto, NamedTuple())
        
        @test h_fixed > 0
        @test h_auto > 0
        
        println("\n    Fixed λ hazard at t=2.5: $(round(h_fixed, digits=4))")
        println("    Auto λ hazard at t=2.5: $(round(h_auto, digits=4))")
        println("\n    ✓ Both fixed and automatic λ produce valid fits")
    end

    @testset "PhaseType proposal with λ selection" begin
        Random.seed!(RNG_SEED + 2)
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
        
        # Fit with PhaseType proposal and λ selection
        h12_sp = Hazard(@formula(0 ~ 1), "sp", 1, 2; degree=2, knots=[1.5, 3.0], boundaryknots=[0.0, 5.0], extrapolation="linear")
        model_spline = multistatemodel(h12_sp; data=panel_data, surrogate=:markov)
        
        fitted = fit(model_spline; 
            proposal=PhaseTypeProposal(n_phases=3), 
            penalty=:auto,
            select_lambda=:pijcv,
            lambda_init=1.0,
            verbose=false, 
            maxiter=MAX_ITER, 
            tol=MCEM_TOL, 
            ess_target_initial=25, 
            max_ess=200, 
            vcov_type=:none
        )
        
        @test isfinite(fitted.loglik.loglik)
        
        if !isnothing(fitted.smoothing_parameters)
            @test all(fitted.smoothing_parameters .> 0.0)
            println("\n    PhaseType: Selected λ = $(round.(fitted.smoothing_parameters, digits=4))")
        end
        
        println("\n    ✓ PhaseType proposal works with λ selection")
    end

    @testset "Penalized spline with covariates" begin
        Random.seed!(RNG_SEED + 3)
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
        
        # Fit penalized spline with covariate
        h12_sp = Hazard(@formula(0 ~ 1 + x), "sp", 1, 2; degree=2, knots=[1.5, 3.0], boundaryknots=[0.0, 5.0], extrapolation="linear")
        model_spline = multistatemodel(h12_sp; data=panel_data, surrogate=:markov)
        
        fitted = fit(model_spline; 
            proposal=:markov, 
            penalty=:auto,
            select_lambda=:pijcv,
            lambda_init=1.0,
            verbose=false, 
            maxiter=MAX_ITER, 
            tol=MCEM_TOL, 
            ess_target_initial=25, 
            max_ess=200, 
            vcov_type=:none
        )
        
        @test isfinite(fitted.loglik.loglik)
        
        # Check covariate effect is recovered
        pars = MultistateModels.get_parameters(fitted, 1, scale=:log)
        n_baseline = fitted.hazards[1].npar_baseline
        fitted_beta = pars[n_baseline + 1]
        @test fitted_beta > 0  # Should be positive like true_beta
        
        println("\n    Fitted β = $(round(fitted_beta, digits=3)) (true = $(true_beta))")
        println("\n    ✓ Penalized spline with covariates works")
    end
end

println("\n" * "="^50)
println("All MCEM λ selection tests completed!")
println("="^50)
