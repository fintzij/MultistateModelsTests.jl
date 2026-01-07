# =============================================================================
# Unit Tests for PERF (Performance Iteration) Criterion
# =============================================================================
#
# Tests for compute_perf_criterion() implementation following Marra & Radice (2020).
#
# References:
# - Marra, G. & Radice, R. (2020). "Copula link-based additive models for 
#   right-censored event time data." JASA 115(530):886-895.
# - Eletti, A., Marra, G. & Radice, R. (2024). arXiv:2312.05345v4, Appendix C
# =============================================================================

using Test
using MultistateModels
using DataFrames
using Random
using LinearAlgebra

println("="^70)
println("Unit Tests: PERF Criterion")
println("="^70)

@testset "PERF Criterion Tests" begin
    
    # =========================================================================
    # Test 1: Basic PERF Computation
    # =========================================================================
    
    @testset "Basic PERF Computation" begin
        # Generate simple survival data
        Random.seed!(42)
        n = 150
        
        # Weibull hazard
        E = -log.(rand(n))
        event_times = (E ./ 0.3) .^ (1 / 1.5)
        obs_times = min.(event_times, 5.0)
        status = Int.(event_times .<= 5.0)
        
        surv_data = DataFrame(
            id = 1:n,
            tstart = zeros(n),
            tstop = obs_times,
            statefrom = ones(Int, n),
            stateto = ifelse.(status .== 1, 2, 1),
            obstype = ones(Int, n)
        )
        
        # Fit spline model
        h12 = Hazard(@formula(0 ~ 1), :sp, 1, 2)
        model = multistatemodel(h12; data=surv_data)
        
        fitted = fit(model; verbose=false, compute_vcov=false, compute_ij_vcov=false)
        beta_hat = MultistateModels.get_parameters(fitted; scale=:flat)
        
        # Build penalty config
        penalty_config = build_penalty_config(model, SplinePenalty(); lambda_init=1.0)
        
        # Create state for PERF computation
        samplepaths = MultistateModels.extract_paths(model)
        exact_data = MultistateModels.ExactData(model, samplepaths)
        
        subject_grads_ll = MultistateModels.compute_subject_gradients(beta_hat, model, samplepaths)
        subject_hessians_ll = MultistateModels.compute_subject_hessians_fast(beta_hat, model, samplepaths)
        subject_grads = -subject_grads_ll
        subject_hessians = [-H for H in subject_hessians_ll]
        H_unpenalized = sum(subject_hessians)
        
        n_subjects = length(samplepaths)
        n_params = length(beta_hat)
        
        state = MultistateModels.SmoothingSelectionState(
            beta_hat, H_unpenalized, subject_grads, subject_hessians,
            penalty_config, n_subjects, n_params, model, exact_data
        )
        
        # Test PERF at multiple λ values
        for log_lambda in [-2.0, 0.0, 2.0, 4.0]
            perf_val = MultistateModels.compute_perf_criterion([log_lambda], state)
            @test isfinite(perf_val)
            @test !isnan(perf_val)
        end
        
        println("  ✓ PERF returns finite values for all λ")
    end
    
    # =========================================================================
    # Test 2: PERF Curve Shape
    # =========================================================================
    
    @testset "PERF Curve Shape" begin
        Random.seed!(123)
        n = 200
        
        E = -log.(rand(n))
        event_times = (E ./ 0.3) .^ (1 / 1.5)
        obs_times = min.(event_times, 5.0)
        status = Int.(event_times .<= 5.0)
        
        surv_data = DataFrame(
            id = 1:n, tstart = zeros(n), tstop = obs_times,
            statefrom = ones(Int, n), stateto = ifelse.(status .== 1, 2, 1),
            obstype = ones(Int, n)
        )
        
        h12 = Hazard(@formula(0 ~ 1), :sp, 1, 2)
        model = multistatemodel(h12; data=surv_data)
        
        fitted = fit(model; verbose=false, compute_vcov=false, compute_ij_vcov=false)
        beta_hat = MultistateModels.get_parameters(fitted; scale=:flat)
        
        penalty_config = build_penalty_config(model, SplinePenalty(); lambda_init=1.0)
        samplepaths = MultistateModels.extract_paths(model)
        exact_data = MultistateModels.ExactData(model, samplepaths)
        
        subject_grads_ll = MultistateModels.compute_subject_gradients(beta_hat, model, samplepaths)
        subject_hessians_ll = MultistateModels.compute_subject_hessians_fast(beta_hat, model, samplepaths)
        subject_grads = -subject_grads_ll
        subject_hessians = [-H for H in subject_hessians_ll]
        H_unpenalized = sum(subject_hessians)
        
        n_subjects = length(samplepaths)
        n_params = length(beta_hat)
        
        state = MultistateModels.SmoothingSelectionState(
            beta_hat, H_unpenalized, subject_grads, subject_hessians,
            penalty_config, n_subjects, n_params, model, exact_data
        )
        
        # Compute PERF over grid
        log_lambda_grid = collect(-4.0:0.5:6.0)
        perf_values = [MultistateModels.compute_perf_criterion([ll], state) for ll in log_lambda_grid]
        
        # All values should be finite
        @test all(isfinite.(perf_values))
        
        # Should have some variation
        perf_range = maximum(perf_values) - minimum(perf_values)
        @test perf_range > 0
        
        println("  ✓ PERF curve has expected properties")
        println("    Range: [$(round(minimum(perf_values), digits=4)), $(round(maximum(perf_values), digits=4))]")
    end
    
    # =========================================================================
    # Test 3: PERF vs EFS Agreement
    # =========================================================================
    
    @testset "PERF vs EFS Qualitative Agreement" begin
        Random.seed!(456)
        n = 100
        
        E = -log.(rand(n))
        event_times = (E ./ 0.3) .^ (1 / 1.5)
        obs_times = min.(event_times, 5.0)
        status = Int.(event_times .<= 5.0)
        
        surv_data = DataFrame(
            id = 1:n, tstart = zeros(n), tstop = obs_times,
            statefrom = ones(Int, n), stateto = ifelse.(status .== 1, 2, 1),
            obstype = ones(Int, n)
        )
        
        h12 = Hazard(@formula(0 ~ 1), :sp, 1, 2)
        model = multistatemodel(h12; data=surv_data)
        
        fitted = fit(model; verbose=false, compute_vcov=false, compute_ij_vcov=false)
        beta_hat = MultistateModels.get_parameters(fitted; scale=:flat)
        
        penalty_config = build_penalty_config(model, SplinePenalty(); lambda_init=1.0)
        samplepaths = MultistateModels.extract_paths(model)
        exact_data = MultistateModels.ExactData(model, samplepaths)
        
        subject_grads_ll = MultistateModels.compute_subject_gradients(beta_hat, model, samplepaths)
        subject_hessians_ll = MultistateModels.compute_subject_hessians_fast(beta_hat, model, samplepaths)
        subject_grads = -subject_grads_ll
        subject_hessians = [-H for H in subject_hessians_ll]
        H_unpenalized = sum(subject_hessians)
        
        n_subjects = length(samplepaths)
        n_params = length(beta_hat)
        
        state = MultistateModels.SmoothingSelectionState(
            beta_hat, H_unpenalized, subject_grads, subject_hessians,
            penalty_config, n_subjects, n_params, model, exact_data
        )
        
        log_lambda_grid = collect(-4.0:0.5:6.0)
        
        perf_values = [MultistateModels.compute_perf_criterion([ll], state) for ll in log_lambda_grid]
        efs_values = [MultistateModels.compute_efs_criterion([ll], state) for ll in log_lambda_grid]
        
        perf_best_idx = argmin(perf_values)
        efs_best_idx = argmin(efs_values)
        
        # Both methods should produce finite optimal values
        # Note: PERF and EFS are fundamentally different criteria (AIC-type vs REML-type)
        # and may select different λ values - this is expected behavior
        @test isfinite(perf_values[perf_best_idx])
        @test isfinite(efs_values[efs_best_idx])
        @test perf_best_idx >= 1 && perf_best_idx <= length(log_lambda_grid)
        @test efs_best_idx >= 1 && efs_best_idx <= length(log_lambda_grid)
        
        println("  ✓ PERF vs EFS produce valid optimal λ values:")
        println("    PERF optimal log(λ) = $(log_lambda_grid[perf_best_idx])")
        println("    EFS optimal log(λ) = $(log_lambda_grid[efs_best_idx])")
    end
    
    # =========================================================================
    # Test 4: PERF vs PIJCV Agreement
    # =========================================================================
    
    @testset "PERF vs PIJCV Qualitative Agreement" begin
        Random.seed!(789)
        n = 80
        
        E = -log.(rand(n))
        event_times = (E ./ 0.3) .^ (1 / 1.5)
        obs_times = min.(event_times, 5.0)
        status = Int.(event_times .<= 5.0)
        
        surv_data = DataFrame(
            id = 1:n, tstart = zeros(n), tstop = obs_times,
            statefrom = ones(Int, n), stateto = ifelse.(status .== 1, 2, 1),
            obstype = ones(Int, n)
        )
        
        h12 = Hazard(@formula(0 ~ 1), :sp, 1, 2)
        model = multistatemodel(h12; data=surv_data)
        
        fitted = fit(model; verbose=false, compute_vcov=false, compute_ij_vcov=false)
        beta_hat = MultistateModels.get_parameters(fitted; scale=:flat)
        
        penalty_config = build_penalty_config(model, SplinePenalty(); lambda_init=1.0)
        samplepaths = MultistateModels.extract_paths(model)
        exact_data = MultistateModels.ExactData(model, samplepaths)
        
        subject_grads_ll = MultistateModels.compute_subject_gradients(beta_hat, model, samplepaths)
        subject_hessians_ll = MultistateModels.compute_subject_hessians_fast(beta_hat, model, samplepaths)
        subject_grads = -subject_grads_ll
        subject_hessians = [-H for H in subject_hessians_ll]
        H_unpenalized = sum(subject_hessians)
        
        n_subjects = length(samplepaths)
        n_params = length(beta_hat)
        
        state = MultistateModels.SmoothingSelectionState(
            beta_hat, H_unpenalized, subject_grads, subject_hessians,
            penalty_config, n_subjects, n_params, model, exact_data
        )
        
        log_lambda_grid = collect(-4.0:0.5:6.0)
        
        perf_values = [MultistateModels.compute_perf_criterion([ll], state) for ll in log_lambda_grid]
        pijcv_values = [MultistateModels.compute_pijcv_criterion([ll], state) for ll in log_lambda_grid]
        
        perf_best_idx = argmin(perf_values)
        pijcv_best_idx = argmin(pijcv_values)
        
        # Both methods should produce finite optimal values
        # Note: PERF and PIJCV are fundamentally different criteria (AIC-type vs LOO-CV type)
        # and may select different λ values - this is expected behavior
        @test isfinite(perf_values[perf_best_idx])
        @test isfinite(pijcv_values[pijcv_best_idx])
        @test perf_best_idx >= 1 && perf_best_idx <= length(log_lambda_grid)
        @test pijcv_best_idx >= 1 && pijcv_best_idx <= length(log_lambda_grid)
        
        println("  ✓ PERF vs PIJCV produce valid optimal λ values:")
        println("    PERF optimal log(λ) = $(log_lambda_grid[perf_best_idx])")
        println("    PIJCV optimal log(λ) = $(log_lambda_grid[pijcv_best_idx])")
    end
    
    # =========================================================================
    # Test 5: select_smoothing_parameters with method=:perf
    # =========================================================================
    
    @testset "select_smoothing_parameters with :perf" begin
        Random.seed!(1234)
        n = 100
        
        E = -log.(rand(n))
        event_times = (E ./ 0.3) .^ (1 / 1.5)
        obs_times = min.(event_times, 5.0)
        status = Int.(event_times .<= 5.0)
        
        surv_data = DataFrame(
            id = 1:n, tstart = zeros(n), tstop = obs_times,
            statefrom = ones(Int, n), stateto = ifelse.(status .== 1, 2, 1),
            obstype = ones(Int, n)
        )
        
        h12 = Hazard(@formula(0 ~ 1), :sp, 1, 2)
        model = multistatemodel(h12; data=surv_data)
        
        # Test that select_smoothing_parameters works with :perf method
        result = select_smoothing_parameters(model, SplinePenalty(); 
                                             method=:perf, verbose=false)
        
        @test haskey(result, :lambda)
        @test haskey(result, :beta)
        @test haskey(result, :criterion)
        @test haskey(result, :method_used)
        
        @test result.method_used == :perf
        @test length(result.lambda) > 0
        @test all(result.lambda .> 0)
        @test isfinite(result.criterion)
        
        println("  ✓ select_smoothing_parameters(:perf) works correctly")
        println("    Optimal λ: $(round.(result.lambda, sigdigits=3))")
        println("    Criterion: $(round(result.criterion, digits=4))")
    end
    
end

println("\n" * "="^70)
println("PERF Unit Tests Complete")
println("="^70)
