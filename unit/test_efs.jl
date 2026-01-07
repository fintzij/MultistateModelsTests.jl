# =============================================================================
# Unit Tests for EFS (Extended Fellner-Schall) Criterion
# =============================================================================
#
# Tests for compute_efs_criterion() and compute_efs_update() implementations
# following Wood & Fasiolo (2017).
#
# References:
# - Wood, S.N. & Fasiolo, M. (2017). "A generalized Fellner-Schall method for 
#   smoothing parameter estimation." Statistics and Computing 27(3):759-773.
# - Eletti, A., Marra, G. & Radice, R. (2024). arXiv:2312.05345v4, Appendix C
# =============================================================================

using Test
using MultistateModels
using DataFrames
using Random
using LinearAlgebra

println("="^70)
println("Unit Tests: EFS (Extended Fellner-Schall) Criterion")
println("="^70)

@testset "EFS Criterion Tests" begin
    
    # =========================================================================
    # Test 1: Basic EFS Criterion Computation
    # =========================================================================
    
    @testset "Basic EFS Criterion Computation" begin
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
        
        # Create state for EFS computation
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
        
        # Test EFS criterion at multiple λ values
        for log_lambda in [-2.0, 0.0, 2.0, 4.0]
            efs_val = MultistateModels.compute_efs_criterion([log_lambda], state)
            @test isfinite(efs_val)
            @test !isnan(efs_val)
        end
        
        println("  ✓ EFS criterion returns finite values for all λ")
    end
    
    # =========================================================================
    # Test 2: EFS Criterion Curve Shape
    # =========================================================================
    
    @testset "EFS Criterion Curve Shape" begin
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
        
        # Compute EFS over grid
        log_lambda_grid = collect(-4.0:0.5:6.0)
        efs_values = [MultistateModels.compute_efs_criterion([ll], state) for ll in log_lambda_grid]
        
        # All values should be finite
        @test all(isfinite.(efs_values))
        
        # Should have some variation
        efs_range = maximum(efs_values) - minimum(efs_values)
        @test efs_range > 0
        
        println("  ✓ EFS criterion curve has expected properties")
        println("    Range: [$(round(minimum(efs_values), digits=4)), $(round(maximum(efs_values), digits=4))]")
    end
    
    # =========================================================================
    # Test 3: EFS Update Function
    # =========================================================================
    
    @testset "EFS Update Function" begin
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
        
        # Test EFS update
        lambda_init = [1.0]
        lambda_new = MultistateModels.compute_efs_update(lambda_init, state)
        
        # Should return positive values
        @test all(lambda_new .> 0)
        # Should be finite
        @test all(isfinite.(lambda_new))
        
        println("  ✓ EFS update produces valid λ values")
        println("    Initial λ: $(lambda_init)")
        println("    Updated λ: $(round.(lambda_new, sigdigits=3))")
    end
    
    # =========================================================================
    # Test 4: EFS vs PERF Comparison
    # =========================================================================
    
    @testset "EFS vs PERF Comparison" begin
        Random.seed!(789)
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
        
        efs_values = [MultistateModels.compute_efs_criterion([ll], state) for ll in log_lambda_grid]
        perf_values = [MultistateModels.compute_perf_criterion([ll], state) for ll in log_lambda_grid]
        
        efs_best_idx = argmin(efs_values)
        perf_best_idx = argmin(perf_values)
        
        # Both methods should produce finite optimal values
        @test isfinite(efs_values[efs_best_idx])
        @test isfinite(perf_values[perf_best_idx])
        @test efs_best_idx >= 1 && efs_best_idx <= length(log_lambda_grid)
        @test perf_best_idx >= 1 && perf_best_idx <= length(log_lambda_grid)
        
        println("  ✓ EFS vs PERF produce valid optimal λ values:")
        println("    EFS optimal log(λ) = $(log_lambda_grid[efs_best_idx])")
        println("    PERF optimal log(λ) = $(log_lambda_grid[perf_best_idx])")
    end
    
    # =========================================================================
    # Test 5: select_smoothing_parameters with method=:efs
    # =========================================================================
    
    @testset "select_smoothing_parameters with :efs" begin
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
        
        # Test that select_smoothing_parameters works with :efs method
        result = select_smoothing_parameters(model, SplinePenalty(); 
                                             method=:efs, verbose=false)
        
        @test haskey(result, :lambda)
        @test haskey(result, :beta)
        @test haskey(result, :criterion)
        @test haskey(result, :method_used)
        
        @test result.method_used == :efs
        @test length(result.lambda) > 0
        @test all(result.lambda .> 0)
        @test isfinite(result.criterion)
        
        println("  ✓ select_smoothing_parameters(:efs) works correctly")
        println("    Optimal λ: $(round.(result.lambda, sigdigits=3))")
        println("    Criterion: $(round(result.criterion, digits=4))")
    end
    
    # =========================================================================
    # Test 6: Compare all three methods (PIJCV, PERF, EFS)
    # =========================================================================
    
    @testset "Compare All Methods" begin
        Random.seed!(5678)
        n = 120
        
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
        
        pijcv_values = [MultistateModels.compute_pijcv_criterion([ll], state) for ll in log_lambda_grid]
        perf_values = [MultistateModels.compute_perf_criterion([ll], state) for ll in log_lambda_grid]
        efs_values = [MultistateModels.compute_efs_criterion([ll], state) for ll in log_lambda_grid]
        
        pijcv_best = log_lambda_grid[argmin(pijcv_values)]
        perf_best = log_lambda_grid[argmin(perf_values)]
        efs_best = log_lambda_grid[argmin(efs_values)]
        
        # All methods should produce finite values
        @test all(isfinite.(pijcv_values))
        @test all(isfinite.(perf_values))
        @test all(isfinite.(efs_values))
        
        println("  ✓ All three methods produce valid results:")
        println("    PIJCV optimal log(λ) = $pijcv_best")
        println("    PERF optimal log(λ) = $perf_best")
        println("    EFS optimal log(λ) = $efs_best")
    end
    
end

println("\n" * "="^70)
println("EFS Unit Tests Complete")
println("="^70)
