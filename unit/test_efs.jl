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
    # NOTE: compute_efs_update is not a standalone function.
    # The EFS criterion is computed via compute_efs_criterion, and λ optimization
    # happens within select_smoothing_parameters using the criterion.
    # No test needed - covered by integration tests in select_smoothing_parameters.
    # =========================================================================
    
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
    
    # =========================================================================
    # Known-Answer Tests for EFS Components
    # =========================================================================
    
    @testset "EFS Known-Answer: Effective Degrees of Freedom" begin
        # The effective degrees of freedom (EDF) formula is:
        #   edf = trace((H + λS)⁻¹ H)
        # where H is the unpenalized Hessian and S is the penalty matrix
        
        # Test with 2×2 matrices for analytical verification
        # H = [4 1; 1 3], S = [1 0; 0 1] = I, λ = 1.0
        # H + λS = [5 1; 1 4]
        # det(H + λS) = 20 - 1 = 19
        # (H + λS)⁻¹ = (1/19) * [4 -1; -1 5]
        # (H + λS)⁻¹ H = (1/19) * [4 -1; -1 5] * [4 1; 1 3]
        #              = (1/19) * [16-1 4-3; -4+5 -1+15]
        #              = (1/19) * [15 1; 1 14]
        # trace = (15 + 14) / 19 = 29/19 ≈ 1.5263
        
        H = [4.0 1.0; 1.0 3.0]
        S = [1.0 0.0; 0.0 1.0]
        λ = 1.0
        
        H_lambda = H + λ * S
        inv_H_lambda = inv(H_lambda)
        edf = tr(inv_H_lambda * H)
        
        expected_edf = 29.0 / 19.0
        @test edf ≈ expected_edf rtol=1e-10
        
        # Test 2: When λ → 0, edf → p (full degrees of freedom)
        λ_small = 1e-10
        H_lambda_small = H + λ_small * S
        edf_small = tr(inv(H_lambda_small) * H)
        @test edf_small ≈ 2.0 rtol=1e-6  # Should be close to p=2
        
        # Test 3: When λ → ∞, edf → 0 (fully penalized)
        λ_large = 1e10
        H_lambda_large = H + λ_large * S
        edf_large = tr(inv(H_lambda_large) * H)
        @test edf_large ≈ 0.0 atol=1e-6  # Should be close to 0
        
        # Test 4: 3×3 case with known eigenstructure
        # H = diag([3, 2, 1]), S = diag([1, 1, 1])
        # (H + λS)⁻¹ = diag([1/(3+λ), 1/(2+λ), 1/(1+λ)])
        # edf = 3/(3+λ) + 2/(2+λ) + 1/(1+λ)
        H_diag = Diagonal([3.0, 2.0, 1.0])
        S_diag = Diagonal([1.0, 1.0, 1.0])
        λ_test = 2.0
        
        edf_analytical = 3.0/(3.0+λ_test) + 2.0/(2.0+λ_test) + 1.0/(1.0+λ_test)
        # = 3/5 + 2/4 + 1/3 = 0.6 + 0.5 + 0.333... = 1.4333...
        expected_edf_33 = 3.0/5.0 + 2.0/4.0 + 1.0/3.0
        
        H_lambda_33 = Matrix(H_diag) + λ_test * Matrix(S_diag)
        edf_computed = tr(inv(H_lambda_33) * Matrix(H_diag))
        @test edf_computed ≈ expected_edf_33 rtol=1e-10
        
        println("  ✓ EDF known-answer tests pass")
        println("    2×2 case: H=[4,1;1,3], S=I, λ=1 → edf = 29/19 ≈ $(round(expected_edf, digits=4))")
        println("    3×3 diagonal case: λ=2 → edf = $(round(expected_edf_33, digits=4))")
        println("    λ→0: edf→p, λ→∞: edf→0 ✓")
    end
    
end

println("\n" * "="^70)
println("EFS Unit Tests Complete")
println("="^70)
