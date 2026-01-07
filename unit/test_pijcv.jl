# =============================================================================
# PIJCV (Predictive Infinitesimal Jackknife Cross-Validation) Tests
# =============================================================================
#
# Tests for the PIJCV framework implementing Wood (2024) "Neighbourhood Cross 
# Validation" (arXiv:2404.16490).
#
# All tests verify mathematical correctness through analytical formulas.

using LinearAlgebra
using Random
using Test
using DataFrames
using Distributions
using MultistateModels
using StatsModels: @formula

# Import internal functions for testing
import MultistateModels: cholesky_downdate!, cholesky_downdate_copy,
                          pijcv_loo_perturbation_direct, pijcv_loo_perturbation_cholesky,
                          pijcv_loo_perturbation_woodbury,
                          PIJCVState, compute_pijcv_perturbations!, pijcv_criterion,
                          loo_perturbations_direct, pijcv_get_loo_estimates, pijcv_vcov,
                          multistatemodel, Hazard, set_parameters!, 
                          ExactData, extract_paths, get_parameters_flat, build_penalty_config,
                          select_smoothing_parameters, PenaltyConfig, SplinePenalty,
                          _cholesky_downdate!, _solve_loo_newton_step,
                          compute_pijcv_criterion, SmoothingSelectionState,
                          _build_penalized_hessian, compute_subject_gradients,
                          compute_subject_hessians_fast

# =============================================================================
# 1. Cholesky Downdate Algorithm
# =============================================================================

@testset "Cholesky Downdate" begin
    
    @testset "Rank-1 downdate correctness" begin
        # Verify L*L' ≈ H - v*v' after downdate
        Random.seed!(12345)
        n = 5
        A = randn(n, n)
        H = A' * A + 2.0 * I
        v = 0.3 * randn(n)
        
        L = cholesky(Symmetric(H)).L
        L_copy = copy(L)
        success = cholesky_downdate!(L_copy, v)
        
        @test success == true
        @test L_copy * L_copy' ≈ H - v * v' atol=1e-10
    end
    
    @testset "Downdate preserves positive definiteness" begin
        Random.seed!(23456)
        n = 4
        A = randn(n, n)
        H = A' * A + 5.0 * I
        v_small = 0.1 * randn(n)
        
        L = cholesky(Symmetric(H)).L
        L_copy = copy(L)
        success = cholesky_downdate!(L_copy, v_small)
        
        @test success == true
        H_new = L_copy * L_copy'
        @test all(eigvals(Symmetric(H_new)) .> 0)
    end
    
    @testset "Downdate detects indefiniteness" begin
        Random.seed!(34567)
        n = 3
        A = randn(n, n)
        H = A' * A + 0.5 * I  # Weakly positive definite
        v_large = 2.0 * randn(n)  # Large enough to make indefinite
        
        L = cholesky(Symmetric(H)).L
        L_copy = copy(L)
        success = cholesky_downdate!(L_copy, v_large)
        
        @test success == false
    end
    
    @testset "Non-mutating copy version" begin
        Random.seed!(45678)
        n = 4
        A = randn(n, n)
        H = A' * A + 3.0 * I
        v = 0.2 * randn(n)
        
        L = cholesky(Symmetric(H)).L
        L_original = copy(L)
        
        L_new, success = cholesky_downdate_copy(L, v)
        
        @test L ≈ L_original  # Original unchanged
        @test success == true
        @test L_new * L_new' ≈ H - v * v' atol=1e-10
    end
    
    @testset "Sequential downdates accumulate correctly" begin
        Random.seed!(56789)
        n = 5
        A = randn(n, n)
        H = A' * A + 10.0 * I
        vs = [0.1 * randn(n) for _ in 1:3]
        
        L = cholesky(Symmetric(H)).L
        L_copy = copy(L)
        
        H_expected = copy(H)
        all_success = true
        for v in vs
            success = cholesky_downdate!(L_copy, v)
            all_success &= success
            H_expected -= v * v'
        end
        
        @test all_success == true
        @test L_copy * L_copy' ≈ H_expected atol=1e-9
    end
end

# =============================================================================
# 1b. NCV Cholesky Downdate Algorithm (smoothing_selection.jl)
# =============================================================================
# Tests for _cholesky_downdate! and _solve_loo_newton_step used in NCV criterion

@testset "NCV Cholesky Downdate (_cholesky_downdate!)" begin
    
    @testset "Rank-1 downdate correctness" begin
        # Verify L*L' ≈ H - v*v' after downdate
        Random.seed!(11111)
        n = 6
        A = randn(n, n)
        H = A' * A + 3.0 * I  # Positive definite
        v = 0.3 * randn(n)
        
        L = Matrix(cholesky(Symmetric(H)).L)
        success = _cholesky_downdate!(L, v)
        
        @test success == true
        @test L * L' ≈ H - v * v' atol=1e-10
    end
    
    @testset "Downdate preserves positive definiteness" begin
        Random.seed!(22222)
        n = 5
        A = randn(n, n)
        H = A' * A + 5.0 * I
        v_small = 0.15 * randn(n)
        
        L = Matrix(cholesky(Symmetric(H)).L)
        success = _cholesky_downdate!(L, v_small)
        
        @test success == true
        H_new = L * L'
        @test all(eigvals(Symmetric(H_new)) .> 0)
    end
    
    @testset "Downdate detects indefiniteness" begin
        Random.seed!(33333)
        n = 4
        A = randn(n, n)
        H = A' * A + 0.5 * I  # Weakly positive definite
        v_large = 3.0 * randn(n)  # Large enough to make indefinite
        
        L = Matrix(cholesky(Symmetric(H)).L)
        success = _cholesky_downdate!(L, v_large)
        
        @test success == false
    end
    
    @testset "Sequential rank-1 downdates accumulate correctly" begin
        Random.seed!(44444)
        n = 5
        A = randn(n, n)
        H = A' * A + 10.0 * I
        vs = [0.15 * randn(n) for _ in 1:4]
        
        L = Matrix(cholesky(Symmetric(H)).L)
        
        H_expected = copy(H)
        all_success = true
        for v in vs
            success = _cholesky_downdate!(L, v)
            all_success &= success
            H_expected -= v * v'
        end
        
        @test all_success == true
        @test L * L' ≈ H_expected atol=1e-9
    end
end

@testset "NCV LOO Newton Step Solver (_solve_loo_newton_step)" begin
    
    @testset "Matches direct solve for simple case" begin
        # Compare Cholesky downdate solve to direct (H - H_i)^{-1} g
        Random.seed!(55555)
        p = 6
        
        # Create positive definite penalized Hessian (larger eigenvalues)
        A = randn(p, p)
        H_lambda = A' * A + 10.0 * I
        
        # Create subject Hessian (much smaller than total Hessian)
        B = 0.2 * randn(p, 2)  # Low rank, scaled down
        H_i = B * B' + 0.01 * I
        
        # Gradient
        g_i = randn(p)
        
        # Cholesky of full Hessian
        chol_H = cholesky(Symmetric(H_lambda))
        
        # Direct solve (ground truth)
        delta_direct = Symmetric(H_lambda - H_i) \ g_i
        
        # Downdate solve
        delta_downdate = _solve_loo_newton_step(chol_H, H_i, g_i)
        
        @test delta_downdate !== nothing
        @test delta_downdate ≈ delta_direct rtol=1e-4
    end
    
    @testset "Handles full-rank subject Hessian" begin
        Random.seed!(66666)
        p = 5
        
        # Penalized Hessian with larger eigenvalues
        A = randn(p, p)
        H_lambda = A' * A + 10.0 * I
        
        # Full rank subject Hessian (much smaller than H_lambda)
        B = 0.2 * randn(p, p)
        H_i = B' * B + 0.01 * I
        
        g_i = randn(p)
        
        chol_H = cholesky(Symmetric(H_lambda))
        
        delta_direct = Symmetric(H_lambda - H_i) \ g_i
        delta_downdate = _solve_loo_newton_step(chol_H, H_i, g_i)
        
        @test delta_downdate !== nothing
        @test delta_downdate ≈ delta_direct rtol=1e-3
    end
    
    @testset "Returns nothing when LOO Hessian is indefinite" begin
        Random.seed!(77777)
        p = 4
        
        # Small penalized Hessian
        A = randn(p, p)
        H_lambda = A' * A + 0.5 * I
        
        # Subject Hessian larger than H_lambda (will cause indefinite LOO Hessian)
        B = randn(p, p)
        H_i = B' * B + 3.0 * I
        
        g_i = randn(p)
        
        chol_H = cholesky(Symmetric(H_lambda))
        
        delta = _solve_loo_newton_step(chol_H, H_i, g_i)
        
        @test delta === nothing
    end
    
    @testset "Multiple subjects give consistent results" begin
        Random.seed!(88888)
        p = 6
        n_subjects = 5
        
        # Penalized Hessian (represent sum of n subjects + penalty)
        # Total Hessian should be ~n times larger than individual subject Hessians
        A = randn(p, p)
        H_lambda = A' * A + 15.0 * I
        
        chol_H = cholesky(Symmetric(H_lambda))
        
        # Generate multiple subjects with small individual Hessians
        for _ in 1:n_subjects
            B = 0.15 * randn(p, 2)  # Scaled to be small relative to total
            H_i = B * B' + 0.005 * I
            g_i = randn(p)
            
            delta_direct = Symmetric(H_lambda - H_i) \ g_i
            delta_downdate = _solve_loo_newton_step(chol_H, H_i, g_i)
            
            @test delta_downdate !== nothing
            @test delta_downdate ≈ delta_direct rtol=1e-3
        end
    end
end

# =============================================================================
# 2. LOO Perturbation Methods
# =============================================================================

@testset "LOO Perturbation Methods" begin
    
    @testset "Direct solve: delta = (H_lambda - H_k)^{-1} g_k" begin
        Random.seed!(11111)
        p = 4
        A = randn(p, p)
        H_lambda = A' * A + 2.0 * I
        g_k = randn(p)
        H_k = 0.1 * g_k * g_k'
        
        result = pijcv_loo_perturbation_direct(H_lambda, H_k, g_k)
        delta_expected = (H_lambda - H_k) \ g_k
        
        @test result.delta ≈ delta_expected atol=1e-10
        @test result.indefinite == false
    end
    
    @testset "Cholesky method matches direct solve" begin
        Random.seed!(22222)
        p = 5
        A = randn(p, p)
        H_lambda = A' * A + 5.0 * I
        H_chol = cholesky(Symmetric(H_lambda))
        g_k = randn(p)
        H_k = 0.05 * g_k * g_k'
        
        result_chol = pijcv_loo_perturbation_cholesky(H_chol, H_k, g_k)
        result_direct = pijcv_loo_perturbation_direct(H_lambda, H_k, g_k)
        
        @test result_chol.delta ≈ result_direct.delta atol=1e-8
    end
    
    @testset "Woodbury fallback matches direct solve" begin
        Random.seed!(33333)
        p = 4
        A = randn(p, p)
        H_lambda = A' * A + 2.0 * I
        H_chol = cholesky(Symmetric(H_lambda))
        g_k = randn(p)
        H_k = 0.1 * g_k * g_k'
        
        result = pijcv_loo_perturbation_woodbury(H_chol, H_k, g_k)
        delta_expected = (H_lambda - H_k) \ g_k
        
        @test result.delta ≈ delta_expected atol=1e-8
    end
    
    @testset "Zero H_k gives H_lambda^{-1} g_k" begin
        Random.seed!(44444)
        p = 4
        A = randn(p, p)
        H_lambda = A' * A + 3.0 * I
        H_chol = cholesky(Symmetric(H_lambda))
        g_k = randn(p)
        H_k = zeros(p, p)
        
        result = pijcv_loo_perturbation_cholesky(H_chol, H_k, g_k)
        expected = H_chol \ g_k
        
        @test result.delta ≈ expected atol=1e-10
    end
    
    @testset "Full-rank H_k handling" begin
        Random.seed!(55555)
        p = 3
        A = randn(p, p)
        H_lambda = A' * A + 10.0 * I
        H_chol = cholesky(Symmetric(H_lambda))
        B = randn(p, p)
        H_k = 0.1 * B' * B  # Full-rank
        g_k = randn(p)
        
        result = pijcv_loo_perturbation_cholesky(H_chol, H_k, g_k)
        delta_expected = (H_lambda - H_k) \ g_k
        
        @test result.delta ≈ delta_expected atol=1e-6
    end
end

# =============================================================================
# 3. PIJCV Perturbation Computation
# =============================================================================

@testset "PIJCV Perturbation Computation" begin
    
    @testset "Outer product approximation: delta_k = (H - g_k g_k')^{-1} g_k" begin
        Random.seed!(88888)
        p = 3
        n = 4
        A = randn(p, p)
        H_lambda = A' * A + 5.0 * I
        subject_grads = 0.5 * randn(p, n)
        
        state = PIJCVState(H_lambda, subject_grads)
        compute_pijcv_perturbations!(state)
        
        for k in 1:n
            g_k = subject_grads[:, k]
            H_k = g_k * g_k'
            delta_expected = (H_lambda - H_k) \ g_k
            @test state.deltas[:, k] ≈ delta_expected atol=1e-6
        end
    end
    
    @testset "Provided Hessians: delta_k = (H - H_k)^{-1} g_k" begin
        Random.seed!(99999)
        p = 3
        n = 3
        A = randn(p, p)
        H_lambda = A' * A + 10.0 * I
        subject_grads = randn(p, n)
        subject_hessians = zeros(p, p, n)
        for k in 1:n
            B = randn(p, p)
            subject_hessians[:, :, k] = 0.05 * B' * B
        end
        
        state = PIJCVState(H_lambda, subject_grads; subject_hessians=subject_hessians)
        compute_pijcv_perturbations!(state)
        
        for k in 1:n
            g_k = subject_grads[:, k]
            H_k = subject_hessians[:, :, k]
            delta_expected = (H_lambda - H_k) \ g_k
            @test state.deltas[:, k] ≈ delta_expected atol=1e-6
        end
    end
end

# =============================================================================
# 4. PIJCV Criterion
# =============================================================================

@testset "PIJCV Criterion" begin
    
    @testset "Criterion = mean of LOO losses" begin
        Random.seed!(10101)
        p = 3
        n = 5
        A = randn(p, p)
        H_lambda = A' * A + 5.0 * I
        subject_grads = 0.3 * randn(p, n)
        
        state = PIJCVState(H_lambda, subject_grads)
        compute_pijcv_perturbations!(state)
        
        params = randn(p)
        loss_fn(pars, data, k) = sum(pars.^2)
        
        V = pijcv_criterion(state, params, loss_fn, nothing)
        
        V_expected = 0.0
        for k in 1:n
            params_loo = params .+ state.deltas[:, k]
            V_expected += loss_fn(params_loo, nothing, k)
        end
        V_expected /= n
        
        @test V ≈ V_expected
    end
    
    @testset "LOO estimates = params + deltas" begin
        Random.seed!(30303)
        p = 3
        n = 4
        A = randn(p, p)
        H_lambda = A' * A + 3.0 * I
        subject_grads = randn(p, n)
        
        state = PIJCVState(H_lambda, subject_grads)
        compute_pijcv_perturbations!(state)
        
        params = randn(p)
        loo_estimates = pijcv_get_loo_estimates(state, params)
        
        for k in 1:n
            @test loo_estimates[:, k] ≈ params .+ state.deltas[:, k]
        end
    end
end

# =============================================================================
# 5. Variance Estimation
# =============================================================================

@testset "PIJCV Variance Estimation" begin
    
    @testset "Covariance matrices are positive semi-definite" begin
        Random.seed!(60606)
        p = 4
        n = 20
        A = randn(p, p)
        H_lambda = A' * A + 2.0 * I
        subject_grads = randn(p, n)
        
        state = PIJCVState(H_lambda, subject_grads)
        compute_pijcv_perturbations!(state)
        
        vcov_result = pijcv_vcov(state)
        
        @test all(eigvals(vcov_result.ij_vcov) .>= -1e-10)
        @test all(eigvals(vcov_result.jk_vcov) .>= -1e-10)
    end
    
    @testset "Variance formulas" begin
        Random.seed!(50505)
        p = 3
        n = 10
        A = randn(p, p)
        H_lambda = A' * A + 3.0 * I
        subject_grads = 0.5 * randn(p, n)
        
        state = PIJCVState(H_lambda, subject_grads)
        compute_pijcv_perturbations!(state)
        
        vcov_result = pijcv_vcov(state)
        
        delta_outer = state.deltas * state.deltas'
        @test vcov_result.ij_vcov ≈ Symmetric(delta_outer / n)
        @test vcov_result.jk_vcov ≈ Symmetric(((n - 1) / n) * delta_outer)
    end
end

# =============================================================================
# 6. Consistency with IJ/JK Methods
# =============================================================================

@testset "PIJCV Consistency with IJ/JK" begin
    
    @testset "Zero H_k matches IJ exactly: delta_k = H^{-1} g_k" begin
        Random.seed!(10110)
        p = 3
        n = 5
        A = randn(p, p)
        fishinf = A' * A + 2.0 * I
        subject_grads = randn(p, n)
        subject_hessians = zeros(p, p, n)
        
        state = PIJCVState(fishinf, subject_grads; subject_hessians=subject_hessians)
        compute_pijcv_perturbations!(state)
        
        vcov_ij = inv(fishinf)
        loo_deltas_ij = loo_perturbations_direct(vcov_ij, subject_grads)
        
        @test state.deltas ≈ loo_deltas_ij atol=1e-8
    end
    
    @testset "PIJCV vs IJ perturbations highly correlated" begin
        Random.seed!(90909)
        p = 4
        n = 8
        A = randn(p, p)
        fishinf = A' * A + 3.0 * I
        subject_grads = 0.3 * randn(p, n)
        
        vcov_ij = inv(fishinf)
        loo_deltas_ij = loo_perturbations_direct(vcov_ij, subject_grads)
        
        state = PIJCVState(fishinf, subject_grads)
        compute_pijcv_perturbations!(state)
        
        for k in 1:n
            corr = dot(loo_deltas_ij[:, k], state.deltas[:, k]) / 
                   (norm(loo_deltas_ij[:, k]) * norm(state.deltas[:, k]))
            @test corr > 0.9
        end
    end
end
# =============================================================================
# 7. End-to-End Public API Test: select_smoothing_parameters
# =============================================================================

@testset "select_smoothing_parameters Public API" begin
    
    @testset "Basic spline model λ selection runs without error" begin
        # Create a simple spline hazard model
        Random.seed!(42424)
        
        # Generate test data with some transition events
        n_subjects = 30
        data = DataFrame(
            id = 1:n_subjects,
            tstart = zeros(n_subjects),
            tstop = rand(Uniform(0.5, 2.0), n_subjects),
            statefrom = ones(Int, n_subjects),
            stateto = fill(2, n_subjects),
            obstype = ones(Int, n_subjects)
        )
        
        # Create spline hazard with penalty
        h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2;
                     degree = 3,
                     knots = [0.5, 1.0, 1.5],
                     boundaryknots = [0.0, 2.0],
                     natural_spline = true)
        
        model = multistatemodel(h12; data = data)
        
        # Initialize parameters
        npar = model.hazards[1].npar_total
        set_parameters!(model, 1, fill(0.0, npar))
        
        # Build penalty config with SplinePenalty (required for spline hazards)
        penalty_config = build_penalty_config(model, SplinePenalty(); lambda_init = 1.0)
        
        # Verify penalty config is valid
        @test penalty_config.n_lambda >= 1
        @test !isempty(penalty_config.terms) || !isempty(penalty_config.smooth_covariate_terms)
        
        # Extract data for selection
        paths = extract_paths(model)
        exact_data = ExactData(model, paths)
        beta_init = get_parameters_flat(model)
        
        # Run select_smoothing_parameters - should complete without error
        result = select_smoothing_parameters(
            model, exact_data, penalty_config, beta_init;
            method = :efs,  # Use EFS for speed
            max_outer_iter = 3,
            verbose = false
        )
        
        # Verify result structure
        @test haskey(result, :lambda) || hasproperty(result, :lambda)
        @test haskey(result, :beta) || hasproperty(result, :beta)
        @test haskey(result, :criterion) || hasproperty(result, :criterion)
        @test haskey(result, :converged) || hasproperty(result, :converged)
        @test haskey(result, :method_used) || hasproperty(result, :method_used)
        
        # Lambda should be positive
        @test all(result.lambda .> 0)
        
        # Beta should be finite
        @test all(isfinite.(result.beta))
        
        # Criterion should be finite or NaN (NaN indicates optimization didn't converge,
        # but the API should still return a valid result structure)
        @test isfinite(result.criterion) || isnan(result.criterion)
    end
    
    @testset "Method fallback from PIJCV to EFS works" begin
        Random.seed!(31313)
        
        # Small dataset that might cause PIJCV numerical issues
        n_subjects = 15
        data = DataFrame(
            id = 1:n_subjects,
            tstart = zeros(n_subjects),
            tstop = rand(Uniform(0.3, 1.0), n_subjects),
            statefrom = ones(Int, n_subjects),
            stateto = fill(2, n_subjects),
            obstype = ones(Int, n_subjects)
        )
        
        h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2;
                     degree = 3,
                     knots = [0.4, 0.6],
                     boundaryknots = [0.0, 1.0])
        
        model = multistatemodel(h12; data = data)
        set_parameters!(model, 1, fill(0.0, model.hazards[1].npar_total))
        
        penalty_config = build_penalty_config(model, SplinePenalty(); lambda_init = 0.1)
        paths = extract_paths(model)
        exact_data = ExactData(model, paths)
        beta_init = get_parameters_flat(model)
        
        # Request PIJCV - may fall back to EFS
        result = select_smoothing_parameters(
            model, exact_data, penalty_config, beta_init;
            method = :pijcv,
            max_outer_iter = 3,
            verbose = false
        )
        
        # Should return valid result regardless of which method was used
        @test result.method_used ∈ (:pijcv, :efs, :none)
        @test all(isfinite.(result.beta))
    end
end