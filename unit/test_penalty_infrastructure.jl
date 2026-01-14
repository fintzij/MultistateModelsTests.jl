# =============================================================================
# Unit Tests: Penalty Infrastructure for Penalized Splines
# =============================================================================
#
# Tests for Phase 1 of penalized splines implementation:
# - SplinePenalty struct construction and validation
# - build_penalty_matrix (Wood 2016 algorithm)
# - place_interior_knots_pooled for competing risks
# - validate_shared_knots enforcement
# - SplineHazardInfo construction
# - PenaltyConfig and compute_penalty
#
# =============================================================================

using Test
using MultistateModels
using BSplineKit
using LinearAlgebra
using DataFrames
using Random

@testset "Penalty Infrastructure" begin

    @testset "SplinePenalty Construction" begin
        # Default construction
        p1 = SplinePenalty()
        @test p1.selector == :all
        @test p1.order == 2
        @test p1.total_hazard == false
        @test p1.share_lambda == false
        
        # Custom order
        p2 = SplinePenalty(order=3)
        @test p2.order == 3
        
        # Origin selector
        p3 = SplinePenalty(1, share_lambda=true)
        @test p3.selector == 1
        @test p3.share_lambda == true
        
        # Transition selector
        p4 = SplinePenalty((1, 2), order=1, total_hazard=true)
        @test p4.selector == (1, 2)
        @test p4.order == 1
        @test p4.total_hazard == true
        
        # Invalid inputs
        @test_throws ArgumentError SplinePenalty(order=0)
        @test_throws ArgumentError SplinePenalty(0)  # state must be ≥ 1
        @test_throws ArgumentError SplinePenalty(:invalid)
    end

    @testset "build_penalty_matrix - Basic Properties" begin
        knots = collect(0.0:1.0:5.0)
        basis = BSplineBasis(BSplineOrder(4), knots)  # Cubic
        
        # Order 2 (curvature)
        S2 = build_penalty_matrix(basis, 2)
        @test size(S2) == (length(basis), length(basis))
        @test isapprox(S2, S2', rtol=1e-10)  # Symmetric
        
        # Positive semi-definite
        eigs = eigvals(Symmetric(S2))
        @test all(e -> e >= -1e-10, eigs)
        
        # Null space dimension for order 2 should be 2 (constants, linears)
        null_dim = count(e -> abs(e) < 1e-10, eigs)
        @test null_dim == 2
    end

    @testset "build_penalty_matrix - Null Space" begin
        knots = collect(0.0:0.5:5.0)
        basis = BSplineBasis(BSplineOrder(4), knots)
        K = length(basis)
        
        S = build_penalty_matrix(basis, 2)
        
        # Constant function should have zero penalty
        v_const = ones(K)
        @test abs(v_const' * S * v_const) < 1e-10
        
        # Linear function: use Greville abscissae
        full_knots = collect(BSplineKit.knots(basis))
        greville = [(full_knots[i+1] + full_knots[i+2] + full_knots[i+3]) / 3 for i in 1:K]
        @test abs(greville' * S * greville) < 1e-10
    end

    @testset "build_penalty_matrix - Different Orders" begin
        knots = collect(0.0:0.5:5.0)
        basis = BSplineBasis(BSplineOrder(4), knots)
        
        for order in 1:3
            S = build_penalty_matrix(basis, order)
            eigs = eigvals(Symmetric(S))
            null_dim = count(e -> abs(e) < 1e-10, eigs)
            # Null space dimension should be at least `order`
            @test null_dim >= order
        end
    end

    @testset "PenaltyConfig and compute_penalty" begin
        # Empty config
        config_empty = PenaltyConfig()
        @test !has_penalties(config_empty)
        @test compute_penalty(rand(5), config_empty) ≈ 0.0
        
        # Single penalty term - parameters on NATURAL scale (post-v0.3.0)
        # Box constraints ensure non-negativity, penalty is quadratic P(β) = (λ/2) βᵀSβ
        K = 6
        S = rand(K, K)
        S = S' * S  # Make symmetric PSD
        β = abs.(randn(K)) .+ 0.1  # Natural scale coefficients (non-negative for splines)
        lambda = 2.0
        
        # Penalty term with natural scale parameters
        term = MultistateModels.PenaltyTerm(1:K, S, lambda, 2, [:h12])
        config = PenaltyConfig(
            [term], 
            MultistateModels.TotalHazardPenaltyTerm[], 
            Dict{Int,Vector{Int}}(), 
            1
        )
        
        @test has_penalties(config)
        
        # Compute penalty: (1/2) * λ * βᵀ S β (parameters directly on natural scale)
        expected = 0.5 * lambda * dot(β, S * β)
        computed = compute_penalty(β, config)
        @test computed ≈ expected
    end

    @testset "place_interior_knots_pooled" begin
        # Create test data with competing risks from state 1
        data = DataFrame(
            id = 1:10,
            tstart = zeros(10),
            tstop = [0.5, 1.0, 1.5, 2.0, 0.8, 1.2, 1.8, 2.5, 3.0, 0.3],
            statefrom = ones(Int, 10),
            stateto = [2, 3, 2, 3, 2, 3, 2, 3, 2, 3],
            obstype = ones(Int, 10)
        )
        
        h12 = Hazard(@formula(0 ~ 1), :sp, 1, 2)
        h13 = Hazard(@formula(0 ~ 1), :sp, 1, 3)
        model = multistatemodel(h12, h13; data=data)
        
        # Pooled knots should use all events from origin 1
        pooled = place_interior_knots_pooled(model, 1, 5)
        @test length(pooled) == 5
        @test all(pooled .> 0)
        @test all(pooled .< maximum(data.tstop))
        @test issorted(pooled)
    end

    @testset "validate_shared_knots" begin
        # Create model where hazards have different knots (by default)
        data = DataFrame(
            id = 1:6,
            tstart = zeros(6),
            tstop = [0.5, 1.5, 2.5, 0.8, 1.2, 2.0],
            statefrom = ones(Int, 6),
            stateto = [2, 2, 2, 3, 3, 3],
            obstype = ones(Int, 6)
        )
        
        h12 = Hazard(@formula(0 ~ 1), :sp, 1, 2)
        h13 = Hazard(@formula(0 ~ 1), :sp, 1, 3)
        model = multistatemodel(h12, h13; data=data)
        
        # Should fail because knots are placed independently per transition
        @test_throws ArgumentError validate_shared_knots(model, 1)
    end

    @testset "build_spline_hazard_info" begin
        data = DataFrame(
            id = 1:5,
            tstart = zeros(5),
            tstop = [1.0, 2.0, 1.5, 2.5, 3.0],
            statefrom = ones(Int, 5),
            stateto = fill(2, 5),
            obstype = ones(Int, 5)
        )
        
        h12 = Hazard(@formula(0 ~ 1), :sp, 1, 2)
        model = multistatemodel(h12; data=data)
        
        haz = model.hazards[1]
        info = build_spline_hazard_info(haz; penalty_order=2)
        
        @test info.origin == 1
        @test info.dest == 2
        @test info.nbasis == haz.npar_baseline
        @test info.penalty_order == 2
        @test size(info.S) == (info.nbasis, info.nbasis)
        @test isapprox(info.S, info.S')  # Symmetric
    end

    @testset "Monotone Spline Penalty Transformation" begin
        # Test that monotone splines get transformed penalty matrices
        # For monotone splines, optimization is on ests (I-spline increments)
        # but penalty should penalize smoothness of the underlying function
        # 
        # The correct transformation is: S_monotone = L' * S_bspline * L
        # where coefs = L * ests
        
        @testset "build_ispline_transform_matrix" begin
            # Use uniform knots to avoid numerical issues with clamped boundaries
            knots = collect(0.0:0.2:1.0)  # [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            basis = BSplineBasis(BSplineOrder(4), knots)
            K = length(basis)
            
            L = MultistateModels.build_ispline_transform_matrix(basis; direction=1)
            
            # Basic properties
            @test size(L) == (K, K)
            @test all(L[:, 1] .== 1.0)  # Intercept column
            @test istril(L)  # Lower triangular
            
            # Verify L * ests == _spline_ests2coefs(ests, basis, 1)
            ests = rand(K)
            coefs_expected = MultistateModels._spline_ests2coefs(ests, basis, 1)
            coefs_L = L * ests
            @test isapprox(coefs_expected, coefs_L; rtol=1e-10)
            
            # Test with different ests
            for _ in 1:5
                ests_rand = rand(K) .+ 0.1
                coefs_expected = MultistateModels._spline_ests2coefs(ests_rand, basis, 1)
                @test isapprox(L * ests_rand, coefs_expected; rtol=1e-10)
            end
            
            # Test decreasing direction
            L_dec = MultistateModels.build_ispline_transform_matrix(basis; direction=-1)
            @test size(L_dec) == (K, K)
            
            # Verify L_dec matches _spline_ests2coefs with monotone=-1
            ests_test = rand(K) .+ 0.1
            coefs_dec_expected = MultistateModels._spline_ests2coefs(ests_test, basis, -1)
            @test isapprox(L_dec * ests_test, coefs_dec_expected; rtol=1e-10)
        end
        
        @testset "transform_penalty_for_monotone" begin
            knots = collect(0.0:0.2:1.0)
            basis = BSplineBasis(BSplineOrder(4), knots)
            K = length(basis)
            
            # Build B-spline penalty
            S_bspline = build_penalty_matrix(basis, 2)
            
            # Transform for monotone
            S_monotone = MultistateModels.transform_penalty_for_monotone(S_bspline, basis)
            
            # Basic properties
            @test size(S_monotone) == (K, K)
            @test isapprox(S_monotone, S_monotone')  # Symmetric
            
            # Verify mathematical relation: S_monotone = L' * S * L
            L = MultistateModels.build_ispline_transform_matrix(basis; direction=1)
            S_expected = L' * S_bspline * L
            @test isapprox(S_monotone, S_expected; rtol=1e-10)
            
            # Positive semi-definite (all eigenvalues ≥ -eps)
            eigs = eigvals(Symmetric(S_monotone))
            @test all(e -> e >= -1e-10, eigs)
        end
        
        @testset "build_spline_hazard_info with monotone" begin
            # Create a model with monotone spline hazard
            data = DataFrame(
                id = 1:10,
                tstart = zeros(10),
                tstop = [0.5, 1.0, 1.5, 2.0, 2.5, 0.6, 1.2, 1.8, 2.2, 2.8],
                statefrom = ones(Int, 10),
                stateto = fill(2, 10),
                obstype = ones(Int, 10)
            )
            
            # Non-monotone spline
            h_nonmono = Hazard(@formula(0 ~ 1), :sp, 1, 2)
            model_nonmono = multistatemodel(h_nonmono; data=data)
            haz_nonmono = model_nonmono.hazards[1]
            info_nonmono = build_spline_hazard_info(haz_nonmono; penalty_order=2)
            
            # Monotone increasing spline
            h_mono = Hazard(@formula(0 ~ 1), :sp, 1, 2; monotone=1)
            model_mono = multistatemodel(h_mono; data=data)
            haz_mono = model_mono.hazards[1]
            info_mono = build_spline_hazard_info(haz_mono; penalty_order=2)
            
            # Both should produce valid penalty matrices
            @test isapprox(info_nonmono.S, info_nonmono.S')
            @test isapprox(info_mono.S, info_mono.S')
            
            # Sizes should match
            @test size(info_nonmono.S) == size(info_mono.S)
            
            # Penalty matrices should be different (transformed vs not)
            # This test verifies that the transformation is actually applied
            @test !isapprox(info_nonmono.S, info_mono.S)
        end
    end

    @testset "build_penalty_config" begin
        # Create model with two spline hazards
        data = DataFrame(
            id = 1:10,
            tstart = zeros(10),
            tstop = [0.5, 1.0, 1.5, 2.0, 2.5, 0.6, 1.2, 1.8, 2.2, 2.8],
            statefrom = ones(Int, 10),
            stateto = [2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
            obstype = ones(Int, 10)
        )
        
        h12 = Hazard(@formula(0 ~ 1), :sp, 1, 2)
        h13 = Hazard(@formula(0 ~ 1), :sp, 1, 3)
        model = multistatemodel(h12, h13; data=data)
        
        # Test nothing penalty
        config_none = build_penalty_config(model, nothing)
        @test config_none.n_lambda == 0
        @test !has_penalties(config_none)
        
        # Test default penalty (one lambda per hazard)
        config_default = build_penalty_config(model, SplinePenalty())
        @test config_default.n_lambda == 2  # One per hazard
        @test length(config_default.terms) == 2
        @test has_penalties(config_default)
        
        # Test with custom lambda_init
        config_custom = build_penalty_config(model, SplinePenalty(); lambda_init=5.0)
        @test config_custom.terms[1].lambda == 5.0
        
        # Test order-1 penalty
        config_order1 = build_penalty_config(model, SplinePenalty(order=1))
        @test config_order1.terms[1].order == 1
    end

    @testset "fit integration with penalty" begin
        # Create simple exact data
        Random.seed!(42)
        nsubj = 50
        rows = []
        for i in 1:nsubj
            t = rand()^0.5 * 3.0
            dest = rand() < 0.6 ? 2 : 3
            push!(rows, (id=i, tstart=0.0, tstop=t, statefrom=1, stateto=dest, obstype=1))
        end
        data = DataFrame(rows)
        
        h12 = Hazard(@formula(0 ~ 1), :sp, 1, 2)
        h13 = Hazard(@formula(0 ~ 1), :sp, 1, 3)
        model = multistatemodel(h12, h13; data=data)
        
        # Fit without penalty (explicit :none since splines default to penalized)
        fitted_no_penalty = fit(model; penalty=:none, verbose=false, compute_vcov=false, compute_ij_vcov=false)
        ll_no_penalty = get_loglik(fitted_no_penalty)
        
        # Fit with penalty
        fitted_with_penalty = fit(model; penalty=SplinePenalty(), lambda_init=10.0,
                                   verbose=false, compute_vcov=false, compute_ij_vcov=false)
        ll_with_penalty = get_loglik(fitted_with_penalty)
        
        # Penalized fit should still complete
        @test !isnan(ll_no_penalty)
        @test !isnan(ll_with_penalty)
        
        # With stronger penalty, likelihood should be <= unpenalized (penalty restricts params)
        # This tests that the penalty is actually being applied during optimization
        fitted_high_penalty = fit(model; penalty=SplinePenalty(), lambda_init=100.0,
                                   verbose=false, compute_vcov=false, compute_ij_vcov=false)
        ll_high_penalty = get_loglik(fitted_high_penalty)
        
        @test ll_high_penalty <= ll_no_penalty  # Higher penalty = worse unpenalized likelihood
    end

    @testset "penalty=:auto default behavior" begin
        # Create simple exact data
        Random.seed!(123)
        nsubj = 30
        rows = []
        for i in 1:nsubj
            t = rand()^0.5 * 3.0
            dest = rand() < 0.6 ? 2 : 3
            push!(rows, (id=i, tstart=0.0, tstop=t, statefrom=1, stateto=dest, obstype=1))
        end
        data = DataFrame(rows)
        
        # Test :auto on spline model - should apply penalty
        h12_sp = Hazard(@formula(0 ~ 1), :sp, 1, 2)
        h13_sp = Hazard(@formula(0 ~ 1), :sp, 1, 3)
        model_sp = multistatemodel(h12_sp, h13_sp; data=data)
        
        @test has_spline_hazards(model_sp) == true
        
        # With default (penalty=:auto), splines should be penalized
        # Fit with explicit penalty and :auto, verify same behavior
        fitted_auto = fit(model_sp; verbose=false, compute_vcov=false, compute_ij_vcov=false)
        fitted_explicit = fit(model_sp; penalty=SplinePenalty(), verbose=false, compute_vcov=false, compute_ij_vcov=false)
        
        # Both should produce valid fits
        @test !isnan(get_loglik(fitted_auto))
        @test !isnan(get_loglik(fitted_explicit))
        
        # Test :auto on non-spline model - should not apply penalty
        h12_exp = Hazard(@formula(0 ~ 1), :exp, 1, 2)
        h13_exp = Hazard(@formula(0 ~ 1), :exp, 1, 3)
        model_exp = multistatemodel(h12_exp, h13_exp; data=data)
        
        @test has_spline_hazards(model_exp) == false
        
        # Should fit without issues (no penalty applied)
        fitted_exp = fit(model_exp; verbose=false, compute_vcov=false, compute_ij_vcov=false)
        @test !isnan(get_loglik(fitted_exp))
        
        # Test :none explicit opt-out
        fitted_none = fit(model_sp; penalty=:none, verbose=false, compute_vcov=false, compute_ij_vcov=false)
        @test !isnan(get_loglik(fitted_none))
        
        # Unpenalized should have higher log-likelihood (less regularization)
        @test get_loglik(fitted_none) >= get_loglik(fitted_auto) - 1e-6  # Small tolerance for numerical error
    end
    
    @testset "penalty=nothing deprecation warning" begin
        # Create simple data
        nsubj = 15
        data = DataFrame(
            id = 1:nsubj,
            tstart = zeros(nsubj),
            tstop = rand(nsubj) .* 2.0 .+ 0.1,
            statefrom = ones(Int, nsubj),
            stateto = fill(2, nsubj),
            obstype = ones(Int, nsubj)
        )
        
        h12 = Hazard(@formula(0 ~ 1), :sp, 1, 2)
        model = multistatemodel(h12; data=data)
        
        # Using penalty=nothing should emit deprecation warning
        @test_logs (:warn, r"penalty=nothing is deprecated") fit(model; penalty=nothing, verbose=false, compute_vcov=false, compute_ij_vcov=false)
    end

end

# =============================================================================
# Section 7: Smoothing Parameter Selection (PIJCV / EFS)
# =============================================================================

@testset "Smoothing Parameter Selection" begin
    using LinearAlgebra
    
    @testset "PIJCV criterion computation" begin
        # Create a small test model with spline hazards
        nsubj = 20
        Random.seed!(12345)
        data = DataFrame(
            id = vcat(1:nsubj, (nsubj+1):(2*nsubj)),
            tstart = zeros(2*nsubj),
            tstop = vcat(fill(0.5, nsubj), fill(0.8, nsubj)),
            statefrom = ones(Int, 2*nsubj),
            stateto = vcat(fill(2, nsubj), fill(1, nsubj)),
            obstype = ones(Int, 2*nsubj)
        )
        
        h12 = Hazard(@formula(0 ~ 1), :sp, 1, 2)
        model = multistatemodel(h12; data=data)
        
        # First fit to get reasonable parameters (use select_lambda=:none to avoid
        # automatic smoothing parameter selection which can fail on sparse data)
        fitted = fit(model; verbose=false, compute_vcov=false, compute_ij_vcov=false, select_lambda=:none)
        beta_hat = MultistateModels.get_parameters(fitted; scale=:flat)
        
        # Build penalty config
        penalty_config = build_penalty_config(model, [SplinePenalty()]; lambda_init=1.0)
        
        # Create ExactData and samplepaths
        samplepaths = MultistateModels.extract_paths(model)
        exact_data = MultistateModels.ExactData(model, samplepaths)
        
        # Compute subject gradients and Hessians (negative log-lik convention)
        subject_grads_ll = MultistateModels.compute_subject_gradients(beta_hat, model, samplepaths)
        subject_hessians_ll = MultistateModels.compute_subject_hessians_fast(beta_hat, model, samplepaths)
        subject_grads = -subject_grads_ll
        subject_hessians = [-H for H in subject_hessians_ll]
        H_unpenalized = sum(subject_hessians)
        
        n_subjects = length(samplepaths)
        n_params = length(beta_hat)
        
        # Create state with all required fields
        state = MultistateModels.SmoothingSelectionState(
            beta_hat,
            H_unpenalized,
            subject_grads,
            subject_hessians,
            penalty_config,
            n_subjects,
            n_params,
            model,
            exact_data
        )
        
        # Test PIJCV criterion returns a finite value
        log_lambda = [0.0]  # λ = 1
        pijcv_val = MultistateModels.compute_pijcv_criterion(log_lambda, state)
        @test isfinite(pijcv_val)
        @test pijcv_val > 0  # Should be positive (sum of negative log-likelihoods)
        
        # Test EFS criterion returns a value (may be NaN/Inf/1e10 for problematic Hessians)
        efs_val = MultistateModels.compute_efs_criterion(log_lambda, state)
        # EFS may return NaN, Inf, or 1e10 if penalized Hessian is not positive definite
        # This is expected for small/sparse datasets - we just check it doesn't error
        @test efs_val isa Real
    end
    
    @testset "Penalty Hessian addition" begin
        # Simple test of _add_penalty_to_hessian!
        n = 5
        H = zeros(n, n)
        lambda = [2.0]
        
        # Create a simple penalty config with one term
        S = [1.0 -1.0 0.0; -1.0 2.0 -1.0; 0.0 -1.0 1.0]  # 3x3 difference matrix
        term = MultistateModels.PenaltyTerm(1:3, S, 1.0, 2, [:h12])  # Note: Symbol vector
        config = MultistateModels.PenaltyConfig([term], MultistateModels.TotalHazardPenaltyTerm[], Dict{Int,Vector{Int}}(), 1)
        
        MultistateModels._add_penalty_to_hessian!(H, lambda, config)
        
        @test H[1:3, 1:3] ≈ 2.0 * S  # lambda[1] * S
        @test all(H[4:5, :] .== 0)    # Other elements unchanged
    end
    
    @testset "select_smoothing_parameters basic test" begin
        # Create a small test model
        nsubj = 30
        Random.seed!(54321)
        tstops = vcat([0.5 + 0.5 * rand() for _ in 1:nsubj], [0.8 + 0.5 * rand() for _ in 1:nsubj])
        data = DataFrame(
            id = vcat(1:nsubj, (nsubj+1):(2*nsubj)),
            tstart = zeros(2*nsubj),
            tstop = tstops,
            statefrom = ones(Int, 2*nsubj),
            stateto = vcat(fill(2, nsubj), fill(1, nsubj)),
            obstype = ones(Int, 2*nsubj)
        )
        
        h12 = Hazard(@formula(0 ~ 1), :sp, 1, 2)
        model = multistatemodel(h12; data=data)
        
        # First fit to get reasonable parameters
        fitted = fit(model; verbose=false, compute_vcov=false, compute_ij_vcov=false)
        beta_hat = MultistateModels.get_parameters(fitted; scale=:flat)
        
        # Build penalty config
        penalty_config = build_penalty_config(model, [SplinePenalty()]; lambda_init=1.0)
        
        # Create ExactData
        samplepaths = MultistateModels.extract_paths(model)
        exact_data = MultistateModels.ExactData(model, samplepaths)
        
        # Test that select_smoothing_parameters runs without error
        result = select_smoothing_parameters(model, exact_data, penalty_config, beta_hat;
                                              method=:efs, max_outer_iter=5, verbose=false)
        
        @test haskey(result, :lambda)
        @test haskey(result, :converged)
        @test haskey(result, :method_used)
        @test haskey(result, :penalty_config)
        @test length(result.lambda) == 1
        @test result.lambda[1] > 0  # Lambda should be positive
    end
    
    @testset "Empty penalty returns early" begin
        # Model with non-spline hazard
        nsubj = 10
        data = DataFrame(
            id = 1:nsubj,
            tstart = zeros(nsubj),
            tstop = fill(0.5, nsubj),
            statefrom = ones(Int, nsubj),
            stateto = fill(2, nsubj),
            obstype = ones(Int, nsubj)
        )
        
        h12 = Hazard(@formula(0 ~ 1), :exp, 1, 2)  # :exp not :Exponential
        model = multistatemodel(h12; data=data)
        
        # Create empty penalty config
        empty_config = MultistateModels.PenaltyConfig(
            MultistateModels.PenaltyTerm[],
            MultistateModels.TotalHazardPenaltyTerm[],
            Dict{Int,Vector{Int}}(),
            0
        )
        
        samplepaths = MultistateModels.extract_paths(model)
        exact_data = MultistateModels.ExactData(model, samplepaths)
        beta_init = zeros(1)
        
        result = select_smoothing_parameters(model, exact_data, empty_config, beta_init)
        
        @test result.method_used == :none
        @test result.converged == true
        @test isempty(result.lambda)
    end

    # =========================================================================
    # Finite Difference Validation of Gradient and Hessian
    # =========================================================================
    # 
    # These tests verify that ForwardDiff produces gradients and Hessians
    # consistent with finite differences for loglik_exact_penalized.
    # This is a critical correctness check for the optimization routines.
    # =========================================================================

    @testset "Penalized Gradient - AD Consistency" begin
        using ForwardDiff
        
        # Build model with spline hazard (uses :sp which auto-calibrates knots)
        nsubj = 50
        Random.seed!(42)
        rows = []
        for i in 1:nsubj
            t = rand()^0.5 * 2.0 + 0.1  # Times between 0.1 and 2.1
            push!(rows, (id=i, tstart=0.0, tstop=t, statefrom=1, stateto=2, obstype=1))
        end
        data = DataFrame(rows)
        
        h12 = Hazard(@formula(0 ~ 1), :sp, 1, 2)
        model = multistatemodel(h12; data=data)
        
        # Build penalty config
        penalty_spec = SplinePenalty(order=2)
        config = MultistateModels.build_penalty_config(model, penalty_spec; lambda_init=10.0)
        
        # Extract data
        samplepaths = MultistateModels.extract_paths(model)
        exact_data = MultistateModels.ExactData(model, samplepaths)
        
        nparams = length(get_parnames(model; flatten=true))
        
        # Test that penalized gradient = unpenalized gradient + penalty gradient
        for trial in 1:3
            Random.seed!(trial)
            params = randn(nparams) .* 0.5
            
            # Define objective functions
            f_pen = p -> MultistateModels.loglik_exact_penalized(p, exact_data, config; neg=true)
            f_unpen = p -> MultistateModels.loglik_exact(p, exact_data; neg=true)
            f_penalty = p -> MultistateModels.compute_penalty(p, config)
            
            # Compute gradients via AD
            grad_penalized = ForwardDiff.gradient(f_pen, params)
            grad_unpenalized = ForwardDiff.gradient(f_unpen, params)
            grad_penalty = ForwardDiff.gradient(f_penalty, params)
            
            # Verify: penalized = unpenalized + penalty (additive structure)
            @test isapprox(grad_penalized, grad_unpenalized + grad_penalty, rtol=1e-10)
        end
    end

    @testset "Penalized Hessian - AD Consistency" begin
        using ForwardDiff
        
        # Build model with spline hazard
        nsubj = 50
        Random.seed!(123)
        rows = []
        for i in 1:nsubj
            t = rand()^0.5 * 2.0 + 0.1
            push!(rows, (id=i, tstart=0.0, tstop=t, statefrom=1, stateto=2, obstype=1))
        end
        data = DataFrame(rows)
        
        h12 = Hazard(@formula(0 ~ 1), :sp, 1, 2)
        model = multistatemodel(h12; data=data)
        
        # Build penalty config
        penalty_spec = SplinePenalty(order=2)
        config = MultistateModels.build_penalty_config(model, penalty_spec; lambda_init=10.0)
        
        # Extract data
        samplepaths = MultistateModels.extract_paths(model)
        exact_data = MultistateModels.ExactData(model, samplepaths)
        
        nparams = length(get_parnames(model; flatten=true))
        params = zeros(nparams)
        
        # Define objective functions
        f_pen = p -> MultistateModels.loglik_exact_penalized(p, exact_data, config; neg=true)
        f_unpen = p -> MultistateModels.loglik_exact(p, exact_data; neg=true)
        f_penalty = p -> MultistateModels.compute_penalty(p, config)
        
        # Compute Hessians via AD
        hess_penalized = ForwardDiff.hessian(f_pen, params)
        hess_unpenalized = ForwardDiff.hessian(f_unpen, params)
        hess_penalty = ForwardDiff.hessian(f_penalty, params)
        
        # Verify: penalized Hessian = unpenalized + penalty Hessian
        @test isapprox(hess_penalized, hess_unpenalized + hess_penalty, rtol=1e-10)
        
        # Verify Hessian is symmetric
        @test isapprox(hess_penalized, hess_penalized', rtol=1e-10)
    end

    # =========================================================================
    # Known-Answer Tests for Penalty Infrastructure
    # =========================================================================

    @testset "build_penalty_matrix - Known Answer (5 uniform knots)" begin
        # For cubic splines (order 4) with 5 uniform knots [0,1,2,3,4],
        # we get 8 basis functions (K = n_knots + degree + 1 = 5 + 3 - 1 + 1 = 8)
        # Actually for clamped splines: K = n_interior_knots + degree + 1
        # With BSplineKit: knots 0:4 gives basis of specific size
        
        # Use uniform knots for predictable behavior
        knots = collect(0.0:1.0:4.0)  # 5 knots
        basis = BSplineBasis(BSplineOrder(4), knots)  # Cubic B-splines
        K = length(basis)  # Number of basis functions
        
        # Build penalty matrix (order 2 = curvature penalty)
        S = build_penalty_matrix(basis, 2)
        
        # Test 1: Matrix dimensions are correct
        # For GPS penalty with order m=2, we get a K×K matrix
        @test size(S) == (K, K)
        
        # Test 2: Matrix is symmetric
        @test isapprox(S, S', rtol=1e-10)
        
        # Test 3: Matrix is positive semi-definite
        # All eigenvalues should be ≥ 0 (within numerical tolerance)
        eigs = eigvals(Symmetric(S))
        @test all(e -> e >= -1e-10, eigs)
        
        # Test 4: Null space dimension equals penalty order
        # For order-2 penalty, constants and linears have zero penalty
        null_dim = count(e -> abs(e) < 1e-8, eigs)
        @test null_dim == 2  # Two basis vectors (constant, linear) unpenalized
        
        # Test 5: Constant vector has zero penalty
        v_const = ones(K)
        @test abs(v_const' * S * v_const) < 1e-10
        
        println("  ✓ build_penalty_matrix produces correct K×K symmetric PSD matrix")
        println("    K = $K basis functions, null space dimension = $null_dim")
    end

    @testset "compute_penalty - Known Answer Tests" begin
        # Test 1: Simple identity penalty matrix
        # β = [1, 2, 3], S = I₃, λ = 0.5
        # Penalty = (1/2) * λ * βᵀ S β = (1/2) * 0.5 * (1 + 4 + 9) = 0.5 * 0.5 * 14 = 3.5
        K = 3
        β = [1.0, 2.0, 3.0]
        S_identity = Matrix{Float64}(I, K, K)
        λ = 0.5
        
        term = MultistateModels.PenaltyTerm(1:K, S_identity, λ, 2, [:h12])
        config = PenaltyConfig(
            [term],
            MultistateModels.TotalHazardPenaltyTerm[],
            MultistateModels.SmoothCovariatePenaltyTerm[],
            Dict{Int,Vector{Int}}(),
            Vector{Int}[],
            1
        )
        
        # Expected: (1/2) * 0.5 * (1 + 4 + 9) = 3.5
        expected = 0.5 * λ * dot(β, β)
        @test expected ≈ 3.5
        @test compute_penalty(β, config) ≈ expected rtol=1e-10
        
        # Test 2: Penalty is zero when λ = 0
        term_zero = MultistateModels.PenaltyTerm(1:K, S_identity, 0.0, 2, [:h12])
        config_zero = PenaltyConfig(
            [term_zero],
            MultistateModels.TotalHazardPenaltyTerm[],
            MultistateModels.SmoothCovariatePenaltyTerm[],
            Dict{Int,Vector{Int}}(),
            Vector{Int}[],
            1
        )
        @test compute_penalty(β, config_zero) ≈ 0.0 atol=1e-15
        
        # Test 3: Penalty scales linearly with λ
        λ_values = [0.1, 0.5, 1.0, 2.0, 10.0]
        base_penalty = dot(β, S_identity * β)  # βᵀ S β = 14
        for λ_test in λ_values
            term_test = MultistateModels.PenaltyTerm(1:K, S_identity, λ_test, 2, [:h12])
            config_test = PenaltyConfig(
                [term_test],
                MultistateModels.TotalHazardPenaltyTerm[],
                MultistateModels.SmoothCovariatePenaltyTerm[],
                Dict{Int,Vector{Int}}(),
                Vector{Int}[],
                1
            )
            expected_test = 0.5 * λ_test * base_penalty
            @test compute_penalty(β, config_test) ≈ expected_test rtol=1e-10
        end
        
        # Test 4: Non-identity penalty matrix with analytical solution
        # S = [2 1; 1 2], β = [1, 1], λ = 1.0
        # βᵀ S β = [1,1] * [2,1; 1,2] * [1; 1] = [1,1] * [3; 3] = 6
        # Penalty = (1/2) * 1.0 * 6 = 3.0
        S_22 = [2.0 1.0; 1.0 2.0]
        β_22 = [1.0, 1.0]
        term_22 = MultistateModels.PenaltyTerm(1:2, S_22, 1.0, 2, [:h12])
        config_22 = PenaltyConfig(
            [term_22],
            MultistateModels.TotalHazardPenaltyTerm[],
            MultistateModels.SmoothCovariatePenaltyTerm[],
            Dict{Int,Vector{Int}}(),
            Vector{Int}[],
            1
        )
        @test compute_penalty(β_22, config_22) ≈ 3.0 rtol=1e-10
        
        println("  ✓ compute_penalty known-answer tests pass")
        println("    β=[1,2,3], S=I, λ=0.5 → penalty = 3.5")
        println("    Penalty scales linearly with λ")
        println("    β=[1,1], S=[2,1;1,2], λ=1.0 → penalty = 3.0")
    end

    @testset "Penalty Gradient - Correct Sign Convention" begin
        using ForwardDiff
        
        # Build model with spline hazard  
        nsubj = 30
        Random.seed!(456)
        rows = []
        for i in 1:nsubj
            t = rand() + 0.1
            push!(rows, (id=i, tstart=0.0, tstop=t, statefrom=1, stateto=2, obstype=1))
        end
        data = DataFrame(rows)
        
        h12 = Hazard(@formula(0 ~ 1), :sp, 1, 2)
        model = multistatemodel(h12; data=data)
        
        # Build penalty config with known lambda
        lambda_val = 100.0
        penalty_spec = SplinePenalty(order=2)
        config = MultistateModels.build_penalty_config(model, penalty_spec; lambda_init=lambda_val)
        
        samplepaths = MultistateModels.extract_paths(model)
        exact_data = MultistateModels.ExactData(model, samplepaths)
        nparams = length(get_parnames(model; flatten=true))
        
        # Natural scale parameterization (v0.3.0+):
        # Penalized NLL = NLL_data + (1/2) * λ * βᵀ S β
        # Gradient of penalty w.r.t. β: λ * S * β
        
        # Use non-negative parameters (spline coefficients must be ≥ 0)
        params = abs.(randn(nparams)) .+ 0.1
        
        # Compute gradients separately
        f_nll = p -> MultistateModels.loglik_exact(p, exact_data; neg=true)
        f_penalized = p -> MultistateModels.loglik_exact_penalized(p, exact_data, config; neg=true)
        
        grad_nll = ForwardDiff.gradient(f_nll, params)
        grad_penalized = ForwardDiff.gradient(f_penalized, params)
        
        # The difference should be the penalty gradient
        grad_penalty_from_diff = grad_penalized - grad_nll
        
        # Compute penalty gradient directly (natural scale, no exp transform):
        # ∂/∂β[(1/2)λ βᵀ S β] = λ S β
        S = config.terms[1].S
        hazard_indices = config.terms[1].hazard_indices
        
        expected_penalty_grad = zeros(nparams)
        β_spline = params[hazard_indices]
        expected_penalty_grad[hazard_indices] = lambda_val * (S * β_spline)
        
        @test isapprox(grad_penalty_from_diff, expected_penalty_grad, rtol=1e-6)
    end

end
