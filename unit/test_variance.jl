# =============================================================================
# Variance-Covariance Unit Tests
# =============================================================================
#
# Quick unit tests for variance estimation functionality:
# 1. get_vcov API works correctly with different types
# 2. JK = ((n-1)/n) * IJ algebraic relationship
# 3. Variance matrices are positive semi-definite
# 4. Warnings for missing variance matrices
# 5. Analytical verification for exponential distribution

# Handle both standalone and suite execution
if !@isdefined(TestFixtures)
    using Test
    using MultistateModels
    using DataFrames
    using StatsModels
    include(joinpath(@__DIR__, "..", "fixtures", "TestFixtures.jl"))
end
# Support both standalone execution and module-based test harness
if !@isdefined(TestFixtures)
    include(joinpath(@__DIR__, "..", "fixtures", "TestFixtures.jl"))
end
using .TestFixtures
using LinearAlgebra
using Logging
using Random
import Distributions

# =============================================================================
# Analytical Variance Test for Exponential Distribution
# =============================================================================
# For exponential MLE with exact event times:
# - True Fisher information: I(λ) = n/λ²
# - True variance of MLE: Var(λ̂) = λ²/n
# - MLE of rate: λ̂ = n / Σtᵢ
# Tests compare observed variance to analytical formula.

@testset "Analytical vcov for exponential" begin
    using MultistateModels: get_vcov
    
    Random.seed!(20240110)
    
    # Known parameters for analytical comparison
    true_rate = 0.5
    n_subj = 100
    
    # Generate exponential event times
    times = rand(Distributions.Exponential(1/true_rate), n_subj)
    
    dat = DataFrame(
        id = 1:n_subj,
        tstart = zeros(n_subj),
        tstop = times,
        statefrom = ones(Int, n_subj),
        stateto = fill(2, n_subj),
        obstype = ones(Int, n_subj)
    )
    
    h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    model = multistatemodel(h12; data=dat)
    # Fit with model-based variance for analytical comparison
    fitted = fit(model; verbose=false, vcov_type=:model)
    
    vcov_model = get_vcov(fitted)
    params = get_parameters(fitted; scale=:natural)
    λ_hat = params.h12[1]
    
    @testset "Variance matches analytical formula" begin
        # For exponential MLE, observed information at λ̂ is: I_obs = n/λ̂²
        # Therefore: Var(λ̂) = λ̂²/n
        # The model-based vcov uses observed Fisher information at MLE
        expected_var = λ_hat^2 / n_subj
        
        # Observed variance should match analytical formula closely
        # (no sampling variation here - this is a deterministic relationship at MLE)
        @test isapprox(vcov_model[1, 1], expected_var; rtol=0.01)
    end
    
    @testset "MLE matches analytical formula" begin
        # MLE for exponential rate: λ̂ = n / Σtᵢ
        expected_mle = n_subj / sum(times)
        
        @test isapprox(λ_hat, expected_mle; rtol=0.001)
    end
    
    @testset "Variance has expected magnitude" begin
        # True asymptotic variance: Var(λ̂) ≈ λ²/n = 0.25/100 = 0.0025
        # The observed variance depends on λ̂, not true λ, but should be similar
        true_asymptotic_var = true_rate^2 / n_subj  # = 0.0025
        
        # With 100 observations, estimated variance should be close to theoretical
        @test isapprox(vcov_model[1, 1], true_asymptotic_var; rtol=0.5)
    end
end

@testset "Vcov matrix properties" begin
    using MultistateModels: get_vcov
    
    Random.seed!(20240111)
    
    # Exponential model - 1 parameter, exact analytical comparison
    @testset "Exponential (1 param) - eigenvalue = analytical variance" begin
        n_subj = 80
        times = rand(n_subj) .* 5.0 .+ 0.5  # Uniform times as "events"
        
        dat = DataFrame(
            id = 1:n_subj,
            tstart = zeros(n_subj),
            tstop = times,
            statefrom = ones(Int, n_subj),
            stateto = fill(2, n_subj),
            obstype = ones(Int, n_subj)
        )
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        model = multistatemodel(h12; data=dat)
        
        # Fit with model-based variance
        fitted_model = fit(model; verbose=false, vcov_type=:model)
        vcov_model = get_vcov(fitted_model)
        
        # Fit with IJ variance (same model, different fit)
        fitted_ij = fit(model; verbose=false, vcov_type=:ij)
        vcov_ij = get_vcov(fitted_ij)
        
        # For 1×1 matrix, eigenvalue = the single element
        eig_model = eigvals(Symmetric(vcov_model))[1]
        eig_ij = eigvals(Symmetric(vcov_ij))[1]
        
        @test eig_model ≈ vcov_model[1, 1] atol=1e-12
        @test eig_ij ≈ vcov_ij[1, 1] atol=1e-12
        
        # Analytical: Var(λ̂) = λ̂²/n at the MLE
        λ_hat = get_parameters(fitted_model; scale=:natural).h12[1]
        expected_var = λ_hat^2 / n_subj
        
        @test isapprox(vcov_model[1, 1], expected_var; rtol=0.01)
    end
    
    # Weibull model with 2 parameters (shape + rate)
    @testset "Weibull (2 params) - structural properties" begin
        h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
        n_subj = 80
        
        dat = DataFrame(
            id = 1:n_subj,
            tstart = zeros(n_subj),
            tstop = rand(n_subj) .* 5.0 .+ 0.5,
            statefrom = ones(Int, n_subj),
            stateto = fill(2, n_subj),
            obstype = ones(Int, n_subj)
        )
        
        model = multistatemodel(h12; data=dat)
        set_parameters!(model, (h12 = [1.5, 0.3],))
        
        # Simulate and fit
        sim_result = simulate(model; paths=false, data=true, nsim=1)
        simdat = sim_result[1, 1]
        
        model_fit = multistatemodel(h12; data=simdat)
        
        # Fit with model-based variance
        fitted_model = fit(model_fit; verbose=false, vcov_type=:model)
        vcov_model = get_vcov(fitted_model)
        
        # Fit with IJ variance
        fitted_ij = fit(model_fit; verbose=false, vcov_type=:ij)
        vcov_ij = get_vcov(fitted_ij)
        
        # Symmetry is structural (must hold exactly)
        @test vcov_model ≈ vcov_model' atol=1e-12
        @test vcov_ij ≈ vcov_ij' atol=1e-12
        
        # All eigenvalues should be positive (positive definiteness)
        eig_model = eigvals(Symmetric(vcov_model))
        eig_ij = eigvals(Symmetric(vcov_ij))
        
        @test all(eig_model .> 0)
        @test all(eig_ij .> 0)
        @test isposdef(Symmetric(vcov_model))
        @test isposdef(Symmetric(vcov_ij))
        
        # Eigenvalue sum = trace (checksum for variance computation)
        @test sum(eig_model) ≈ tr(vcov_model) rtol=1e-10
        @test sum(eig_ij) ≈ tr(vcov_ij) rtol=1e-10
        
        # Dimensions match number of parameters
        n_params = length(get_parameters(fitted_model; scale=:flat))
        @test n_params == 2
        @test size(vcov_model) == (n_params, n_params)
        @test size(vcov_ij) == (n_params, n_params)
    end
end

@testset "get_vcov API" begin
    using MultistateModels: get_vcov
    
    @testset "returns correct variance type" begin
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        dat = DataFrame(
            id = [1, 2, 3, 4, 5],
            tstart = zeros(5),
            tstop = [2.0, 3.0, 4.0, 5.0, 6.0],
            statefrom = ones(Int, 5),
            stateto = [2, 2, 2, 2, 2],
            obstype = ones(Int, 5)
        )
        model = multistatemodel(h12; data=dat)
        
        # Fit with each variance type and verify
        fitted_model = fit(model; verbose=false, vcov_type=:model)
        fitted_ij = fit(model; verbose=false, vcov_type=:ij)
        fitted_jk = fit(model; verbose=false, vcov_type=:jk)
        
        # All should have computed vcov
        @test !isnothing(get_vcov(fitted_model))
        @test !isnothing(get_vcov(fitted_ij))
        @test !isnothing(get_vcov(fitted_jk))
        
        # vcov_type should be set correctly
        @test fitted_model.vcov_type == :model
        @test fitted_ij.vcov_type == :ij
        @test fitted_jk.vcov_type == :jk
        
        # All should have same dimensions
        @test size(get_vcov(fitted_model)) == size(get_vcov(fitted_ij)) == size(get_vcov(fitted_jk))
        
        # All should be square with size = number of parameters
        @test size(get_vcov(fitted_model), 1) == size(get_vcov(fitted_model), 2)
    end
    
    @testset "returns nothing when variance not computed" begin
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        dat = DataFrame(
            id = [1, 2, 3],
            tstart = zeros(3),
            tstop = [2.0, 3.0, 4.0],
            statefrom = ones(Int, 3),
            stateto = [2, 2, 2],
            obstype = ones(Int, 3)
        )
        model = multistatemodel(h12; data=dat)
        
        # Fit without variance computation
        fitted = fit(model; verbose=false, vcov_type=:none)
        
        @test fitted.vcov_type == :none
        
        # Suppress warnings about missing variance matrices
        with_logger(NullLogger()) do
            @test isnothing(get_vcov(fitted))
        end
    end
end

@testset "JK = ((n-1)/n) * IJ algebraic identity" begin
    using MultistateModels: get_vcov
    
    h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
    n_subj = 50
    dat = DataFrame(
        id = 1:n_subj,
        tstart = zeros(n_subj),
        tstop = rand(n_subj) .* 5.0 .+ 1.0,
        statefrom = ones(Int, n_subj),
        stateto = fill(2, n_subj),
        obstype = ones(Int, n_subj)
    )
    
    model = multistatemodel(h12; data=dat)
    set_parameters!(model, (h12 = [1.2, 0.15],))
    
    # Simulate and fit
    sim_result = simulate(model; paths=false, data=true, nsim=1)
    simdat = sim_result[1, 1]
    
    model_fit = multistatemodel(h12; data=simdat)
    
    # Fit with IJ variance
    fitted_ij = fit(model_fit; verbose=false, vcov_type=:ij)
    vcov_ij = get_vcov(fitted_ij)
    
    # Fit with JK variance
    fitted_jk = fit(model_fit; verbose=false, vcov_type=:jk)
    vcov_jk = get_vcov(fitted_jk)
    
    # Relationship should hold exactly (algebraic identity)
    n = n_subj
    expected_jk = ((n - 1) / n) * vcov_ij
    
    @test isapprox(vcov_jk, expected_jk; atol=1e-12)
end

@testset "Variance matrices positive semi-definite" begin
    using MultistateModels: get_vcov
    
    h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
    n_subj = 30
    dat = DataFrame(
        id = 1:n_subj,
        tstart = zeros(n_subj),
        tstop = rand(n_subj) .* 5.0 .+ 1.0,
        statefrom = ones(Int, n_subj),
        stateto = fill(2, n_subj),
        obstype = ones(Int, n_subj)
    )
    
    model = multistatemodel(h12; data=dat)
    set_parameters!(model, (h12 = [1.1, 0.20],))
    
    sim_result = simulate(model; paths=false, data=true, nsim=1)
    simdat = sim_result[1, 1]
    
    model_fit = multistatemodel(h12; data=simdat)
    
    # Fit with each variance type
    fitted_model = fit(model_fit; verbose=false, vcov_type=:model)
    fitted_ij = fit(model_fit; verbose=false, vcov_type=:ij)
    fitted_jk = fit(model_fit; verbose=false, vcov_type=:jk)
    
    vcov_model = get_vcov(fitted_model)
    vcov_ij = get_vcov(fitted_ij)
    vcov_jk = get_vcov(fitted_jk)
    
    # Check positive semi-definiteness (eigenvalues >= 0 with tolerance for numerical errors)
    @test isposdef(Symmetric(vcov_model + sqrt(eps()) * I))
    @test isposdef(Symmetric(vcov_ij + sqrt(eps()) * I))
    @test isposdef(Symmetric(vcov_jk + sqrt(eps()) * I))
end

@testset "Variance with panel data (Markov)" begin
    using MultistateModels: get_vcov
    
    h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
    
    n_subj = 50
    nobs = 3
    dat = DataFrame(
        id = repeat(1:n_subj, inner=nobs),
        tstart = repeat([0.0, 2.0, 4.0], n_subj),
        tstop = repeat([2.0, 4.0, 6.0], n_subj),
        statefrom = repeat([1, 1, 1], n_subj),
        stateto = repeat([1, 1, 1], n_subj),
        obstype = repeat([2, 2, 2], n_subj)  # Panel data
    )
    
    model = multistatemodel(h12, h23; data=dat)
    set_parameters!(model, (h12 = [0.2], h23 = [0.15]))
    
    sim_result = simulate(model; paths=false, data=true, nsim=1, autotmax=false)
    simdat = sim_result[1]
    
    model_fit = multistatemodel(h12, h23; data=simdat)
    
    # Fit with model-based and IJ variance
    fitted_model = fit(model_fit; verbose=false, vcov_type=:model)
    fitted_ij = fit(model_fit; verbose=false, vcov_type=:ij)
    
    vcov_model = get_vcov(fitted_model)
    vcov_ij = get_vcov(fitted_ij)
    
    @test !isnothing(vcov_model)
    @test !isnothing(vcov_ij)
    @test size(vcov_model) == (2, 2)
    @test size(vcov_ij) == (2, 2)
    
    # Check positive semi-definiteness (stronger than diagonal > 0)
    @test isposdef(Symmetric(vcov_model + sqrt(eps()) * I))
    @test isposdef(Symmetric(vcov_ij + sqrt(eps()) * I))
end

# =============================================================================
# compute_subject_hessians variants consistency tests
# =============================================================================
# These tests verify that all implementation variants produce identical results

@testset "compute_subject_hessians variants consistency" begin
    using MultistateModels: compute_subject_hessians, compute_subject_hessians_batched,
                           compute_subject_hessians_threaded, compute_subject_hessians_fast,
                           extract_paths
    
    Random.seed!(20240112)
    
    @testset "All variants produce identical results (exponential)" begin
        # Simple exponential model
        n_subj = 30
        times = rand(Distributions.Exponential(2.0), n_subj)
        
        dat = DataFrame(
            id = 1:n_subj,
            tstart = zeros(n_subj),
            tstop = times,
            statefrom = ones(Int, n_subj),
            stateto = fill(2, n_subj),
            obstype = ones(Int, n_subj)
        )
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        model = multistatemodel(h12; data=dat)
        samplepaths = extract_paths(model)
        params = model.parameters.flat
        
        # Compute with all variants
        hess_seq = compute_subject_hessians(params, model, samplepaths)
        hess_batched = compute_subject_hessians_batched(params, model, samplepaths)
        hess_threaded = compute_subject_hessians_threaded(params, model, samplepaths)
        hess_fast = compute_subject_hessians_fast(params, model, samplepaths)
        
        # All should have same length
        @test length(hess_seq) == length(hess_batched) == length(hess_threaded) == length(hess_fast) == n_subj
        
        # All should produce identical values (to machine precision)
        for i in 1:n_subj
            @test hess_seq[i] ≈ hess_batched[i] atol=1e-12
            @test hess_seq[i] ≈ hess_threaded[i] atol=1e-12
            @test hess_seq[i] ≈ hess_fast[i] atol=1e-12
        end
    end
    
    @testset "All variants produce identical results (Weibull with covariates)" begin
        # More complex model: Weibull with covariates
        n_subj = 25
        
        dat = DataFrame(
            id = 1:n_subj,
            tstart = zeros(n_subj),
            tstop = rand(n_subj) .* 5.0 .+ 0.5,
            statefrom = ones(Int, n_subj),
            stateto = fill(2, n_subj),
            obstype = ones(Int, n_subj),
            x = randn(n_subj)
        )
        
        h12 = Hazard(@formula(0 ~ 1 + x), "wei", 1, 2)
        model = multistatemodel(h12; data=dat)
        samplepaths = extract_paths(model)
        params = model.parameters.flat
        
        # Should have 3 parameters: shape, rate, x coefficient
        @test length(params) == 3
        
        # Compute with all variants
        hess_seq = compute_subject_hessians(params, model, samplepaths)
        hess_batched = compute_subject_hessians_batched(params, model, samplepaths)
        hess_threaded = compute_subject_hessians_threaded(params, model, samplepaths)
        hess_fast = compute_subject_hessians_fast(params, model, samplepaths)
        
        # Check dimensions
        @test all(size(H) == (3, 3) for H in hess_seq)
        @test all(size(H) == (3, 3) for H in hess_batched)
        @test all(size(H) == (3, 3) for H in hess_threaded)
        @test all(size(H) == (3, 3) for H in hess_fast)
        
        # All should produce identical values
        for i in 1:n_subj
            @test hess_seq[i] ≈ hess_batched[i] atol=1e-10
            @test hess_seq[i] ≈ hess_threaded[i] atol=1e-10
            @test hess_seq[i] ≈ hess_fast[i] atol=1e-10
        end
    end
    
    @testset "Analytical verification: exponential Hessian = -1/λ²" begin
        # For exponential: ℓᵢ(λ) = log(λ) - λtᵢ, so Hᵢ = -1/λ²
        n_subj = 20
        λ_test = 0.4
        
        dat = DataFrame(
            id = 1:n_subj,
            tstart = zeros(n_subj),
            tstop = rand(n_subj) .* 5.0 .+ 0.5,
            statefrom = ones(Int, n_subj),
            stateto = fill(2, n_subj),
            obstype = ones(Int, n_subj)
        )
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        model = multistatemodel(h12; data=dat)
        set_parameters!(model, (h12 = [λ_test],))
        
        samplepaths = extract_paths(model)
        params = model.parameters.flat
        
        expected_hessian = -1.0 / λ_test^2
        
        # Test all variants against analytical formula
        for (name, hess_fn) in [
            ("sequential", () -> compute_subject_hessians(params, model, samplepaths)),
            ("batched", () -> compute_subject_hessians_batched(params, model, samplepaths)),
            ("threaded", () -> compute_subject_hessians_threaded(params, model, samplepaths)),
            ("fast", () -> compute_subject_hessians_fast(params, model, samplepaths))
        ]
            hessians = hess_fn()
            for (i, H) in enumerate(hessians)
                @test H[1,1] ≈ expected_hessian atol=1e-10
            end
        end
    end
    
    @testset "Sum of subject Hessians equals total Hessian" begin
        using ForwardDiff
        using MultistateModels: loglik_exact, ExactData
        
        n_subj = 15
        dat = DataFrame(
            id = 1:n_subj,
            tstart = zeros(n_subj),
            tstop = rand(n_subj) .* 5.0 .+ 0.5,
            statefrom = ones(Int, n_subj),
            stateto = fill(2, n_subj),
            obstype = ones(Int, n_subj)
        )
        
        h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
        model = multistatemodel(h12; data=dat)
        samplepaths = extract_paths(model)
        params = model.parameters.flat
        
        # Compute total Hessian directly
        data = ExactData(model, samplepaths)
        total_hess = ForwardDiff.hessian(p -> loglik_exact(p, data; neg=false), params)
        
        # Compute sum of subject Hessians
        subject_hessians = compute_subject_hessians_fast(params, model, samplepaths)
        sum_hess = sum(subject_hessians)
        
        # They should be equal
        @test total_hess ≈ sum_hess atol=1e-10
    end
end

# =============================================================================
# Constrained Variance Functions (merged from test_constrained_variance.jl)
# =============================================================================
#
# Tests the reduced Hessian approach for variance estimation under constraints:
#   Var(θ̂) = Z(Z'HZ)⁻¹Z'
# where Z spans the null space of the active constraint Jacobian.
#
# Reference: Item #27 in CODEBASE_REFACTORING_GUIDE.md
# =============================================================================

# Import internal functions for constrained variance testing
import MultistateModels: identify_active_constraints, compute_constraint_jacobian,
    identify_bound_parameters, compute_null_space_basis, compute_constrained_vcov

@testset "identify_active_constraints" begin
    @testset "All constraints active (equality constraints)" begin
        cons_fn = θ -> [θ[1] + θ[2] - 1.0]
        constraints = (cons_fn = cons_fn, lcons = [0.0], ucons = [0.0])
        
        θ = [0.5, 0.5]
        active = identify_active_constraints(θ, constraints)
        @test active[1] == true
        @test sum(active) == 1
    end
    
    @testset "No constraints active (interior point)" begin
        cons_fn = θ -> [θ[1] + θ[2]]
        constraints = (cons_fn = cons_fn, lcons = [0.0], ucons = [1.0])
        
        θ = [0.2, 0.2]
        active = identify_active_constraints(θ, constraints)
        @test active[1] == false
    end
    
    @testset "Inequality at lower bound" begin
        cons_fn = θ -> [θ[1] + θ[2]]
        constraints = (cons_fn = cons_fn, lcons = [0.0], ucons = [1.0])
        
        θ = [0.0, 0.0]
        active = identify_active_constraints(θ, constraints)
        @test active[1] == true
    end
    
    @testset "Inequality at upper bound" begin
        cons_fn = θ -> [θ[1] + θ[2]]
        constraints = (cons_fn = cons_fn, lcons = [0.0], ucons = [1.0])
        
        θ = [0.5, 0.5]
        active = identify_active_constraints(θ, constraints)
        @test active[1] == true
    end
    
    @testset "Multiple constraints with mixed activity" begin
        cons_fn = θ -> [θ[1] - θ[2], θ[1] + θ[2]]
        constraints = (cons_fn = cons_fn, lcons = [0.0, -Inf], ucons = [0.0, 2.0])
        
        θ = [0.5, 0.5]
        active = identify_active_constraints(θ, constraints)
        @test active[1] == true   # equality satisfied
        @test active[2] == false  # interior of inequality
    end
    
    @testset "Tolerance handling" begin
        cons_fn = θ -> [θ[1]]
        constraints = (cons_fn = cons_fn, lcons = [0.0], ucons = [0.0])
        
        θ = [1e-5]
        active_tight = identify_active_constraints(θ, constraints; tol=1e-6)
        @test active_tight[1] == false
        
        active_loose = identify_active_constraints(θ, constraints; tol=1e-4)
        @test active_loose[1] == true
    end
end

@testset "compute_constraint_jacobian" begin
    @testset "Linear constraint" begin
        cons_fn = θ -> [θ[1] + 2θ[2] - 3θ[3]]
        constraints = (cons_fn = cons_fn,)
        
        θ = [1.0, 2.0, 3.0]
        J = compute_constraint_jacobian(θ, constraints)
        
        @test size(J) == (1, 3)
        @test J[1, 1] ≈ 1.0
        @test J[1, 2] ≈ 2.0
        @test J[1, 3] ≈ -3.0
    end
    
    @testset "Nonlinear constraint" begin
        cons_fn = θ -> [θ[1]^2 + θ[2]^2 - 1]
        constraints = (cons_fn = cons_fn,)
        
        θ = [0.6, 0.8]
        J = compute_constraint_jacobian(θ, constraints)
        
        @test size(J) == (1, 2)
        @test J[1, 1] ≈ 2 * 0.6
        @test J[1, 2] ≈ 2 * 0.8
    end
    
    @testset "Multiple constraints" begin
        cons_fn = θ -> [θ[1] + θ[2], θ[1] - θ[2]]
        constraints = (cons_fn = cons_fn,)
        
        θ = [1.0, 2.0]
        J = compute_constraint_jacobian(θ, constraints)
        
        @test size(J) == (2, 2)
        @test J ≈ [1.0 1.0; 1.0 -1.0]
    end
end

@testset "identify_bound_parameters" begin
    @testset "No parameters at bounds" begin
        θ = [0.5, 1.0, 1.5]
        lb = [0.0, 0.0, 0.0]
        ub = [2.0, 2.0, 2.0]
        
        at_bounds = identify_bound_parameters(θ, lb, ub)
        @test sum(at_bounds) == 0
    end
    
    @testset "Parameter at lower bound" begin
        θ = [0.0, 1.0, 1.5]
        lb = [0.0, 0.0, 0.0]
        ub = [2.0, 2.0, 2.0]
        
        at_bounds = identify_bound_parameters(θ, lb, ub)
        @test at_bounds[1] == true
        @test at_bounds[2] == false
    end
    
    @testset "Multiple parameters at bounds" begin
        θ = [0.0, 2.0, 1.5]
        lb = [0.0, 0.0, 0.0]
        ub = [2.0, 2.0, 2.0]
        
        at_bounds = identify_bound_parameters(θ, lb, ub)
        @test sum(at_bounds) == 2
        @test at_bounds[1] == true
        @test at_bounds[2] == true
        @test at_bounds[3] == false
    end
end

@testset "compute_null_space_basis" begin
    @testset "Empty matrix returns identity" begin
        J = Matrix{Float64}(undef, 0, 3)
        Z = compute_null_space_basis(J)
        
        @test size(Z) == (3, 3)
        @test Z ≈ Matrix(I, 3, 3)
    end
    
    @testset "Full column rank has trivial null space" begin
        J = Matrix{Float64}(I, 2, 2)
        Z = compute_null_space_basis(J)
        
        @test size(Z) == (2, 0)
    end
    
    @testset "Single linear constraint" begin
        J = [1.0 1.0 1.0]
        Z = compute_null_space_basis(J)
        
        @test size(Z) == (3, 2)
        @test Z' * Z ≈ Matrix(I, 2, 2) atol=1e-10
        @test norm(J * Z) ≈ 0.0 atol=1e-10
    end
end

@testset "compute_constrained_vcov" begin
    @testset "No active constraints reduces to standard inverse" begin
        H = -[2.0 0.0; 0.0 3.0]
        J_active = Matrix{Float64}(undef, 0, 2)
        
        vcov = compute_constrained_vcov(H, J_active)
        expected = inv(-H)
        @test vcov ≈ expected atol=1e-10
    end
    
    @testset "Single equality constraint" begin
        H = -[4.0 0.0; 0.0 4.0]
        J_active = [1.0 -1.0]
        
        vcov = compute_constrained_vcov(H, J_active)
        
        @test issymmetric(vcov)
        @test vcov[1, 1] ≈ vcov[2, 2] atol=1e-10
    end
    
    @testset "Variance in constrained direction is zero" begin
        H = -[2.0 0.0; 0.0 2.0]
        J_active = [1.0 0.0]
        
        vcov = compute_constrained_vcov(H, J_active)
        
        @test abs(vcov[1, 1]) < 1e-10
        @test vcov[2, 2] > 0
    end
end
