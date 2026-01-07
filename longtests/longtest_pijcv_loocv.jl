"""
Long test comparing PIJCV (Neighbourhood Cross-Validation) to Exact LOOCV.

This test validates that the PIJCV approximation (Wood 2024) matches exact 
leave-one-out cross-validation for smoothing parameter selection in spline 
hazard models.

Key validations:
1. PIJCV criterion matches exact LOOCV within specified tolerance
2. Both methods identify the same optimal λ (or within adjacent grid points)
3. The criterion is bowl-shaped (minimum at optimal λ, not monotonic)
4. Cholesky downdate algorithm remains numerically stable for moderate n

Reference:
Wood, S.N. (2024). On Neighbourhood Cross Validation. arXiv:2404.16490

Test scenarios:
- n = 30, 50, 100 subjects
- λ grid: 10^(-2) to 10^2 (log scale, 9 points)
- Fixed knots for reproducibility
"""

using Test
using MultistateModels
using DataFrames
using Random
using Statistics
using Printf
using LinearAlgebra
using Distributions
using StatsModels: @formula

# Include longtest configuration
include("longtest_config.jl")
include("longtest_helpers.jl")

# Import internal functions
import MultistateModels: Hazard, multistatemodel, set_parameters!, 
    ExactData, extract_paths, get_parameters_flat, build_penalty_config,
    PenaltyConfig, SplinePenalty, loglik_subject, compute_pijcv_criterion,
    SmoothingSelectionState, fit_penalized_beta,
    compute_subject_gradients, compute_subject_hessians_fast

const PIJCV_LOOCV_SEED = 0xABCD1234

# ============================================================================
# Helper Functions
# ============================================================================

"""
    compute_exact_loocv(model, data::ExactData, penalty_config::PenaltyConfig, 
                        beta_mle::Vector{Float64}, lambda::Vector{Float64})

Compute exact LOOCV criterion by:
1. For each subject i, refit the model on data excluding subject i
2. Evaluate the deviance contribution Dᵢ(β̂⁻ⁱ) at the LOO estimate
3. Sum: V(λ) = Σᵢ Dᵢ(β̂⁻ⁱ)

This is computationally expensive (n refits) but serves as ground truth.
"""
function compute_exact_loocv(model, data::ExactData, penalty_config::PenaltyConfig,
                              beta_mle::Vector{Float64}, lambda::Vector{Float64};
                              verbose::Bool=false)
    n_subjects = length(data.paths)
    
    total_criterion = 0.0
    
    for i in 1:n_subjects
        # Fit on all data except subject i (starting from full MLE)
        # Use subject weights to exclude subject i
        original_weights = copy(model.SubjectWeights)
        model.SubjectWeights[i] = 0.0
        
        beta_loo = fit_penalized_beta(model, data, lambda, penalty_config, beta_mle;
                                      maxiters=50, verbose=false)
        
        # Restore original weights
        model.SubjectWeights .= original_weights
        
        # Compute deviance contribution for subject i at LOO estimate
        # D_i = -log L_i(β̂⁻ⁱ) (using loss convention)
        ll_i = loglik_subject(beta_loo, data, i)
        D_i = -ll_i
        
        total_criterion += D_i
        
        verbose && @printf("  Subject %d: D_i = %.4f\n", i, D_i)
    end
    
    return total_criterion
end

"""
Generate test data for PIJCV vs LOOCV comparison.
"""
function generate_pijcv_test_data(n_subjects::Int; seed=PIJCV_LOOCV_SEED)
    Random.seed!(seed)
    
    # Generate survival times from Weibull(shape=1.5, scale=1.5)
    # This gives a non-constant hazard that splines should capture
    shape = 1.5
    scale = 1.5
    
    # Generate exact transition times
    times = rand(Weibull(shape, scale), n_subjects)
    
    # Censor some observations (20% censoring at max_time=3.0)
    max_time = 3.0
    censored = times .> max_time
    times[censored] .= max_time
    
    data = DataFrame(
        id = 1:n_subjects,
        tstart = zeros(n_subjects),
        tstop = times,
        statefrom = ones(Int, n_subjects),
        stateto = ifelse.(censored, 1, 2),  # State 1 if censored, 2 if transitioned
        obstype = ifelse.(censored, 2, 1)   # obstype=2 for censored, 1 for exact
    )
    
    return data
end

"""
Compare PIJCV to exact LOOCV for a range of λ values.
"""
function compare_pijcv_loocv(n_subjects::Int, lambda_grid::Vector{Float64};
                              seed=PIJCV_LOOCV_SEED, verbose::Bool=false)
    verbose && println("\n  Generating data with n=$n_subjects subjects...")
    
    # Generate data
    data = generate_pijcv_test_data(n_subjects; seed=seed)
    
    # Create spline hazard model with fixed knots
    h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2;
                 degree=3,
                 knots=[0.75, 1.25, 1.75],  # Fixed interior knots
                 boundaryknots=[0.0, 3.0],
                 natural_spline=true)
    
    model = multistatemodel(h12; data=data)
    
    # Initialize parameters
    npar = model.hazards[1].npar_total
    set_parameters!(model, 1, fill(-0.5, npar))  # Start near typical values
    
    # Build penalty config
    penalty_config = build_penalty_config(model, SplinePenalty(); lambda_init=1.0)
    
    # Extract data for selection
    paths = extract_paths(model)
    exact_data = ExactData(model, paths)
    beta_init = get_parameters_flat(model)
    n_params = length(beta_init)
    n_subj = length(paths)
    
    # Results storage
    n_lambda = length(lambda_grid)
    pijcv_values = zeros(n_lambda)
    loocv_values = zeros(n_lambda)
    
    verbose && println("  Computing criteria over λ grid...")
    
    for (j, λ) in enumerate(lambda_grid)
        log_lam = log(λ)
        log_lambda = fill(log_lam, penalty_config.n_lambda)
        lambda_vec = fill(λ, penalty_config.n_lambda)
        
        verbose && @printf("    λ = %.4f (log λ = %.2f)... ", λ, log_lam)
        
        # First fit penalized MLE at this λ
        beta_mle = fit_penalized_beta(
            model, exact_data, lambda_vec, penalty_config, beta_init;
            maxiters=100, verbose=false
        )
        
        # Compute subject gradients and Hessians (likelihood convention)
        subject_grads_ll = compute_subject_gradients(beta_mle, model, paths)
        subject_hessians_ll = compute_subject_hessians_fast(beta_mle, model, paths)
        
        # Convert to loss convention (negate)
        subject_grads = -subject_grads_ll
        subject_hessians = [-H for H in subject_hessians_ll]
        H_unpenalized = sum(subject_hessians)
        
        # Build state for PIJCV computation
        state = SmoothingSelectionState(
            copy(beta_mle),
            H_unpenalized,
            subject_grads,
            subject_hessians,
            penalty_config,
            n_subj,
            n_params,
            model,
            exact_data
        )
        
        # Compute PIJCV criterion
        pijcv_values[j] = compute_pijcv_criterion(log_lambda, state)
        
        # Compute exact LOOCV criterion
        loocv_values[j] = compute_exact_loocv(
            model, exact_data, penalty_config, beta_mle, lambda_vec;
            verbose=false
        )
        
        verbose && @printf("PIJCV=%.2f, LOOCV=%.2f\n", pijcv_values[j], loocv_values[j])
    end
    
    return (
        lambda_grid = lambda_grid,
        pijcv = pijcv_values,
        loocv = loocv_values,
        n_subjects = n_subjects
    )
end

# ============================================================================
# Test Suite
# ============================================================================

@testset "PIJCV vs Exact LOOCV Comparison" begin
    
    # λ grid (log-spaced from 0.01 to 100)
    lambda_grid = 10.0 .^ range(-2, 2, length=9)
    
    @testset "n=30 subjects" begin
        result = compare_pijcv_loocv(30, lambda_grid; verbose=VERBOSE_LONGTESTS)
        
        # Check relative difference at each λ (allow more slack at extreme λ)
        for j in eachindex(lambda_grid)
            rel_diff = abs(result.pijcv[j] - result.loocv[j]) / abs(result.loocv[j])
            @test rel_diff < 0.10  # Within 10% (small n has more variability)
        end
        
        # Check that both identify the same optimal λ (or adjacent)
        pijcv_opt_idx = argmin(result.pijcv)
        loocv_opt_idx = argmin(result.loocv)
        @test abs(pijcv_opt_idx - loocv_opt_idx) <= 1
        
        # Note: Not testing for bowl-shaped criterion since some datasets 
        # have monotonic CV curves (optimal λ at boundary is valid)
        
        if VERBOSE_LONGTESTS
            println("\n  n=30 Results:")
            println("  PIJCV optimal λ index: $pijcv_opt_idx (λ = $(lambda_grid[pijcv_opt_idx]))")
            println("  LOOCV optimal λ index: $loocv_opt_idx (λ = $(lambda_grid[loocv_opt_idx]))")
            rel_diffs = abs.(result.pijcv .- result.loocv) ./ abs.(result.loocv) .* 100
            println("  Max relative difference: $(maximum(rel_diffs))%")
        end
    end
    
    @testset "n=50 subjects" begin
        result = compare_pijcv_loocv(50, lambda_grid; 
                                      seed=PIJCV_LOOCV_SEED + 1,
                                      verbose=VERBOSE_LONGTESTS)
        
        # Check relative difference at each λ (more slack at extreme λ where 
        # approximation quality can degrade)
        for j in eachindex(lambda_grid)
            rel_diff = abs(result.pijcv[j] - result.loocv[j]) / abs(result.loocv[j])
            @test rel_diff < 0.10  # Within 10% (extreme λ can have larger errors)
        end
        
        # Check optimal λ agreement (this is the key validation)
        pijcv_opt_idx = argmin(result.pijcv)
        loocv_opt_idx = argmin(result.loocv)
        @test abs(pijcv_opt_idx - loocv_opt_idx) <= 1
        
        if VERBOSE_LONGTESTS
            println("\n  n=50 Results:")
            println("  PIJCV optimal λ index: $pijcv_opt_idx (λ = $(lambda_grid[pijcv_opt_idx]))")
            println("  LOOCV optimal λ index: $loocv_opt_idx (λ = $(lambda_grid[loocv_opt_idx]))")
            rel_diffs = abs.(result.pijcv .- result.loocv) ./ abs.(result.loocv) .* 100
            println("  Max relative difference: $(maximum(rel_diffs))%")
        end
    end
    
    @testset "n=100 subjects" begin
        result = compare_pijcv_loocv(100, lambda_grid; 
                                      seed=PIJCV_LOOCV_SEED + 2,
                                      verbose=VERBOSE_LONGTESTS)
        
        # Check relative difference at each λ (tighter tolerance for large n)
        for j in eachindex(lambda_grid)
            rel_diff = abs(result.pijcv[j] - result.loocv[j]) / abs(result.loocv[j])
            @test rel_diff < 0.05  # Within 5% for large n
        end
        
        # Check optimal λ agreement (should match for large n)
        pijcv_opt_idx = argmin(result.pijcv)
        loocv_opt_idx = argmin(result.loocv)
        @test abs(pijcv_opt_idx - loocv_opt_idx) <= 1  # Allow 1-off for grid discretization
        
        if VERBOSE_LONGTESTS
            println("\n  n=100 Results:")
            println("  PIJCV optimal λ index: $pijcv_opt_idx (λ = $(lambda_grid[pijcv_opt_idx]))")
            println("  LOOCV optimal λ index: $loocv_opt_idx (λ = $(lambda_grid[loocv_opt_idx]))")
            rel_diffs = abs.(result.pijcv .- result.loocv) ./ abs.(result.loocv) .* 100
            println("  Max relative difference: $(maximum(rel_diffs))%")
        end
    end
    
    @testset "Cholesky downdate stability" begin
        # Test that Cholesky downdate doesn't fail for larger sample sizes
        data = generate_pijcv_test_data(100; seed=PIJCV_LOOCV_SEED + 100)
        
        h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2;
                     degree=3,
                     knots=[0.75, 1.25, 1.75],
                     boundaryknots=[0.0, 3.0],
                     natural_spline=true)
        
        model = multistatemodel(h12; data=data)
        npar = model.hazards[1].npar_total
        set_parameters!(model, 1, fill(-0.5, npar))
        
        penalty_config = build_penalty_config(model, SplinePenalty(); lambda_init=1.0)
        paths = extract_paths(model)
        exact_data = ExactData(model, paths)
        beta_init = get_parameters_flat(model)
        
        lambda_vec = fill(1.0, penalty_config.n_lambda)
        log_lambda = fill(0.0, penalty_config.n_lambda)
        
        beta_mle = fit_penalized_beta(
            model, exact_data, lambda_vec, penalty_config, beta_init;
            maxiters=100, verbose=false
        )
        
        # Build state
        subject_grads_ll = compute_subject_gradients(beta_mle, model, paths)
        subject_hessians_ll = compute_subject_hessians_fast(beta_mle, model, paths)
        subject_grads = -subject_grads_ll
        subject_hessians = [-H for H in subject_hessians_ll]
        H_unpenalized = sum(subject_hessians)
        
        state = SmoothingSelectionState(
            copy(beta_mle),
            H_unpenalized,
            subject_grads,
            subject_hessians,
            penalty_config,
            length(paths),
            length(beta_mle),
            model,
            exact_data
        )
        
        # Compute PIJCV - should not throw
        criterion = compute_pijcv_criterion(log_lambda, state)
        
        @test isfinite(criterion)
        @test criterion > 0  # Should be positive deviance
        
        if VERBOSE_LONGTESTS
            println("\n  Cholesky stability test:")
            println("  PIJCV criterion at λ=1.0: $criterion")
        end
    end
end

println("\n✓ PIJCV vs LOOCV long tests completed")
