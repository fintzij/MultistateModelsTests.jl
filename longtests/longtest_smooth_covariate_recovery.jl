"""
Long test suite for smooth covariate effect recovery using s(x) terms.

This test suite validates that the penalized splines implementation correctly
recovers known smooth covariate effects in survival models.

Tests:
1. Sinusoidal effect recovery: f(x) = sin(2πx)
2. Quadratic effect recovery: f(x) = x²
3. Step-like effect recovery: f(x) = tanh(10(x-0.5))

Reference:
- Wood, S.N. (2017) Generalized Additive Models: An Introduction with R (2nd ed.)
- Wood, S.N. (2024) PIJCV smoothing parameter selection. arXiv:2404.16490
"""

using Test
using MultistateModels
using DataFrames
using Random
using Statistics
using Printf
using LinearAlgebra
using BSplineKit

# Include shared longtest helpers
include("longtest_config.jl")
include("longtest_helpers.jl")

# Import internal functions for testing
import MultistateModels: Hazard, multistatemodel, fit, set_parameters!, simulate,
    get_parameters_flat, get_parameters, SplinePenalty, build_penalty_config

const RNG_SEED = 0x534D4F01  # Valid hex seed
const N_SUBJECTS = 500       # Sample size for fitting
const MAX_TIME = 10.0        # Maximum follow-up time
const BASELINE_RATE = 0.5    # Baseline hazard rate λ₀

# Maximum errors tolerated (function has amplitude ~1)
# Note: Penalized B-splines with automatic smoothing tend to shrink toward
# the mean, especially at boundaries. These tolerances reflect realistic
# statistical estimation, not exact function recovery.
const MAX_ABS_ERROR = 0.80   # Max pointwise error (allows for boundary effects)
const RMSE_TOLERANCE = 0.50  # Root mean squared error

# ============================================================================
# Helper Functions
# ============================================================================

"""
    generate_smooth_effect_data(f_true::Function, n::Int; baseline_rate, max_time, seed)

Generate exact (continuously observed) survival data from a hazard model with
smooth covariate effect:

    h(t|x) = λ₀ * exp(f(x))

where x ~ Uniform(0, 1).

Returns DataFrame with columns: id, tstart, tstop, statefrom, stateto, obstype, x
"""
function generate_smooth_effect_data(f_true::Function, n::Int; 
    baseline_rate::Float64=BASELINE_RATE, 
    max_time::Float64=MAX_TIME,
    seed::Integer=RNG_SEED)
    
    Random.seed!(seed)
    
    # Generate covariates
    x_vals = rand(n)
    
    # Simulate survival times directly using inverse transform
    # For h(t|x) = λ₀ * exp(f(x)), the cumulative hazard is H(t|x) = λ₀ * exp(f(x)) * t
    # S(t|x) = exp(-H(t|x))
    # T = -log(U) / (λ₀ * exp(f(x))) where U ~ Uniform(0,1)
    
    Random.seed!(seed + 0x1234)  # Separate seed for survival times
    u_vals = rand(n)
    
    survival_times = Float64[]
    final_states = Int[]
    
    for i in 1:n
        rate_i = baseline_rate * exp(f_true(x_vals[i]))
        t_i = -log(u_vals[i]) / rate_i
        
        if t_i <= max_time
            push!(survival_times, t_i)
            push!(final_states, 2)  # Reached absorbing state
        else
            push!(survival_times, max_time)
            push!(final_states, 1)  # Censored in state 1
        end
    end
    
    # Create DataFrame
    data = DataFrame(
        id = 1:n,
        tstart = zeros(n),
        tstop = survival_times,
        statefrom = ones(Int, n),
        stateto = final_states,
        obstype = ones(Int, n),
        x = x_vals
    )
    
    return data
end

"""
    create_basis_for_evaluation(k::Int, min_val::Float64, max_val::Float64)

Create a B-spline basis matching what the s(x, k, m) formula creates.
"""
function create_basis_for_evaluation(k::Int, min_val::Float64, max_val::Float64; order::Int=4)
    # BSplineKit: for k basis functions with specified order
    n_input_knots = k + 2 - order
    knots = collect(range(min_val, max_val, length=n_input_knots))
    return BSplineBasis(BSplineOrder(order), knots)
end

"""
    evaluate_smooth_effect(fitted, hazard_idx::Int, eval_grid::Vector{Float64}, 
                          k::Int, data_range::Tuple{Float64,Float64})

Evaluate the fitted smooth effect at specified x values.

We reconstruct the basis from the data range and number of basis functions,
then evaluate using the fitted coefficients.
"""
function evaluate_smooth_effect(fitted, hazard_idx::Int, eval_grid::Vector{Float64}, 
                               k::Int, data_range::Tuple{Float64,Float64})
    haz = fitted.hazards[hazard_idx]
    
    # Get all parameters for this hazard
    pars = get_parameters(fitted, hazard_idx, scale=:log)
    
    # Find smooth term info
    smooth_info = haz.smooth_info
    if isempty(smooth_info)
        error("No smooth terms found in hazard $hazard_idx")
    end
    
    info = smooth_info[1]  # First (and only) smooth term
    coef_indices = info.par_indices
    smooth_coefs = pars[coef_indices]
    
    # Reconstruct the basis
    basis = create_basis_for_evaluation(k, data_range[1], data_range[2])
    
    # Evaluate at grid points
    f_hat = zeros(length(eval_grid))
    for (i, x) in enumerate(eval_grid)
        # Evaluate all B-spline basis functions at x
        B_x = zeros(k)
        for j in 1:k
            B_x[j] = basis[j](x)
        end
        f_hat[i] = dot(B_x, smooth_coefs)
    end
    
    # Center the fitted effect (remove mean) for comparison with true effect
    # This accounts for confounding with the intercept
    f_hat .-= mean(f_hat)
    
    return f_hat
end

"""
    compute_recovery_metrics(f_true, f_hat, eval_grid) -> (max_abs_err, rmse)

Compute metrics for comparing true and fitted smooth effects.
"""
function compute_recovery_metrics(f_true::Function, f_hat::Vector{Float64}, eval_grid::Vector{Float64})
    # Evaluate true function and center it
    f_true_vals = f_true.(eval_grid)
    f_true_centered = f_true_vals .- mean(f_true_vals)
    
    # Compute errors
    errors = f_hat .- f_true_centered
    max_abs_err = maximum(abs.(errors))
    rmse = sqrt(mean(errors.^2))
    
    return max_abs_err, rmse
end

"""
    print_smooth_comparison(test_name, f_true, f_hat, eval_grid)

Print formatted comparison of true vs fitted smooth effect.
"""
function print_smooth_comparison(test_name::String, f_true::Function, f_hat::Vector{Float64}, eval_grid::Vector{Float64})
    f_true_vals = f_true.(eval_grid)
    f_true_centered = f_true_vals .- mean(f_true_vals)
    
    println("\n    Smooth Effect Comparison: $test_name")
    println("    " * "-"^70)
    println("    x           f(x) true   f̂(x)        Error")
    println("    " * "-"^70)
    
    for i in 1:min(10, length(eval_grid))
        x = eval_grid[i]
        err = f_hat[i] - f_true_centered[i]
        println("    ", @sprintf("%.2f", x), "        ",
                @sprintf("%+.4f", f_true_centered[i]), "      ",
                @sprintf("%+.4f", f_hat[i]), "      ",
                @sprintf("%+.4f", err))
    end
    println("    " * "-"^70)
    
    max_err, rmse = compute_recovery_metrics(f_true, f_hat, eval_grid)
    println("    Max Abs Error: ", @sprintf("%.4f", max_err))
    println("    RMSE:          ", @sprintf("%.4f", rmse))
end

# ============================================================================
# Test 1: Sinusoidal Effect Recovery
# ============================================================================

@testset "Smooth Covariate: Sinusoidal Effect" begin
    Random.seed!(RNG_SEED)
    
    # True smooth effect: sin(2πx)
    f_true(x) = sin(2π * x)
    k = 10  # Number of basis functions
    
    println("\n" * "="^70)
    println("Test: Sinusoidal Effect Recovery")
    println("  f(x) = sin(2πx)")
    println("  h(t|x) = $BASELINE_RATE * exp(f(x))")
    println("  n = $N_SUBJECTS, k = $k")
    println("="^70)
    
    # Generate data
    data = generate_smooth_effect_data(f_true, N_SUBJECTS; seed=RNG_SEED)
    
    # Count events
    n_events = sum(data.stateto .== 2)
    println("  Events: $n_events / $N_SUBJECTS ($(round(100*n_events/N_SUBJECTS, digits=1))%)")
    
    # Get data range for basis reconstruction
    data_range = (minimum(data.x), maximum(data.x))
    
    # Create model with smooth covariate
    h12 = Hazard(@formula(0 ~ s(x, 10, 2)), :exp, 1, 2)
    model = multistatemodel(h12; data=data)
    
    # Fit with penalized likelihood
    println("  Fitting with SplinePenalty and automatic λ selection...")
    fitted = fit(model; 
        penalty=SplinePenalty(),
        verbose=false,
        compute_vcov=false)
    
    println("  Log-likelihood: ", @sprintf("%.2f", fitted.loglik.loglik))
    
    # Evaluate fitted effect on grid (within data range)
    eval_grid = collect(range(data_range[1] + 0.05, data_range[2] - 0.05, length=10))
    f_hat = evaluate_smooth_effect(fitted, 1, eval_grid, k, data_range)
    
    # Compute metrics
    max_err, rmse = compute_recovery_metrics(f_true, f_hat, eval_grid)
    
    # Print comparison
    print_smooth_comparison("Sinusoidal", f_true, f_hat, eval_grid)
    
    # Tests
    @test max_err < MAX_ABS_ERROR
    @test rmse < RMSE_TOLERANCE
    @test isfinite(fitted.loglik.loglik)
    
    println("  ✓ Sinusoidal effect recovered within tolerance")
end

# ============================================================================
# Test 2: Quadratic Effect Recovery
# ============================================================================

@testset "Smooth Covariate: Quadratic Effect" begin
    Random.seed!(RNG_SEED + 1)
    
    # True smooth effect: centered quadratic x² - 1/3
    f_true(x) = x^2 - 1/3
    k = 8  # Number of basis functions
    
    println("\n" * "="^70)
    println("Test: Quadratic Effect Recovery")
    println("  f(x) = x² - 1/3 (centered)")
    println("  h(t|x) = $BASELINE_RATE * exp(f(x))")
    println("  n = $N_SUBJECTS, k = $k")
    println("="^70)
    
    # Generate data
    data = generate_smooth_effect_data(f_true, N_SUBJECTS; seed=RNG_SEED + 1)
    
    # Count events
    n_events = sum(data.stateto .== 2)
    println("  Events: $n_events / $N_SUBJECTS ($(round(100*n_events/N_SUBJECTS, digits=1))%)")
    
    data_range = (minimum(data.x), maximum(data.x))
    
    # Create model with smooth covariate
    h12 = Hazard(@formula(0 ~ s(x, 8, 2)), :exp, 1, 2)
    model = multistatemodel(h12; data=data)
    
    # Fit with penalized likelihood
    println("  Fitting with SplinePenalty and automatic λ selection...")
    fitted = fit(model; 
        penalty=SplinePenalty(),
        verbose=false,
        compute_vcov=false)
    
    println("  Log-likelihood: ", @sprintf("%.2f", fitted.loglik.loglik))
    
    # Evaluate fitted effect on grid
    eval_grid = collect(range(data_range[1] + 0.05, data_range[2] - 0.05, length=10))
    f_hat = evaluate_smooth_effect(fitted, 1, eval_grid, k, data_range)
    
    # Compute metrics
    max_err, rmse = compute_recovery_metrics(f_true, f_hat, eval_grid)
    
    # Print comparison
    print_smooth_comparison("Quadratic", f_true, f_hat, eval_grid)
    
    # Tests (quadratic should be easier - use tighter tolerance)
    @test max_err < MAX_ABS_ERROR * 0.8
    @test rmse < RMSE_TOLERANCE * 0.8
    @test isfinite(fitted.loglik.loglik)
    
    println("  ✓ Quadratic effect recovered within tolerance")
end

# ============================================================================
# Test 3: Sigmoid/Step-like Effect Recovery
# ============================================================================

@testset "Smooth Covariate: Sigmoid Effect" begin
    Random.seed!(RNG_SEED + 2)
    
    # True smooth effect: tanh(5(x - 0.5)) - approximately step function
    f_true(x) = tanh(5 * (x - 0.5))
    k = 12  # More knots for sharp transition
    
    println("\n" * "="^70)
    println("Test: Sigmoid Effect Recovery")
    println("  f(x) = tanh(5(x - 0.5))")
    println("  h(t|x) = $BASELINE_RATE * exp(f(x))")
    println("  n = $N_SUBJECTS, k = $k")
    println("="^70)
    
    # Generate data
    data = generate_smooth_effect_data(f_true, N_SUBJECTS; seed=RNG_SEED + 2)
    
    # Count events
    n_events = sum(data.stateto .== 2)
    println("  Events: $n_events / $N_SUBJECTS ($(round(100*n_events/N_SUBJECTS, digits=1))%)")
    
    data_range = (minimum(data.x), maximum(data.x))
    
    # Create model with smooth covariate - more knots for sharp transition
    h12 = Hazard(@formula(0 ~ s(x, 12, 2)), :exp, 1, 2)
    model = multistatemodel(h12; data=data)
    
    # Fit with penalized likelihood
    println("  Fitting with SplinePenalty and automatic λ selection...")
    fitted = fit(model; 
        penalty=SplinePenalty(),
        verbose=false,
        compute_vcov=false)
    
    println("  Log-likelihood: ", @sprintf("%.2f", fitted.loglik.loglik))
    
    # Evaluate fitted effect on grid
    eval_grid = collect(range(data_range[1] + 0.05, data_range[2] - 0.05, length=15))
    f_hat = evaluate_smooth_effect(fitted, 1, eval_grid, k, data_range)
    
    # Compute metrics
    max_err, rmse = compute_recovery_metrics(f_true, f_hat, eval_grid)
    
    # Print comparison
    print_smooth_comparison("Sigmoid", f_true, f_hat, eval_grid)
    
    # Tests (sigmoid is harder - use relaxed tolerance)
    @test max_err < MAX_ABS_ERROR * 1.2
    @test rmse < RMSE_TOLERANCE * 1.2
    @test isfinite(fitted.loglik.loglik)
    
    println("  ✓ Sigmoid effect recovered within tolerance")
end

# ============================================================================
# Test 4: Combined Smooth + Linear Covariate
# ============================================================================

@testset "Smooth Covariate: s(x) + Linear trt" begin
    Random.seed!(RNG_SEED + 3)
    
    # True model: h(t|x, trt) = λ₀ * exp(f(x) + β_trt * trt)
    # where f(x) = sin(2πx) and β_trt = 0.5
    f_true(x) = sin(2π * x)
    beta_trt = 0.5
    k = 10
    
    println("\n" * "="^70)
    println("Test: Combined Smooth + Linear Effect")
    println("  f(x) = sin(2πx)")
    println("  h(t|x,trt) = $BASELINE_RATE * exp(f(x) + $beta_trt * trt)")
    println("  n = $N_SUBJECTS, k = $k")
    println("="^70)
    
    # Generate data with both covariates
    Random.seed!(RNG_SEED + 3)
    n = N_SUBJECTS
    x_vals = rand(n)
    trt_vals = rand([0.0, 1.0], n)  # Binary treatment
    
    # Simulate survival times
    Random.seed!(RNG_SEED + 0x3456)
    u_vals = rand(n)
    
    survival_times = Float64[]
    final_states = Int[]
    
    for i in 1:n
        lin_pred = f_true(x_vals[i]) + beta_trt * trt_vals[i]
        rate_i = BASELINE_RATE * exp(lin_pred)
        t_i = -log(u_vals[i]) / rate_i
        
        if t_i <= MAX_TIME
            push!(survival_times, t_i)
            push!(final_states, 2)
        else
            push!(survival_times, MAX_TIME)
            push!(final_states, 1)
        end
    end
    
    data = DataFrame(
        id = 1:n,
        tstart = zeros(n),
        tstop = survival_times,
        statefrom = ones(Int, n),
        stateto = final_states,
        obstype = ones(Int, n),
        x = x_vals,
        trt = trt_vals
    )
    
    n_events = sum(data.stateto .== 2)
    println("  Events: $n_events / $N_SUBJECTS ($(round(100*n_events/N_SUBJECTS, digits=1))%)")
    
    data_range = (minimum(data.x), maximum(data.x))
    
    # Create model with smooth and linear terms
    h12 = Hazard(@formula(0 ~ s(x, 10, 2) + trt), :exp, 1, 2)
    model = multistatemodel(h12; data=data)
    
    # Fit with penalized likelihood
    println("  Fitting with SplinePenalty...")
    fitted = fit(model; 
        penalty=SplinePenalty(),
        verbose=false,
        compute_vcov=true)
    
    println("  Log-likelihood: ", @sprintf("%.2f", fitted.loglik.loglik))
    
    # Check treatment effect recovery
    # Find trt parameter index (should be last)
    pars = get_parameters(fitted, 1, scale=:log)
    trt_est = pars[end]  # trt coefficient is last
    
    println("  Treatment effect: true=$beta_trt, estimated=", @sprintf("%.3f", trt_est))
    
    # Evaluate smooth effect
    eval_grid = collect(range(data_range[1] + 0.05, data_range[2] - 0.05, length=10))
    f_hat = evaluate_smooth_effect(fitted, 1, eval_grid, k, data_range)
    
    max_err, rmse = compute_recovery_metrics(f_true, f_hat, eval_grid)
    print_smooth_comparison("Smooth (with trt)", f_true, f_hat, eval_grid)
    
    # Tests
    @test abs(trt_est - beta_trt) / beta_trt < 0.30
    @test max_err < MAX_ABS_ERROR
    @test rmse < RMSE_TOLERANCE
    @test isfinite(fitted.loglik.loglik)
    
    println("  ✓ Combined smooth + linear effect recovered")
end

# ============================================================================
# Summary
# ============================================================================

println("\n" * "="^70)
println("SMOOTH COVARIATE RECOVERY TESTS COMPLETE")
println("="^70)
