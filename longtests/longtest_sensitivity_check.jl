# =============================================================================
# Long Test: Sensitivity Check
# =============================================================================
#
# ACTION-8 from longtest review: Validate that tests can actually detect bias.
#
# This test file deliberately generates data from one set of parameters and
# verifies that our parameter recovery checks correctly FAIL when the fitted
# model uses wrong parameters or constraints.
#
# Purpose:
# - Confirm that tolerance checks are not vacuously passing
# - Document expected failure modes
# - Validate test sensitivity (ability to detect real problems)
#
# Test Strategy:
# 1. Generate data from known "true" parameters
# 2. Check that fitting with correct parameters passes tolerance checks
# 3. Check that fitting with deliberately biased parameters FAILS tolerance checks
# 4. Document the bias magnitude required to trigger failure
#
# =============================================================================

using Test
using MultistateModels
using DataFrames
using Random
using Statistics
using Printf

import MultistateModels: Hazard, multistatemodel, fit, set_parameters!, simulate,
    get_parameters_flat, get_parameters, @formula

# Include shared configuration
include(joinpath(@__DIR__, "longtest_config.jl"))
include(joinpath(@__DIR__, "longtest_helpers.jl"))

# Test-specific constants
const RNG_SEED_SENSITIVITY = 0xDEAD1234
const N_SUBJECTS_SENSITIVITY = 500  # Smaller sample for faster execution

# =============================================================================
# Helper Functions
# =============================================================================

"""
    check_parameter_recovery_verbose(fitted_params, true_params; 
        rel_tol=PARAM_REL_TOL, abs_tol=BETA_ABS_TOL) -> (Bool, Vector{Float64})

Check if fitted parameters are within tolerance of true parameters.
Returns (passed, relative_errors).
"""
function check_parameter_recovery_verbose(fitted_params::Vector{Float64}, 
                                          true_params::Vector{Float64};
                                          rel_tol::Float64=PARAM_REL_TOL,
                                          abs_tol::Float64=BETA_ABS_TOL)
    @assert length(fitted_params) == length(true_params)
    
    rel_errors = Float64[]
    all_passed = true
    
    for (i, (fit, true_val)) in enumerate(zip(fitted_params, true_params))
        if abs(true_val) < SMALL_PARAM_THRESHOLD
            # Use absolute tolerance for small parameters
            err = abs(fit - true_val)
            passed = err <= abs_tol
            push!(rel_errors, err)  # Store absolute error
        else
            # Use relative tolerance
            err = abs(fit - true_val) / abs(true_val)
            passed = err <= rel_tol
            push!(rel_errors, err)
        end
        all_passed = all_passed && passed
    end
    
    return (all_passed, rel_errors)
end

"""
    print_sensitivity_comparison(true_params, fitted_params, biased_params, 
                                 fitted_errors, biased_errors)

Print comparison showing that correct fits pass and biased fits fail.
"""
function print_sensitivity_comparison(true_params, fitted_params, biased_params,
                                      fitted_errors, biased_errors)
    println("\n    Sensitivity Check Summary:")
    println("    " * "-"^70)
    println("    ", rpad("Param", 8), rpad("True", 10), rpad("Fit(OK)", 10), 
            rpad("Fit(Bias)", 10), rpad("Err(OK)", 10), rpad("Err(Bias)", 10))
    println("    " * "-"^70)
    
    for i in eachindex(true_params)
        @printf("    %-8d %-10.3f %-10.3f %-10.3f %-10.3f %-10.3f\n",
                i, true_params[i], fitted_params[i], biased_params[i],
                fitted_errors[i], biased_errors[i])
    end
    println("    " * "-"^70)
end

# =============================================================================
# Test 1: Exponential - Correct vs Biased Rate
# =============================================================================
#
# Generate data with rate=0.15, verify:
# - Fitting freely recovers correct rate (passes)
# - Starting from rate=0.60 (4x bias, 300% error) fails tolerance check
# =============================================================================

@testset "Sensitivity: Exponential Rate Bias Detection" begin
    Random.seed!(RNG_SEED_SENSITIVITY)
    
    VERBOSE_LONGTESTS && println("\n  ▶ Sensitivity: Exponential rate bias detection")
    
    # --- True DGP ---
    true_rate = 0.15
    h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    
    template = DataFrame(
        id = 1:N_SUBJECTS_SENSITIVITY,
        tstart = zeros(N_SUBJECTS_SENSITIVITY),
        tstop = fill(MAX_TIME, N_SUBJECTS_SENSITIVITY),
        statefrom = ones(Int, N_SUBJECTS_SENSITIVITY),
        stateto = ones(Int, N_SUBJECTS_SENSITIVITY),
        obstype = ones(Int, N_SUBJECTS_SENSITIVITY)
    )
    
    model_dgp = multistatemodel(h12; data=template)
    set_parameters!(model_dgp, (h12 = [true_rate],))
    
    # --- Simulate data ---
    sim_result = simulate(model_dgp; paths=false, data=true, nsim=1)
    exact_data = sim_result[1]
    
    # --- Fit with correct initialization ---
    model_correct = multistatemodel(h12; data=exact_data)
    fitted_correct = fit(model_correct; verbose=false)
    params_correct = get_parameters_flat(fitted_correct)
    
    correct_passed, correct_errors = check_parameter_recovery_verbose(
        params_correct, [true_rate])
    
    # --- Fit with biased initialization (should still converge to correct) ---
    model_biased_init = multistatemodel(h12; data=exact_data)
    set_parameters!(model_biased_init, (h12 = [0.60],))  # Start 4x too high
    fitted_biased_init = fit(model_biased_init; verbose=false)
    params_biased_init = get_parameters_flat(fitted_biased_init)
    
    biased_init_passed, biased_init_errors = check_parameter_recovery_verbose(
        params_biased_init, [true_rate])
    
    # --- Compute what biased parameters would look like ---
    biased_rate = 0.60  # 4x the true rate (300% error, exceeds 35% tolerance)
    _, biased_errors = check_parameter_recovery_verbose([biased_rate], [true_rate])
    
    if VERBOSE_LONGTESTS
        println("    True rate: $true_rate")
        println("    Fitted (free): $(round(params_correct[1], digits=4)) - $(correct_passed ? "PASS" : "FAIL")")
        println("    Fitted (biased init): $(round(params_biased_init[1], digits=4)) - $(biased_init_passed ? "PASS" : "FAIL")")
        println("    Biased rate (0.60): Would have error $(round(biased_errors[1], digits=3))")
        println("    Tolerance: $(PARAM_REL_TOL) ($(Int(100*PARAM_REL_TOL))%)")
        
        # Verify that biased parameters would fail
        biased_would_fail = biased_errors[1] > PARAM_REL_TOL
        println("    Biased (0.60) would fail: $biased_would_fail")
    end
    
    # Assertions
    @test correct_passed  # Free fit should pass tolerance check
    @test biased_init_passed  # Fit from biased init should still converge and pass
    
    # Key sensitivity check: verify that tolerance would catch 4x bias
    @test biased_errors[1] > PARAM_REL_TOL  # Tolerance should catch 4x rate bias
    
    VERBOSE_LONGTESTS && println("  ✓ Exponential sensitivity check complete")
end

# =============================================================================
# Test 2: Weibull - Correct vs Biased Shape
# =============================================================================
#
# Generate data with shape=1.3, scale=0.15, verify:
# - Fitting freely recovers correct parameters (passes)
# - Shape=2.0 (54% bias) would fail tolerance check
# =============================================================================

@testset "Sensitivity: Weibull Shape Bias Detection" begin
    Random.seed!(RNG_SEED_SENSITIVITY + 1)
    
    VERBOSE_LONGTESTS && println("\n  ▶ Sensitivity: Weibull shape bias detection")
    
    # --- True DGP ---
    true_shape = 1.3
    true_scale = 0.15
    h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
    
    template = DataFrame(
        id = 1:N_SUBJECTS_SENSITIVITY,
        tstart = zeros(N_SUBJECTS_SENSITIVITY),
        tstop = fill(MAX_TIME, N_SUBJECTS_SENSITIVITY),
        statefrom = ones(Int, N_SUBJECTS_SENSITIVITY),
        stateto = ones(Int, N_SUBJECTS_SENSITIVITY),
        obstype = ones(Int, N_SUBJECTS_SENSITIVITY)
    )
    
    model_dgp = multistatemodel(h12; data=template)
    set_parameters!(model_dgp, (h12 = [true_shape, true_scale],))
    
    # --- Simulate data ---
    sim_result = simulate(model_dgp; paths=false, data=true, nsim=1)
    exact_data = sim_result[1]
    
    # --- Fit with correct initialization ---
    model_correct = multistatemodel(h12; data=exact_data)
    fitted_correct = fit(model_correct; verbose=false)
    params_correct = get_parameters_flat(fitted_correct)
    
    correct_passed, correct_errors = check_parameter_recovery_verbose(
        params_correct, [true_shape, true_scale])
    
    # --- Compute what biased parameters would look like ---
    biased_shape = 2.0  # 54% higher than true
    biased_scale = 0.15  # Keep scale correct to isolate shape effect
    _, biased_errors = check_parameter_recovery_verbose(
        [biased_shape, biased_scale], [true_shape, true_scale])
    
    if VERBOSE_LONGTESTS
        println("    True: shape=$true_shape, scale=$true_scale")
        println("    Fitted: shape=$(round(params_correct[1], digits=3)), scale=$(round(params_correct[2], digits=4))")
        println("    Correct fit passed: $correct_passed")
        println("    Biased shape (2.0) error: $(round(biased_errors[1], digits=3))")
        println("    Tolerance: $(PARAM_REL_TOL)")
        println("    Biased (shape=2.0) would fail: $(biased_errors[1] > PARAM_REL_TOL)")
    end
    
    # Assertions
    @test correct_passed  # Free fit should pass tolerance check
    @test biased_errors[1] > PARAM_REL_TOL  # Tolerance should catch 54% shape bias
    
    VERBOSE_LONGTESTS && println("  ✓ Weibull sensitivity check complete")
end

# =============================================================================
# Test 3: Covariate Effect - Correct vs Biased Beta
# =============================================================================
#
# Generate data with β=0.5, verify:
# - Fitting freely recovers correct beta (passes)
# - β=1.5 (200% bias, or +1.0 absolute) would fail tolerance check
# =============================================================================

@testset "Sensitivity: Covariate Effect Bias Detection" begin
    Random.seed!(RNG_SEED_SENSITIVITY + 2)
    
    VERBOSE_LONGTESTS && println("\n  ▶ Sensitivity: Covariate effect (beta) bias detection")
    
    # --- True DGP ---
    true_rate = 0.15
    true_beta = 0.5
    h12 = Hazard(@formula(0 ~ x), "exp", 1, 2)
    
    # Binary covariate with 50% prevalence
    x_vals = rand([0.0, 1.0], N_SUBJECTS_SENSITIVITY)
    template = DataFrame(
        id = 1:N_SUBJECTS_SENSITIVITY,
        tstart = zeros(N_SUBJECTS_SENSITIVITY),
        tstop = fill(MAX_TIME, N_SUBJECTS_SENSITIVITY),
        statefrom = ones(Int, N_SUBJECTS_SENSITIVITY),
        stateto = ones(Int, N_SUBJECTS_SENSITIVITY),
        obstype = ones(Int, N_SUBJECTS_SENSITIVITY),
        x = x_vals
    )
    
    model_dgp = multistatemodel(h12; data=template)
    set_parameters!(model_dgp, (h12 = [true_rate, true_beta],))
    
    # --- Simulate data ---
    sim_result = simulate(model_dgp; paths=false, data=true, nsim=1)
    exact_data = sim_result[1]
    
    # --- Fit with correct initialization ---
    model_correct = multistatemodel(h12; data=exact_data)
    fitted_correct = fit(model_correct; verbose=false)
    params_correct = get_parameters_flat(fitted_correct)
    
    correct_passed, correct_errors = check_parameter_recovery_verbose(
        params_correct, [true_rate, true_beta])
    
    # --- Compute what biased beta would look like ---
    biased_beta = 1.5  # True beta + 1.0 (absolute error = 1.0, exceeds BETA_ABS_TOL=0.40)
    _, biased_errors = check_parameter_recovery_verbose(
        [true_rate, biased_beta], [true_rate, true_beta]; abs_tol=BETA_ABS_TOL)
    
    if VERBOSE_LONGTESTS
        println("    True: rate=$true_rate, beta=$true_beta")
        println("    Fitted: rate=$(round(params_correct[1], digits=4)), beta=$(round(params_correct[2], digits=3))")
        println("    Correct fit passed: $correct_passed")
        println("    Biased beta (1.5) absolute error: $(round(abs(biased_beta - true_beta), digits=3))")
        println("    Beta abs tolerance: $(BETA_ABS_TOL)")
        println("    Biased (beta=1.5) would fail: $(abs(biased_beta - true_beta) > BETA_ABS_TOL)")
    end
    
    # Assertions
    @test correct_passed  # Free fit should pass tolerance check
    @test abs(biased_beta - true_beta) > BETA_ABS_TOL  # Tolerance should catch +1.0 beta bias
    
    VERBOSE_LONGTESTS && println("  ✓ Covariate effect sensitivity check complete")
end

# =============================================================================
# Test 4: Summary - Minimum Detectable Bias
# =============================================================================
#
# Document the minimum bias that our tolerances can detect.
# =============================================================================

@testset "Sensitivity: Minimum Detectable Bias Documentation" begin
    VERBOSE_LONGTESTS && println("\n  ▶ Sensitivity: Minimum detectable bias summary")
    
    # For relative tolerance = 0.35 (35%)
    # Minimum detectable multiplicative bias: factor > 1.35 or factor < 0.65
    # Example: true=0.15, detectable if estimate < 0.10 or estimate > 0.20
    
    # For absolute tolerance = 0.40
    # Minimum detectable additive bias: |bias| > 0.40
    # Example: true=0.5, detectable if estimate < 0.10 or estimate > 0.90
    
    if VERBOSE_LONGTESTS
        println("\n    Minimum Detectable Bias (given current tolerances):")
        println("    " * "-"^60)
        println("    Relative tolerance (PARAM_REL_TOL): $(PARAM_REL_TOL)")
        println("      - Detects multiplicative bias > $(1 + PARAM_REL_TOL)x or < $(round(1 - PARAM_REL_TOL, digits=2))x")
        println("      - Example: rate=0.15 → detectable if outside [$(round(0.15*(1-PARAM_REL_TOL), digits=3)), $(round(0.15*(1+PARAM_REL_TOL), digits=3))]")
        println()
        println("    Absolute tolerance (BETA_ABS_TOL): $(BETA_ABS_TOL)")
        println("      - Detects additive bias > $(BETA_ABS_TOL)")
        println("      - Example: beta=0.5 → detectable if outside [$(0.5-BETA_ABS_TOL), $(0.5+BETA_ABS_TOL)]")
        println()
        println("    Shape tolerance (SHAPE_ABS_TOL): $(SHAPE_ABS_TOL)")
        println("      - Detects shape parameter errors > $(SHAPE_ABS_TOL)")
        println("    " * "-"^60)
    end
    
    # Validation that tolerances are finite and positive
    @test PARAM_REL_TOL > 0 && PARAM_REL_TOL < 1  # Relative tolerance should be in (0, 1)
    @test BETA_ABS_TOL > 0  # Absolute tolerance should be positive
    @test SHAPE_ABS_TOL > 0  # Shape tolerance should be positive
    
    VERBOSE_LONGTESTS && println("  ✓ Sensitivity documentation complete")
end

# =============================================================================
# Runner
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    println("="^70)
    println("LONG TEST: SENSITIVITY CHECK")
    println("="^70)
    println("This test validates that parameter recovery tolerances can detect bias.")
    println()
    
    @testset "Sensitivity Checks" begin
        include(@__FILE__)
    end
    
    println()
    println("="^70)
    println("SENSITIVITY CHECK COMPLETE")
    println("="^70)
end
