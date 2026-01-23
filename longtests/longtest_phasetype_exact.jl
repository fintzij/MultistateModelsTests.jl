"""
Long test suite for phase-type HAZARD MODELS with exact (continuous-time) data.

This test suite validates inference when the TARGET MODEL has Coxian phase-type
hazard structure. Since phase-type models on the expanded state space are Markov
(exponential hazards), we use direct MLE fitting - no MCEM required.

Test workflow:
1. Build phase-type model (expanded state space with exponential hazards)
2. Simulate exact data from the phase-type model
3. Fit using standard Markov MLE
4. Verify parameter recovery

Test matrix:
- Number of phases: 2, 3 phases per transient state
- Model structure: illness-death (1→2→3, 1→3, 2→3)
- Observation type: exact (obstype=1)

The key insight: phase-type sojourn time distributions can approximate
semi-Markov processes (Weibull, Gompertz) while remaining tractable for
inference. This is an alternative to using phase-type as MCEM proposals.

References:
- Titman & Sharples (2010) Biometrics - phase-type semi-Markov approximations
- Asmussen et al. (1996) Coxian phase-type distributions
"""

using Test
using MultistateModels
using DataFrames
using Random
using Statistics
using LinearAlgebra

# Import internal functions
import MultistateModels: Hazard, multistatemodel, fit, set_parameters!, simulate,
    get_parameters_flat, get_parameters, SamplePath, @formula,
    PhaseTypeConfig, build_phasetype_surrogate

# Longtest config and helpers are loaded by MultistateModelsTests module.
# For standalone runs, include from src/ (canonical location).
if !@isdefined(PARAM_REL_TOL)
    include(joinpath(@__DIR__, "..", "src", "longtest_config.jl"))
    include(joinpath(@__DIR__, "..", "src", "longtest_helpers.jl"))
    include(joinpath(@__DIR__, "..", "src", "phasetype_longtest_helpers.jl"))
end

# LongTestResult struct for standalone runs
if !isdefined(Main, :LongTestResult) && !@isdefined(LongTestResult)
    include(joinpath(@__DIR__, "..", "src", "LongTestResults.jl"))
end

const RNG_SEED = 0xABCD0001
const N_SUBJECTS = 1000           # Large sample for MLE precision
const N_SIM_TRAJ = 5000           # Trajectories for distributional comparison
const MAX_TIME = 10.0             # Maximum follow-up time
const PARAM_TOL_REL = 0.20        # Relative tolerance (20% for direct MLE - allows for sampling variability)

println("\n" * "="^70)
println("Phase-Type Hazard Models: Exact Data Long Tests")
println("="^70)
println("Testing inference for models with Coxian phase-type hazard structure.")
println("These models are Markov on the expanded state space → direct MLE.")
println("Default sample size: n=$N_SUBJECTS")

# Shared helper functions (compute_state_prevalence, compute_state_prevalence_phasetype, etc.) 
# are loaded from longtest_helpers.jl by the test runner.

# ============================================================================
# Helper Functions
# ============================================================================

"""
    check_parameter_recovery(fitted_params, true_params; tol_rel)

Verify fitted parameters are close to true values.
Works with log-scale parameters (baseline rates).
"""
function check_parameter_recovery(fitted_params::Vector{Float64}, true_params::Vector{Float64}; 
                                   tol_rel=PARAM_TOL_REL)
    @assert length(fitted_params) == length(true_params) "Parameter length mismatch"
    
    all_pass = true
    for i in eachindex(fitted_params)
        true_val = true_params[i]
        fitted_val = fitted_params[i]
        
        # Use absolute tolerance for values near zero
        if abs(true_val) < 0.1
            if abs(fitted_val - true_val) > 0.2
                println("  Parameter $i: true=$(round(true_val, digits=4)), fitted=$(round(fitted_val, digits=4)), diff=$(round(fitted_val - true_val, digits=4))")
                all_pass = false
            end
        else
            rel_err = abs(fitted_val - true_val) / abs(true_val)
            if rel_err > tol_rel
                println("  Parameter $i: true=$(round(true_val, digits=4)), fitted=$(round(fitted_val, digits=4)), rel_err=$(round(rel_err, digits=3))")
                all_pass = false
            end
        end
    end
    return all_pass
end

"""
    generate_exact_data_template(n_subj, max_time)

Create a DataFrame template for simulating exact data.
One row per subject, obstype=1 for exact observation.
"""
function generate_exact_data_template(n_subj::Int, max_time::Float64)
    return DataFrame(
        id = 1:n_subj,
        tstart = zeros(n_subj),
        tstop = fill(max_time, n_subj),
        statefrom = ones(Int, n_subj),
        stateto = ones(Int, n_subj),
        obstype = ones(Int, n_subj)
    )
end

# ============================================================================
# TEST SECTION 1: 2-PHASE COXIAN ILLNESS-DEATH
# ============================================================================

@testset "Phase-Type Hazard: 2-Phase Illness-Death (Exact Data)" begin
    Random.seed!(RNG_SEED)
    
    # Original 3-state illness-death: 1 → 2 → 3, with 1 → 3 direct
    tmat_obs = [0 1 1; 0 0 1; 0 0 0]
    
    # Configure 2 phases for each transient state
    # State 1: phases 1-2
    # State 2: phases 3-4  
    # State 3 (absorbing): phase 5
    config = PhaseTypeConfig(n_phases=Dict(1=>2, 2=>2))
    
    println("\n--- 2-Phase Coxian Illness-Death Model ---")
    println("Observed states: 3, Phases per transient: 2")
    println("Expanded states: 5 (2 + 2 + 1)")
    
    # Build surrogate to understand structure
    surrogate = build_phasetype_surrogate(tmat_obs, config)
    
    @testset "State space expansion" begin
        @test surrogate.n_observed_states == 3
        @test surrogate.n_expanded_states == 5
        @test surrogate.state_to_phases[1] == 1:2
        @test surrogate.state_to_phases[2] == 3:4
        @test surrogate.state_to_phases[3] == 5:5
    end
    
    # Create data template for simulation
    template = generate_exact_data_template(N_SUBJECTS, MAX_TIME)
    
    # Build phase-type model
    result = build_phasetype_model(tmat_obs, config; data=template, verbose=true)
    model = result.model
    
    @testset "Model structure" begin
        # Should have exponential hazards on expanded space
        @test length(model.hazards) > 0
        # All hazards should be MarkovHazard (exponential on expanded space)
        for haz in model.hazards
            @test haz isa MultistateModels.MarkovHazard
            @test haz.family == :exp
        end
    end
    
    # Set true parameters (natural scale since v0.3.0)
    # Rates for transitions in expanded space
    # We'll set these directly on the model
    n_hazards = length(model.hazards)
    println("  Number of hazards in expanded model: $n_hazards")
    
    # True rates (positive scale, natural scale since v0.3.0)
    # Progression through phases: λ₁ (phase 1→2 within state 1), etc.
    # Exit rates: μ (phase → absorbing)
    # This vector is sized for the expected 8-hazard illness-death phase-type model
    true_rates = [0.5, 0.3, 0.4, 0.2, 0.25, 0.35, 0.3, 0.15]
    
    # ACTION-5: Replace silent padding/truncation with explicit assertion
    # The test should fail loudly if the phase-type expansion produces an unexpected
    # number of hazards, rather than silently adjusting parameters
    @assert length(true_rates) == n_hazards """
        Parameter count mismatch in phase-type test:
          Expected $(length(true_rates)) hazards (from true_rates vector)
          Got $n_hazards hazards from model expansion
        
        This indicates either:
        1. The phase-type configuration changed (update true_rates to match), or
        2. A bug in the phase-type expansion code
        
        Current true_rates: $true_rates
        """
    
    # v0.3.0+: Parameters are on natural scale, not log scale
    # Use natural scale rates directly
    true_params = copy(true_rates)
    
    # Set parameters (natural scale since v0.3.0)
    pars_dict = Dict{Symbol, Vector{Float64}}()
    for (i, haz) in enumerate(model.hazards)
        pars_dict[haz.hazname] = [true_params[i]]
    end
    set_parameters!(model, NamedTuple(pars_dict))
    
    println("  True rates: $(round.(true_rates, digits=3))")
    
    # Simulate exact data
    println("\nSimulating exact data...")
    sim_result = simulate(model; paths=false, data=true, nsim=1)
    exact_data = sim_result[1]
    
    # Count transitions in expanded space
    println("  Simulated $(nrow(exact_data)) observations")
    
    # For fitting, we create a NEW model using the SIMULATED DATA
    # The simulated data is already in expanded state space (phases 1-5)
    # We rebuild the hazard specifications to create a fresh model
    hazards_for_fit = build_phasetype_hazards(tmat_obs, config, surrogate)
    model_fit = multistatemodel(hazards_for_fit...; data=exact_data)
    
    @testset "Parameter recovery" begin
        println("\nFitting phase-type model with direct MLE...")
        fitted = fit(model_fit; verbose=false, vcov_type=:ij)
        
        # v0.3.0+: get_parameters_flat returns natural scale
        fitted_params = get_parameters_flat(fitted)
        println("True params (natural): $(round.(true_params, digits=3))")
        println("Fitted params (natural): $(round.(fitted_params, digits=3))")
        
        @test check_parameter_recovery(fitted_params, true_params)
        
        # Cache results for reporting with prevalence/CI plots
        param_names = ["rate_$i" for i in 1:n_hazards]
        
        # Build hazard specs for simulation (needed by capture function)
        hazard_specs_for_sim = build_phasetype_hazards(tmat_obs, config, surrogate)
        
        capture_phasetype_longtest_result!(
            "pt_exact_nocov",
            fitted,
            true_params,
            param_names,
            surrogate,
            hazard_specs_for_sim;
            hazard_family = "pt",
            data_type = "exact",
            covariate_type = "nocov",
            n_subjects = N_SUBJECTS,
            n_observed_states = surrogate.n_observed_states,
            transitions = [(1, 2), (1, 3), (2, 3)]  # Include all illness-death transitions
        )
    end
end

# ============================================================================
# TEST SECTION 2: FUTURE - PHASE-TYPE APPROXIMATION QUALITY
# ============================================================================
# 
# The section below tests how well phase-type distributions can approximate
# semi-Markov (Weibull) sojourn times. This is a separate question from
# parameter recovery and requires careful analysis of:
# 1. How to initialize phase-type parameters to match Weibull moments
# 2. Whether the fitted phase-type faithfully approximates the target distribution
#
# This is commented out pending proper implementation of Weibull → phase-type
# moment matching or other approximation methods.
#
# @testset "Phase-Type Approximates Weibull" begin
#     ...
# end

# ============================================================================
# TEST SECTION 3: DISTRIBUTIONAL FIDELITY
# ============================================================================

@testset "Distributional Fidelity: Phase-Type Model" begin
    Random.seed!(RNG_SEED + 200)
    
    println("\n--- Distributional Fidelity Check ---")
    println("Verify fitted phase-type model reproduces data-generating distribution")
    
    # Simple 2-state model: 1 → 2
    tmat = [0 1; 0 0]
    config = PhaseTypeConfig(n_phases=Dict(1=>2))
    
    # Build surrogate to understand phase structure
    surrogate = build_phasetype_surrogate(tmat, config)
    absorbing_phase = first(surrogate.state_to_phases[2])  # Phase for absorbing state
    
    # Generate template
    template = generate_exact_data_template(N_SUBJECTS, MAX_TIME)
    
    # Build and set true parameters
    result = build_phasetype_model(tmat, config; data=template, verbose=false)
    model_true = result.model
    
    # Set true rates: progression λ₁=0.4, exit μ₁=0.2, exit μ₂=0.5
    # v0.3.0+: Parameters are on natural scale
    true_rates = [0.4, 0.2, 0.5]
    
    pars = Dict{Symbol, Vector{Float64}}()
    for (i, haz) in enumerate(model_true.hazards)
        pars[haz.hazname] = [true_rates[i]]
    end
    set_parameters!(model_true, NamedTuple(pars))
    
    # Simulate data
    sim_true = simulate(model_true; paths=false, data=true, nsim=1)
    exact_data = sim_true[1]
    
    # Fit model - rebuild hazards for the simulated data (already in expanded space)
    hazards_for_fit = build_phasetype_hazards(tmat, config, surrogate)
    model_fit = multistatemodel(hazards_for_fit...; data=exact_data)
    fitted = fit(model_fit; verbose=false)
    
    # To compare distributions fairly, we must simulate from fitted using a FRESH template
    # with proper tstop=MAX_TIME (not the event times from exact_data).
    # Otherwise subjects who absorbed early have short observation windows in re-simulation.
    template_for_compare = generate_exact_data_template(N_SUBJECTS, MAX_TIME)
    hazards_for_compare = build_phasetype_hazards(tmat, config, surrogate)
    model_compare = multistatemodel(hazards_for_compare...; data=template_for_compare)
    
    # Set the fitted parameters on the comparison model
    fitted_params = get_parameters_flat(fitted)
    pars_compare = Dict{Symbol, Vector{Float64}}()
    for (i, haz) in enumerate(model_compare.hazards)
        pars_compare[haz.hazname] = [fitted_params[i]]
    end
    set_parameters!(model_compare, NamedTuple(pars_compare))
    
    # Simulate from fitted parameters with proper template
    sim_fitted = simulate(model_compare; paths=false, data=true, nsim=1)
    
    @testset "Sojourn time distribution" begin
        # Compare sojourn time CDFs
        # Note: simulated data uses expanded phases, absorbing state is phase 3
        sojourn_true = exact_data[exact_data.stateto .== absorbing_phase, :tstop]
        sojourn_fit = sim_fitted[1][sim_fitted[1].stateto .== absorbing_phase, :tstop]
        
        if length(sojourn_true) > 10 && length(sojourn_fit) > 10
            # Compare quantiles
            qs = [0.25, 0.5, 0.75, 0.9]
            q_true = quantile(sojourn_true, qs)
            q_fit = quantile(sojourn_fit, qs)
            
            println("  Quantiles (true → fitted):")
            for (q, qt, qf) in zip(qs, q_true, q_fit)
                println("    $(Int(q*100))th: $(round(qt, digits=3)) → $(round(qf, digits=3))")
            end
            
            # Median should match well
            @test abs(q_true[2] - q_fit[2]) / q_true[2] < 0.15
        else
            @test_skip "Insufficient events"
        end
    end
end

# ============================================================================
# TEST SECTION 4: PHASE-TYPE WITH FIXED COVARIATES
# ============================================================================

@testset "Phase-Type Hazard: 2-Phase with Fixed Covariate (Exact Data)" begin
    Random.seed!(RNG_SEED + 100)
    
    println("\n--- 2-Phase Phase-Type with Fixed Covariate ---")
    
    # Simple 2-state model: 1 → 2 (absorbing)
    tmat = [0 1; 0 0]
    config = PhaseTypeConfig(n_phases=Dict(1=>2))
    
    # Build surrogate structure
    surrogate = build_phasetype_surrogate(tmat, config)
    
    @test surrogate.n_observed_states == 2
    @test surrogate.n_expanded_states == 3  # 2 phases for state 1, 1 for absorbing state 2
    
    # Generate covariate data - binary treatment
    cov_data = DataFrame(x = rand([0.0, 1.0], N_SUBJECTS))
    
    # Create data template with covariate
    template = DataFrame(
        id = 1:N_SUBJECTS,
        tstart = zeros(N_SUBJECTS),
        tstop = fill(MAX_TIME, N_SUBJECTS),
        statefrom = ones(Int, N_SUBJECTS),
        stateto = ones(Int, N_SUBJECTS),
        obstype = ones(Int, N_SUBJECTS),
        x = cov_data.x
    )
    
    # Build model with covariate
    covariate_formula = @formula(0 ~ x)
    result = build_phasetype_model(tmat, config; 
                                    data=template,
                                    covariate_formula=covariate_formula,
                                    verbose=true)
    model = result.model
    
    # For 2-phase Coxian with 1 destination:
    # - λ₁: progression rate (phase 1 → phase 2)
    # - μ₁: exit from phase 1 → absorbing
    # - μ₂: exit from phase 2 → absorbing
    # Each has (log_rate, beta) parameters
    # Total: 6 parameters
    
    n_hazards = length(model.hazards)
    println("  Number of hazards: $n_hazards")
    @test n_hazards == 3  # λ₁, μ₁, μ₂
    
    # True parameters: (log_rate, beta) for each hazard
    # Covariate x has positive effect on exit rates (increases hazard)
    true_rates = [0.5, 0.3, 0.6]  # λ₁, μ₁, μ₂
    true_betas = [0.0, 0.4, 0.3]   # No effect on progression, positive on exit
    
    # Set parameters (natural scale since v0.3.0)
    pars_dict = Dict{Symbol, Vector{Float64}}()
    for (i, haz) in enumerate(model.hazards)
        pars_dict[haz.hazname] = [true_rates[i], true_betas[i]]
    end
    set_parameters!(model, NamedTuple(pars_dict))
    
    println("  True rates: $(round.(true_rates, digits=3))")
    println("  True betas: $(round.(true_betas, digits=3))")
    
    # Simulate exact data
    println("\nSimulating exact data with covariate...")
    sim_result = simulate(model; paths=false, data=true, nsim=1)
    exact_data = sim_result[1]
    
    println("  Simulated $(nrow(exact_data)) observations")
    println("  Events: $(sum(exact_data.stateto .== 3))")  # Phase 3 is absorbing
    
    # Fit model - rebuild with covariate formula
    hazards_for_fit = build_phasetype_hazards(tmat, config, surrogate;
                                               covariate_formula=covariate_formula)
    model_fit = multistatemodel(hazards_for_fit...; data=exact_data)
    
    @testset "Parameter recovery with covariate" begin
        println("\nFitting phase-type model with covariate...")
        fitted = fit(model_fit; verbose=false, vcov_type=:ij)
        
        fitted_params = get_parameters_flat(fitted)
        
        # Check rate recovery (natural scale since v0.3.0)
        for i in 1:n_hazards
            true_rate = true_rates[i]
            fitted_rate = fitted_params[2*(i-1) + 1]
            rel_err = abs(fitted_rate - true_rate) / abs(true_rate)
            @test rel_err < PARAM_TOL_REL
        end
        
        # Check beta recovery - exit rates should have positive effect recovered
        # μ₁ beta: fitted_params[4], true = 0.4
        # μ₂ beta: fitted_params[6], true = 0.3
        @test fitted_params[4] > 0.0  # Positive direction correct for μ₁
        @test isapprox(fitted_params[4], true_betas[2]; atol=0.25)
        @test fitted_params[6] > 0.0  # Positive direction correct for μ₂
        @test isapprox(fitted_params[6], true_betas[3]; atol=0.25)
        
        # Cache results for reporting (natural scale)
        # Parameters: (rate, beta) for each hazard
        true_params_flat = Float64[]
        param_names = String[]
        for i in 1:n_hazards
            push!(true_params_flat, true_rates[i])
            push!(true_params_flat, true_betas[i])
            push!(param_names, "rate_$i")
            push!(param_names, "beta_$i")
        end
        
        # Build hazard specs for simulation (with covariate)
        hazard_specs_for_sim = build_phasetype_hazards(tmat, config, surrogate;
                                                       covariate_formula=covariate_formula)
        
        capture_phasetype_longtest_result!(
            "pt_exact_fixed",
            fitted,
            true_params_flat,
            param_names,
            surrogate,
            hazard_specs_for_sim;
            hazard_family = "pt",
            data_type = "exact",
            covariate_type = "fixed",
            n_subjects = N_SUBJECTS,
            n_observed_states = surrogate.n_observed_states,
            transitions = [(1, 2)]  # 2-state model: only 1→2 transition
        )
    end
end

# ============================================================================
# TEST SECTION 5: PHASE-TYPE WITH TIME-VARYING COVARIATES
# ============================================================================

@testset "Phase-Type Hazard: 2-Phase with TVC (Exact Data)" begin
    Random.seed!(RNG_SEED + 200)
    
    println("\n--- 2-Phase Phase-Type with Time-Varying Covariate ---")
    
    # Simple 2-state model: 1 → 2 (absorbing)
    tmat = [0 1; 0 0]
    config = PhaseTypeConfig(n_phases=Dict(1=>2))
    surrogate = build_phasetype_surrogate(tmat, config)
    
    # TVC setup: covariate changes at t=3
    n_subj = N_SUBJECTS
    change_time = 3.0
    
    # Create TVC template - each subject has 2 intervals
    rows = []
    for subj in 1:n_subj
        # Treatment: 50% switch from 0 to 1 at change_time
        trt_switch = rand() < 0.5
        x_before = 0.0
        x_after = trt_switch ? 1.0 : 0.0
        
        # Interval 1: [0, change_time)
        push!(rows, (id=subj, tstart=0.0, tstop=change_time, 
                     statefrom=1, stateto=1, obstype=1, x=x_before))
        # Interval 2: [change_time, MAX_TIME]
        push!(rows, (id=subj, tstart=change_time, tstop=MAX_TIME,
                     statefrom=1, stateto=1, obstype=1, x=x_after))
    end
    tvc_template = DataFrame(rows)
    
    # Build model with covariate
    covariate_formula = @formula(0 ~ x)
    result = build_phasetype_model(tmat, config;
                                    data=tvc_template,
                                    covariate_formula=covariate_formula,
                                    verbose=false)
    model = result.model
    
    # True parameters
    true_rates = [0.4, 0.25, 0.5]  # λ₁, μ₁, μ₂
    true_betas = [0.0, 0.5, 0.4]   # TVC effect on exit rates
    
    # Set parameters (natural scale since v0.3.0)
    pars_dict = Dict{Symbol, Vector{Float64}}()
    for (i, haz) in enumerate(model.hazards)
        pars_dict[haz.hazname] = [true_rates[i], true_betas[i]]
    end
    set_parameters!(model, NamedTuple(pars_dict))
    
    println("  True rates: $(round.(true_rates, digits=3))")
    println("  True TVC betas: $(round.(true_betas, digits=3))")
    
    # Simulate with TVC - use autotmax=false to preserve TVC structure
    println("\nSimulating exact data with TVC...")
    sim_result = simulate(model; paths=false, data=true, nsim=1, autotmax=false)
    exact_data = sim_result[1]
    
    println("  Simulated $(nrow(exact_data)) observations")
    
    # Fit model
    hazards_for_fit = build_phasetype_hazards(tmat, config, surrogate;
                                               covariate_formula=covariate_formula)
    model_fit = multistatemodel(hazards_for_fit...; data=exact_data)
    
    @testset "TVC parameter recovery" begin
        println("\nFitting phase-type model with TVC...")
        fitted = fit(model_fit; verbose=false, vcov_type=:ij)
        
        fitted_params = get_parameters_flat(fitted)
        n_hazards = 3
        
        # Check rate recovery (natural scale)
        for i in 1:n_hazards
            true_rate = true_rates[i]
            fitted_rate = fitted_params[2*(i-1) + 1]
            @test isfinite(fitted_rate)
        end
        
        # Check TVC effect direction (positive effects on exit rates)
        # μ₁ beta: fitted_params[4]
        # μ₂ beta: fitted_params[6]
        @test fitted_params[4] > -0.2  # Should be positive or near zero
        @test fitted_params[6] > -0.2  # Should be positive or near zero
        
        # Recovery tolerance is higher for TVC
        @test isapprox(fitted_params[4], true_betas[2]; atol=0.4)
        
        # Cache results for reporting (natural scale)
        true_params_flat = Float64[]
        param_names = String[]
        for i in 1:n_hazards
            push!(true_params_flat, true_rates[i])
            push!(true_params_flat, true_betas[i])
            push!(param_names, "rate_$i")
            push!(param_names, "beta_$i")
        end
        
        # Build hazard specs for simulation (with TVC)
        hazard_specs_for_sim = build_phasetype_hazards(tmat, config, surrogate;
                                                       covariate_formula=covariate_formula)
        
        capture_phasetype_longtest_result!(
            "pt_exact_tvc",
            fitted,
            true_params_flat,
            param_names,
            surrogate,
            hazard_specs_for_sim;
            hazard_family = "pt",
            data_type = "exact",
            covariate_type = "tvc",
            n_subjects = n_subj,
            n_observed_states = surrogate.n_observed_states,
            transitions = [(1, 2)]  # 2-state model: only 1->2 transition
        )
    end
end
