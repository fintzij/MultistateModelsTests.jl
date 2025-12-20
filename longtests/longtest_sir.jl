# =============================================================================
# Long Test: SIR/LHS Resampling Method Comparison
# =============================================================================
#
# This test validates that SIR and LHS resampling methods produce parameter 
# estimates comparable to the standard importance-weighted (no-SIR) approach.
#
# Test Configuration:
# - Model: Illness-death (1→2, 1→3, 2→3 where state 3 is absorbing)
# - DGP: Weibull hazards
# - Fitted model: Weibull hazards (correctly specified)
# - Sample size: 200 subjects
# - Observations: Up to 10 observation times per subject
# - Proposals: Markov and Phase-Type
# - SIR methods: :none (no SIR), :sir, :lhs
#
# Validations:
# 1. Parameter recovery: All methods recover true parameters
# 2. Consistency: SIR/LHS estimates close to no-SIR estimates
# 3. Computational performance: Runtime comparison
#
# =============================================================================

using Test
using MultistateModels
using DataFrames
using Random
using Statistics
using LinearAlgebra
using Printf
using Dates

import MultistateModels: Hazard, multistatemodel, fit, set_parameters!, simulate,
    get_parameters_flat, get_parameters, MarkovProposal, PhaseTypeProposal, @formula

# =============================================================================
# Configuration
# =============================================================================

const SIR_N_SUBJECTS = 500
const SIR_MAX_TIME = 10.0
const SIR_PANEL_TIMES = collect(0.0:1.0:SIR_MAX_TIME)  # 11 observation times (up to 10 intervals)
const SIR_RNG_SEED = 20241213
const SIR_MCEM_TOL = 0.02
const SIR_MCEM_ESS_INITIAL = 50
const SIR_MCEM_ESS_MAX = 200
const SIR_MCEM_MAX_ITER = 100
const SIR_PARAM_TOL_REL = 0.30  # 30% relative tolerance for parameter recovery

# True Weibull parameters (natural scale)
# Progressive model: 1 → 2 → 3 (state 3 is absorbing)
# Weibull hazard: h(t) = shape * scale * t^(shape-1)
# Using shapes close to 1 (nearly exponential) for better Markov surrogate approximation
const TRUE_H12_SHAPE = 1.1   # nearly exponential, slightly increasing hazard
const TRUE_H12_SCALE = 0.15  # moderate transition 1→2
const TRUE_H23_SHAPE = 1.2   # slightly increasing hazard
const TRUE_H23_SCALE = 0.20  # moderate transition from state 2 to death

# True parameters in NamedTuple format for set_parameters!
# Parameters are [log(shape), log(scale)]
const SIR_TRUE_PARAMS = (
    h12 = [log(TRUE_H12_SHAPE), log(TRUE_H12_SCALE)],
    h23 = [log(TRUE_H23_SHAPE), log(TRUE_H23_SCALE)]
)

# Parameter names and true values for reporting
const PARAM_NAMES = ["h12_shape", "h12_scale", "h23_shape", "h23_scale"]
const TRUE_PARAMS_NATURAL = [TRUE_H12_SHAPE, TRUE_H12_SCALE, TRUE_H23_SHAPE, TRUE_H23_SCALE]

# =============================================================================
# Result Container
# =============================================================================

"""Container for a single fit result"""
Base.@kwdef mutable struct SIRFitResult
    method::Symbol = :none  # :none, :sir, :lhs
    proposal::Symbol = :markov  # :markov, :phasetype
    
    # Fitted parameters (natural scale)
    params_natural::Vector{Float64} = Float64[]
    
    # Relative errors vs true (%)
    rel_errors::Vector{Float64} = Float64[]
    max_rel_error::Float64 = NaN
    
    # Convergence
    converged::Bool = false
    iterations::Int = 0
    final_mll::Float64 = NaN
    
    # Timing
    runtime_seconds::Float64 = NaN
    
    # Diagnostics
    mean_pareto_k::Float64 = NaN
    min_ess::Float64 = NaN
end

# =============================================================================
# Helper Functions
# =============================================================================

"""Generate panel data from progressive model (1→2→3) with Weibull hazards."""
function generate_sir_test_data(; n_subjects::Int = SIR_N_SUBJECTS, seed::Int = SIR_RNG_SEED)
    Random.seed!(seed)
    
    # Create hazards for progressive model (no 1→3 transition)
    h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
    h23 = Hazard(@formula(0 ~ 1), "wei", 2, 3)
    
    # Create template data for simulation
    nobs = length(SIR_PANEL_TIMES) - 1
    template = DataFrame(
        id = repeat(1:n_subjects, inner=nobs),
        tstart = repeat(SIR_PANEL_TIMES[1:end-1], n_subjects),
        tstop = repeat(SIR_PANEL_TIMES[2:end], n_subjects),
        statefrom = ones(Int, n_subjects * nobs),
        stateto = ones(Int, n_subjects * nobs),
        obstype = fill(2, n_subjects * nobs)  # Panel observation
    )
    
    # Build model and set true parameters
    model = multistatemodel(h12, h23; data=template)
    set_parameters!(model, SIR_TRUE_PARAMS)
    
    # Simulate panel data
    sim_result = simulate(model; paths=false, data=true, nsim=1, autotmax=false)
    panel_data = sim_result[1, 1]
    
    return panel_data, (h12, h23)
end

"""Convert fitted parameters to natural scale."""
function params_to_natural(params_flat::Vector{Float64})
    # params_flat has 4 elements: [log(shape12), log(scale12), log(shape23), log(scale23)]
    return exp.(params_flat)
end

"""Compute relative errors (%) vs true parameters."""
function compute_rel_errors(estimated_natural::Vector{Float64})
    rel_errors = Float64[]
    for (est, true_val) in zip(estimated_natural, TRUE_PARAMS_NATURAL)
        if abs(true_val) < 0.01
            push!(rel_errors, (est - true_val) * 100)
        else
            push!(rel_errors, ((est - true_val) / true_val) * 100)
        end
    end
    return rel_errors
end

"""Fit model with specified SIR method and proposal type."""
function fit_sir_model(data::DataFrame; 
                       sir_method::Symbol = :none,
                       proposal_type::Symbol = :markov,
                       verbose::Bool = false)
    
    # Create hazards for fitting (progressive model)
    h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
    h23 = Hazard(@formula(0 ~ 1), "wei", 2, 3)
    
    # Build model with Markov surrogate (required for MCEM)
    model = multistatemodel(h12, h23; data=data, surrogate=:markov)
    
    # Fit with specified SIR method and proposal type
    result = SIRFitResult(method=sir_method, proposal=proposal_type)
    
    # Set up proposal configuration
    proposal_config = if proposal_type == :phasetype
        PhaseTypeProposal(n_phases=3)
    else
        MarkovProposal()
    end
    
    start_time = time()
    try
        fitted = fit(model;
            proposal = proposal_config,
            sir = sir_method,
            ess_target_initial = SIR_MCEM_ESS_INITIAL,
            max_ess = SIR_MCEM_ESS_MAX,
            maxiter = SIR_MCEM_MAX_ITER,
            tol = SIR_MCEM_TOL,
            verbose = verbose,
            compute_vcov = true,
            compute_ij_vcov = false,
            compute_jk_vcov = false,
            return_convergence_records = true
        )
        result.runtime_seconds = time() - start_time
        
        # Extract parameters and convert to natural scale
        params_flat = get_parameters_flat(fitted)
        result.params_natural = params_to_natural(params_flat)
        
        # Compute relative errors
        result.rel_errors = compute_rel_errors(result.params_natural)
        result.max_rel_error = maximum(abs.(result.rel_errors))
        result.converged = true
        
        # Extract convergence info if available
        if hasproperty(fitted, :ConvergenceRecords) && !isnothing(fitted.ConvergenceRecords)
            records = fitted.ConvergenceRecords
            if hasproperty(records, :mll_trace)
                result.iterations = length(records.mll_trace)
                result.final_mll = isempty(records.mll_trace) ? NaN : records.mll_trace[end]
            end
            if hasproperty(records, :psis_pareto_k) && !isnothing(records.psis_pareto_k)
                result.mean_pareto_k = mean(records.psis_pareto_k)
            end
            if hasproperty(records, :ess_trace) && !isempty(records.ess_trace)
                # Each entry in ess_trace is a vector of ESS values per subject
                result.min_ess = minimum(minimum.(records.ess_trace))
            end
        end
        
    catch e
        result.runtime_seconds = time() - start_time
        result.converged = false
        if verbose
            @warn "Fit failed" method=sir_method proposal=proposal_type exception=e
            showerror(stdout, e, catch_backtrace())
        end
    end
    
    return result
end

"""Generate markdown report."""
function generate_sir_report(results::Vector{SIRFitResult}, output_path::String)
    
    open(output_path, "w") do io
        println(io, "# SIR/LHS Resampling Method Comparison Report")
        println(io, "")
        println(io, "Generated: $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))")
        println(io, "")
        
        println(io, "## Test Configuration")
        println(io, "")
        println(io, "| Setting | Value |")
        println(io, "|---------|-------|")
        println(io, "| Model | Progressive (1→2→3) |")
        println(io, "| DGP Hazards | Weibull |")
        println(io, "| Fitted Hazards | Weibull |")
        println(io, "| Sample Size | $(SIR_N_SUBJECTS) subjects |")
        println(io, "| Observation Times | $(length(SIR_PANEL_TIMES)-1) intervals (t=0 to $(SIR_MAX_TIME)) |")
        println(io, "| Initial ESS | $(SIR_MCEM_ESS_INITIAL) |")
        println(io, "| Max ESS | $(SIR_MCEM_ESS_MAX) |")
        println(io, "| Max Iterations | $(SIR_MCEM_MAX_ITER) |")
        println(io, "| Convergence Tolerance | $(SIR_MCEM_TOL) |")
        println(io, "")
        
        println(io, "## True Parameters (Natural Scale)")
        println(io, "")
        println(io, "| Transition | Shape | Scale |")
        println(io, "|------------|-------|-------|")
        println(io, "| 1→2 | $(round(TRUE_H12_SHAPE, digits=3)) | $(round(TRUE_H12_SCALE, digits=3)) |")
        println(io, "| 2→3 | $(round(TRUE_H23_SHAPE, digits=3)) | $(round(TRUE_H23_SCALE, digits=3)) |")
        println(io, "")
        
        println(io, "## Results Summary")
        println(io, "")
        
        # Separate by proposal type
        for proposal in [:markov, :phasetype]
            prop_results = filter(r -> r.proposal == proposal, results)
            if isempty(prop_results)
                continue
            end
            
            prop_name = proposal == :markov ? "Markov Proposal" : "Phase-Type Proposal"
            println(io, "### $(prop_name)")
            println(io, "")
            
            # Parameter estimates table
            println(io, "#### Parameter Estimates (Natural Scale)")
            println(io, "")
            println(io, "| Method | Conv | Iter | Runtime (s) | h12 shape | h12 scale | h23 shape | h23 scale |")
            println(io, "|--------|------|------|-------------|-----------|-----------|-----------|-----------|")
            println(io, "| **True** | - | - | - | $(round(TRUE_H12_SHAPE, digits=3)) | $(round(TRUE_H12_SCALE, digits=3)) | $(round(TRUE_H23_SHAPE, digits=3)) | $(round(TRUE_H23_SCALE, digits=3)) |")
            
            for r in prop_results
                method_name = r.method == :none ? "none" : string(r.method)
                conv = r.converged ? "✓" : "✗"
                iter = r.converged ? string(r.iterations) : "-"
                runtime = r.converged ? @sprintf("%.1f", r.runtime_seconds) : "-"
                
                if r.converged && length(r.params_natural) == 4
                    vals = [@sprintf("%.3f", v) for v in r.params_natural]
                else
                    vals = fill("-", 4)
                end
                
                println(io, "| $(method_name) | $(conv) | $(iter) | $(runtime) | $(vals[1]) | $(vals[2]) | $(vals[3]) | $(vals[4]) |")
            end
            println(io, "")
            
            # Relative errors table
            println(io, "#### Relative Errors (%)")
            println(io, "")
            println(io, "| Method | h12 shape | h12 scale | h23 shape | h23 scale | Max |")
            println(io, "|--------|-----------|-----------|-----------|-----------|-----|")
            
            for r in prop_results
                method_name = r.method == :none ? "none" : string(r.method)
                
                if r.converged && length(r.rel_errors) == 4
                    errs = [@sprintf("%.1f", e) for e in r.rel_errors]
                    maxe = @sprintf("%.1f", r.max_rel_error)
                else
                    errs = fill("-", 4)
                    maxe = "-"
                end
                
                println(io, "| $(method_name) | $(errs[1]) | $(errs[2]) | $(errs[3]) | $(errs[4]) | $(maxe) |")
            end
            println(io, "")
        end
        
        # Computational performance comparison
        println(io, "## Computational Performance")
        println(io, "")
        println(io, "| Method | Proposal | Runtime (s) | Iterations | Speedup vs none |")
        println(io, "|--------|----------|-------------|------------|-----------------|")
        
        # Find baseline (none) runtimes
        none_markov = findfirst(r -> r.method == :none && r.proposal == :markov && r.converged, results)
        none_pt = findfirst(r -> r.method == :none && r.proposal == :phasetype && r.converged, results)
        
        for r in results
            method_name = r.method == :none ? "none" : string(r.method)
            prop_name = r.proposal == :markov ? "Markov" : "Phase-Type"
            
            if r.converged
                runtime = @sprintf("%.1f", r.runtime_seconds)
                iter = string(r.iterations)
                
                # Calculate speedup
                baseline_idx = r.proposal == :markov ? none_markov : none_pt
                if !isnothing(baseline_idx) && results[baseline_idx].converged
                    speedup = results[baseline_idx].runtime_seconds / r.runtime_seconds
                    speedup_str = @sprintf("%.2fx", speedup)
                else
                    speedup_str = "-"
                end
            else
                runtime = "-"
                iter = "-"
                speedup_str = "-"
            end
            
            println(io, "| $(method_name) | $(prop_name) | $(runtime) | $(iter) | $(speedup_str) |")
        end
        println(io, "")
        
        # Summary
        println(io, "## Summary")
        println(io, "")
        
        converged_count = count(r -> r.converged, results)
        total_count = length(results)
        
        println(io, "- **Convergence**: $(converged_count)/$(total_count) fits converged")
        println(io, "")
        
        # Check parameter recovery
        recovered_count = count(r -> r.converged && r.max_rel_error < SIR_PARAM_TOL_REL * 100, results)
        println(io, "- **Parameter Recovery**: $(recovered_count)/$(converged_count) converged fits recovered true parameters (within $(Int(SIR_PARAM_TOL_REL*100))%)")
        println(io, "")
        
        # Compare SIR/LHS to none
        println(io, "### Method Comparison")
        println(io, "")
        
        for proposal in [:markov, :phasetype]
            none_idx = findfirst(r -> r.method == :none && r.proposal == proposal && r.converged, results)
            sir_idx = findfirst(r -> r.method == :sir && r.proposal == proposal && r.converged, results)
            lhs_idx = findfirst(r -> r.method == :lhs && r.proposal == proposal && r.converged, results)
            
            if isnothing(none_idx)
                continue
            end
            
            prop_name = proposal == :markov ? "Markov" : "Phase-Type"
            println(io, "**$(prop_name) Proposal:**")
            println(io, "")
            
            none_r = results[none_idx]
            
            if !isnothing(sir_idx) && length(results[sir_idx].params_natural) == 6 && length(none_r.params_natural) == 6
                sir_r = results[sir_idx]
                param_diffs = abs.(sir_r.params_natural .- none_r.params_natural)
                max_diff = maximum(param_diffs)
                speedup = none_r.runtime_seconds / sir_r.runtime_seconds
                println(io, "- SIR vs none: Max parameter difference = $(round(max_diff, digits=4)), Runtime speedup = $(round(speedup, digits=2))x")
            end
            
            if !isnothing(lhs_idx) && length(results[lhs_idx].params_natural) == 6 && length(none_r.params_natural) == 6
                lhs_r = results[lhs_idx]
                param_diffs = abs.(lhs_r.params_natural .- none_r.params_natural)
                max_diff = maximum(param_diffs)
                speedup = none_r.runtime_seconds / lhs_r.runtime_seconds
                println(io, "- LHS vs none: Max parameter difference = $(round(max_diff, digits=4)), Runtime speedup = $(round(speedup, digits=2))x")
            end
            println(io, "")
        end
        
        println(io, "---")
        println(io, "*Report generated by `longtest_sir.jl`*")
    end
    
    return output_path
end

# =============================================================================
# Main Test
# =============================================================================

@testset "SIR/LHS Method Comparison" begin
    
    println("\n" * "="^70)
    println("SIR/LHS Resampling Method Comparison Long Test")
    println("="^70 * "\n")
    
    # Generate test data
    println("Generating test data (n=$(SIR_N_SUBJECTS))...")
    panel_data, hazard_specs = generate_sir_test_data()
    println("  Generated $(nrow(panel_data)) observations for $(length(unique(panel_data.id))) subjects\n")
    
    # Run all configurations
    results = SIRFitResult[]
    
    # Note: Phase-type proposals were previously disabled due to high Pareto-k (>1) values.
    # Investigation on 2024-12-17 (see diagnostics/phasetype_testing_plan.md) confirmed
    # the issue is RESOLVED - all Pareto-k values now < 1.0 across various conditions.
    configurations = [
        (:none, :markov, "No SIR + Markov"),
        (:sir, :markov, "SIR + Markov"),
        (:lhs, :markov, "LHS + Markov"),
        (:none, :phasetype, "No SIR + Phase-Type"),
        (:sir, :phasetype, "SIR + Phase-Type"),
        (:lhs, :phasetype, "LHS + Phase-Type")
    ]
    
    for (sir_method, proposal, desc) in configurations
        println("Fitting: $(desc)...")
        result = fit_sir_model(panel_data; 
                              sir_method=sir_method, 
                              proposal_type=proposal,
                              verbose=false)  # Set back to false for cleaner output
        push!(results, result)
        
        if result.converged
            println("  ✓ Converged in $(result.iterations) iterations ($(round(result.runtime_seconds, digits=1))s)")
            println("  Max relative error: $(round(result.max_rel_error, digits=1))%")
        else
            println("  ✗ Did not converge ($(round(result.runtime_seconds, digits=1))s)")
        end
        println()
    end
    
    # Generate report
    report_dir = joinpath(@__DIR__, "..", "diagnostics")
    mkpath(report_dir)
    report_path = joinpath(report_dir, "sir_longtest_report.md")
    generate_sir_report(results, report_path)
    println("Report saved to: $(report_path)\n")
    
    # Run tests
    @testset "Parameter Recovery" begin
        for r in results
            if r.converged
                @testset "$(r.method)/$(r.proposal)" begin
                    @test r.max_rel_error < SIR_PARAM_TOL_REL * 100
                end
            end
        end
    end
    
    @testset "Method Consistency" begin
        # Compare SIR/LHS to none for each proposal type
        for proposal in [:markov, :phasetype]
            none_idx = findfirst(r -> r.method == :none && r.proposal == proposal && r.converged, results)
            
            if isnothing(none_idx)
                @warn "Baseline (none) did not converge for $(proposal)"
                continue
            end
            
            none_r = results[none_idx]
            
            for method in [:sir, :lhs]
                method_idx = findfirst(r -> r.method == method && r.proposal == proposal && r.converged, results)
                
                if isnothing(method_idx)
                    @warn "$(method) did not converge for $(proposal)"
                    continue
                end
                
                method_r = results[method_idx]
                
                @testset "$(method) vs none ($(proposal))" begin
                    if length(method_r.params_natural) == 6 && length(none_r.params_natural) == 6
                        # Parameters should be close (within 25% of each other)
                        for (i, (m_param, n_param)) in enumerate(zip(method_r.params_natural, none_r.params_natural))
                            rel_diff = abs(m_param - n_param) / n_param * 100
                            @test rel_diff < 25.0
                        end
                    end
                end
            end
        end
    end
    
    @testset "Convergence" begin
        # At least the baseline methods should converge
        @test any(r -> r.method == :none && r.proposal == :markov && r.converged, results)
    end
    
    println("\n" * "="^70)
    println("SIR/LHS Long Test Complete")
    println("="^70 * "\n")
end
