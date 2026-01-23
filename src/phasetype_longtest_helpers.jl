"""
Phase-type longtest infrastructure.

This file contains helper functions for building phase-type models directly
from transition matrices and PhaseTypeConfig. This is an alternative API
used only for longtests - production code uses Hazard(:pt, ...) + multistatemodel().

These functions were moved from src/phasetype/expansion.jl during package streamlining.

This file is included by the MultistateModelsTests module after longtest_helpers.jl.
"""

using StatsModels: FormulaTerm

import MultistateModels: Hazard, multistatemodel, HazardFunction, 
    PhaseTypeSurrogate, PhaseTypeConfig, build_phasetype_surrogate, @formula

# =============================================================================
# Phase-Type Model Building for Inference (Longtest Infrastructure)
# =============================================================================

"""
    build_phasetype_hazards(tmat::Matrix{Int64}, config::PhaseTypeConfig, 
                            surrogate::PhaseTypeSurrogate;
                            covariate_formula::Union{Nothing, FormulaTerm} = nothing)

Generate Hazard specifications for the expanded phase-type model.

For each transition in the expanded state space, creates an exponential hazard.
The expanded model has two types of transitions:
1. **Progression transitions** (λᵢ): Rate of moving from phase i to phase i+1 within a state
2. **Absorption/exit transitions** (μᵢ): Rate of exiting from phase i to another observed state

# Arguments
- `tmat::Matrix{Int64}`: Original transition matrix
- `config::PhaseTypeConfig`: Configuration with number of phases
- `surrogate::PhaseTypeSurrogate`: Pre-built surrogate with state mappings
- `covariate_formula`: Optional formula for covariates (applied to ALL rates)

# Returns
- `Vector{HazardFunction}`: Vector of Hazard specifications for the expanded model
"""
function build_phasetype_hazards(tmat::Matrix{Int64}, config::PhaseTypeConfig,
                                 surrogate::PhaseTypeSurrogate;
                                 covariate_formula::Union{Nothing, FormulaTerm} = nothing)
    
    n_observed = size(tmat, 1)
    n_expanded = surrogate.n_expanded_states
    
    hazards = HazardFunction[]
    
    # Identify transient states (those with outgoing transitions)
    is_absorbing = [all(tmat[s, :] .== 0) for s in 1:n_observed]
    
    for obs_state in 1:n_observed
        if is_absorbing[obs_state]
            continue  # No transitions from absorbing states
        end
        
        phases = surrogate.state_to_phases[obs_state]
        n_phases = length(phases)
        
        # Find destination states from this observed state
        dest_states = findall(tmat[obs_state, :] .> 0)
        n_dests = length(dest_states)
        
        # 1. Progression transitions (λ): phase i → phase i+1 within this state
        for local_i in 1:(n_phases - 1)
            phase_from = phases[local_i]
            phase_to = phases[local_i + 1]
            
            if covariate_formula === nothing
                h = Hazard(@formula(0 ~ 1), "exp", phase_from, phase_to)
            else
                h = Hazard(covariate_formula, "exp", phase_from, phase_to)
            end
            push!(hazards, h)
        end
        
        # 2. Absorption/exit transitions (μ): from each phase to each destination state
        for local_i in 1:n_phases
            phase_from = phases[local_i]
            
            for dest_state in dest_states
                # Transition goes to first phase of destination state
                dest_phase = first(surrogate.state_to_phases[dest_state])
                
                if covariate_formula === nothing
                    h = Hazard(@formula(0 ~ 1), "exp", phase_from, dest_phase)
                else
                    h = Hazard(covariate_formula, "exp", phase_from, dest_phase)
                end
                push!(hazards, h)
            end
        end
    end
    
    return hazards
end

"""
    build_expanded_tmat(tmat::Matrix{Int64}, surrogate::PhaseTypeSurrogate)

Build the expanded transition matrix for the phase-type model.
"""
function build_expanded_tmat(tmat::Matrix{Int64}, surrogate::PhaseTypeSurrogate)
    n_observed = size(tmat, 1)
    n_expanded = surrogate.n_expanded_states
    
    tmat_exp = zeros(Int64, n_expanded, n_expanded)
    trans_num = 1
    
    # Identify transient states
    is_absorbing = [all(tmat[s, :] .== 0) for s in 1:n_observed]
    
    for obs_state in 1:n_observed
        if is_absorbing[obs_state]
            continue
        end
        
        phases = surrogate.state_to_phases[obs_state]
        n_phases = length(phases)
        dest_states = findall(tmat[obs_state, :] .> 0)
        
        # Progression transitions
        for local_i in 1:(n_phases - 1)
            tmat_exp[phases[local_i], phases[local_i + 1]] = trans_num
            trans_num += 1
        end
        
        # Exit transitions  
        for local_i in 1:n_phases
            for dest_state in dest_states
                dest_phase = first(surrogate.state_to_phases[dest_state])
                tmat_exp[phases[local_i], dest_phase] = trans_num
                trans_num += 1
            end
        end
    end
    
    return tmat_exp
end

"""
    build_phasetype_emat(data::DataFrame, surrogate::PhaseTypeSurrogate,
                         CensoringPatterns::Matrix{Float64})

Build the emission matrix for the phase-type model.
"""
function build_phasetype_emat(data::DataFrame, surrogate::PhaseTypeSurrogate,
                              CensoringPatterns::Matrix{Float64})
    
    n_obs = nrow(data)
    n_expanded = surrogate.n_expanded_states
    n_observed = surrogate.n_observed_states
    
    emat = zeros(Float64, n_obs, n_expanded)
    
    for i in 1:n_obs
        obstype = data.obstype[i]
        
        if obstype == 1
            # Exact transition observation - transition always goes to FIRST phase of destination
            obs_state = data.stateto[i]
            first_phase = first(surrogate.state_to_phases[obs_state])
            emat[i, first_phase] = 1.0
        elseif obstype == 2
            # Panel observation - any phase of observed state is possible
            obs_state = data.stateto[i]
            for p in surrogate.state_to_phases[obs_state]
                emat[i, p] = 1.0
            end
        elseif obstype == 0
            # Fully censored - all phases possible
            emat[i, :] .= 1.0
        else
            # Partial censoring - use CensoringPatterns
            pattern_idx = obstype - 2
            for s in 1:n_observed
                state_prob = size(CensoringPatterns, 2) > s ? CensoringPatterns[pattern_idx, s + 1] : 0.0
                if state_prob > 0
                    for p in surrogate.state_to_phases[s]
                        emat[i, p] = state_prob
                    end
                end
            end
        end
    end
    
    return emat
end

"""
    expand_data_states!(data::DataFrame, surrogate::PhaseTypeSurrogate)

Expand statefrom and stateto columns to use phase indices.
"""
function expand_data_states!(data::DataFrame, surrogate::PhaseTypeSurrogate)
    data.statefrom = [first(surrogate.state_to_phases[s]) for s in data.statefrom]
    data.stateto = [first(surrogate.state_to_phases[s]) for s in data.stateto]
    return data
end

"""
    build_phasetype_model(tmat::Matrix{Int64}, config::PhaseTypeConfig;
                          data::DataFrame,
                          covariate_formula::Union{Nothing, FormulaTerm} = nothing,
                          SubjectWeights::Union{Nothing, Vector{Float64}} = nothing,
                          CensoringPatterns::Union{Nothing, Matrix{<:Real}} = nothing,
                          verbose::Bool = false)

Build a multistate Markov model on the expanded phase-type state space.

This creates a standard MultistateModel that can be fitted using the
existing `fit()` infrastructure.
"""
function build_phasetype_model(tmat::Matrix{Int64}, config::PhaseTypeConfig;
                               data::DataFrame,
                               covariate_formula::Union{Nothing, FormulaTerm} = nothing,
                               SubjectWeights::Union{Nothing, Vector{Float64}} = nothing,
                               CensoringPatterns::Union{Nothing, Matrix{<:Real}} = nothing,
                               verbose::Bool = false)
    
    # Step 1: Build the phase-type surrogate
    surrogate = build_phasetype_surrogate(tmat, config)
    
    if verbose
        println("Phase-type model structure:")
        println("  Observed states: $(surrogate.n_observed_states)")
        println("  Expanded states: $(surrogate.n_expanded_states)")
        for s in 1:surrogate.n_observed_states
            phases = surrogate.state_to_phases[s]
            println("  State $s → phases $(first(phases)):$(last(phases)) ($(length(phases)) phases)")
        end
    end
    
    # Step 2: Build expanded transition matrix
    tmat_expanded = build_expanded_tmat(tmat, surrogate)
    
    # Step 3: Generate hazard specifications for expanded model
    hazards = build_phasetype_hazards(tmat, config, surrogate;
                                       covariate_formula = covariate_formula)
    
    if verbose
        println("  Number of hazards: $(length(hazards))")
    end
    
    # Step 4: Prepare data - expand state references to phases
    data_expanded = copy(data)
    expand_data_states!(data_expanded, surrogate)
    
    # Step 5: Prepare censoring patterns for expanded state space
    n_expanded = surrogate.n_expanded_states
    if CensoringPatterns !== nothing
        n_patterns = size(CensoringPatterns, 1)
        CensoringPatterns_expanded = zeros(Float64, n_patterns, n_expanded + 1)
        CensoringPatterns_expanded[:, 1] = CensoringPatterns[:, 1]
        
        for p in 1:n_patterns
            for s in 1:size(tmat, 1)
                state_prob = CensoringPatterns[p, s + 1]
                for phase in surrogate.state_to_phases[s]
                    CensoringPatterns_expanded[p, phase + 1] = state_prob
                end
            end
        end
    else
        CensoringPatterns_expanded = nothing
    end
    
    # Step 6: Build the emission matrix
    emat_expanded = build_phasetype_emat(data, surrogate, 
                                          CensoringPatterns === nothing ? 
                                          Matrix{Float64}(undef, 0, size(tmat, 1)) : 
                                          Float64.(CensoringPatterns))
    
    # Step 7: Build the multistate model
    # Note: Always use verbose=false for model construction since this builds exponential 
    # hazards on the expanded phase space. The standard validation would incorrectly warn
    # about "missing transitions" when template data has no actual transitions yet (they're
    # simulated later). Users see structure info via the earlier verbose print statements.
    model = multistatemodel(hazards...; 
                           data = data_expanded,
                           SubjectWeights = SubjectWeights,
                           CensoringPatterns = CensoringPatterns_expanded,
                           EmissionMatrix = emat_expanded,
                           verbose = false)
    
    return (
        model = model,
        surrogate = surrogate,
        tmat_expanded = tmat_expanded,
        tmat_original = tmat
    )
end
