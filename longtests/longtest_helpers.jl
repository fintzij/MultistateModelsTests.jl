# =============================================================================
# Long Test Helper Functions
# =============================================================================
#
# Shared utility functions for inference long tests.
# These functions are included in the MultistateModelsTests module context,
# so they have access to DataFrames, Distributions, LinearAlgebra, Printf,
# Random, and Statistics via the module's imports.
#
# When running standalone (outside module), ensure these are imported first.
# =============================================================================

# Ensure required packages are available (safe to call multiple times)
using DataFrames
using Dates
using Distributions
using JSON3
using LinearAlgebra
using Printf
using Random
using Statistics

mutable struct TestResult
    test_name::String
    rel_errors::Dict{Symbol, Float64}
    max_rel_error::Float64
    passed::Bool
    
    function TestResult(test_name::String)
        new(test_name, Dict{Symbol, Float64}(), NaN, false)
    end
end

const PASS_THRESHOLD = 0.15

# =============================================================================
# Relative Error Computation
# =============================================================================

"""
    compute_relative_error(true_val, est_val)

Compute relative error as percentage. For values near zero (|true_val| < 0.01),
returns absolute error × 100 to avoid division issues.
"""
function compute_relative_error(true_val::Float64, est_val::Float64)
    if isnan(true_val) || isnan(est_val)
        return NaN
    end
    if abs(true_val) < 0.01
        return (est_val - true_val) * 100
    end
    return ((est_val - true_val) / true_val) * 100
end

"""
    extract_ci(se, est; level=0.95)

Compute confidence interval from standard error and estimate.
"""
function extract_ci(se::Float64, est::Float64; level=0.95)
    if isnan(se) || isnan(est)
        return (NaN, NaN)
    end
    z = quantile(Normal(), 1 - (1 - level) / 2)
    return (est - z * se, est + z * se)
end

# =============================================================================
# Data Template Generation
# =============================================================================

"""
    create_baseline_template(n_subjects; max_time=MAX_TIME)

Create a basic data template for models without covariates.
All subjects start in state 1 at time 0.
"""
function create_baseline_template(n_subjects::Int; max_time::Float64=MAX_TIME)
    return DataFrame(
        id = 1:n_subjects,
        tstart = zeros(n_subjects),
        tstop = fill(max_time, n_subjects),
        statefrom = ones(Int, n_subjects),
        stateto = ones(Int, n_subjects),
        obstype = ones(Int, n_subjects)
    )
end

"""
    create_tfc_template(n_subjects; max_time=MAX_TIME, x_prob=0.5)

Create data template with time-fixed binary covariate x.
x is randomly assigned with probability x_prob of being 1.
"""
function create_tfc_template(n_subjects::Int; max_time::Float64=MAX_TIME, x_prob::Float64=0.5)
    x_vals = rand([0.0, 1.0], n_subjects)
    return DataFrame(
        id = 1:n_subjects,
        tstart = zeros(n_subjects),
        tstop = fill(max_time, n_subjects),
        statefrom = ones(Int, n_subjects),
        stateto = ones(Int, n_subjects),
        obstype = ones(Int, n_subjects),
        x = x_vals
    )
end

"""
    create_tvc_template(n_subjects; max_time=MAX_TIME, changepoint=TVC_CHANGEPOINT)

Create data template with time-varying binary covariate x.
Each subject has x=0 for t < changepoint, x=1 for t ≥ changepoint.
Returns a template with 2 rows per subject.
"""
function create_tvc_template(n_subjects::Int; max_time::Float64=MAX_TIME, 
                             changepoint::Float64=TVC_CHANGEPOINT)
    ids = repeat(1:n_subjects, inner=2)
    tstart = repeat([0.0, changepoint], n_subjects)
    tstop = repeat([changepoint, max_time], n_subjects)
    x_vals = repeat([0.0, 1.0], n_subjects)
    
    return DataFrame(
        id = ids,
        tstart = tstart,
        tstop = tstop,
        statefrom = ones(Int, 2*n_subjects),
        stateto = ones(Int, 2*n_subjects),
        obstype = ones(Int, 2*n_subjects),
        x = x_vals
    )
end

# =============================================================================
# Panel Data Conversion
# =============================================================================

"""
    create_panel_data(paths, panel_times, n_states; phase_to_state=nothing)

Convert simulated sample paths to panel observations at fixed times.

# Arguments
- `paths`: Vector of SamplePath objects
- `panel_times`: Times at which to observe state (e.g., [1,2,3,...,10])
- `n_states`: Number of observed states (for determining absorbing state)
- `phase_to_state`: Optional mapping from phase indices to observed state indices

# Returns
DataFrame with panel observations (obstype=2)
"""
function create_panel_data(paths::Vector, panel_times::Vector{Float64}, n_states::Int;
                           phase_to_state::Union{Nothing, Dict{Int,Int}}=nothing)
    panel_rows = DataFrame[]
    
    for (subj_id, path) in enumerate(paths)
        for i in 1:(length(panel_times) - 1)
            t_start = panel_times[i]
            t_stop = panel_times[i + 1]
            
            # Get state at t_start and t_stop
            idx_start = searchsortedlast(path.times, t_start)
            idx_stop = searchsortedlast(path.times, t_stop)
            
            if idx_start >= 1 && idx_stop >= 1
                state_start = path.states[idx_start]
                state_stop = path.states[idx_stop]
                
                # Map phases to observed states if provided
                if !isnothing(phase_to_state)
                    state_start = get(phase_to_state, state_start, state_start)
                    state_stop = get(phase_to_state, state_stop, state_stop)
                end
                
                # Only include if not already absorbed at start
                if state_start < n_states
                    push!(panel_rows, DataFrame(
                        id = subj_id,
                        tstart = t_start,
                        tstop = t_stop,
                        statefrom = state_start,
                        stateto = state_stop,
                        obstype = 2  # Panel observation
                    ))
                end
            end
        end
    end
    
    df = isempty(panel_rows) ? DataFrame() : vcat(panel_rows...)
    if !isempty(df)
        # Re-index IDs to be contiguous 1..N
        old_ids = unique(df.id)
        id_map = Dict(old_id => new_id for (new_id, old_id) in enumerate(old_ids))
        df.id = [id_map[id] for id in df.id]
    end
    return df
end

"""
    create_panel_data_with_covariate(paths, panel_times, n_states, x_vals;
                                      phase_to_state=nothing)

Convert simulated sample paths to panel observations, preserving time-fixed covariate.
"""
function create_panel_data_with_covariate(paths::Vector, panel_times::Vector{Float64}, 
                                          n_states::Int, x_vals::Vector{Float64};
                                          phase_to_state::Union{Nothing, Dict{Int,Int}}=nothing)
    panel_rows = DataFrame[]
    
    for (subj_id, path) in enumerate(paths)
        x = x_vals[subj_id]
        
        for i in 1:(length(panel_times) - 1)
            t_start = panel_times[i]
            t_stop = panel_times[i + 1]
            
            idx_start = searchsortedlast(path.times, t_start)
            idx_stop = searchsortedlast(path.times, t_stop)
            
            if idx_start >= 1 && idx_stop >= 1
                state_start = path.states[idx_start]
                state_stop = path.states[idx_stop]
                
                if !isnothing(phase_to_state)
                    state_start = get(phase_to_state, state_start, state_start)
                    state_stop = get(phase_to_state, state_stop, state_stop)
                end
                
                if state_start < n_states
                    push!(panel_rows, DataFrame(
                        id = subj_id,
                        tstart = t_start,
                        tstop = t_stop,
                        statefrom = state_start,
                        stateto = state_stop,
                        obstype = 2,
                        x = x
                    ))
                end
            end
        end
    end
    
    df = isempty(panel_rows) ? DataFrame() : vcat(panel_rows...)
    if !isempty(df)
        # Re-index IDs to be contiguous 1..N
        old_ids = unique(df.id)
        id_map = Dict(old_id => new_id for (new_id, old_id) in enumerate(old_ids))
        df.id = [id_map[id] for id in df.id]
    end
    return df
end

"""
    create_panel_data_with_tvc(paths, panel_times, n_states; 
                                changepoint=TVC_CHANGEPOINT, phase_to_state=nothing)

Convert simulated sample paths to panel observations with time-varying covariate.
Covariate x = 0 for t < changepoint, x = 1 for t ≥ changepoint.
"""
function create_panel_data_with_tvc(paths::Vector, panel_times::Vector{Float64}, 
                                    n_states::Int; changepoint::Float64=TVC_CHANGEPOINT,
                                    phase_to_state::Union{Nothing, Dict{Int,Int}}=nothing)
    panel_rows = DataFrame[]
    
    for (subj_id, path) in enumerate(paths)
        for i in 1:(length(panel_times) - 1)
            t_start = panel_times[i]
            t_stop = panel_times[i + 1]
            
            idx_start = searchsortedlast(path.times, t_start)
            idx_stop = searchsortedlast(path.times, t_stop)
            
            if idx_start >= 1 && idx_stop >= 1
                state_start = path.states[idx_start]
                state_stop = path.states[idx_stop]
                
                if !isnothing(phase_to_state)
                    state_start = get(phase_to_state, state_start, state_start)
                    state_stop = get(phase_to_state, state_stop, state_stop)
                end
                
                if state_start < n_states
                    # Determine x value based on panel interval
                    # Use midpoint of interval to determine covariate value
                    t_mid = (t_start + t_stop) / 2
                    x = t_mid >= changepoint ? 1.0 : 0.0
                    
                    push!(panel_rows, DataFrame(
                        id = subj_id,
                        tstart = t_start,
                        tstop = t_stop,
                        statefrom = state_start,
                        stateto = state_stop,
                        obstype = 2,
                        x = x
                    ))
                end
            end
        end
    end
    
    df = isempty(panel_rows) ? DataFrame() : vcat(panel_rows...)
    if !isempty(df)
        # Re-index IDs to be contiguous 1..N
        old_ids = unique(df.id)
        id_map = Dict(old_id => new_id for (new_id, old_id) in enumerate(old_ids))
        df.id = [id_map[id] for id in df.id]
    end
    return df
end

# =============================================================================
# Prevalence and Cumulative Incidence Computation
# =============================================================================

"""
    compute_state_prevalence(paths, eval_times, n_states)

Compute state prevalence at each evaluation time from sample paths.
Returns matrix of size (n_times, n_states).
"""
function compute_state_prevalence(paths::Vector, eval_times::Vector{Float64}, n_states::Int)
    n_times = length(eval_times)
    prevalence = zeros(Float64, n_times, n_states)
    n_paths = length(paths)
    
    for path in paths
        for (t_idx, t) in enumerate(eval_times)
            state_idx = searchsortedlast(path.times, t)
            if state_idx >= 1
                state = path.states[state_idx]
                if state <= n_states
                    prevalence[t_idx, state] += 1.0
                end
            end
        end
    end
    
    prevalence ./= n_paths
    return prevalence
end

"""
    count_transitions(paths::Vector, n_states::Int)

Count total transitions between each pair of states from sample paths.
Returns a matrix of size (n_states, n_states) with transition counts.
"""
function count_transitions(paths::Vector, n_states::Int)
    counts = zeros(Int, n_states, n_states)
    for path in paths
        for i in 1:(length(path.states) - 1)
            from_state = path.states[i]
            to_state = path.states[i+1]
            if from_state <= n_states && to_state <= n_states
                counts[from_state, to_state] += 1
            end
        end
    end
    return counts
end

"""
    compute_state_prevalence_phasetype(paths, eval_times, n_states, phase_to_state)

Compute state prevalence, mapping phases to observed states.
"""
function compute_state_prevalence_phasetype(paths::Vector, eval_times::Vector{Float64}, 
                                            n_states::Int, phase_to_state::Dict{Int, Int})
    n_times = length(eval_times)
    prevalence = zeros(Float64, n_times, n_states)
    n_paths = length(paths)
    
    for path in paths
        for (t_idx, t) in enumerate(eval_times)
            state_idx = searchsortedlast(path.times, t)
            if state_idx >= 1
                phase = path.states[state_idx]
                state = get(phase_to_state, phase, phase)
                if state <= n_states
                    prevalence[t_idx, state] += 1.0
                end
            end
        end
    end
    
    prevalence ./= n_paths
    return prevalence
end

"""
    compute_prevalence_from_data(exact_data, eval_times, n_states)

Compute state prevalence from exact observed data.
"""
function compute_prevalence_from_data(exact_data::DataFrame, eval_times::Vector{Float64}, 
                                      n_states::Int)
    n_times = length(eval_times)
    prevalence = zeros(Float64, n_times, n_states)
    
    subjects = unique(exact_data.id)
    n_subjects = length(subjects)
    
    for subj_id in subjects
        subj_data = filter(row -> row.id == subj_id, exact_data)
        
        for (t_idx, t) in enumerate(eval_times)
            state = nothing
            for row in eachrow(subj_data)
                if row.tstart <= t < row.tstop
                    state = row.statefrom
                    break
                elseif t >= row.tstop
                    state = row.stateto
                end
            end
            
            if !isnothing(state) && state <= n_states
                prevalence[t_idx, state] += 1.0
            end
        end
    end
    
    prevalence ./= n_subjects
    return prevalence
end

"""
    compute_cumulative_incidence(paths, eval_times, from_state, to_state)

Compute cumulative incidence of transitions from_state → to_state.
"""
function compute_cumulative_incidence(paths::Vector, eval_times::Vector{Float64},
                                      from_state::Int, to_state::Int)
    n_times = length(eval_times)
    cumincid = zeros(Float64, n_times)
    n_paths = length(paths)
    
    for path in paths
        transition_time = Inf
        for i in 1:(length(path.states) - 1)
            if path.states[i] == from_state && path.states[i+1] == to_state
                transition_time = path.times[i+1]
                break
            end
        end
        
        for (t_idx, t) in enumerate(eval_times)
            if transition_time <= t
                cumincid[t_idx] += 1.0
            end
        end
    end
    
    cumincid ./= n_paths
    return cumincid
end

"""
    compute_cumincid_from_data(exact_data, eval_times, from_state, to_state)

Compute cumulative incidence from exact observed data.
"""
function compute_cumincid_from_data(exact_data::DataFrame, eval_times::Vector{Float64},
                                    from_state::Int, to_state::Int)
    n_times = length(eval_times)
    cumincid = zeros(Float64, n_times)
    
    subjects = unique(exact_data.id)
    n_subjects = length(subjects)
    
    for subj_id in subjects
        subj_data = filter(row -> row.id == subj_id, exact_data)
        
        transition_time = Inf
        for row in eachrow(subj_data)
            if row.statefrom == from_state && row.stateto == to_state
                transition_time = row.tstop
                break
            end
        end
        
        for (t_idx, t) in enumerate(eval_times)
            if transition_time <= t
                cumincid[t_idx] += 1.0
            end
        end
    end
    
    cumincid ./= n_subjects
    return cumincid
end

"""
    compute_phasetype_cumincid(paths, eval_times, from_phases, to_phases)

Compute cumulative incidence for phase-type models.
Track proportion who transitioned from any phase in from_phases to any phase in to_phases.
"""
function compute_phasetype_cumincid(paths::Vector, eval_times::Vector{Float64},
                                    from_phases::Vector{Int}, to_phases::Vector{Int})
    n_times = length(eval_times)
    cumincid = zeros(Float64, n_times)
    n_paths = length(paths)
    
    for path in paths
        first_trans_time = Inf
        for i in 1:(length(path.states) - 1)
            if path.states[i] in from_phases && path.states[i + 1] in to_phases
                first_trans_time = path.times[i + 1]
                break
            end
        end
        
        for (t_idx, t) in enumerate(eval_times)
            if first_trans_time <= t
                cumincid[t_idx] += 1.0
            end
        end
    end
    
    cumincid ./= n_paths
    return cumincid
end

# =============================================================================
# Phase-Type Model Helpers
# =============================================================================

"""
    make_phase_to_state_map(n_phases)

Create mapping from phase indices to observed state indices for phase-type models.
For 3-state progressive model with n_phases per transient state:
- Phases 1:n_phases → State 1
- Phases (n_phases+1):(2*n_phases) → State 2  
- Phase 2*n_phases+1 → State 3 (absorbing)
"""
function make_phase_to_state_map(n_phases::Int)
    phase_to_state = Dict{Int, Int}()
    for p in 1:n_phases
        phase_to_state[p] = 1
    end
    for p in (n_phases + 1):(2 * n_phases)
        phase_to_state[p] = 2
    end
    phase_to_state[2 * n_phases + 1] = 3
    return phase_to_state
end

# =============================================================================
# Spline Knot Computation
# =============================================================================

"""
    compute_sojourn_quantiles(rate; quantiles=[0.2, 0.4, 0.6, 0.8])

Compute quantiles of exponential sojourn distribution for spline knot placement.
"""
function compute_sojourn_quantiles_exp(rate::Float64; quantiles::Vector{Float64}=[0.2, 0.4, 0.6, 0.8])
    return quantile.(Exponential(1/rate), quantiles)
end

"""
    compute_sojourn_quantiles_wei(shape, scale; quantiles=[0.2, 0.4, 0.6, 0.8])

Compute quantiles of Weibull sojourn distribution for spline knot placement.
Weibull parameterized as h(t) = shape * scale * t^(shape-1)
"""
function compute_sojourn_quantiles_wei(shape::Float64, scale::Float64; 
                                       quantiles::Vector{Float64}=[0.2, 0.4, 0.6, 0.8])
    # Weibull distribution with shape α and scale λ
    # S(t) = exp(-(λt)^α)
    return quantile.(Weibull(shape, 1/scale), quantiles)
end

"""
    compute_spline_knots(sojourn_quantiles; min_knot=0.5, max_knot=12.0)

Compute interior knots for spline baseline, clipped to reasonable range.
"""
function compute_spline_knots(sojourn_quantiles::Vector{Float64}; 
                              min_knot::Float64=0.5, max_knot::Float64=12.0)
    knots = clamp.(sojourn_quantiles, min_knot, max_knot)
    return unique(sort(knots))
end

# =============================================================================
# Result Processing
# =============================================================================

"""
    finalize_result!(result::TestResult)

Compute max relative error and pass/fail status for a test result.
"""
function finalize_result!(result::TestResult)
    rel_errs = filter(!isnan, collect(values(result.rel_errors)))
    if isempty(rel_errs)
        result.max_rel_error = NaN
        result.passed = false
    else
        result.max_rel_error = maximum(abs.(rel_errs))
        result.passed = result.max_rel_error <= PASS_THRESHOLD
    end
    return result
end

# =============================================================================
# LongTestResult Capture Functions
# =============================================================================
#
# These functions capture comprehensive test results for the reporting system.
# Each test should call capture_longtest_result!() after fitting to save
# all relevant data to the cache for display in long_tests.qmd.
#
# Note: LongTestResults.jl is included at module level in MultistateModelsTests.jl
# so types like LongTestResult are available here.
# =============================================================================

# Import internal functions needed for capture
import MultistateModels: get_parameters_flat, multistatemodel, set_parameters!, simulate

# Use constants from longtest_config.jl (included before this file)
# N_SUBJECTS, MAX_TIME, EVAL_TIMES, N_SIM_TRAJ, RNG_SEED are defined there

"""
    _flat_to_named(flat_params, hazards)

Convert flat parameter vector to NamedTuple keyed by hazard names.
"""
function _flat_to_named(flat_params::Vector{Float64}, hazards)
    params = Dict{Symbol, Vector{Float64}}()
    idx = 1
    for haz in hazards
        npar = haz.npar_total
        params[haz.hazname] = flat_params[idx:idx+npar-1]
        idx += npar
    end
    return NamedTuple(params)
end

"""
    check_param_recovery(true_val, est_val; is_beta=false, is_shape=false) -> Bool

Check if a parameter was recovered within tolerance.

# Arguments
- `true_val`: True parameter value
- `est_val`: Estimated parameter value  
- `is_beta`: Whether this is a beta (covariate) parameter (uses BETA_ABS_TOL)
- `is_shape`: Whether this is a shape parameter (uses SHAPE_ABS_TOL)

Shape and beta parameters use absolute tolerance since they can be near zero.
Other parameters use relative tolerance unless the true value is small.
"""
function check_param_recovery(true_val::Float64, est_val::Float64; 
                              is_beta::Bool=false, is_shape::Bool=false)
    if is_beta
        return abs(est_val - true_val) <= BETA_ABS_TOL
    elseif is_shape
        return abs(est_val - true_val) <= SHAPE_ABS_TOL
    else
        # For small parameters, use absolute tolerance to avoid massive relative errors
        if abs(true_val) < SMALL_PARAM_THRESHOLD
            return abs(est_val - true_val) <= PARAM_REL_TOL
        else
            return abs((est_val - true_val) / true_val) <= PARAM_REL_TOL
        end
    end
end

"""
    capture_longtest_result!(
        test_name::String,
        fitted,
        true_params_named::NamedTuple,
        param_names::Vector{String},
        hazard_specs;
        hazard_family::String,
        data_type::String,
        covariate_type::String,
        data::DataFrame,
        n_sim::Int=N_SIM_TRAJ,
        eval_times::Vector{Float64}=EVAL_TIMES,
        beta_param_names::Vector{String}=String[],
        shape_param_names::Vector{String}=String[]
    ) -> LongTestResult

Capture comprehensive long test results for reporting.

# Arguments
- `test_name`: Unique test identifier (e.g., "exp_exact_nocov")
- `fitted`: Fitted MultistateModelFitted object
- `true_params_named`: True parameters as NamedTuple by hazard
- `param_names`: Vector of parameter names for display
- `hazard_specs`: Tuple of Hazard objects used in model construction
- `hazard_family`: One of "exp", "wei", "gom", "pt", "sp"
- `data_type`: One of "exact", "panel", "mcem"
- `covariate_type`: One of "nocov", "fixed", "tvc"
- `data`: The simulated data used for fitting
- `n_sim`: Number of simulations for prevalence/CI comparison (default: N_SIM_TRAJ from config)
- `eval_times`: Times at which to evaluate prevalence/CI (default: EVAL_TIMES from config)
- `beta_param_names`: Names of beta (covariate) parameters for tolerance checking
- `shape_param_names`: Names of shape parameters for tolerance checking

# Returns
Populated LongTestResult, also saved to cache.
"""
function capture_longtest_result!(
    test_name::String,
    fitted,
    true_params_named::NamedTuple,
    param_names::Vector{String},
    hazard_specs;
    hazard_family::String,
    data_type::String,
    covariate_type::String,
    data::DataFrame,
    n_sim::Int=N_SIM_TRAJ,
    eval_times::Vector{Float64}=EVAL_TIMES,
    beta_param_names::Vector{String}=String[],
    shape_param_names::Vector{String}=String[],
    n_states::Int=3,
    transitions::Vector{Tuple{Int,Int}}=[(1, 2), (2, 3)]
)
    result = LongTestResult(
        test_name = test_name,
        test_description = "$(hazard_family) - $(data_type) - $(covariate_type)",
        hazard_family = hazard_family,
        data_type = data_type,
        covariate_type = covariate_type,
        n_subjects = length(unique(data.id)),
        n_simulations = n_sim,
        n_states = n_states
    )
    
    # Flatten true params
    true_flat = Float64[]
    for haz in fitted.hazards
        append!(true_flat, true_params_named[haz.hazname])
    end
    
    # Get fitted params (use get_parameters with :flat scale via import)
    fitted_flat = get_parameters_flat(fitted)
    
    # Get SEs (handle missing vcov or negative diagonal elements safely)
    if isnothing(fitted.vcov)
        ses = fill(NaN, length(fitted_flat))
    else
        diag_vals = diag(fitted.vcov)
        ses = [v >= 0 ? sqrt(v) : NaN for v in diag_vals]
    end
    
    # Store parameters and check recovery
    all_passed = true
    for (i, name) in enumerate(param_names)
        is_beta = name in beta_param_names
        is_shape = name in shape_param_names
        passed = check_param_recovery(true_flat[i], fitted_flat[i]; is_beta=is_beta, is_shape=is_shape)
        all_passed = all_passed && passed
        
        result.true_params[name] = true_flat[i]
        result.estimated_params[name] = fitted_flat[i]
        result.standard_errors[name] = ses[i]
        result.ci_lower[name] = fitted_flat[i] - 1.96 * ses[i]
        result.ci_upper[name] = fitted_flat[i] + 1.96 * ses[i]
        result.param_passed[name] = passed
    end
    result.passed = all_passed
    
    # Data summary
    result.data_summary = summarize_data(data)
    
    # Simulate from true and fitted for comparison
    max_time = maximum(eval_times)
    
    # Create appropriate template based on covariate type
    if covariate_type == "nocov"
        template = DataFrame(
            id = 1:n_sim,
            tstart = zeros(n_sim),
            tstop = fill(max_time, n_sim),
            statefrom = ones(Int, n_sim),
            stateto = ones(Int, n_sim),
            obstype = ones(Int, n_sim)
        )
    elseif covariate_type == "fixed"
        # Time-fixed covariate: half with x=0, half with x=1
        template = DataFrame(
            id = 1:n_sim,
            tstart = zeros(n_sim),
            tstop = fill(max_time, n_sim),
            statefrom = ones(Int, n_sim),
            stateto = ones(Int, n_sim),
            obstype = ones(Int, n_sim),
            x = repeat([0.0, 1.0], n_sim ÷ 2)
        )
    elseif covariate_type == "tvc"
        # Time-varying covariate: 2 rows per subject, x changes at TVC_CHANGEPOINT
        ids = repeat(1:n_sim, inner=2)
        tstart = repeat([0.0, TVC_CHANGEPOINT], n_sim)
        tstop = repeat([TVC_CHANGEPOINT, max_time], n_sim)
        x_vals = repeat([0.0, 1.0], n_sim)
        template = DataFrame(
            id = ids,
            tstart = tstart,
            tstop = tstop,
            statefrom = ones(Int, 2*n_sim),
            stateto = ones(Int, 2*n_sim),
            obstype = ones(Int, 2*n_sim),
            x = x_vals
        )
    else
        error("Unknown covariate_type: $covariate_type")
    end
    
    model_true = multistatemodel(hazard_specs...; data=template)
    set_parameters!(model_true, true_params_named)
    
    model_fitted = multistatemodel(hazard_specs...; data=template)
    fitted_named = _flat_to_named(fitted_flat, model_fitted.hazards)
    set_parameters!(model_fitted, fitted_named)
    
    Random.seed!(RNG_SEED + 5000)
    paths_true = simulate(model_true; paths=true, data=false, nsim=1)[1]
    Random.seed!(RNG_SEED + 5001)
    paths_fitted = simulate(model_fitted; paths=true, data=false, nsim=1)[1]
    
    # Compute state prevalence
    result.prevalence_times = copy(eval_times)
    for s in 1:n_states
        prev_true = compute_state_prevalence(paths_true, s, eval_times)
        prev_fitted = compute_state_prevalence(paths_fitted, s, eval_times)
        
        result.prevalence_true[string(s)] = prev_true.mean
        result.prevalence_true_lower[string(s)] = prev_true.lower
        result.prevalence_true_upper[string(s)] = prev_true.upper
        result.prevalence_fitted[string(s)] = prev_fitted.mean
        result.prevalence_fitted_lower[string(s)] = prev_fitted.lower
        result.prevalence_fitted_upper[string(s)] = prev_fitted.upper
    end
    
    # Compute cumulative incidence for specified transitions
    result.cumulative_incidence_times = copy(eval_times)
    for (from, to) in transitions
        key = "$(from)→$(to)"
        ci_true = compute_cumulative_incidence(paths_true, from, to, eval_times)
        ci_fitted = compute_cumulative_incidence(paths_fitted, from, to, eval_times)
        
        result.cumulative_incidence_true[key] = ci_true.mean
        result.cumulative_incidence_true_lower[key] = ci_true.lower
        result.cumulative_incidence_true_upper[key] = ci_true.upper
        result.cumulative_incidence_fitted[key] = ci_fitted.mean
        result.cumulative_incidence_fitted_lower[key] = ci_fitted.lower
        result.cumulative_incidence_fitted_upper[key] = ci_fitted.upper
    end
    
    # Note: Caching disabled - results are returned but not saved to disk.
    # See TEST_OVERHAUL_PROMPT.md for rationale.
    # save_longtest_result(result)  # DISABLED
    
    return result
end

"""
    capture_simple_longtest_result!(
        test_name::String,
        fitted,
        true_params_flat::Vector{Float64},
        param_names::Vector{String};
        hazard_family::String,
        data_type::String,
        covariate_type::String,
        n_subjects::Int,
        n_states::Int=3
    ) -> LongTestResult

Simplified result capture for tests that don't fit the standard 3-state progressive model.
Captures parameter recovery only, without prevalence/CI simulation.

Useful for phase-type tests with expanded state spaces.
"""
function capture_simple_longtest_result!(
    test_name::String,
    fitted,
    true_params_flat::Vector{Float64},
    param_names::Vector{String};
    hazard_family::String,
    data_type::String,
    covariate_type::String,
    n_subjects::Int,
    n_states::Int=3
)
    result = LongTestResult(
        test_name = test_name,
        test_description = "$(hazard_family) - $(data_type) - $(covariate_type)",
        hazard_family = hazard_family,
        data_type = data_type,
        covariate_type = covariate_type,
        n_subjects = n_subjects,
        n_simulations = 0,  # No simulation comparison
        n_states = n_states
    )
    
    # Get fitted params
    fitted_flat = get_parameters_flat(fitted)
    
    # Get SEs
    if isnothing(fitted.vcov)
        ses = fill(NaN, length(fitted_flat))
    else
        diag_vals = diag(fitted.vcov)
        ses = [v >= 0 ? sqrt(v) : NaN for v in diag_vals]
    end
    
    # Store parameters and check recovery
    all_passed = true
    for (i, name) in enumerate(param_names)
        true_val = true_params_flat[i]
        est_val = fitted_flat[i]
        
        # Use relative tolerance for non-zero, absolute for near-zero
        if abs(true_val) < 0.1
            passed = abs(est_val - true_val) <= 0.2
        else
            passed = abs((est_val - true_val) / true_val) <= PARAM_REL_TOL
        end
        
        all_passed = all_passed && passed
        
        result.true_params[name] = true_val
        result.estimated_params[name] = est_val
        result.standard_errors[name] = ses[i]
        result.ci_lower[name] = est_val - 1.96 * ses[i]
        result.ci_upper[name] = est_val + 1.96 * ses[i]
    result.param_passed[name] = passed
    end
    result.passed = all_passed
    
    # Note: Caching disabled - results are returned but not saved to disk.
    # See TEST_OVERHAUL_PROMPT.md for rationale.
    # save_longtest_result(result)  # DISABLED
    
    return result
end
