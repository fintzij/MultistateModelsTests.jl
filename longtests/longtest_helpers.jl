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

# z-value for 99% CI (two-sided)
# Using 99% CI for coverage checks reduces spurious failures when running many tests
const Z_99 = 2.576

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
    check_param_recovery(true_val, est_val; is_beta=false, is_shape=false, beta_tol=BETA_ABS_TOL) -> Bool

Check if a parameter was recovered within tolerance.

# Arguments
- `true_val`: True parameter value
- `est_val`: Estimated parameter value  
- `is_beta`: Whether this is a beta (covariate) parameter (uses beta_tol)
- `is_shape`: Whether this is a shape parameter (uses SHAPE_ABS_TOL)
- `beta_tol`: Absolute tolerance for beta parameters (default: BETA_ABS_TOL)

Shape and beta parameters use absolute tolerance since they can be near zero.
Other parameters use relative tolerance unless the true value is small.
"""
function check_param_recovery(true_val::Float64, est_val::Float64; 
                              is_beta::Bool=false, is_shape::Bool=false,
                              beta_tol::Float64=BETA_ABS_TOL)
    if is_beta
        return abs(est_val - true_val) <= beta_tol
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
    transitions::Vector{Tuple{Int,Int}}=[(1, 2), (2, 3)],
    beta_abs_tol::Float64=BETA_ABS_TOL
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
    # CRITICAL: Use 99% CI (z=Z_99=2.576) for coverage checks to reduce spurious failures
    # With 18 tests and multiple parameters, 95% CI would give ~5% false failures per param
    
    all_passed = true
    for (i, name) in enumerate(param_names)
        result.true_params[name] = true_flat[i]
        result.estimated_params[name] = fitted_flat[i]
        result.standard_errors[name] = ses[i]
        
        # Use 99% CI for coverage check (more appropriate for validation)
        ci_lo = fitted_flat[i] - Z_99 * ses[i]
        ci_hi = fitted_flat[i] + Z_99 * ses[i]
        result.ci_lower[name] = ci_lo
        result.ci_upper[name] = ci_hi
        
        # Check CI coverage: does the 99% CI contain the true value?
        # This is the ONLY criterion for pass/fail - CI coverage is the statistically
        # valid test. Tolerance-based checks are informative but not required.
        ci_covers_truth = (ci_lo <= true_flat[i] <= ci_hi)
        
        # Parameter passes if 99% CI covers the true value
        result.param_passed[name] = ci_covers_truth
        all_passed = all_passed && ci_covers_truth
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
    
    # Compute OBSERVED prevalence from the actual data used for fitting
    observed_prev = compute_observed_prevalence(data, eval_times, n_states)
    for s in 1:n_states
        result.prevalence_observed[string(s)] = observed_prev[s]
    end
    
    # Compute prevalence from simulated trajectories (truth and estimated)
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
        
        # Compute OBSERVED cumulative incidence from actual fitting data
        ci_observed = compute_observed_cumulative_incidence(data, from, to, eval_times)
        result.cumulative_incidence_observed[key] = ci_observed
        
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
        param_names::Vector{String},
        est_params_flat::Union{Vector{Float64}, Nothing}=nothing,
        ses_flat::Union{Vector{Float64}, Nothing}=nothing;
        hazard_family::String,
        data_type::String,
        covariate_type::String,
        n_subjects::Int,
        n_states::Int=3,
        model_true=nothing,
        model_fitted=nothing,
        phase_to_state::Union{Vector{Int}, Nothing}=nothing,
        transitions::Vector{Tuple{Int,Int}}=Tuple{Int,Int}[],
        n_sim::Int=1000,
        eval_times::Vector{Float64}=collect(0.0:0.5:10.0)
    ) -> LongTestResult

Simplified result capture for tests that don't fit the standard 3-state progressive model.

When model_true and model_fitted are provided, also computes prevalence/cumulative
incidence for plotting. For phase-type models, phase_to_state maps phase indices
to observed states for collapsing.

# Arguments
- `test_name`: Identifier for the test
- `fitted`: Fitted model object
- `true_params_flat`: Vector of true parameter values
- `param_names`: Names for the parameters
- `est_params_flat`: (optional) Explicit estimates; if not provided, extracts from fitted model
- `ses_flat`: (optional) Explicit standard errors for the explicit estimates

# Keyword Arguments  
- `hazard_family`: e.g., "pt", "wei", "gom"
- `data_type`: e.g., "exact", "panel"
- `covariate_type`: e.g., "fixed", "tvc"
- `n_subjects`: Number of subjects
- `n_states`: Number of observed states (default 3)

# Optional Simulation Arguments (for generating plot data)
- `model_true`: Model with true parameters set (for simulation)
- `model_fitted`: Model with fitted parameters set (for simulation)
- `phase_to_state`: Vector mapping phase indices to observed states (for phase-type models)
- `transitions`: List of (from, to) observed state pairs for cumulative incidence
- `n_sim`: Number of simulations (default 1000)
- `eval_times`: Times for evaluation (default 0:0.5:10)
"""
function capture_simple_longtest_result!(
    test_name::String,
    fitted,
    true_params_flat::Vector{Float64},
    param_names::Vector{String},
    est_params_flat::Union{Vector{Float64}, Nothing}=nothing,
    ses_flat::Union{Vector{Float64}, Nothing}=nothing;
    hazard_family::String,
    data_type::String,
    covariate_type::String,
    n_subjects::Int,
    n_states::Int=3,
    model_true=nothing,
    model_fitted=nothing,
    phase_to_state::Union{Vector{Int}, Nothing}=nothing,
    transitions::Vector{Tuple{Int,Int}}=Tuple{Int,Int}[],
    n_sim::Int=1000,
    eval_times::Vector{Float64}=collect(0.0:0.5:10.0)
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
    
    # Get fitted params - use explicit estimates if provided, otherwise extract from model
    if isnothing(est_params_flat)
        fitted_flat = get_parameters_flat(fitted)
    else
        fitted_flat = est_params_flat
    end
    
    # Get variance-covariance matrix - prefer model-based vcov, fall back to IJ (robust)
    # IJ variance is particularly useful when model has constraints
    vcov_matrix = if !isnothing(fitted.vcov)
        fitted.vcov
    elseif hasproperty(fitted, :ij_vcov) && !isnothing(fitted.ij_vcov)
        fitted.ij_vcov
    else
        nothing
    end
    
    # Get SEs - priority:
    # 1. Use explicit SEs if provided (e.g., from delta method for derived params)
    # 2. Extract from vcov if no explicit estimates provided
    # 3. Fall back to NaN
    if !isnothing(ses_flat)
        # Explicit SEs provided (e.g., from delta method for identifiable params)
        ses = ses_flat
    elseif isnothing(est_params_flat) && !isnothing(vcov_matrix)
        # No explicit estimates, so we can use vcov directly
        diag_vals = diag(vcov_matrix)
        ses = [v >= 0 ? sqrt(v) : NaN for v in diag_vals]
    else
        # Either no vcov, or explicit estimates without explicit SEs
        ses = fill(NaN, length(fitted_flat))
    end
    
    # Store parameters and check recovery
    # When SEs are available, use 99% CI coverage (Z_99 = 2.576)
    # When SEs are not available, use relative error tolerance
    has_ses = !any(isnan.(ses))
    rel_err_tolerance = 0.30  # 30% relative error tolerance when no SEs
    
    all_passed = true
    for (i, name) in enumerate(param_names)
        true_val = true_params_flat[i]
        est_val = fitted_flat[i]
        
        result.true_params[name] = true_val
        result.estimated_params[name] = est_val
        result.standard_errors[name] = ses[i]
        
        # Compute CI (will be NaN if no SEs)
        ci_lo = est_val - Z_99 * ses[i]
        ci_hi = est_val + Z_99 * ses[i]
        result.ci_lower[name] = ci_lo
        result.ci_upper[name] = ci_hi
        
        # Check parameter recovery
        if has_ses
            # Use 99% CI coverage when SEs are available
            param_passes = (ci_lo <= true_val <= ci_hi)
        else
            # Use relative error tolerance when no SEs
            if abs(true_val) > 0.01
                rel_err = abs(est_val - true_val) / abs(true_val)
                param_passes = rel_err < rel_err_tolerance
            else
                # For values near zero, use absolute error
                param_passes = abs(est_val - true_val) < rel_err_tolerance
            end
        end
        
        result.param_passed[name] = param_passes
        all_passed = all_passed && param_passes
    end
    result.passed = all_passed
    
    # If simulation models provided, compute prevalence/cumulative incidence for plots
    if !isnothing(model_true) && !isnothing(model_fitted)
        result.n_simulations = n_sim
        
        # Simulate paths from both models
        Random.seed!(RNG_SEED + 8000)
        paths_true = simulate(model_true; paths=true, data=false, nsim=1)[1]
        Random.seed!(RNG_SEED + 8001)
        paths_fitted = simulate(model_fitted; paths=true, data=false, nsim=1)[1]
        
        result.prevalence_times = copy(eval_times)
        result.cumulative_incidence_times = copy(eval_times)
        
        # Compute prevalence for each observed state
        for obs_state in 1:n_states
            if isnothing(phase_to_state)
                # Direct state model (no phase expansion) - treat state as single phase
                prev_true = _compute_phasetype_prevalence(paths_true, [obs_state], eval_times)
                prev_fitted = _compute_phasetype_prevalence(paths_fitted, [obs_state], eval_times)
            else
                # Phase-type: collapse phases to observed states
                phases_for_state = findall(p -> p == obs_state, phase_to_state)
                prev_true = _compute_phasetype_prevalence(paths_true, phases_for_state, eval_times)
                prev_fitted = _compute_phasetype_prevalence(paths_fitted, phases_for_state, eval_times)
            end
            
            key = string(obs_state)
            result.prevalence_true[key] = prev_true.mean
            result.prevalence_true_lower[key] = prev_true.lower
            result.prevalence_true_upper[key] = prev_true.upper
            result.prevalence_fitted[key] = prev_fitted.mean
            result.prevalence_fitted_lower[key] = prev_fitted.lower
            result.prevalence_fitted_upper[key] = prev_fitted.upper
        end
        
        # Compute cumulative incidence for specified transitions
        for (from_state, to_state) in transitions
            key = "$(from_state)→$(to_state)"
            
            if isnothing(phase_to_state)
                # Direct state model
                ci_true = compute_cumulative_incidence(paths_true, from_state, to_state, eval_times)
                ci_fitted = compute_cumulative_incidence(paths_fitted, from_state, to_state, eval_times)
            else
                # Phase-type: find phases for each state
                from_phases = findall(p -> p == from_state, phase_to_state)
                to_phases = findall(p -> p == to_state, phase_to_state)
                ci_true = _compute_phasetype_cumincid(paths_true, from_phases, to_phases, eval_times)
                ci_fitted = _compute_phasetype_cumincid(paths_fitted, from_phases, to_phases, eval_times)
            end
            
            result.cumulative_incidence_true[key] = ci_true.mean
            result.cumulative_incidence_true_lower[key] = ci_true.lower
            result.cumulative_incidence_true_upper[key] = ci_true.upper
            result.cumulative_incidence_fitted[key] = ci_fitted.mean
            result.cumulative_incidence_fitted_lower[key] = ci_fitted.lower
            result.cumulative_incidence_fitted_upper[key] = ci_fitted.upper
        end
    end
    
    # Save to cache for reports
    save_longtest_result(result; force=true)
    
    return result
end
"""
    capture_phasetype_longtest_result!(
        test_name::String,
        fitted,
        true_params_flat::Vector{Float64},
        param_names::Vector{String},
        surrogate,
        hazard_specs;
        hazard_family::String="pt",
        data_type::String,
        covariate_type::String,
        n_subjects::Int,
        n_observed_states::Int=3,
        n_sim::Int=N_SIM_TRAJ,
        eval_times::Vector{Float64}=EVAL_TIMES,
        max_time::Float64=MAX_TIME,
        transitions::Vector{Tuple{Int,Int}}=[(1, 2), (2, 3)]
    ) -> LongTestResult

Capture long test results for phase-type hazard models.

Phase-type models operate on an expanded state space with latent phases.
This function simulates from both true and fitted models on the expanded space,
then collapses results to observed states for plotting comparison.

# Arguments
- `test_name`: Unique test identifier (e.g., "pt_exact_nocov")
- `fitted`: Fitted MultistateModelFitted object (on expanded state space)
- `true_params_flat`: True parameter values as flat vector
- `param_names`: Vector of parameter names for display
- `surrogate`: PhaseTypeSurrogate with state mappings
- `hazard_specs`: Tuple of Hazard objects used in model construction
- `hazard_family`: One of "pt" or "phasetype" (default: "pt")
- `data_type`: One of "exact", "panel", "mixed"
- `covariate_type`: One of "nocov", "fixed", "tvc"
- `n_subjects`: Number of subjects in the original test
- `n_observed_states`: Number of observed (non-phase) states (default: 3)
- `n_sim`: Number of simulations for prevalence/CI comparison
- `eval_times`: Times at which to evaluate prevalence/CI
- `max_time`: Maximum simulation time
- `transitions`: List of (from, to) observed state transitions for CI plots

# Returns
Populated LongTestResult with phase-collapsed prevalence/CI, saved to cache.

# Notes
The key insight is that we simulate on the expanded phase space then collapse:
- Phase indices are mapped to observed states via `surrogate.phase_to_state`
- Prevalence is computed as: P(observed state s) = sum of P(phase p) for all p in state s
- Cumulative incidence tracks first transition between observed states
"""
function capture_phasetype_longtest_result!(
    test_name::String,
    fitted,
    true_params_flat::Vector{Float64},
    param_names::Vector{String},
    surrogate,
    hazard_specs;
    hazard_family::String="pt",
    data_type::String,
    covariate_type::String,
    n_subjects::Int,
    n_observed_states::Int=3,
    n_sim::Int=N_SIM_TRAJ,
    eval_times::Vector{Float64}=EVAL_TIMES,
    max_time::Float64=MAX_TIME,
    transitions::Vector{Tuple{Int,Int}}=[(1, 2), (2, 3)]
)
    result = LongTestResult(
        test_name = test_name,
        test_description = "Phase-type: $(data_type) - $(covariate_type)",
        hazard_family = hazard_family,
        data_type = data_type,
        covariate_type = covariate_type,
        n_subjects = n_subjects,
        n_simulations = n_sim,
        n_states = n_observed_states
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
    
    # Store parameters and check recovery using CI coverage only
    all_passed = true
    for (i, name) in enumerate(param_names)
        true_val = true_params_flat[i]
        est_val = fitted_flat[i]
        
        result.true_params[name] = true_val
        result.estimated_params[name] = est_val
        result.standard_errors[name] = ses[i]
        
        # Use 99% CI for coverage check
        ci_lo = est_val - Z_99 * ses[i]
        ci_hi = est_val + Z_99 * ses[i]
        result.ci_lower[name] = ci_lo
        result.ci_upper[name] = ci_hi
        
        # Check CI coverage: does the 99% CI contain the true value?
        ci_covers_truth = (ci_lo <= true_val <= ci_hi)
        result.param_passed[name] = ci_covers_truth
        all_passed = all_passed && ci_covers_truth
    end
    result.passed = all_passed
    
    # Build phase_to_state Vector from surrogate
    phase_to_state = surrogate.phase_to_state  # Vector{Int}: phase index → observed state
    
    # Create simulation template on expanded state space
    # Template starts all subjects in first phase of state 1
    first_phase = first(surrogate.state_to_phases[1])
    
    if covariate_type == "nocov"
        template = DataFrame(
            id = 1:n_sim,
            tstart = zeros(n_sim),
            tstop = fill(max_time, n_sim),
            statefrom = fill(first_phase, n_sim),
            stateto = fill(first_phase, n_sim),
            obstype = ones(Int, n_sim)
        )
    elseif covariate_type == "fixed"
        # Time-fixed covariate: half with x=0, half with x=1
        template = DataFrame(
            id = 1:n_sim,
            tstart = zeros(n_sim),
            tstop = fill(max_time, n_sim),
            statefrom = fill(first_phase, n_sim),
            stateto = fill(first_phase, n_sim),
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
            statefrom = fill(first_phase, 2*n_sim),
            stateto = fill(first_phase, 2*n_sim),
            obstype = ones(Int, 2*n_sim),
            x = x_vals
        )
    else
        error("Unknown covariate_type: $covariate_type")
    end
    
    # Build true model and set parameters
    model_true = multistatemodel(hazard_specs...; data=template)
    true_params_named = _flat_to_named(true_params_flat, model_true.hazards)
    set_parameters!(model_true, true_params_named)
    
    # Build fitted model and set parameters
    model_fitted = multistatemodel(hazard_specs...; data=template)
    fitted_params_named = _flat_to_named(fitted_flat, model_fitted.hazards)
    set_parameters!(model_fitted, fitted_params_named)
    
    # Simulate paths from both models
    Random.seed!(RNG_SEED + 7000)
    paths_true = simulate(model_true; paths=true, data=false, nsim=1)[1]
    Random.seed!(RNG_SEED + 7001)
    paths_fitted = simulate(model_fitted; paths=true, data=false, nsim=1)[1]
    
    # Compute state prevalence (collapsed from phases to observed states)
    result.prevalence_times = copy(eval_times)
    
    # Compute OBSERVED prevalence from the actual data used for fitting
    # The data is in PHASE space, so we collapse to observed states using phase_to_state
    observed_prev = _compute_phasetype_observed_prevalence(fitted.data, eval_times, n_observed_states, phase_to_state)
    for s in 1:n_observed_states
        result.prevalence_observed[string(s)] = observed_prev[s]
    end
    
    for obs_state in 1:n_observed_states
        # Get all phases that belong to this observed state
        phases_for_state = findall(p -> p == obs_state, phase_to_state)
        
        # Compute prevalence by summing across all phases in this observed state
        prev_true = _compute_phasetype_prevalence(paths_true, phases_for_state, eval_times)
        prev_fitted = _compute_phasetype_prevalence(paths_fitted, phases_for_state, eval_times)
        
        key = string(obs_state)
        result.prevalence_true[key] = prev_true.mean
        result.prevalence_true_lower[key] = prev_true.lower
        result.prevalence_true_upper[key] = prev_true.upper
        result.prevalence_fitted[key] = prev_fitted.mean
        result.prevalence_fitted_lower[key] = prev_fitted.lower
        result.prevalence_fitted_upper[key] = prev_fitted.upper
    end
    
    # Compute cumulative incidence for observed state transitions
    # We track when a path first transitions from any phase in state A to any phase in state B
    result.cumulative_incidence_times = copy(eval_times)
    
    for (from_state, to_state) in transitions
        key = "$(from_state)→$(to_state)"
        
        # Compute OBSERVED cumulative incidence from fitting data (collapsing phases)
        ci_observed = _compute_phasetype_observed_cumincid(fitted.data, from_state, to_state, 
                                                           eval_times, phase_to_state)
        result.cumulative_incidence_observed[key] = ci_observed
        
        # Get phases for each observed state
        from_phases = findall(p -> p == from_state, phase_to_state)
        to_phases = findall(p -> p == to_state, phase_to_state)
        
        ci_true = _compute_phasetype_cumincid(paths_true, from_phases, to_phases, eval_times)
        ci_fitted = _compute_phasetype_cumincid(paths_fitted, from_phases, to_phases, eval_times)
        
        result.cumulative_incidence_true[key] = ci_true.mean
        result.cumulative_incidence_true_lower[key] = ci_true.lower
        result.cumulative_incidence_true_upper[key] = ci_true.upper
        result.cumulative_incidence_fitted[key] = ci_fitted.mean
        result.cumulative_incidence_fitted_lower[key] = ci_fitted.lower
        result.cumulative_incidence_fitted_upper[key] = ci_fitted.upper
    end
    
    # Save to cache for reports
    save_longtest_result(result; force=true)
    
    return result
end

"""
    _compute_phasetype_prevalence(paths, phases, times)

Compute prevalence for a set of phases at each time point.
Used to collapse phase-level prevalence to observed state prevalence.

Returns (mean, lower, upper) vectors.
"""
function _compute_phasetype_prevalence(paths::Vector, phases::Vector{Int}, times::Vector{Float64})
    n_paths = length(paths)
    n_times = length(times)
    
    prev = zeros(n_times)
    
    for (i, t) in enumerate(times)
        n_in_phases = 0
        for path in paths
            # Find state at time t
            idx = searchsortedlast(path.times, t)
            if idx >= 1 && path.states[idx] in phases
                n_in_phases += 1
            end
        end
        prev[i] = n_in_phases / n_paths
    end
    
    # Normal approximation CI
    se = sqrt.(prev .* (1 .- prev) ./ n_paths)
    lower = max.(0.0, prev .- 1.96 .* se)
    upper = min.(1.0, prev .+ 1.96 .* se)
    
    return (mean=prev, lower=lower, upper=upper)
end

"""
    _compute_phasetype_cumincid(paths, from_phases, to_phases, times)

Compute cumulative incidence for transitions between phase sets.
Tracks the first time a path transitions from any phase in from_phases
to any phase in to_phases.

Returns (mean, lower, upper) vectors.
"""
function _compute_phasetype_cumincid(paths::Vector, from_phases::Vector{Int}, 
                                      to_phases::Vector{Int}, times::Vector{Float64})
    n_paths = length(paths)
    n_times = length(times)
    
    ci = zeros(n_times)
    
    for (i, t) in enumerate(times)
        n_transitioned = 0
        for path in paths
            # Check if transition from any from_phase to any to_phase occurred by time t
            for j in 2:length(path.states)
                if path.states[j-1] in from_phases && path.states[j] in to_phases
                    if path.times[j] <= t
                        n_transitioned += 1
                        break
                    end
                end
            end
        end
        ci[i] = n_transitioned / n_paths
    end
    
    # Normal approximation CI
    se = sqrt.(ci .* (1 .- ci) ./ n_paths)
    lower = max.(0.0, ci .- 1.96 .* se)
    upper = min.(1.0, ci .+ 1.96 .* se)
    
    return (mean=ci, lower=lower, upper=upper)
end

"""
    _compute_phasetype_observed_prevalence(data::DataFrame, times::Vector{Float64}, 
                                           n_observed_states::Int, phase_to_state::Vector{Int})

Compute observed state prevalence from phase-space data by collapsing phases to observed states.

For phase-type models, the fitting data is in expanded phase space (phases 1, 2, 3, ...).
This function maps each phase to its observed state using phase_to_state and computes
the proportion of subjects in each observed state at each time point.

# Arguments
- `data`: DataFrame with columns id, tstart, tstop, statefrom, stateto (in PHASE space)
- `times`: Time points at which to evaluate prevalence
- `n_observed_states`: Number of observed (collapsed) states
- `phase_to_state`: Vector mapping phase index → observed state index

# Returns
Dict mapping observed state (Int) to Vector of prevalence values at each time.
"""
function _compute_phasetype_observed_prevalence(data::DataFrame, times::Vector{Float64}, 
                                                 n_observed_states::Int, phase_to_state::Vector{Int})
    result = Dict{Int, Vector{Float64}}()
    
    all_ids = unique(data.id)
    n_subjects = length(all_ids)
    
    for obs_state in 1:n_observed_states
        prev = zeros(length(times))
        
        for (i, t) in enumerate(times)
            n_in_state = 0
            
            for id in all_ids
                subject_data = filter(row -> row.id == id, data)
                
                # Find the phase at time t
                current_phase = nothing
                
                for row in eachrow(subject_data)
                    if row.tstart <= t < row.tstop
                        current_phase = row.statefrom
                        break
                    elseif t >= row.tstop
                        current_phase = row.stateto
                    end
                end
                
                # Map phase to observed state and count
                if !isnothing(current_phase) && current_phase <= length(phase_to_state)
                    current_obs_state = phase_to_state[current_phase]
                    if current_obs_state == obs_state
                        n_in_state += 1
                    end
                end
            end
            
            prev[i] = n_in_state / n_subjects
        end
        result[obs_state] = prev
    end
    
    return result
end

"""
    _compute_phasetype_observed_cumincid(data, from_state, to_state, times, phase_to_state)

Compute crude observed cumulative incidence from phase-space data by collapsing to observed states.

For phase-type models, the fitting data is in expanded phase space. This function maps phases
to observed states and tracks the FIRST direct transition from from_state to to_state.

For panel data, a direct transition is recorded when the observed state at tstart is from_state
and the observed state at tstop is to_state. This means if a subject goes 1→2→3 between 
observations but only state 1 and state 3 are observed, it appears as a 1→3 transition.

For illness-death models with both 1→3 and 2→3 transitions, we track:
- 1→2: First observation interval where obs_from=1 and obs_to=2
- 1→3: First observation interval where obs_from=1 and obs_to=3 (direct observation)
- 2→3: First observation interval where obs_from=2 and obs_to=3

# Arguments
- `data`: DataFrame with columns id, tstart, tstop, statefrom, stateto (in PHASE space)
- `from_state`: Starting observed state
- `to_state`: Target observed state
- `times`: Time points at which to evaluate cumulative incidence
- `phase_to_state`: Vector mapping phase index → observed state index

# Returns
Vector of cumulative incidence values at each time point.
"""
function _compute_phasetype_observed_cumincid(data::DataFrame, from_state::Int, to_state::Int,
                                               times::Vector{Float64}, phase_to_state::Vector{Int})
    all_ids = unique(data.id)
    n_subjects = length(all_ids)
    
    ci = zeros(length(times))
    
    for (i, t) in enumerate(times)
        n_transitioned = 0
        
        for id in all_ids
            subject_data = filter(row -> row.id == id, data)
            subject_data = sort(subject_data, :tstart)  # Ensure chronological order
            
            # Track subject's observed state history to find first entry to to_state from from_state
            # For panel data: we see (state at t1, state at t2) for each interval
            transitioned = false
            
            for row in eachrow(subject_data)
                if row.tstop > t
                    break  # Haven't reached this time yet
                end
                
                # Map phases to observed states
                phase_from = row.statefrom
                phase_to = row.stateto
                
                # Handle phase indices that might be out of bounds
                if phase_from > length(phase_to_state) || phase_to > length(phase_to_state)
                    continue
                end
                
                obs_from = phase_to_state[phase_from]
                obs_to = phase_to_state[phase_to]
                
                # Check if this is the target transition in observed state space
                if obs_from == from_state && obs_to == to_state && obs_from != obs_to
                    transitioned = true
                    break  # Found first transition
                end
            end
            
            if transitioned
                n_transitioned += 1
            end
        end
        
        ci[i] = n_transitioned / n_subjects
    end
    
    return ci
end