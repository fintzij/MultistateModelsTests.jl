# =============================================================================
# Long Test Results Infrastructure
# =============================================================================
#
# Standardized result storage and loading for long test reports.
# Results are saved to cache/longtest_results/<test_name>.json
#
# Naming convention: {family}_{datatype}_{covtype}.json
#   - family: exp, wei, gom, pt, sp
#   - datatype: exact, panel, mcem
#   - covtype: nocov, fixed, tvc
#
# This module provides:
# - LongTestResult struct for standardized result format
# - save_longtest_result() for saving results from tests
# - load_longtest_result() for loading in reports
# - Utility functions for cumulative incidence and prevalence computation
# =============================================================================

# Note: JSON3, Dates, and DataFrames are imported by the parent module (MultistateModelsTests)

export LongTestResult, save_longtest_result, load_longtest_result,
       compute_cumulative_incidence, compute_state_prevalence,
       list_available_results, compute_observed_prevalence,
       result_to_parameter_table, get_results_by_category,
       summarize_data

const LONGTEST_RESULTS_DIR = normpath(joinpath(@__DIR__, "..", "cache", "longtest_results"))

# Valid values for categorical fields
const VALID_FAMILIES = ["exp", "wei", "gom", "pt", "sp"]
const VALID_DATA_TYPES = ["exact", "panel", "mcem"]
const VALID_COV_TYPES = ["nocov", "fixed", "tvc"]

# =============================================================================
# Result Structure
# =============================================================================

"""
    LongTestResult

Standardized result format for long test results.

## Fields
### Metadata
- `test_name`: Unique identifier (e.g., "exp_exact_nocov")
- `test_description`: Human-readable description
- `hazard_family`: One of "exp", "wei", "gom", "pt", "sp"
- `data_type`: One of "exact", "panel", "mcem"
- `covariate_type`: One of "nocov", "fixed", "tvc"
- `passed`: Whether the test passed parameter recovery checks

### Configuration
- `n_subjects`: Number of subjects (should be 1000)
- `n_simulations`: Number of simulation replicates for comparison
- `n_states`: Number of states (should be 3)

### Parameters (keyed by parameter name)
- `true_params`: True parameter values (estimation scale)
- `estimated_params`: Estimated parameter values
- `standard_errors`: Standard errors
- `ci_lower`, `ci_upper`: 95% confidence interval bounds

### Data Summary (for visualization)
- `data_summary`: Dict with observation counts, state occupancy, etc.

### Simulation Comparison
- Cumulative incidence: For transitions 1→2 and 2→3 only
- State prevalence: For states 1, 2, 3
"""
Base.@kwdef mutable struct LongTestResult
    # Metadata
    test_name::String
    test_description::String = ""
    timestamp::String = string(now())
    git_commit::String = ""
    git_branch::String = ""
    
    # Test categorization (NEW)
    hazard_family::String = ""      # exp, wei, gom, pt, sp
    data_type::String = ""          # exact, panel, mcem
    covariate_type::String = ""     # nocov, fixed, tvc
    passed::Bool = false            # Overall pass/fail
    
    # Model configuration
    n_subjects::Int = 1000
    n_simulations::Int = 1000
    n_states::Int = 3
    hazard_families::Vector{String} = String[]  # Legacy, use hazard_family instead
    
    # Parameter recovery (keyed by parameter name)
    true_params::Dict{String, Float64} = Dict{String, Float64}()
    estimated_params::Dict{String, Float64} = Dict{String, Float64}()
    standard_errors::Dict{String, Float64} = Dict{String, Float64}()
    ci_lower::Dict{String, Float64} = Dict{String, Float64}()
    ci_upper::Dict{String, Float64} = Dict{String, Float64}()
    param_passed::Dict{String, Bool} = Dict{String, Bool}()  # Per-param pass/fail
    
    # Data summary for visualization (NEW)
    # Contains: n_transitions_12, n_transitions_23, state_counts_at_times, etc.
    data_summary::Dict{String, Any} = Dict{String, Any}()
    
    # Simulation comparison - cumulative incidence
    # Keys are "1→2" and "2→3" only (NOT "1→3")
    cumulative_incidence_times::Vector{Float64} = Float64[]
    cumulative_incidence_true::Dict{String, Vector{Float64}} = Dict{String, Vector{Float64}}()
    cumulative_incidence_fitted::Dict{String, Vector{Float64}} = Dict{String, Vector{Float64}}()
    cumulative_incidence_true_lower::Dict{String, Vector{Float64}} = Dict{String, Vector{Float64}}()
    cumulative_incidence_true_upper::Dict{String, Vector{Float64}} = Dict{String, Vector{Float64}}()
    cumulative_incidence_fitted_lower::Dict{String, Vector{Float64}} = Dict{String, Vector{Float64}}()
    cumulative_incidence_fitted_upper::Dict{String, Vector{Float64}} = Dict{String, Vector{Float64}}()
    
    # Simulation comparison - state prevalence
    # Keys are "1", "2", "3"
    prevalence_times::Vector{Float64} = Float64[]
    prevalence_observed::Dict{String, Vector{Float64}} = Dict{String, Vector{Float64}}()
    prevalence_true::Dict{String, Vector{Float64}} = Dict{String, Vector{Float64}}()
    prevalence_fitted::Dict{String, Vector{Float64}} = Dict{String, Vector{Float64}}()
    prevalence_true_lower::Dict{String, Vector{Float64}} = Dict{String, Vector{Float64}}()
    prevalence_true_upper::Dict{String, Vector{Float64}} = Dict{String, Vector{Float64}}()
    prevalence_fitted_lower::Dict{String, Vector{Float64}} = Dict{String, Vector{Float64}}()
    prevalence_fitted_upper::Dict{String, Vector{Float64}} = Dict{String, Vector{Float64}}()
    
    # Test outcomes
    fitting_time_seconds::Float64 = 0.0
    all_tests_passed::Bool = false  # Legacy, use passed instead
    n_tests_passed::Int = 0
    n_tests_failed::Int = 0
    failure_messages::Vector{String} = String[]
end

# =============================================================================
# Git Utilities
# =============================================================================

function get_git_info()
    try
        cd(normpath(joinpath(@__DIR__, "..", ".."))) do
            hash = strip(read(`git rev-parse --short HEAD`, String))
            branch = strip(read(`git rev-parse --abbrev-ref HEAD`, String))
            return (hash=hash, branch=branch)
        end
    catch
        return (hash="unknown", branch="unknown")
    end
end

# =============================================================================
# Save/Load Functions
# =============================================================================

"""
    save_longtest_result(result::LongTestResult)

Save a long test result to JSON in the cache directory.
NaN values are converted to null for JSON compatibility.
"""
function save_longtest_result(result::LongTestResult)
    # Ensure directory exists
    mkpath(LONGTEST_RESULTS_DIR)
    
    # Add git info if not set
    if isempty(result.git_commit)
        git = get_git_info()
        result.git_commit = git.hash
        result.git_branch = git.branch
    end
    
    # Update timestamp
    result.timestamp = string(now())
    
    # Convert to dict and sanitize NaN values
    result_dict = _result_to_dict(result)
    
    # Save to JSON
    filepath = joinpath(LONGTEST_RESULTS_DIR, "$(result.test_name).json")
    open(filepath, "w") do io
        JSON3.pretty(io, result_dict)
    end
    
    return filepath
end

"""
    _sanitize_value(x)

Convert NaN to nothing for JSON compatibility.
"""
_sanitize_value(x::Float64) = isnan(x) ? nothing : x
_sanitize_value(x::Int) = x
_sanitize_value(x::Bool) = x
_sanitize_value(x::String) = x
_sanitize_value(x::Nothing) = nothing
_sanitize_value(x::Vector{Float64}) = [_sanitize_value(v) for v in x]
_sanitize_value(x::Vector{String}) = x
_sanitize_value(x::Vector) = [_sanitize_value(v) for v in x]
function _sanitize_value(x::Dict{String, T}) where T
    Dict{String, Any}(k => _sanitize_value(v) for (k, v) in x)
end
function _sanitize_value(x::Dict{String, Any})
    Dict{String, Any}(k => _sanitize_value(v) for (k, v) in x)
end

"""
    _result_to_dict(result::LongTestResult) -> Dict{String, Any}

Convert LongTestResult to a JSON-serializable dictionary.
"""
function _result_to_dict(result::LongTestResult)
    d = Dict{String, Any}()
    
    # String fields
    for field in [:test_name, :test_description, :timestamp, :git_commit, :git_branch,
                  :hazard_family, :data_type, :covariate_type]
        d[string(field)] = getfield(result, field)
    end
    
    # Numeric scalar fields
    d["n_subjects"] = result.n_subjects
    d["n_simulations"] = result.n_simulations
    d["n_states"] = result.n_states
    d["n_tests_passed"] = result.n_tests_passed
    d["n_tests_failed"] = result.n_tests_failed
    d["fitting_time_seconds"] = _sanitize_value(result.fitting_time_seconds)
    
    # Boolean fields
    d["passed"] = result.passed
    d["all_tests_passed"] = result.all_tests_passed
    
    # Vector fields
    d["hazard_families"] = result.hazard_families
    d["failure_messages"] = result.failure_messages
    d["cumulative_incidence_times"] = _sanitize_value(result.cumulative_incidence_times)
    d["prevalence_times"] = _sanitize_value(result.prevalence_times)
    
    # Dict{String, Float64} fields - sanitize NaN
    for field in [:true_params, :estimated_params, :standard_errors, :ci_lower, :ci_upper]
        d[string(field)] = _sanitize_value(getfield(result, field))
    end
    
    # Dict{String, Bool} fields
    d["param_passed"] = result.param_passed
    
    # Dict{String, Any} fields
    d["data_summary"] = _sanitize_value(result.data_summary)
    
    # Dict{String, Vector{Float64}} fields - sanitize NaN
    for field in [:cumulative_incidence_true, :cumulative_incidence_fitted,
                  :cumulative_incidence_true_lower, :cumulative_incidence_true_upper,
                  :cumulative_incidence_fitted_lower, :cumulative_incidence_fitted_upper,
                  :prevalence_observed, :prevalence_true, :prevalence_fitted,
                  :prevalence_true_lower, :prevalence_true_upper,
                  :prevalence_fitted_lower, :prevalence_fitted_upper]
        d[string(field)] = _sanitize_value(getfield(result, field))
    end
    
    return d
end

"""
    load_longtest_result(test_name::String) -> Union{LongTestResult, Nothing}

Load a long test result from the cache directory.
"""
function load_longtest_result(test_name::String)
    filepath = joinpath(LONGTEST_RESULTS_DIR, "$(test_name).json")
    if !isfile(filepath)
        return nothing
    end
    
    try
        json = JSON3.read(read(filepath, String))
        return _json_to_result(json)
    catch e
        @warn "Failed to load longtest result for $test_name: $e"
        return nothing
    end
end

"""
    list_available_results() -> Vector{String}

List all available long test results in the cache.
"""
function list_available_results()
    if !isdir(LONGTEST_RESULTS_DIR)
        return String[]
    end
    
    files = filter(f -> endswith(f, ".json"), readdir(LONGTEST_RESULTS_DIR))
    return [replace(f, ".json" => "") for f in files]
end

# Helper to convert JSON back to result struct
function _json_to_result(json)
    result = LongTestResult(test_name = string(get(json, :test_name, "unknown")))
    
    # Copy string fields
    for field in [:test_description, :timestamp, :git_commit, :git_branch,
                  :hazard_family, :data_type, :covariate_type]
        if haskey(json, field) && !isnothing(json[field])
            setfield!(result, field, string(json[field]))
        end
    end
    
    # Copy numeric scalar fields
    for field in [:n_subjects, :n_simulations, :n_states, :n_tests_passed, :n_tests_failed]
        if haskey(json, field) && !isnothing(json[field])
            setfield!(result, field, Int(json[field]))
        end
    end
    
    # Copy float fields
    for field in [:fitting_time_seconds]
        if haskey(json, field) && !isnothing(json[field])
            setfield!(result, field, Float64(json[field]))
        end
    end
    
    # Copy boolean fields
    for field in [:passed, :all_tests_passed]
        if haskey(json, field) && !isnothing(json[field])
            setfield!(result, field, Bool(json[field]))
        end
    end
    
    # Copy vector fields
    for field in [:hazard_families, :cumulative_incidence_times, :prevalence_times, :failure_messages]
        if haskey(json, field) && !isnothing(json[field])
            setfield!(result, field, collect(json[field]))
        end
    end
    
    # Copy dict fields (String -> Float64)
    for field in [:true_params, :estimated_params, :standard_errors, :ci_lower, :ci_upper]
        if haskey(json, field) && !isnothing(json[field])
            d = Dict{String, Float64}()
            for (k, v) in pairs(json[field])
                if !isnothing(v)
                    d[string(k)] = Float64(v)
                end
            end
            setfield!(result, field, d)
        end
    end
    
    # Copy dict fields (String -> Bool)
    for field in [:param_passed]
        if haskey(json, field) && !isnothing(json[field])
            d = Dict{String, Bool}()
            for (k, v) in pairs(json[field])
                if !isnothing(v)
                    d[string(k)] = Bool(v)
                end
            end
            setfield!(result, field, d)
        end
    end
    
    # Copy dict fields (String -> Vector{Float64})
    for field in [:cumulative_incidence_true, :cumulative_incidence_fitted,
                  :cumulative_incidence_true_lower, :cumulative_incidence_true_upper,
                  :cumulative_incidence_fitted_lower, :cumulative_incidence_fitted_upper,
                  :prevalence_observed, :prevalence_true, :prevalence_fitted,
                  :prevalence_true_lower, :prevalence_true_upper,
                  :prevalence_fitted_lower, :prevalence_fitted_upper]
        if haskey(json, field) && !isnothing(json[field])
            d = Dict{String, Vector{Float64}}()
            for (k, v) in pairs(json[field])
                if !isnothing(v)
                    d[string(k)] = collect(Float64, v)
                end
            end
            setfield!(result, field, d)
        end
    end
    
    # Copy data_summary (Dict{String, Any})
    if haskey(json, :data_summary) && !isnothing(json[:data_summary])
        d = Dict{String, Any}()
        for (k, v) in pairs(json[:data_summary])
            d[string(k)] = v
        end
        result.data_summary = d
    end
    
    return result
end

# =============================================================================
# Simulation Comparison Utilities
# =============================================================================

"""
    compute_cumulative_incidence(paths, from_state, to_state, times)

Compute cumulative incidence of transition from `from_state` to `to_state`.

Returns (mean, lower_95, upper_95) vectors over time.
"""
function compute_cumulative_incidence(paths::Vector, from_state::Int, to_state::Int, 
                                       times::Vector{Float64})
    n_paths = length(paths)
    n_times = length(times)
    
    # For each path, determine if/when the transition occurred
    ci = zeros(n_times)
    
    for (i, t) in enumerate(times)
        n_transitioned = 0
        for path in paths
            # Check if transition from_state → to_state occurred by time t
            for j in 2:length(path.states)
                if path.states[j-1] == from_state && path.states[j] == to_state
                    if path.times[j] <= t
                        n_transitioned += 1
                        break
                    end
                end
            end
        end
        ci[i] = n_transitioned / n_paths
    end
    
    # Bootstrap confidence intervals (simple percentile method)
    # For computational efficiency, use normal approximation
    se = sqrt.(ci .* (1 .- ci) ./ n_paths)
    lower = max.(0.0, ci .- 1.96 .* se)
    upper = min.(1.0, ci .+ 1.96 .* se)
    
    return (mean=ci, lower=lower, upper=upper)
end

"""
    compute_state_prevalence(paths, state, times)

Compute prevalence (proportion in state) at each time point.

Returns (mean, lower_95, upper_95) vectors over time.
"""
function compute_state_prevalence(paths::Vector, state::Int, times::Vector{Float64})
    n_paths = length(paths)
    n_times = length(times)
    
    prev = zeros(n_times)
    
    for (i, t) in enumerate(times)
        n_in_state = 0
        for path in paths
            # Find state at time t
            idx = searchsortedlast(path.times, t)
            if idx >= 1 && path.states[idx] == state
                n_in_state += 1
            end
        end
        prev[i] = n_in_state / n_paths
    end
    
    # Normal approximation CI
    se = sqrt.(prev .* (1 .- prev) ./ n_paths)
    lower = max.(0.0, prev .- 1.96 .* se)
    upper = min.(1.0, prev .+ 1.96 .* se)
    
    return (mean=prev, lower=lower, upper=upper)
end

"""
    compute_observed_prevalence(data::DataFrame, times::Vector{Float64}, n_states::Int)

Compute observed state prevalence from panel data.
"""
function compute_observed_prevalence(data::DataFrame, times::Vector{Float64}, n_states::Int)
    result = Dict{Int, Vector{Float64}}()
    
    for s in 1:n_states
        prev = zeros(length(times))
        for (i, t) in enumerate(times)
            # Find observations that contain time t
            relevant = filter(row -> row.tstart <= t < row.tstop, eachrow(data))
            if !isempty(relevant)
                # Use statefrom for observations containing t
                n_in_state = count(row -> row.statefrom == s, relevant)
                prev[i] = n_in_state / length(relevant)
            end
        end
        result[s] = prev
    end
    
    return result
end

# =============================================================================
# Parameter Table Helpers
# =============================================================================

"""
    result_to_parameter_table(result::LongTestResult) -> DataFrame

Convert a LongTestResult to a parameter comparison DataFrame.
"""
function result_to_parameter_table(result::LongTestResult)
    params = sort(collect(keys(result.true_params)))
    
    df = DataFrame(
        Parameter = String[],
        True = Float64[],
        Estimated = Float64[],
        SE = Float64[],
        CI_Lower = Float64[],
        CI_Upper = Float64[],
        RelError = Float64[],
        Covered = String[]
    )
    
    for p in params
        true_val = get(result.true_params, p, NaN)
        est_val = get(result.estimated_params, p, NaN)
        se = get(result.standard_errors, p, NaN)
        ci_lo = get(result.ci_lower, p, NaN)
        ci_hi = get(result.ci_upper, p, NaN)
        
        # Relative error
        rel_err = abs(true_val) > 0.01 ? 
            abs(est_val - true_val) / abs(true_val) * 100 :
            abs(est_val - true_val) * 100
        
        # Coverage check
        covered = (ci_lo <= true_val <= ci_hi) ? "✓" : "✗"
        
        push!(df, (p, true_val, est_val, se, ci_lo, ci_hi, rel_err, covered))
    end
    
    return df
end
"""
    get_results_by_category() -> Dict

Load all results and organize by hazard family.
Returns a Dict with keys being hazard families and values being vectors of results.
"""
function get_results_by_category()
    results_by_family = Dict{String, Vector{LongTestResult}}()
    
    for family in VALID_FAMILIES
        results_by_family[family] = LongTestResult[]
    end
    results_by_family["other"] = LongTestResult[]  # For results without proper category
    
    for test_name in list_available_results()
        result = load_longtest_result(test_name)
        if !isnothing(result)
            family = result.hazard_family
            if family in VALID_FAMILIES
                push!(results_by_family[family], result)
            else
                push!(results_by_family["other"], result)
            end
        end
    end
    
    # Sort each family's results by data_type then covariate_type
    for (family, results) in results_by_family
        sort!(results, by = r -> (r.data_type, r.covariate_type))
    end
    
    return results_by_family
end

"""
    get_test_inventory() -> DataFrame

Return a DataFrame summarizing all available test results.
"""
function get_test_inventory()
    df = DataFrame(
        Test = String[],
        Family = String[],
        DataType = String[],
        Covariates = String[],
        NSubjects = Int[],
        Passed = String[],
        Timestamp = String[]
    )
    
    for test_name in sort(list_available_results())
        result = load_longtest_result(test_name)
        if !isnothing(result)
            push!(df, (
                result.test_name,
                result.hazard_family,
                result.data_type,
                result.covariate_type,
                result.n_subjects,
                result.passed ? "✅" : "❌",
                result.timestamp
            ))
        end
    end
    
    return df
end

"""
    summarize_data(data::DataFrame) -> Dict{String, Any}

Compute data summary for visualization: state counts, transitions, etc.
"""
function summarize_data(data::DataFrame)
    summary = Dict{String, Any}()
    
    # Total observations
    summary["n_obs"] = nrow(data)
    summary["n_subjects"] = length(unique(data.id))
    
    # State counts at observation end times
    if hasproperty(data, :stateto)
        for s in unique(data.stateto)
            summary["state_$(s)_count"] = count(==(s), data.stateto)
        end
    end
    
    # Transition counts (for exact data with obstype == 1)
    exact_obs = filter(row -> row.obstype == 1, eachrow(data))
    if !isempty(exact_obs)
        for row in exact_obs
            if row.statefrom != row.stateto
                key = "trans_$(row.statefrom)_$(row.stateto)"
                summary[key] = get(summary, key, 0) + 1
            end
        end
    end
    
    # Observation times
    if hasproperty(data, :tstop)
        summary["max_time"] = maximum(data.tstop)
        summary["obs_times"] = sort(unique(data.tstop))
    end
    
    return summary
end