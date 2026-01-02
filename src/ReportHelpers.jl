# =============================================================================
# Report Helpers
# =============================================================================
# 
# Helper functions for Quarto reports to read cached test results
# and display status information.
#
# This module reads from the cache created by scripts/test_cache.jl
# =============================================================================

module ReportHelpers

using JSON3
using Dates
using DataFrames
using CairoMakie

const CACHE_DIR = normpath(joinpath(@__DIR__, "..", "cache"))
const LONGTEST_RESULTS_DIR = normpath(joinpath(@__DIR__, "..", "cache", "longtest_results"))
const PACKAGE_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

export load_test_cache, get_cache_status, cache_status_badge, 
       results_to_dataframe, get_git_info, get_category_result,
       # Long test result exports
       load_longtest_result, list_longtest_results,
       plot_parameter_recovery, plot_state_prevalence, plot_cumulative_incidence,
       longtest_result_to_parameter_table,
       # New exports for comprehensive long tests
       plot_data_summary, plot_validation_panel, format_test_name,
       create_test_inventory_df

# =============================================================================
# Cache Loading
# =============================================================================

"""
    load_test_cache() -> Union{JSON3.Object, Nothing}

Load the test cache JSON file. Returns nothing if cache doesn't exist.
"""
function load_test_cache()
    cache_path = joinpath(CACHE_DIR, "test_cache.json")
    if !isfile(cache_path)
        return nothing
    end
    try
        return JSON3.read(read(cache_path, String))
    catch e
        @warn "Failed to load test cache: $e"
        return nothing
    end
end

"""
    get_cache_status() -> NamedTuple

Get a summary of the cache status for display in reports.
"""
function get_cache_status()
    cache = load_test_cache()
    
    if isnothing(cache)
        return (
            exists = false,
            last_updated = "never",
            git_commit = "unknown",
            git_branch = "unknown",
            total_passed = 0,
            total_failed = 0,
            total_errors = 0,
            categories = Symbol[],
            is_stale = true
        )
    end
    
    # Current git state
    current_hash = get_current_git_hash()
    cache_hash = get(cache, :git_commit, "unknown")
    
    # Sum up results
    results = get(cache, :test_results, Dict())
    total_passed = sum(get(r, :passed, 0) for r in values(results); init=0)
    total_failed = sum(get(r, :failed, 0) for r in values(results); init=0)
    total_errors = sum(get(r, :errors, 0) for r in values(results); init=0)
    
    return (
        exists = true,
        last_updated = get(cache, :last_updated, "unknown"),
        git_commit = cache_hash,
        git_branch = get(cache, :git_branch, "unknown"),
        current_hash = current_hash,
        total_passed = total_passed,
        total_failed = total_failed,
        total_errors = total_errors,
        categories = collect(Symbol.(keys(results))),
        is_stale = cache_hash != current_hash
    )
end

"""
    get_current_git_hash() -> String

Get the current short git hash.
"""
function get_current_git_hash()
    try
        cd(PACKAGE_ROOT) do
            strip(read(`git rev-parse --short HEAD`, String))
        end
    catch
        "unknown"
    end
end

"""
    get_git_info() -> NamedTuple

Get git information for display in reports.
"""
function get_git_info()
    try
        cd(PACKAGE_ROOT) do
            hash = strip(read(`git rev-parse --short HEAD`, String))
            branch = strip(read(`git rev-parse --abbrev-ref HEAD`, String))
            # Get list of changed files since last commit
            changed = strip(read(`git diff --name-only HEAD`, String))
            changed_files = isempty(changed) ? String[] : split(changed, '\n')
            
            return (
                hash = hash,
                branch = branch,
                changed_files = changed_files,
                has_uncommitted = !isempty(changed_files)
            )
        end
    catch e
        return (
            hash = "unknown",
            branch = "unknown",
            changed_files = String[],
            has_uncommitted = false
        )
    end
end

# =============================================================================
# Report Display Helpers
# =============================================================================

"""
    cache_status_badge() -> String

Returns a Markdown badge showing cache status.
"""
function cache_status_badge()
    status = get_cache_status()
    
    if !status.exists
        return "🔴 **No cache** - run tests and record results"
    elseif status.is_stale
        return "🟡 **Cache stale** ($(status.git_commit) → $(status.current_hash))"
    else
        return "🟢 **Cache current** ($(status.git_commit), updated $(status.last_updated))"
    end
end

"""
    results_to_dataframe() -> DataFrame

Convert cached test results to a DataFrame for display.
"""
function results_to_dataframe()
    cache = load_test_cache()
    
    if isnothing(cache)
        return DataFrame(
            Category = Symbol[],
            Passed = Int[],
            Failed = Int[],
            Errors = Int[],
            Timestamp = String[],
            Status = String[]
        )
    end
    
    results = get(cache, :test_results, Dict())
    rows = []
    
    for (cat, result) in pairs(results)
        passed = get(result, :passed, 0)
        failed = get(result, :failed, 0)
        errors = get(result, :errors, 0)
        
        status = if errors > 0
            "❌ Error"
        elseif failed > 0
            "⚠️ Failed"
        else
            "✅ Pass"
        end
        
        push!(rows, (
            Category = Symbol(cat),
            Passed = passed,
            Failed = failed,
            Errors = errors,
            Timestamp = get(result, :timestamp, "unknown"),
            Status = status
        ))
    end
    
    if isempty(rows)
        return DataFrame(
            Category = Symbol[],
            Passed = Int[],
            Failed = Int[],
            Errors = Int[],
            Timestamp = String[],
            Status = String[]
        )
    end
    
    return DataFrame(rows)
end

"""
    get_category_result(category::Symbol) -> Union{NamedTuple, Nothing}

Get results for a specific test category.
"""
function get_category_result(category::Symbol)
    cache = load_test_cache()
    isnothing(cache) && return nothing
    
    results = get(cache, :test_results, Dict())
    cat_str = String(category)
    
    if haskey(results, cat_str)
        r = results[cat_str]
        return (
            passed = get(r, :passed, 0),
            failed = get(r, :failed, 0),
            errors = get(r, :errors, 0),
            timestamp = get(r, :timestamp, "unknown"),
            git_commit = get(r, :git_commit, "unknown")
        )
    end
    return nothing
end

# =============================================================================
# Long Test Result Loading and Plotting
# =============================================================================

"""
    list_longtest_results() -> Vector{String}

List all available long test results in the cache.
"""
function list_longtest_results()
    if !isdir(LONGTEST_RESULTS_DIR)
        return String[]
    end
    files = filter(f -> endswith(f, ".json"), readdir(LONGTEST_RESULTS_DIR))
    return [replace(f, ".json" => "") for f in files]
end

"""
    load_longtest_result(test_name::String) -> Union{JSON3.Object, Nothing}

Load a long test result from the cache.
"""
function load_longtest_result(test_name::String)
    filepath = joinpath(LONGTEST_RESULTS_DIR, "$(test_name).json")
    if !isfile(filepath)
        return nothing
    end
    try
        return JSON3.read(read(filepath, String))
    catch e
        @warn "Failed to load longtest result for $test_name: $e"
        return nothing
    end
end

"""
    longtest_result_to_parameter_table(result) -> DataFrame

Convert a long test result to a parameter comparison DataFrame.
Handles missing/null values gracefully by converting to NaN.
"""
function longtest_result_to_parameter_table(result)
    true_params = result[:true_params]
    est_params = result[:estimated_params]
    ses = result[:standard_errors]
    ci_lo = result[:ci_lower]
    ci_hi = result[:ci_upper]
    
    params = sort(collect(String.(keys(true_params))))
    
    df = DataFrame(
        Parameter = String[],
        True = Float64[],
        Estimated = Float64[],
        SE = Float64[],
        CI_Lower = Float64[],
        CI_Upper = Float64[],
        RelError_pct = Float64[],
        Covered = String[]
    )
    
    # Helper to safely convert to Float64, handling Nothing/null
    _safe_float(x) = isnothing(x) ? NaN : Float64(x)
    
    for p in params
        true_val = _safe_float(true_params[Symbol(p)])
        est_val = _safe_float(est_params[Symbol(p)])
        se = _safe_float(ses[Symbol(p)])
        ci_l = _safe_float(ci_lo[Symbol(p)])
        ci_h = _safe_float(ci_hi[Symbol(p)])
        
        # Relative error (percentage) - handle NaN
        rel_err = if isnan(true_val) || isnan(est_val)
            NaN
        elseif abs(true_val) > 0.01
            abs(est_val - true_val) / abs(true_val) * 100
        else
            abs(est_val - true_val) * 100
        end
        
        # Coverage check - handle NaN
        covered = if isnan(ci_l) || isnan(ci_h) || isnan(true_val)
            "?"
        elseif ci_l <= true_val <= ci_h
            "✓"
        else
            "✗"
        end
        
        push!(df, (p, true_val, est_val, se, ci_l, ci_h, rel_err, covered))
    end
    
    return df
end

"""
    plot_state_prevalence(result; figsize=(800, 400)) -> Figure

Plot state prevalence over time comparing true vs fitted model simulations.
Returns a CairoMakie Figure.

Colors: Red = true model, Blue = fitted model
"""
function plot_state_prevalence(result; figsize=(800, 400))
    if !haskey(result, :prevalence_times) || isnothing(result[:prevalence_times])
        return nothing
    end
    
    if !haskey(result, :prevalence_true) || isnothing(result[:prevalence_true])
        return nothing
    end

    times = collect(Float64, result[:prevalence_times])
    if isempty(times)
        return nothing
    end

    n_states = result[:n_states]
    
    fig = Figure(size=figsize)
    
    for s in 1:n_states
        ax = Axis(fig[1, s], 
            title = "State $s",
            xlabel = s == 2 ? "Time" : "",
            ylabel = s == 1 ? "Prevalence" : ""
        )
        
        sk = string(s)
        
        # Check if data exists for this state
        has_plots = false
        if haskey(result[:prevalence_true], sk)
            # True model (red)
            prev_true = collect(Float64, result[:prevalence_true][sk])
            
            if !isempty(prev_true)
                prev_true_lo = collect(Float64, result[:prevalence_true_lower][sk])
                prev_true_hi = collect(Float64, result[:prevalence_true_upper][sk])
                
                band!(ax, times, prev_true_lo, prev_true_hi, color=(:red, 0.2))
                lines!(ax, times, prev_true, color=:red, linewidth=2, label="True")
                has_plots = true
            end
            
            # Fitted model (blue)
            if haskey(result, :prevalence_fitted) && haskey(result[:prevalence_fitted], sk)
                prev_fit = collect(Float64, result[:prevalence_fitted][sk])
                if !isempty(prev_fit)
                    prev_fit_lo = collect(Float64, result[:prevalence_fitted_lower][sk])
                    prev_fit_hi = collect(Float64, result[:prevalence_fitted_upper][sk])
                    
                    band!(ax, times, prev_fit_lo, prev_fit_hi, color=(:blue, 0.2))
                    lines!(ax, times, prev_fit, color=:blue, linewidth=2, linestyle=:dash, label="Fitted")
                end
            end
        end
        
        ylims!(ax, 0, 1)
        
        if s == n_states && has_plots
            axislegend(ax, position=:rt)
        end
    end
    
    return fig
end

"""
    plot_cumulative_incidence(result; transitions=[(1,2), (2,3)], figsize=(700, 300)) -> Figure

Plot cumulative incidence over time comparing true vs fitted model simulations.
Returns a CairoMakie Figure.

Note: Only transitions 1→2 and 2→3 are shown (no 1→3 for progressive models).

Colors: Red = true model, Blue = fitted model
"""
function plot_cumulative_incidence(result; transitions=[(1,2), (2,3)], figsize=(700, 300))
    if !haskey(result, :cumulative_incidence_times)
        return nothing
    end

    times = collect(Float64, result[:cumulative_incidence_times])
    
    fig = Figure(size=figsize)
    
    for (i, (from, to)) in enumerate(transitions)
        ax = Axis(fig[1, i], 
            title = "$from → $to",
            xlabel = "Time",
            ylabel = i == 1 ? "Cumulative Incidence" : ""
        )
        
        key = "$from→$to"
        
        # Check if this transition exists in results
        if !haskey(result[:cumulative_incidence_true], key)
            continue
        end
        
        # True model (red)
        ci_true = collect(Float64, result[:cumulative_incidence_true][key])
        ci_true_lo = collect(Float64, result[:cumulative_incidence_true_lower][key])
        ci_true_hi = collect(Float64, result[:cumulative_incidence_true_upper][key])
        
        band!(ax, times, ci_true_lo, ci_true_hi, color=(:red, 0.2))
        lines!(ax, times, ci_true, color=:red, linewidth=2, label="True")
        
        # Fitted model (blue)
        ci_fit = collect(Float64, result[:cumulative_incidence_fitted][key])
        ci_fit_lo = collect(Float64, result[:cumulative_incidence_fitted_lower][key])
        ci_fit_hi = collect(Float64, result[:cumulative_incidence_fitted_upper][key])
        
        band!(ax, times, ci_fit_lo, ci_fit_hi, color=(:blue, 0.2))
        lines!(ax, times, ci_fit, color=:blue, linewidth=2, linestyle=:dash, label="Fitted")
        
        ylims!(ax, 0, 1)
        
        if i == length(transitions)
            axislegend(ax, position=:rb)
        end
    end
    
    return fig
end

"""
    plot_parameter_recovery(result; figsize=(600, 400)) -> Figure

Plot parameter recovery: true vs estimated with confidence intervals.
"""
function plot_parameter_recovery(result; figsize=(600, 400))
    true_params = result[:true_params]
    est_params = result[:estimated_params]
    ci_lo = result[:ci_lower]
    ci_hi = result[:ci_upper]
    
    params = sort(collect(String.(keys(true_params))))
    n_params = length(params)
    
    fig = Figure(size=figsize)
    ax = Axis(fig[1, 1], 
        title = "Parameter Recovery",
        xlabel = "True Value",
        ylabel = "Estimated Value"
    )
    
    # Get all values for axis limits
    all_true = [Float64(true_params[Symbol(p)]) for p in params]
    all_est = [Float64(est_params[Symbol(p)]) for p in params]
    all_lo = [Float64(ci_lo[Symbol(p)]) for p in params]
    all_hi = [Float64(ci_hi[Symbol(p)]) for p in params]
    
    # Identity line
    min_val = min(minimum(all_true), minimum(all_lo))
    max_val = max(maximum(all_true), maximum(all_hi))
    margin = 0.1 * (max_val - min_val)
    
    lines!(ax, [min_val - margin, max_val + margin], [min_val - margin, max_val + margin], 
           color=:gray, linestyle=:dash, linewidth=1)
    
    # Points with error bars
    for (i, p) in enumerate(params)
        true_val = Float64(true_params[Symbol(p)])
        est_val = Float64(est_params[Symbol(p)])
        lo = Float64(ci_lo[Symbol(p)])
        hi = Float64(ci_hi[Symbol(p)])
        
        # Vertical error bar (CI on estimated)
        lines!(ax, [true_val, true_val], [lo, hi], color=:blue, linewidth=2)
        scatter!(ax, [true_val], [est_val], color=:blue, markersize=10)
        
        # Label
        text!(ax, true_val, hi + 0.02 * (max_val - min_val), 
              text=replace(p, "_" => "\n"), fontsize=8, align=(:center, :bottom))
    end
    
    return fig
end

# =============================================================================
# New Long Test Report Functions
# =============================================================================

"""
    plot_data_summary(result; figsize=(600, 300)) -> Figure

Plot data summary: state counts and transition counts.
"""
function plot_data_summary(result; figsize=(600, 300))
    fig = Figure(size=figsize)
    
    data_summary = get(result, :data_summary, Dict())
    
    # State counts at end of observation
    ax1 = Axis(fig[1, 1], 
        title = "Final State Distribution",
        xlabel = "State",
        ylabel = "Count"
    )
    
    state_counts = Int[]
    state_labels = String[]
    for s in 1:3
        key = "state_$(s)_count"
        if haskey(data_summary, key)
            push!(state_counts, Int(data_summary[key]))
            push!(state_labels, "State $s")
        end
    end
    
    if !isempty(state_counts)
        barplot!(ax1, 1:length(state_counts), state_counts, 
                 color=[:steelblue, :orange, :green][1:length(state_counts)])
        ax1.xticks = (1:length(state_counts), state_labels)
    end
    
    # Transition counts
    ax2 = Axis(fig[1, 2], 
        title = "Observed Transitions",
        xlabel = "Transition",
        ylabel = "Count"
    )
    
    trans_counts = Int[]
    trans_labels = String[]
    for (from, to) in [(1, 2), (2, 3)]
        key = "trans_$(from)_$(to)"
        if haskey(data_summary, key)
            push!(trans_counts, Int(data_summary[key]))
            push!(trans_labels, "$from→$to")
        end
    end
    
    if !isempty(trans_counts)
        barplot!(ax2, 1:length(trans_counts), trans_counts,
                 color=[:steelblue, :orange][1:length(trans_counts)])
        ax2.xticks = (1:length(trans_counts), trans_labels)
    end
    
    return fig
end

"""
    plot_validation_panel(result; figsize=(900, 600)) -> Figure

Create a 2x2 validation panel with prevalence (3 states) and cumulative incidence.
"""
function plot_validation_panel(result; figsize=(900, 600))
    if !haskey(result, :prevalence_times) || !haskey(result, :cumulative_incidence_times)
        return nothing
    end

    times_prev = collect(Float64, result[:prevalence_times])
    times_ci = collect(Float64, result[:cumulative_incidence_times])
    
    if isempty(times_prev) || isempty(times_ci)
        return nothing
    end
    
    fig = Figure(size=figsize)
    
    # Row 1: Prevalence for states 1, 2, 3
    for s in 1:3
        ax = Axis(fig[1, s], 
            title = "State $s Prevalence",
            xlabel = "",
            ylabel = s == 1 ? "Prevalence" : ""
        )
        
        sk = string(s)
        
        has_plots = false
        if haskey(result, :prevalence_true) && haskey(result[:prevalence_true], sk)
            # True model (red)
            prev_true = collect(Float64, result[:prevalence_true][sk])
            
            if !isempty(prev_true)
                prev_true_lo = collect(Float64, result[:prevalence_true_lower][sk])
                prev_true_hi = collect(Float64, result[:prevalence_true_upper][sk])
                
                band!(ax, times_prev, prev_true_lo, prev_true_hi, color=(:red, 0.2))
                lines!(ax, times_prev, prev_true, color=:red, linewidth=2, label="True")
                has_plots = true
            end
            
            # Fitted model (blue)
            if haskey(result, :prevalence_fitted) && haskey(result[:prevalence_fitted], sk)
                prev_fit = collect(Float64, result[:prevalence_fitted][sk])
                if !isempty(prev_fit)
                    prev_fit_lo = collect(Float64, result[:prevalence_fitted_lower][sk])
                    prev_fit_hi = collect(Float64, result[:prevalence_fitted_upper][sk])
                    
                    band!(ax, times_prev, prev_fit_lo, prev_fit_hi, color=(:blue, 0.2))
                    lines!(ax, times_prev, prev_fit, color=:blue, linewidth=2, linestyle=:dash, label="Fitted")
                end
            end
        end
        
        ylims!(ax, 0, 1)
        if s == 3 && has_plots
            axislegend(ax, position=:rt)
        end
    end
    
    # Row 2: Cumulative incidence for 1→2 and 2→3
    for (i, (from, to)) in enumerate([(1, 2), (2, 3)])
        ax = Axis(fig[2, i], 
            title = "Cumulative Incidence: $from→$to",
            xlabel = "Time",
            ylabel = i == 1 ? "Cumulative Incidence" : ""
        )
        
        key = "$from→$to"
        
        has_ci_plots = false
        if haskey(result, :cumulative_incidence_true) && haskey(result[:cumulative_incidence_true], key)
            # True model (red)
            ci_true = collect(Float64, result[:cumulative_incidence_true][key])
            
            if !isempty(ci_true)
                ci_true_lo = collect(Float64, result[:cumulative_incidence_true_lower][key])
                ci_true_hi = collect(Float64, result[:cumulative_incidence_true_upper][key])
                
                band!(ax, times_ci, ci_true_lo, ci_true_hi, color=(:red, 0.2))
                lines!(ax, times_ci, ci_true, color=:red, linewidth=2, label="True")
                has_ci_plots = true
            end
            
            # Fitted model (blue)
            if haskey(result, :cumulative_incidence_fitted) && haskey(result[:cumulative_incidence_fitted], key)
                ci_fit = collect(Float64, result[:cumulative_incidence_fitted][key])
                if !isempty(ci_fit)
                    ci_fit_lo = collect(Float64, result[:cumulative_incidence_fitted_lower][key])
                    ci_fit_hi = collect(Float64, result[:cumulative_incidence_fitted_upper][key])
                    
                    band!(ax, times_ci, ci_fit_lo, ci_fit_hi, color=(:blue, 0.2))
                    lines!(ax, times_ci, ci_fit, color=:blue, linewidth=2, linestyle=:dash, label="Fitted")
                end
            end
        end
        
        ylims!(ax, 0, 1)
        if i == 2 && has_ci_plots
            axislegend(ax, position=:rb)
        end
    end
    
    # Add label for third column of row 2 (empty, for layout symmetry)
    Label(fig[2, 3], "", tellwidth=false, tellheight=false)
    
    return fig
end

"""
    format_test_name(result) -> String

Format test name for display (e.g., "Exponential - Exact Data - No Covariates").
"""
function format_test_name(result)
    family_names = Dict(
        "exp" => "Exponential",
        "wei" => "Weibull", 
        "gom" => "Gompertz",
        "pt" => "Phase-Type",
        "sp" => "Spline"
    )
    
    data_type_names = Dict(
        "exact" => "Exact Data",
        "panel" => "Panel Data (Markov)",
        "mcem" => "Panel Data (MCEM)"
    )
    
    cov_type_names = Dict(
        "nocov" => "No Covariates",
        "fixed" => "Fixed Covariates",
        "tvc" => "Time-Varying Covariates"
    )
    
    family = get(family_names, get(result, :hazard_family, ""), "Unknown")
    datatype = get(data_type_names, get(result, :data_type, ""), "Unknown")
    covtype = get(cov_type_names, get(result, :covariate_type, ""), "Unknown")
    
    return "$family - $datatype - $covtype"
end

"""
    create_test_inventory_df(results_dir::String=LONGTEST_RESULTS_DIR) -> DataFrame

Create a DataFrame with inventory of all long test results.
"""
function create_test_inventory_df(results_dir::String=LONGTEST_RESULTS_DIR)
    df = DataFrame(
        TestName = String[],
        Family = String[],
        DataType = String[],
        Covariates = String[],
        NSubjects = Int[],
        Status = String[],
        Timestamp = String[]
    )
    
    if !isdir(results_dir)
        return df
    end
    
    for f in filter(f -> endswith(f, ".json"), readdir(results_dir))
        filepath = joinpath(results_dir, f)
        try
            result = JSON3.read(read(filepath, String))
            
            passed = get(result, :passed, get(result, :all_tests_passed, false))
            status = passed ? "✅ Pass" : "❌ Fail"
            
            push!(df, (
                replace(f, ".json" => ""),
                get(result, :hazard_family, ""),
                get(result, :data_type, ""),
                get(result, :covariate_type, ""),
                get(result, :n_subjects, 0),
                status,
                get(result, :timestamp, "")
            ))
        catch e
            @warn "Failed to read $f: $e"
        end
    end
    
    # Sort by family, then data type, then covariate type
    sort!(df, [:Family, :DataType, :Covariates])
    
    return df
end

end # module
