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

const CACHE_DIR = normpath(joinpath(@__DIR__, "..", "cache"))
const PACKAGE_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

export load_test_cache, get_cache_status, cache_status_badge, 
       results_to_dataframe, get_git_info, get_category_result

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

end # module
