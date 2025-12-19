# =============================================================================
# Test Cache System (Simplified)
# =============================================================================
# 
# This module tracks which source files have changed since the last test run.
# It does NOT automatically run tests (that's slow and complex).
# Instead, it:
#   1. Tracks git hashes for source modules
#   2. Reports which test categories are stale
#   3. Stores test results when tests are run manually
#   4. Provides status for reports to display
#
# Usage:
#   # Check what's changed
#   julia> include("scripts/test_cache.jl")
#   julia> show_cache_status()
#   
#   # After running tests manually, record results
#   julia> record_test_result(:hazards, passed=45, failed=0, errors=0)
#   
#   # From reports, check if cache is fresh
#   julia> is_cache_fresh()
# =============================================================================

using JSON3
using Dates

const CACHE_DIR = joinpath(@__DIR__, "..", "cache")
const PACKAGE_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const TEST_ROOT = joinpath(@__DIR__, "..")

# =============================================================================
# Dependency Mapping
# =============================================================================

const DEPENDENCY_MAP = Dict(
    "src/hazard/" => [:hazards, :simulation, :splines],
    "src/types/" => [:hazards, :simulation, :modelgeneration],
    "src/inference/" => [:mcem, :sir, :variance],
    "src/phasetype/" => [:phasetype],
    "src/simulation/" => [:simulation],
    "src/likelihood/" => [:mcem, :hazards, :variance],
    "src/surrogate/" => [:mcem, :surrogates],
    "src/utilities/" => [:hazards, :simulation, :mcem, :initialization],
    "src/construction/" => [:modelgeneration, :hazards],
    "src/output/" => [:variance],
)

const ALL_CATEGORIES = [:hazards, :splines, :simulation, :mcem, :phasetype, 
                        :sir, :variance, :initialization, :modelgeneration, 
                        :surrogates, :helpers, :reconstructor]

# =============================================================================
# Git Operations
# =============================================================================

function get_current_commit()
    try
        cd(PACKAGE_ROOT) do
            strip(read(`git rev-parse --short HEAD`, String))
        end
    catch
        "unknown"
    end
end

function get_current_branch()
    try
        cd(PACKAGE_ROOT) do
            strip(read(`git rev-parse --abbrev-ref HEAD`, String))
        end
    catch
        "unknown"
    end
end

function get_file_hash(path::String)
    full_path = joinpath(PACKAGE_ROOT, path)
    if !ispath(full_path)
        return "missing"
    end
    
    try
        cd(PACKAGE_ROOT) do
            # Get hash of directory tree
            strip(read(`git rev-parse HEAD:$path`, String))
        end
    catch
        # Fallback to mtime-based hash
        if isdir(full_path)
            mtimes = [stat(joinpath(root, f)).mtime 
                      for (root, _, files) in walkdir(full_path) 
                      for f in files if endswith(f, ".jl")]
            return string(hash(mtimes))
        elseif isfile(full_path)
            return string(hash(stat(full_path).mtime))
        end
        return "unknown"
    end
end

function get_source_hashes()
    Dict(path => get_file_hash(path) for path in keys(DEPENDENCY_MAP))
end

# =============================================================================
# Cache Storage
# =============================================================================

function cache_path()
    joinpath(CACHE_DIR, "test_cache.json")
end

function load_cache()
    path = cache_path()
    if !isfile(path)
        return nothing
    end
    try
        # Parse JSON and convert to mutable Dict structure
        raw = JSON3.read(read(path, String))
        
        # Deep convert to standard Dict
        cache = Dict{Symbol,Any}(
            :source_hashes => Dict{String,String}(String(k) => String(v) for (k,v) in pairs(raw.source_hashes)),
            :test_results => Dict{Symbol,Any}(),
            :last_updated => String(raw.last_updated),
            :git_commit => String(get(raw, :git_commit, "unknown")),
            :git_branch => String(get(raw, :git_branch, "unknown"))
        )
        
        # Convert test results
        if haskey(raw, :test_results)
            for (cat, result) in pairs(raw.test_results)
                cache[:test_results][Symbol(cat)] = Dict{Symbol,Any}(
                    :passed => Int(result.passed),
                    :failed => Int(result.failed),
                    :errors => Int(result.errors),
                    :timestamp => String(result.timestamp),
                    :git_commit => String(get(result, :git_commit, "unknown"))
                )
            end
        end
        
        cache
    catch e
        @warn "Failed to load cache: $e"
        nothing
    end
end

function save_cache(cache)
    mkpath(CACHE_DIR)
    open(cache_path(), "w") do io
        JSON3.pretty(io, cache)
    end
end

function init_cache()
    Dict(
        :source_hashes => get_source_hashes(),
        :test_results => Dict{Symbol,Any}(),
        :last_updated => string(now()),
        :git_commit => get_current_commit(),
        :git_branch => get_current_branch()
    )
end

# =============================================================================
# Change Detection
# =============================================================================

function get_stale_categories()
    cache = load_cache()
    if isnothing(cache)
        return Set(ALL_CATEGORIES)  # All stale if no cache
    end
    
    current_hashes = get_source_hashes()
    old_hashes = cache[:source_hashes]
    
    stale = Set{Symbol}()
    for (path, new_hash) in current_hashes
        old_hash = get(old_hashes, path, "")
        if old_hash != new_hash
            categories = get(DEPENDENCY_MAP, path, Symbol[])
            union!(stale, categories)
        end
    end
    
    stale
end

function is_cache_fresh()
    isempty(get_stale_categories())
end

# =============================================================================
# Result Recording
# =============================================================================

function record_test_result(category::Symbol; passed::Int, failed::Int, errors::Int)
    cache = load_cache()
    if isnothing(cache)
        cache = init_cache()
    end
    
    cache[:test_results][category] = Dict(
        :passed => passed,
        :failed => failed,
        :errors => errors,
        :timestamp => string(now()),
        :git_commit => get_current_commit()
    )
    cache[:last_updated] = string(now())
    cache[:source_hashes] = get_source_hashes()
    cache[:git_commit] = get_current_commit()
    
    save_cache(cache)
    @info "Recorded results for :$category"
end

function update_source_hashes()
    cache = load_cache()
    if isnothing(cache)
        cache = init_cache()
    end
    cache[:source_hashes] = get_source_hashes()
    cache[:last_updated] = string(now())
    cache[:git_commit] = get_current_commit()
    save_cache(cache)
    @info "Source hashes updated"
end

# =============================================================================
# Status Display
# =============================================================================

function show_cache_status()
    println("\n" * "="^60)
    println("Test Cache Status")
    println("="^60)
    
    cache = load_cache()
    
    println("Git: $(get_current_branch()) @ $(get_current_commit())")
    println()
    
    if isnothing(cache)
        println("❌ No cache exists")
        println("\nAll test categories are stale.")
        println("\nRun tests and record results with:")
        println("  record_test_result(:hazards, passed=N, failed=N, errors=N)")
    else
        println("Last updated: $(cache[:last_updated])")
        println("Cache commit: $(get(cache, :git_commit, "unknown"))")
        println()
        
        stale = get_stale_categories()
        
        if isempty(stale)
            println("✅ All categories up to date")
        else
            println("⚠️  Stale categories: $(join(stale, ", "))")
        end
        
        println("\nTest Results:")
        results = get(cache, :test_results, Dict())
        if isempty(results)
            println("  (no results recorded)")
        else
            for (cat, r) in pairs(results)
                status = r[:errors] > 0 ? "❌" : r[:failed] > 0 ? "⚠️" : "✅"
                println("  $status $cat: $(r[:passed]) passed, $(r[:failed]) failed, $(r[:errors]) errors")
            end
        end
    end
    
    println("="^60 * "\n")
end

# =============================================================================
# Export for reports
# =============================================================================

function get_cache_for_reports()
    cache = load_cache()
    stale = get_stale_categories()
    
    (
        exists = !isnothing(cache),
        is_fresh = isempty(stale),
        stale_categories = collect(stale),
        git_commit = get_current_commit(),
        git_branch = get_current_branch(),
        last_updated = isnothing(cache) ? "never" : cache[:last_updated],
        test_results = isnothing(cache) ? Dict() : get(cache, :test_results, Dict())
    )
end

# Run status check when script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    show_cache_status()
end
