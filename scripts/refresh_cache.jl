#!/usr/bin/env julia
# =============================================================================
# Test Cache Status & Management
# =============================================================================
# 
# This script displays the current test cache status and helps manage results.
#
# Usage:
#   julia --project=MultistateModelsTests scripts/refresh_cache.jl
#   julia --project=MultistateModelsTests scripts/refresh_cache.jl --update-hashes
#
# =============================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include("test_cache.jl")

function main()
    # Show cache status
    show_cache_status()
    
    # Check for --update-hashes flag
    if "--update-hashes" in ARGS
        println("Updating source hashes to mark cache as current...")
        update_source_hashes()
        println()
        show_cache_status()
        return
    end
    
    # Get stale categories
    stale = get_stale_categories()
    
    if !isempty(stale)
        println("To mark tests as passing after manual verification:")
        for cat in stale
            println("  record_test_result(:$cat, passed=N, failed=0, errors=0)")
        end
        println()
        println("Or to mark all source changes as tested (updates hashes only):")
        println("  julia scripts/refresh_cache.jl --update-hashes")
    end
end

main()
