#!/usr/bin/env julia
"""
Parallel test runner - runs long tests in separate processes simultaneously.
Usage: julia --project=. scripts/run_tests_parallel.jl [--workers N]
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Distributed
using Printf

# Parse arguments
local n_workers = 4  # default
for (i, arg) in enumerate(ARGS)
    if arg == "--workers" && i < length(ARGS)
        global n_workers = parse(Int, ARGS[i+1])
    end
end

println("Setting up $n_workers worker processes...")
addprocs(n_workers; exeflags="--project=$(joinpath(@__DIR__, ".."))")

# Load packages on all workers first
@everywhere begin
    using Test
    using MultistateModels
    using StatsModels
    using DataFrames
    using Distributions
    using Random
    using LinearAlgebra
    using Logging
end

# Define paths for workers
const TESTS_DIR = joinpath(@__DIR__, "..")
const FIXTURES_PATH = joinpath(TESTS_DIR, "fixtures", "TestFixtures.jl")
const HELPERS_PATH = joinpath(TESTS_DIR, "longtests", "longtest_helpers.jl")

# Load fixtures and helpers on all workers
@everywhere TESTS_DIR = $TESTS_DIR
@everywhere FIXTURES_PATH = $FIXTURES_PATH  
@everywhere HELPERS_PATH = $HELPERS_PATH

@everywhere begin
    include(FIXTURES_PATH)
    using .TestFixtures
    include(HELPERS_PATH)
end

# Load on main process too
include(FIXTURES_PATH)
using .TestFixtures
include(HELPERS_PATH)

# ============================================================================
# Run unit tests first (sequentially on main process - they're fast)
# ============================================================================
println("\n" * "="^70)
println("PHASE 1: Running Unit Tests (sequential)")
println("="^70 * "\n")

unit_tests = [
    "test_hazards.jl",
    "test_helpers.jl",
    "test_initialization.jl",
    "test_mcem.jl",
    "test_mll_consistency.jl",
    "test_modelgeneration.jl",
    "test_observation_weights_emat.jl",
    "test_per_transition_obstype.jl",
    "test_phasetype.jl",
    "test_phasetype_emission_expansion.jl",
    "test_phasetype_panel_expansion.jl",
    "test_pijcv.jl",
    "test_reconstructor.jl",
    "test_reversible_tvc_loglik.jl",
    "test_simulation.jl",
    "test_sir.jl",
    "test_splines.jl",
    "test_subject_weights.jl",
    "test_surrogates.jl",
    "test_variance.jl"
]

unit_results = Dict{String, Symbol}()
for test in unit_tests
    print("  Running $test...")
    flush(stdout)
    try
        @testset "$test" begin
            include(joinpath(TESTS_DIR, "unit", test))
        end
        unit_results[test] = :passed
        println(" ✓")
    catch e
        unit_results[test] = :failed
        println(" ✗")
        @error "Unit test failed" test exception=(e, catch_backtrace())
    end
end

# ============================================================================
# Run long tests in parallel
# ============================================================================
println("\n" * "="^70)
println("PHASE 2: Running Long Tests (parallel on $n_workers workers)")
println("="^70 * "\n")

long_tests = [
    "longtest_exact_markov.jl",
    "longtest_mcem.jl",
    "longtest_mcem_splines.jl",
    "longtest_mcem_tvc.jl",
    "longtest_phasetype.jl",
    "longtest_robust_markov_phasetype.jl",
    "longtest_robust_parametric.jl",
    "longtest_simulation_distribution.jl",
    "longtest_simulation_tvc.jl",
    "longtest_sir.jl",
    "longtest_variance_validation.jl"
]

# Function to run a single long test
@everywhere function run_longtest(test_file::String)
    test_path = joinpath(TESTS_DIR, "longtests", test_file)
    result = Dict{String, Any}(
        "test" => test_file,
        "worker" => myid(),
        "status" => :unknown,
        "time" => 0.0,
        "error" => nothing
    )
    
    start_time = time()
    try
        @testset "$test_file" begin
            include(test_path)
        end
        result["status"] = :passed
    catch e
        result["status"] = :failed
        result["error"] = sprint(showerror, e, catch_backtrace())
    end
    result["time"] = time() - start_time
    
    return result
end

# Run long tests in parallel using pmap
println("Starting $(length(long_tests)) long tests on $(nworkers()) workers...")
println()

start_total = time()
results = pmap(run_longtest, long_tests)
total_time = time() - start_total

# ============================================================================
# Print summary
# ============================================================================
println("\n" * "="^70)
println("TEST SUMMARY")
println("="^70 * "\n")

println("Unit Tests:")
n_unit_passed = count(v -> v == :passed, values(unit_results))
n_unit_failed = count(v -> v == :failed, values(unit_results))
println("  Passed: $n_unit_passed / $(length(unit_tests))")
if n_unit_failed > 0
    println("  Failed:")
    for (test, status) in unit_results
        status == :failed && println("    - $test")
    end
end

println("\nLong Tests:")
for r in results
    status_sym = r["status"] == :passed ? "✓" : "✗"
    time_str = @sprintf("%.1fs", r["time"])
    println("  $status_sym $(r["test"]) (worker $(r["worker"]), $time_str)")
    if r["status"] == :failed && r["error"] !== nothing
        # Print first few lines of error
        error_lines = split(r["error"], '\n')
        for line in error_lines[1:min(5, length(error_lines))]
            println("      $line")
        end
    end
end

n_long_passed = count(r -> r["status"] == :passed, results)
n_long_failed = count(r -> r["status"] == :failed, results)
println("\n  Total: $n_long_passed passed, $n_long_failed failed")
println("  Total parallel time: $(round(total_time, digits=1))s")

# Estimate sequential time
seq_time = sum(r["time"] for r in results)
println("  Estimated sequential time: $(round(seq_time, digits=1))s")
println("  Speedup: $(round(seq_time/total_time, digits=2))x")

# Exit with appropriate code
exit_code = (n_unit_failed + n_long_failed) > 0 ? 1 : 0
println("\nExiting with code $exit_code")

# Clean up workers
rmprocs(workers())

exit(exit_code)
