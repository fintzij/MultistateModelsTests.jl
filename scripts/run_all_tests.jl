
using Test
using Pkg

# Activate the test environment
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using MultistateModels
using StatsModels
using DataFrames
using Distributions
using Random
using LinearAlgebra
using Logging

# Load fixtures
include(joinpath(@__DIR__, "..", "fixtures", "TestFixtures.jl"))
using .TestFixtures

# Load cache system
include(joinpath(@__DIR__, "test_cache.jl"))

println("Running all tests...")

# Map files to categories
const FILE_TO_CATEGORY = Dict(
    "test_hazards.jl" => :hazards,
    "test_splines.jl" => :splines,
    "test_simulation.jl" => :simulation,
    "test_mcem.jl" => :mcem,
    "test_phasetype.jl" => :phasetype,
    "test_sir.jl" => :sir,
    "test_variance.jl" => :variance,
    "test_initialization.jl" => :initialization,
    "test_modelgeneration.jl" => :modelgeneration,
    "test_surrogates.jl" => :surrogates,
    "test_helpers.jl" => :helpers,
    "test_reconstructor.jl" => :reconstructor
)

# Accumulator for results
category_results = Dict{Symbol, Vector{Int}}() # passed, failed, errors

function accumulate_result!(category, passed, failed, errors)
    if !haskey(category_results, category)
        category_results[category] = [0, 0, 0]
    end
    category_results[category][1] += passed
    category_results[category][2] += failed
    category_results[category][3] += errors
end

function count_tests(ts::Test.DefaultTestSet)
    passed = ts.n_passed
    failed = 0
    errors = 0
    
    for res in ts.results
        if res isa Test.Fail
            failed += 1
        elseif res isa Test.Error
            errors += 1
        elseif res isa Test.DefaultTestSet
            p, f, e = count_tests(res)
            passed += p
            failed += f
            errors += e
        end
    end
    return passed, failed, errors
end

@testset "All Tests" begin

    @testset "Unit Tests" begin
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
        
        for test in unit_tests
            println("  Running $test...")
            
            ts = @testset "$test" begin
                include(joinpath(@__DIR__, "..", "unit", test))
            end
            
            if haskey(FILE_TO_CATEGORY, test)
                p, f, e = count_tests(ts)
                accumulate_result!(FILE_TO_CATEGORY[test], p, f, e)
            end
        end
    end

    @testset "Long Tests" begin
        # Load helpers for long tests
        println("  Loading long test helpers...")
        include(joinpath(@__DIR__, "..", "longtests", "longtest_helpers.jl"))

        long_tests = [
            "longtest_exact_markov.jl",
            "longtest_mcem.jl",
            "longtest_mcem_splines.jl",
            "longtest_mcem_tvc.jl",
            "longtest_phasetype.jl",  # includes _exact and _panel
            "longtest_robust_markov_phasetype.jl",
            "longtest_robust_parametric.jl",
            "longtest_simulation_distribution.jl",
            "longtest_simulation_tvc.jl",
            "longtest_sir.jl",
            "longtest_variance_validation.jl"
        ]

        for test in long_tests
            println("  Running $test...")
            # Long tests are not mapped to categories yet, but we run them.
            include(joinpath(@__DIR__, "..", "longtests", test))
        end
    end

end

# Update cache
println("Updating test cache...")
for (cat, counts) in category_results
    record_test_result(cat, passed=counts[1], failed=counts[2], errors=counts[3])
end

println("Running Benchmarks...")
include(joinpath(@__DIR__, "..", "benchmarks", "run_benchmarks.jl"))

println("All tests and benchmarks completed.")
