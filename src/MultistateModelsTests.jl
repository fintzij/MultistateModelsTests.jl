module MultistateModelsTests

using DataFrames
using Distributions
using LinearAlgebra
using Logging
using MultistateModels
using Random
using Statistics
using Test

const TEST_LEVEL = get(ENV, "MSM_TEST_LEVEL", "quick")
const SUPPRESS_EXPECTED_WARNINGS = get(ENV, "MSM_SUPPRESS_WARNINGS", "true") == "true"

"""
Custom logger that filters expected warnings during tests.
"""
struct TestFilterLogger <: AbstractLogger
    wrapped::AbstractLogger
end

Logging.min_enabled_level(l::TestFilterLogger) = Logging.min_enabled_level(l.wrapped)
Logging.shouldlog(l::TestFilterLogger, args...) = Logging.shouldlog(l.wrapped, args...)
Logging.catch_exceptions(l::TestFilterLogger) = Logging.catch_exceptions(l.wrapped)

function Logging.handle_message(l::TestFilterLogger, level, message, _module, group, id, file, line; kwargs...)
    msg_str = string(message)
    # Filter out expected warnings that clutter test output
    if level == Logging.Warn
        if contains(msg_str, "Using :surrogate method on Markov model") ||
           contains(msg_str, "IJ variance-covariance matrix was not computed") ||
           contains(msg_str, "Jackknife variance-covariance matrix was not computed") ||
           contains(msg_str, "MCEM did not converge") ||
           contains(msg_str, "The maximum number of iterations")
            return nothing  # Suppress
        end
    end
    Logging.handle_message(l.wrapped, level, message, _module, group, id, file, line; kwargs...)
end

function should_run_longtest(name::String)
    only_test = get(ENV, "MSM_LONGTEST_ONLY", "")
    if !isempty(only_test)
        return lowercase(only_test) == lowercase(name)
    end
    env_key = "MSM_LONGTEST_" * uppercase(replace(name, "_" => "_"))
    return lowercase(get(ENV, env_key, "true")) == "true"
end

# Bring in fixtures (now lives within the test package)
include(joinpath(@__DIR__, "..", "fixtures", "TestFixtures.jl"))
using .TestFixtures

# Paths into the new package layout
const UNIT_DIR = joinpath(@__DIR__, "..", "unit")
const INTEGRATION_DIR = joinpath(@__DIR__, "..", "integration")
const LONGTESTS_DIR = joinpath(@__DIR__, "..", "longtests")

function runtests()
    # Deterministic RNG so regression failures are reproducible in CI.
    Random.seed!(52787)
    
    # Optionally suppress expected warnings
    if SUPPRESS_EXPECTED_WARNINGS
        old_logger = global_logger(TestFilterLogger(global_logger()))
    end

    @testset "Unit Tests" begin
        include(joinpath(UNIT_DIR, "test_modelgeneration.jl"))
        include(joinpath(UNIT_DIR, "test_hazards.jl"))
        include(joinpath(UNIT_DIR, "test_helpers.jl"))
        include(joinpath(UNIT_DIR, "test_simulation.jl"))
        include(joinpath(UNIT_DIR, "test_pijcv.jl"))
        include(joinpath(UNIT_DIR, "test_phasetype.jl"))
        include(joinpath(UNIT_DIR, "test_splines.jl"))
        include(joinpath(UNIT_DIR, "test_surrogates.jl"))
        include(joinpath(UNIT_DIR, "test_mcem.jl"))
        include(joinpath(UNIT_DIR, "test_sir.jl"))
        include(joinpath(UNIT_DIR, "test_mll_consistency.jl"))
        include(joinpath(UNIT_DIR, "test_reconstructor.jl"))
        include(joinpath(UNIT_DIR, "test_reversible_tvc_loglik.jl"))
        include(joinpath(UNIT_DIR, "test_initialization.jl"))
        include(joinpath(UNIT_DIR, "test_variance.jl"))
        include(joinpath(INTEGRATION_DIR, "test_parallel_likelihood.jl"))
        include(joinpath(INTEGRATION_DIR, "test_parameter_ordering.jl"))
    end

    if TEST_LEVEL == "full"
        @info "Running full test suite including long statistical tests..."

        # Load shared configuration and helper functions for long tests
        include(joinpath(LONGTESTS_DIR, "longtest_config.jl"))
        include(joinpath(LONGTESTS_DIR, "longtest_helpers.jl"))

        only_test = get(ENV, "MSM_LONGTEST_ONLY", "")
        if !isempty(only_test)
            @info "Running only: $only_test"
        end

        # Long test suite definitions with progress tracking
        long_tests = [
            ("exact_data", "Long Tests - Exact Data", "longtest_exact_markov.jl"),
            ("mcem_parametric", "Long Tests - Panel Data MCEM (Parametric Hazards)", "longtest_mcem.jl"),
            ("mcem_splines", "Long Tests - Panel Data MCEM (Spline Hazards)", "longtest_mcem_splines.jl"),
            ("mcem_tvc", "Long Tests - Panel Data MCEM (Time-Varying Covariates)", "longtest_mcem_tvc.jl"),
            ("sim_dist", "Long Tests - Simulation Distributional Fidelity", "longtest_simulation_distribution.jl"),
            ("sim_tvc", "Long Tests - Simulation (Time-Varying Covariates)", "longtest_simulation_tvc.jl"),
            ("robust_exact", "Long Tests - Robust Exact Data (Tight Tolerances)", "longtest_robust_parametric.jl"),
            ("markov_phasetype_validation", "Long Tests - Markov/PhaseType Validation", "longtest_robust_markov_phasetype.jl"),
            ("phasetype", "Long Tests - Phase-Type Hazard Models (Exact + Panel Data)", "longtest_phasetype.jl"),
            ("variance_validation", "Long Tests - Variance Estimation Validation", "longtest_variance_validation.jl"),
        ]
        
        total_long_tests = count(t -> should_run_longtest(t[1]), long_tests)
        current_test = 0
        
        for (key, name, file) in long_tests
            if should_run_longtest(key)
                current_test += 1
                println("\n" * "="^70)
                @info "[$current_test/$total_long_tests] Starting: $name"
                println("="^70)
                start_time = time()
                
                @testset "$name" begin
                    include(joinpath(LONGTESTS_DIR, file))
                end
                
                elapsed = round(time() - start_time; digits=1)
                @info "[$current_test/$total_long_tests] Completed: $name ($(elapsed)s)"
            end
        end
        
        println("\n" * "="^70)
        @info "All long tests completed!"
        println("="^70)
    else
        @info "Running quick tests only. Set MSM_TEST_LEVEL=full for complete suite."
    end

    return nothing
end

# Diagnostics directory path
const DIAGNOSTICS_DIR = joinpath(@__DIR__, "..", "diagnostics")

"""
    generate_simulation_diagnostics()

Regenerate all simulation diagnostic plots (hazard/cumulative hazard/survival curves
and simulation distribution validation). Outputs PNG files to 
`MultistateModelsTests/diagnostics/assets/`.

Requires CairoMakie (loaded on-demand).
"""
function generate_simulation_diagnostics()
    script = joinpath(DIAGNOSTICS_DIR, "generate_model_diagnostics.jl")
    @info "Running simulation diagnostics generator..."
    include(script)
    Main.generate_all()
    @info "Diagnostics saved to $(joinpath(DIAGNOSTICS_DIR, "assets"))"
end

"""
    diagnostics_path()

Return the path to the diagnostics directory.
"""
diagnostics_path() = DIAGNOSTICS_DIR

"""
    diagnostics_assets_path()

Return the path to the diagnostics assets directory containing generated plots.
"""
diagnostics_assets_path() = joinpath(DIAGNOSTICS_DIR, "assets")

export runtests, generate_simulation_diagnostics, diagnostics_path, diagnostics_assets_path

end # module
