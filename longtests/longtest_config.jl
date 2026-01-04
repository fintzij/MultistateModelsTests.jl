# =============================================================================
# Long Test Configuration
# =============================================================================
# 
# Shared configuration for all inference long tests.
# All tests use a 3-state progressive model: State 1 → State 2 → State 3
# =============================================================================

using Test

# Verbose mode for progress tracking
const VERBOSE_LONGTESTS = get(ENV, "MSM_VERBOSE_LONGTESTS", "true") == "true"

# Sample size (same for all tests)
const N_SUBJECTS = 1000

# Time settings
const MAX_TIME = 15.0
# IMPORTANT: Panel observations MUST start at t=0 for AFT+TVC models.
# If panel starts later (e.g., t=1), the covariate effect from t=0 to t=1
# cannot be captured, causing systematic bias in AFT parameter estimation.
const PANEL_TIMES = collect(0.0:1.0:10.0)  # Observations at t = 0, 1, 2, ..., 10
const TVC_CHANGEPOINT = 5.0                 # Time-varying covariate changes at t=5
const EVAL_TIMES = collect(0.0:0.5:MAX_TIME)

# Simulation settings
const N_SIM_TRAJ = 5000  # Number of trajectories for prevalence/cumincid plots
const RNG_SEED = 2882347045

# Pass criteria
const PASS_THRESHOLD = 10.0  # Max |relative error| ≤ 10%

# Parameter recovery tolerances for capture_longtest_result!
const PARAM_REL_TOL = 0.35   # 35% relative error for main params (MCEM has more MC noise)
const BETA_ABS_TOL = 0.40    # 0.40 absolute tolerance for beta (covariate) params
const SHAPE_ABS_TOL = 0.40   # Absolute tolerance for shape parameters (log-scale values near 0)
const SMALL_PARAM_THRESHOLD = 0.5  # Below this, use absolute tolerance

# MCEM settings (Markov proposals for all) - relaxed settings that work reliably
const MCEM_TOL = 0.05
const MCEM_ESS_INITIAL = 30
const MCEM_ESS_MAX = 500
const MCEM_MAX_ITER = 30

# Spline settings
const SPLINE_DEGREE = 1  # Linear between knots (degree 1)

# Output settings
const OUTPUT_DIR = joinpath(@__DIR__, "reports")
const ASSETS_DIR = joinpath(OUTPUT_DIR, "assets", "diagnostics")

# =============================================================================
# TestResult Structure
# =============================================================================

"""
    TestResult

Container for a single test's results including parameters, estimates, 
errors, and diagnostic data for plotting.

Uses @kwdef for keyword-argument construction.
"""
Base.@kwdef mutable struct TestResult
    # Test identification
    name::String
    family::String
    parameterization::Symbol = :none     # :none, :ph, :aft
    covariates::Symbol = :none           # :none, :tfc, :tvc
    data_type::Symbol = :exact           # :exact, :panel
    n_subjects::Int = N_SUBJECTS
    
    # Parameters
    true_params::Dict{String, Float64} = Dict{String, Float64}()
    estimated_params::Dict{String, Float64} = Dict{String, Float64}()
    rel_errors::Dict{String, Float64} = Dict{String, Float64}()
    
    # Pass/fail
    max_rel_error::Float64 = NaN
    passed::Bool = false
    
    # Diagnostic data
    eval_times::Vector{Float64} = Float64[]
    prevalence_true::Union{Nothing, Matrix{Float64}} = nothing
    prevalence_observed::Union{Nothing, Matrix{Float64}} = nothing
    prevalence_fitted::Union{Nothing, Matrix{Float64}} = nothing
    cumincid_12_true::Union{Nothing, Vector{Float64}} = nothing
    cumincid_12_observed::Union{Nothing, Vector{Float64}} = nothing
    cumincid_12_fitted::Union{Nothing, Vector{Float64}} = nothing
    cumincid_23_true::Union{Nothing, Vector{Float64}} = nothing
    cumincid_23_observed::Union{Nothing, Vector{Float64}} = nothing
    cumincid_23_fitted::Union{Nothing, Vector{Float64}} = nothing
end

# Global results storage
const ALL_RESULTS = TestResult[]

# =============================================================================
# Test Categories for Reporting
# =============================================================================

const FAMILY_NAMES = Dict(
    "exp" => "Exponential",
    "wei" => "Weibull", 
    "gom" => "Gompertz",
    "phasetype" => "Phase-Type",
    "spline" => "Spline"
)

const PARAM_NAMES = Dict(
    "none" => "Baseline",
    "ph" => "PH",
    "aft" => "AFT"
)

const COV_NAMES = Dict(
    "none" => "No Covariates",
    "tfc" => "Time-Fixed",
    "tvc" => "Time-Varying"
)

const DATA_NAMES = Dict(
    "exact" => "Exact",
    "panel" => "Panel"
)

# =============================================================================
# Verbose Test Macro
# =============================================================================

"""
    @verbose_testset name expr

A wrapper around @testset that prints progress when VERBOSE_LONGTESTS is true.
Shows start time, test name, and elapsed time on completion.
"""
macro verbose_testset(name, expr)
    quote
        local test_name = $(esc(name))
        if VERBOSE_LONGTESTS
            local start_time = time()
            println("  ▶ Starting: ", test_name)
            flush(stdout)
        end
        local result = @testset $name $(esc(expr))
        if VERBOSE_LONGTESTS
            local elapsed = round(time() - start_time; digits=1)
            println("  ✓ Completed: ", test_name, " (", elapsed, "s)")
            flush(stdout)
        end
        result
    end
end
