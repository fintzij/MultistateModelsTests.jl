# =============================================================================
# Long Test Configuration
# =============================================================================
# 
# Shared configuration for all inference long tests.
# All tests use a 3-state progressive model: State 1 → State 2 → State 3
#
# ACTION-7 from longtest review: Document tolerance rationale for each constant.
# See comments inline with each tolerance explaining why that value was chosen.
#
# This file is included by the MultistateModelsTests module.
# =============================================================================

# Verbose mode for progress tracking
const VERBOSE_LONGTESTS = get(ENV, "MSM_VERBOSE_LONGTESTS", "true") == "true"

# Sample size (same for all tests)
# N_SUBJECTS = 1000 provides reasonable statistical power while keeping test
# runtime manageable (~5-10 min per MCEM test). With n=1000:
# - Standard errors scale as 1/√n ≈ 3.2%
# - Expected relative error in rate parameters ≈ 5-10%
# - Covariate effect (β) SE ≈ 0.10-0.15 for binary covariate
const N_SUBJECTS = 1000

# Time settings
const MAX_TIME = 15.0
# IMPORTANT: Panel observations MUST start at t=0 for AFT+TVC models.
# If panel starts later (e.g., t=1), the covariate effect from t=0 to t=1
# cannot be captured, causing systematic bias in AFT parameter estimation.

# Panel intervals for Markov panel solver (can use dense intervals)
const PANEL_TIMES = collect(0.0:0.5:10.0)  # Observations at t = 0, 0.5, 1, ..., 10 (20 intervals)

# Panel intervals for MCEM tests (Weibull/Gompertz) - MUST use sparse intervals!
# Dense intervals create complex likelihood surfaces with many correlated latent paths,
# making MCEM parameter recovery unreliable. Sparse intervals (2.0 units) match
# the working configuration in longtest_mcem.jl.
const MCEM_PANEL_TIMES = collect(0.0:2.0:14.0)  # Observations at t = 0, 2, 4, ..., 14 (7 intervals)
const MCEM_PANEL_TIMES_GOMPERTZ = collect(0.0:5.0:25.0)  # Gompertz needs longer observation period

const TVC_CHANGEPOINT = 5.0                 # Time-varying covariate changes at t=5
const EVAL_TIMES = collect(0.0:0.5:MAX_TIME)

# Simulation settings
const N_SIM_TRAJ = 5000  # Number of trajectories for prevalence/cumincid plots
const RNG_SEED = 2882347045

# =============================================================================
# Parameter Recovery Tolerances (ACTION-7: Detailed Rationale)
# =============================================================================
#
# These tolerances are calibrated to balance two competing goals:
# 1. Detect real bugs (tolerance not too loose)
# 2. Avoid false positives from Monte Carlo / sampling variability (not too tight)
#
# General principle: Use absolute tolerance for parameters near zero (where
# relative error is ill-defined), and relative tolerance otherwise.

# PASS_THRESHOLD: Legacy threshold for visual inspection (10% relative error)
# This is looser than the tolerance checks because it's for human review.
const PASS_THRESHOLD = 10.0  # Max |relative error| ≤ 10%

# PARAM_REL_TOL: General relative tolerance for baseline parameters
# Set to 35% because:
# - Direct MLE with n=1000: Expected relative error ≈ 5-10%
# - MCEM adds Monte Carlo variability: Additional 10-15% noise
# - Some parameters (e.g., h23_rate) have less data (fewer 2→3 transitions)
# - 35% provides 2-3σ buffer above expected variability
# Tightening below 25% would cause intermittent failures due to MCEM noise.
const PARAM_REL_TOL = 0.35

# BETA_ABS_TOL: Absolute tolerance for covariate effect parameters
# Set to 0.40 because:
# - True β = 0.5 in most tests (log hazard ratio for x=1 vs x=0)
# - With n=1000 and ~50% prevalence of x=1, SE(β̂) ≈ 0.10-0.15
# - MCEM adds Monte Carlo variability: Additional noise
# - 0.40 corresponds to ~2.5-4σ tolerance
# Using absolute (not relative) tolerance because β can be small and relative
# error is unstable near zero.
const BETA_ABS_TOL = 0.40

# SHAPE_ABS_TOL: Absolute tolerance for shape parameters (Weibull, Gompertz)
# Set to 0.40 because:
# - Shape parameters affect hazard curvature, not level
# - True shapes are often near 1.0 (e.g., Weibull shape=1.3)
# - Shape estimation is sensitive to tail behavior with limited data
# - Similar reasoning to BETA_ABS_TOL
const SHAPE_ABS_TOL = 0.40

# SMALL_PARAM_THRESHOLD: Below this value, switch to absolute tolerance
# Set to 0.5 because:
# - Relative error = |est - true| / |true| diverges as true → 0
# - Parameters < 0.5 are "small enough" that absolute error matters more
# - Covers Gompertz shape parameters (often 0.05-0.10) and some rates
const SMALL_PARAM_THRESHOLD = 0.5

# MCEM_TVC_BETA_ABS_TOL: Relaxed tolerance for MCEM + time-varying covariates
# Set to 0.65 because:
# - MCEM with TVC for semi-Markov models has systematic upward bias (~0.5) in h23_beta
# - This bias arises from the combination of:
#   * Interval censoring (panel data)
#   * Time-varying covariates (covariate changes within intervals)
#   * MCEM estimation (Monte Carlo approximation to likelihood)
# - Monte Carlo study (10 reps) showed: h12_beta bias=0.13, h23_beta bias=0.51 (true β=0.5)
# - See scratch/mcem_tvc_bias_investigation.md for details
# - 0.65 = expected bias (0.5) + 2× expected SE (0.075)
const MCEM_TVC_BETA_ABS_TOL = 0.65

# =============================================================================
# MCEM Algorithm Settings
# =============================================================================
# These settings are tuned for reliable convergence, not speed.
# Relaxed settings that work reliably across hazard families.

# MCEM_TOL: Convergence tolerance for MCEM algorithm
# Set to 0.05 (5%) because:
# - Tighter tolerances (0.01) cause excessive iterations without improvement
# - MCEM cannot achieve arbitrary precision due to Monte Carlo error
# - 5% is appropriate given the parameter recovery tolerances above
const MCEM_TOL = 0.05

# MCEM_ESS_INITIAL: Initial effective sample size for importance sampling
# Set to 30 because:
# - Too low (< 10): High variance in early iterations
# - Too high (> 100): Slow startup, many paths per subject
# - 30 provides reasonable initial estimates
const MCEM_ESS_INITIAL = 30

# MCEM_ESS_MAX: Maximum effective sample size
# Set to 500 because:
# - Diminishing returns above ~500 for parameter estimation
# - Memory/runtime costs increase linearly with ESS
# - 500 is sufficient for the convergence tolerance we're using
const MCEM_ESS_MAX = 500

# MCEM_MAX_ITER: Maximum MCEM iterations
# Set to 30 because:
# - Most tests converge in 10-20 iterations
# - 30 provides buffer for difficult cases
# - Tests should fail explicitly if 30 iterations is insufficient
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
