# Debug script to trace exponential simulation diagnostics
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using MultistateModels
using MultistateModels: Hazard, multistatemodel, set_parameters!, simulate_path, 
    eval_hazard, eval_cumhaz, CachedTransformStrategy, get_hazard_params
using DataFrames
using Random
using Statistics

println("=" ^ 70)
println("DEBUG: Exponential Simulation Diagnostics")
println("=" ^ 70)

# Configuration from simulation_diagnostics.qmd
const COVARIATE_VALUE = 1.5
const rate = 0.35
const horizon = 5.0

# Build model WITHOUT covariates (the failing case)
data = DataFrame(
    id = [1], tstart = [0.0], tstop = [horizon],
    statefrom = [1], stateto = [2], obstype = [1]
)

hazard = Hazard(@formula(0 ~ 1), "exp", 1, 2; linpred_effect = :ph)
model = multistatemodel(hazard; data = data)

# Set parameters: log(rate) for exponential
pars = [log(rate)]
hazname = model.hazards[1].hazname
println("\nSetting parameters for hazard '$(hazname)': $pars")
println("Natural scale rate should be: $(exp(pars[1])) = $rate")

set_parameters!(model, NamedTuple{(hazname,)}((pars,)))

# Get hazard params and verify
h = model.hazards[1]
params = get_hazard_params(model.parameters)[1]
println("\nHazard params from model: $params")

# Evaluate hazard at t=1.0 (should be constant = rate for exponential)
subjdat_row = model.data[1, :]
covars = Float64[]  # no covariates
haz_val = eval_hazard(h, 1.0, params, subjdat_row)
println("\nHazard at t=1.0: $haz_val (expected: $rate)")

# Evaluate cumulative hazard from 0 to 1.0 (should be rate * 1.0 = rate)
cumhaz_val = eval_cumhaz(h, 0.0, 1.0, params, subjdat_row)
println("Cumulative hazard [0, 1.0]: $cumhaz_val (expected: $rate)")

# Expected CDF at t=1.0
expected_cdf_1 = 1 - exp(-rate * 1.0)
println("Expected CDF at t=1.0: $expected_cdf_1")

# Now simulate and check
println("\n" * "=" ^ 70)
println("SIMULATION TEST")
println("=" ^ 70)

Random.seed!(12345)
nsim = 10000
strategy = CachedTransformStrategy()

durations = Float64[]
while length(durations) < nsim
    path = simulate_path(model, 1; strategy = strategy, rng = Random.default_rng())
    if path.states[end] != path.states[1]
        push!(durations, path.times[end] - path.times[1])
    end
end

# Empirical mean should be 1/rate for exponential
empirical_mean = mean(durations)
expected_mean = 1 / rate
println("\nEmpirical mean: $empirical_mean")
println("Expected mean (1/rate): $expected_mean")
println("Ratio: $(empirical_mean / expected_mean)")

# Check CDF at a few points
for t in [0.5, 1.0, 2.0, 3.0]
    empirical_cdf = count(d -> d <= t, durations) / length(durations)
    expected_cdf = 1 - exp(-rate * t)
    diff = abs(empirical_cdf - expected_cdf)
    println("t=$t: empirical_cdf=$empirical_cdf, expected_cdf=$expected_cdf, diff=$diff")
end

# The key question: is the simulated rate matching what we set?
# For exponential, median = ln(2) / rate
empirical_median = median(durations)
expected_median = log(2) / rate
println("\nEmpirical median: $empirical_median")
println("Expected median (ln(2)/rate): $expected_median")

# If empirical median is different, what rate does it correspond to?
implied_rate = log(2) / empirical_median
println("Implied rate from empirical median: $implied_rate")
println("Set rate: $rate")
