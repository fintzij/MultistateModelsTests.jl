# Quick validation of the simulation diagnostics fix
# Tests that increased horizons eliminate right-truncation bias

using Pkg
Pkg.activate(".")

using MultistateModels
using Random
using DataFrames
using StatsModels: @formula
using Statistics

println("="^70)
println("SIMULATION DIAGNOSTIC VALIDATION")
println("="^70)

# Test configuration  
rate = 0.35
horizon_short = 5.0    # Original (too short)
horizon_long = 100.0   # Fixed (sufficient)
nsim = 5000
rng = Random.MersenneTwister(12345)

# Build exponential model
function build_exp_model(horizon)
    data = DataFrame(
        id = [1], tstart = [0.0], tstop = [horizon],
        statefrom = [1], stateto = [2], obstype = [1]
    )
    hazard = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    model = multistatemodel(hazard; data = data)
    set_parameters!(model, NamedTuple{(model.hazards[1].hazname,)}(([log(rate)],)))
    return model
end

# Simulate and compute empirical CDF error
function simulate_and_evaluate(model, nsim, rng)
    durations = Float64[]
    n_censored = 0
    strategy = CachedTransformStrategy()
    
    for _ in 1:nsim*2  # Allow for some censoring
        length(durations) >= nsim && break
        path = simulate_path(model, 1; strategy = strategy, rng = rng)
        if path.states[end] != path.states[1]
            push!(durations, path.times[end] - path.times[1])
        else
            n_censored += 1
        end
    end
    
    truncation_rate = n_censored / (length(durations) + n_censored)
    
    # Compute max CDF diff at quantiles
    sorted = sort(durations)
    n = length(sorted)
    ecdf = (1:n) ./ n
    tcdf = [1.0 - exp(-rate * t) for t in sorted]
    max_diff = maximum(abs.(ecdf .- tcdf))
    
    return max_diff, truncation_rate
end

println("\nTest 1: Short horizon ($(horizon_short)) - SHOULD FAIL")
model_short = build_exp_model(horizon_short)
rng = Random.MersenneTwister(12345)
max_diff_short, trunc_short = simulate_and_evaluate(model_short, nsim, rng)
println("  Truncation rate: $(round(trunc_short*100, digits=1))%")
println("  Max CDF diff: $(round(max_diff_short, digits=4))")
println("  Status: ", max_diff_short > 0.01 ? "FAIL ✗ (as expected)" : "PASS ✓")

println("\nTest 2: Long horizon ($(horizon_long)) - SHOULD PASS")
model_long = build_exp_model(horizon_long)
rng = Random.MersenneTwister(12345)
max_diff_long, trunc_long = simulate_and_evaluate(model_long, nsim, rng)
println("  Truncation rate: $(round(trunc_long*100, digits=2))%")
println("  Max CDF diff: $(round(max_diff_long, digits=4))")
println("  Status: ", max_diff_long < 0.01 ? "PASS ✓" : "FAIL ✗")

println("\n" * "="^70)
if max_diff_long < 0.01 && max_diff_short > 0.01
    println("VALIDATION PASSED: Increased horizon fixes right-truncation bias")
else
    println("VALIDATION FAILED: Unexpected results")
end
println("="^70)
