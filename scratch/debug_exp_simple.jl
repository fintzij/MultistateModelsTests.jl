# Simple diagnostic: trace exponential simulation without full model
# This avoids heavy compilation by testing the core math directly

using Pkg
Pkg.activate(".")

using MultistateModels
using Random
using Statistics

println("="^70)
println("SIMPLE DEBUG: Exponential Simulation Core Math")
println("="^70)

# Test 1: Verify the inverse CDF sampling works correctly
println("\n--- Test 1: Direct inverse CDF sampling for Exponential ---")

rate = 0.35
n_samples = 100000

# Inverse CDF for Exponential: F^{-1}(u) = -log(1-u)/rate
Random.seed!(12345)
u = rand(n_samples)
times = -log.(1.0 .- u) ./ rate

empirical_mean = mean(times)
theoretical_mean = 1.0 / rate
empirical_median = median(times)
theoretical_median = log(2) / rate

println("Rate: $rate")
println("Theoretical mean: $(theoretical_mean)")
println("Empirical mean: $(empirical_mean)")
println("Mean error: $(abs(empirical_mean - theoretical_mean))")
println("Theoretical median: $(theoretical_median)")
println("Empirical median: $(empirical_median)")
println("Median error: $(abs(empirical_median - theoretical_median))")

# Check CDF match
t_grid = [0.5, 1.0, 2.0, 3.0, 5.0]
println("\n--- CDF Comparison ---")
println("Time\tTheoretical\tEmpirical\tDiff")
for t in t_grid
    theor_cdf = 1.0 - exp(-rate * t)
    emp_cdf = mean(times .<= t)
    println("$(t)\t$(round(theor_cdf, digits=4))\t\t$(round(emp_cdf, digits=4))\t\t$(round(abs(theor_cdf - emp_cdf), digits=5))")
end

# Test 2: Check the hazard math matches
println("\n--- Test 2: Hazard/Cumulative Hazard verification ---")

# For exponential, h(t) = rate, H(t) = rate * t
for t in [1.0, 2.0, 5.0]
    h_expected = rate
    H_expected = rate * t
    cdf_from_H = 1.0 - exp(-H_expected)
    println("t=$t: h(t)=$h_expected, H(t)=$H_expected, F(t)=$(round(cdf_from_H, digits=4))")
end

println("\n--- Test 3: Verify simulation_diagnostics.qmd formula ---")

# The formula from simulation_diagnostics.qmd for exponential:
# elseif family == "exp"
#     cum_hazard = rate * t  # PH: rate * exp(linear_pred)
#     return 1 - exp(-cum_hazard)

# With no covariates, linear_pred = intercept only
# If we set rate directly, then: cum_hazard = rate * t

# Check: What if the package uses log(rate) as the parameter?
# i.e., if parameter = log(0.35), then rate = exp(log(0.35)) = 0.35
log_rate = log(rate)
println("If package stores log(rate): $(log_rate)")
println("Then rate = exp($(log_rate)) = $(exp(log_rate))")

# For AFT: the parameterization is different
# T = exp(mu + sigma * W) where W is standard Gumbel
# For exponential, this becomes T ~ Exp(rate) where rate = exp(-mu/sigma)
# Or equivalently, mu = -log(rate), sigma = 1

mu_aft = -log(rate)
println("\nFor AFT with mu = -log(rate) = $mu_aft, sigma = 1:")
println("We get: T ~ Exp(rate = $(exp(-mu_aft))) which should equal $rate")

println("\n--- Test Complete ---")
