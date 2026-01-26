# =============================================================================
# MultistateModels.jl vs mgcv PAM Comparison
# =============================================================================
#
# Direct comparison of our spline hazard estimates against R mgcv's PAM.
# This generates data in Julia, exports it, fits with both packages, and
# compares the estimated hazard curves and smoothing parameters.
#
# Updated 2026-01-24: Uses new fit() API with penalty=:auto, select_lambda=:pijcv
# =============================================================================

using MultistateModels
using DataFrames
using CSV
using JSON
using Random
using Statistics
using LinearAlgebra

import MultistateModels: get_parameters

# =============================================================================
# 1. Generate exact observation data with known hazard
# =============================================================================

println("=" ^ 70)
println("MultistateModels.jl vs mgcv PAM Comparison")
println("=" ^ 70)

Random.seed!(2024)

# True hazard: Weibull with time-varying pattern
# h(t) = (k/λ) * (t/λ)^(k-1) with k=1.5, λ=5
# This gives an increasing hazard that's easy to recover
weibull_k = 1.5
weibull_lambda = 5.0

true_hazard(t) = (weibull_k / weibull_lambda) * (t / weibull_lambda)^(weibull_k - 1)
true_cumhaz(t) = (t / weibull_lambda)^weibull_k

# Generate event times using inverse transform
function simulate_weibull_times(n, k, λ; max_time=15.0)
    times = Float64[]
    status = Int[]
    for _ in 1:n
        U = rand()
        t = λ * (-log(U))^(1/k)
        if t <= max_time
            push!(times, t)
            push!(status, 1)
        else
            push!(times, max_time)
            push!(status, 0)  # right-censored
        end
    end
    return times, status
end

n_subjects = 500
times, status = simulate_weibull_times(n_subjects, weibull_k, weibull_lambda)

println("\n--- Data Summary ---")
println("Sample size: $n_subjects")
println("Events: $(sum(status))")
println("Censored: $(sum(1 .- status))")
println("Time range: [$(round(minimum(times), digits=3)), $(round(maximum(times), digits=3))]")
println("True hazard: Weibull(k=$weibull_k, λ=$weibull_lambda)")

# Create MultistateModels data format
# For exact observations: obstype=1, statefrom→stateto is the transition
# For censored: obstype=1 with statefrom=stateto (stayed in state)
data = DataFrame(
    id = 1:n_subjects,
    tstart = zeros(n_subjects),
    tstop = times,
    statefrom = ones(Int, n_subjects),
    stateto = ifelse.(status .== 1, 2, 1),  # 1→2 if event, 1→1 if censored
    obstype = ones(Int, n_subjects),  # All exact observations
)

# =============================================================================
# 2. Fit with MultistateModels.jl SplineHazard
# =============================================================================

println("\n--- Fitting with MultistateModels.jl ---")

# Create spline hazard with explicit interior knots
# Place knots at quantiles of event times
event_times = times[status .== 1]
n_interior_knots = 6
knot_quantiles = range(0.1, 0.9, length=n_interior_knots)
interior_knots = quantile(event_times, knot_quantiles)

println("Interior knots at quantiles: ", round.(interior_knots, digits=3))

# Use new API with boundaryknots
h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; degree=3, knots=interior_knots, 
             boundaryknots=[0.0, maximum(times)*1.01])
model = multistatemodel(h12; data=data)

# Fit the model with automatic smoothing selection (PIJCV)
fitted_model = fit(model; penalty=:auto, select_lambda=:pijcv, verbose=true, vcov_type=:ij)

# Get fitted parameters using the nested structure
fitted_params = get_parameters(fitted_model)
spline_coeffs = fitted_params.h12

println("Spline coefficients: $(length(spline_coeffs)) parameters")
println("Coefficients: ", round.(spline_coeffs, digits=4))

# Get the fitted hazard object
hazard_obj = fitted_model.hazards[1]
knots = hazard_obj.knots
println("Full knots (including boundaries): ", round.(knots, digits=3))

# Print smoothing info
if !isnothing(fitted_model.smoothing_parameters)
    println("Selected λ: ", round.(fitted_model.smoothing_parameters, digits=4))
end
if !isnothing(fitted_model.edf)
    println("EDF: ", fitted_model.edf)
end

# =============================================================================
# 3. Evaluate Julia fitted hazard
# =============================================================================

println("\n--- Evaluating Julia fit ---")

# Evaluation grid
t_eval = range(0.5, maximum(times) * 0.95, length=50)

# Evaluate hazard at each time point using the nested parameter structure
h_julia = Float64[]
for t in t_eval
    # The hazard function takes (t, pars, covars) where pars is the nested params for this hazard
    h_val = hazard_obj(t, spline_coeffs, NamedTuple())
    push!(h_julia, h_val)
end

h_true = true_hazard.(t_eval)

rmse_julia = sqrt(mean((h_julia .- h_true).^2))
println("Julia RMSE vs truth: $(round(rmse_julia, digits=6))")

# =============================================================================
# 4. Export data for R comparison
# =============================================================================

println("\n--- Exporting data for R ---")

fixtures_dir = joinpath(@__DIR__, "..", "fixtures")

export_data = DataFrame(
    id = 1:n_subjects,
    time = times,
    status = status
)

export_path = joinpath(fixtures_dir, "comparison_data.csv")
CSV.write(export_path, export_data)
println("Data exported to: $export_path")

# Export evaluation grid and true hazard
export_hazard = DataFrame(
    t = collect(t_eval),
    h_true = h_true,
    h_julia = h_julia
)

export_hazard_path = joinpath(fixtures_dir, "comparison_hazard_grid.csv")
CSV.write(export_hazard_path, export_hazard)

# Export Julia results
julia_results = Dict(
    "spline_coeffs" => spline_coeffs,
    "knots" => knots,
    "interior_knots" => interior_knots,
    "degree" => hazard_obj.degree,
    "natural_spline" => hasfield(typeof(hazard_obj), :natural_spline) ? hazard_obj.natural_spline : false,
    "n_events" => sum(status),
    "n_censored" => sum(1 .- status),
    "rmse" => rmse_julia,
    "lambda" => isnothing(fitted_model.smoothing_parameters) ? [] : collect(fitted_model.smoothing_parameters),
    "edf" => isnothing(fitted_model.edf) ? NaN : (isa(fitted_model.edf, NamedTuple) ? fitted_model.edf.total : fitted_model.edf)
)

julia_results_path = joinpath(fixtures_dir, "julia_fit_results.json")
open(julia_results_path, "w") do f
    JSON.print(f, julia_results, 2)
end

println("Julia results exported to: $julia_results_path")

# =============================================================================
# 5. Generate and run R comparison script
# =============================================================================

r_script = """
# =============================================================================
# R mgcv PAM Comparison Script
# =============================================================================

library(mgcv)
library(survival)
library(jsonlite)

setwd("$(fixtures_dir)")

# Load Julia's data
data <- read.csv("comparison_data.csv")
hazard_grid <- read.csv("comparison_hazard_grid.csv")
julia_results <- fromJSON("julia_fit_results.json")

cat("\\n=== R mgcv PAM Fitting ===\\n")
cat(sprintf("n=%d, events=%d, censored=%d\\n", 
            nrow(data), sum(data\$status), sum(1 - data\$status)))

# -----------------------------------------------------------------------------
# Transform to piecewise-exponential format
# -----------------------------------------------------------------------------

n_intervals <- 50
cut_points <- seq(0, max(data\$time) * 1.01, length.out = n_intervals + 1)

ped_list <- lapply(1:nrow(data), function(i) {
  t_i <- data\$time[i]
  d_i <- data\$status[i]
  
  intervals <- which(cut_points[-length(cut_points)] < t_i)
  if (length(intervals) == 0) return(NULL)
  
  data.frame(
    id = i,
    tstart = cut_points[intervals],
    tend = pmin(cut_points[intervals + 1], t_i),
    interval = intervals,
    ped_status = c(rep(0, length(intervals) - 1), d_i),
    offset = log(pmin(cut_points[intervals + 1], t_i) - cut_points[intervals])
  )
})

ped <- do.call(rbind, ped_list)
ped\$tend_mid <- (ped\$tstart + ped\$tend) / 2

cat(sprintf("PED rows: %d\\n", nrow(ped)))

# -----------------------------------------------------------------------------
# Fit PAM with REML (standard) - match Julia's knot count
# -----------------------------------------------------------------------------

# Use k = number of Julia basis functions + 1 (for mgcv's constraint)
k_mgcv <- length(julia_results\$spline_coeffs) + 1

cat(sprintf("\\nFitting with k=%d basis functions (to match Julia's %d coefficients)...\\n",
            k_mgcv, length(julia_results\$spline_coeffs)))

cat("Fitting with REML...\\n")
pam_reml <- gam(
  ped_status ~ s(tend_mid, k = k_mgcv, bs = "ps"),
  family = poisson(),
  offset = offset,
  data = ped,
  method = "REML"
)

cat("Fitting with NCV (PIJCV)...\\n")
pam_ncv <- gam(
  ped_status ~ s(tend_mid, k = k_mgcv, bs = "ps"),
  family = poisson(),
  offset = offset,
  data = ped,
  method = "NCV"
)

# Also fit with GCV for comparison
cat("Fitting with GCV...\\n")
pam_gcv <- gam(
  ped_status ~ s(tend_mid, k = k_mgcv, bs = "ps"),
  family = poisson(),
  offset = offset,
  data = ped,
  method = "GCV.Cp"
)

# -----------------------------------------------------------------------------
# Extract fitted hazards at evaluation points
# -----------------------------------------------------------------------------

pred_data <- data.frame(
  tend_mid = hazard_grid\$t,
  offset = 0
)

h_reml <- predict(pam_reml, newdata = pred_data, type = "response")
h_ncv <- predict(pam_ncv, newdata = pred_data, type = "response")
h_gcv <- predict(pam_gcv, newdata = pred_data, type = "response")

# -----------------------------------------------------------------------------
# Compare to truth
# -----------------------------------------------------------------------------

rmse_reml <- sqrt(mean((h_reml - hazard_grid\$h_true)^2))
rmse_ncv <- sqrt(mean((h_ncv - hazard_grid\$h_true)^2))
rmse_gcv <- sqrt(mean((h_gcv - hazard_grid\$h_true)^2))
rmse_julia <- sqrt(mean((hazard_grid\$h_julia - hazard_grid\$h_true)^2))

cat(sprintf("\\n=== RMSE vs True Hazard ===\\n"))
cat(sprintf("Julia:     %.6f\\n", rmse_julia))
cat(sprintf("mgcv REML: %.6f\\n", rmse_reml))
cat(sprintf("mgcv NCV:  %.6f\\n", rmse_ncv))
cat(sprintf("mgcv GCV:  %.6f\\n", rmse_gcv))

cat(sprintf("\\n=== Smoothing Parameters ===\\n"))
cat(sprintf("REML: sp=%.4f, EDF=%.2f\\n", pam_reml\$sp[1], sum(pam_reml\$edf)))
cat(sprintf("NCV:  sp=%.4f, EDF=%.2f\\n", pam_ncv\$sp[1], sum(pam_ncv\$edf)))
cat(sprintf("GCV:  sp=%.4f, EDF=%.2f\\n", pam_gcv\$sp[1], sum(pam_gcv\$edf)))

# -----------------------------------------------------------------------------
# Export results
# -----------------------------------------------------------------------------

results <- list(
  reml = list(
    sp = pam_reml\$sp[1],
    edf = sum(pam_reml\$edf),
    rmse = rmse_reml,
    fitted_hazard = as.vector(h_reml)
  ),
  ncv = list(
    sp = pam_ncv\$sp[1],
    edf = sum(pam_ncv\$edf),
    rmse = rmse_ncv,
    fitted_hazard = as.vector(h_ncv)
  ),
  gcv = list(
    sp = pam_gcv\$sp[1],
    edf = sum(pam_gcv\$edf),
    rmse = rmse_gcv,
    fitted_hazard = as.vector(h_gcv)
  ),
  julia = list(
    rmse = rmse_julia,
    fitted_hazard = hazard_grid\$h_julia
  ),
  t_eval = hazard_grid\$t,
  h_true = hazard_grid\$h_true
)

write_json(results, "mgcv_fit_results.json", pretty = TRUE, auto_unbox = TRUE)
cat("\\nResults exported to mgcv_fit_results.json\\n")
"""

r_script_path = joinpath(fixtures_dir, "run_mgcv_comparison.R")
open(r_script_path, "w") do f
    write(f, r_script)
end

println("\n--- Running R comparison ---")
cd(fixtures_dir)

# Run the R script
run(`Rscript run_mgcv_comparison.R`)

# =============================================================================
# 6. Load and display final comparison
# =============================================================================

println("\n" * "=" ^ 70)
println("FINAL COMPARISON SUMMARY")
println("=" ^ 70)

# Load R results
r_results = JSON.parsefile(joinpath(fixtures_dir, "mgcv_fit_results.json"))

println("\nRMSE vs True Hazard (Weibull k=$weibull_k, λ=$weibull_lambda):")
println("  MultistateModels.jl: $(round(rmse_julia, digits=6))")
println("  mgcv REML:           $(round(r_results["reml"]["rmse"], digits=6))")
println("  mgcv NCV (PIJCV):    $(round(r_results["ncv"]["rmse"], digits=6))")
println("  mgcv GCV:            $(round(r_results["gcv"]["rmse"], digits=6))")

println("\nSmoothing Selection:")
println("  Julia (PIJCV): λ=$(isnothing(fitted_model.smoothing_parameters) ? "N/A" : round.(fitted_model.smoothing_parameters, digits=4)), EDF=$(isnothing(fitted_model.edf) ? "N/A" : round(isa(fitted_model.edf, NamedTuple) ? fitted_model.edf.total : fitted_model.edf, digits=2))")
println("  mgcv REML: sp=$(round(r_results["reml"]["sp"], digits=4)), EDF=$(round(r_results["reml"]["edf"], digits=2))")
println("  mgcv NCV:  sp=$(round(r_results["ncv"]["sp"], digits=4)), EDF=$(round(r_results["ncv"]["edf"], digits=2))")
println("  mgcv GCV:  sp=$(round(r_results["gcv"]["sp"], digits=4)), EDF=$(round(r_results["gcv"]["edf"], digits=2))")

println("\nJulia Spline Configuration:")
println("  Basis dimension: $(length(spline_coeffs))")
println("  Degree: $(hazard_obj.degree)")

# Calculate relative performance
best_rmse = minimum([rmse_julia, r_results["reml"]["rmse"], r_results["ncv"]["rmse"], r_results["gcv"]["rmse"]])
println("\nRelative Performance (RMSE / Best RMSE):")
println("  MultistateModels.jl: $(round(rmse_julia / best_rmse, digits=3))")
println("  mgcv REML:           $(round(r_results["reml"]["rmse"] / best_rmse, digits=3))")
println("  mgcv NCV:            $(round(r_results["ncv"]["rmse"] / best_rmse, digits=3))")
println("  mgcv GCV:            $(round(r_results["gcv"]["rmse"] / best_rmse, digits=3))")

println("\n" * "=" ^ 70)
