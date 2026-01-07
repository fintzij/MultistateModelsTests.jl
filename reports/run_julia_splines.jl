#!/usr/bin/env julia
# Pre-render script for spline comparison benchmark
# Run this before rendering the Quarto document

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using MultistateModels
using Random
using DataFrames
using CSV
using Printf

# Configuration
n = 100
true_shape = 1.5
true_rate = 0.3
max_time = 5.0
seed = 12345

println("=== Simulating Data ===")
Random.seed!(seed)

# Simulate Weibull survival data
E = -log.(rand(n))
event_times = (E ./ true_rate) .^ (1 / true_shape)
obs_times = min.(event_times, max_time)
status = Int.(event_times .<= max_time)

println("n = $n")
println("Events: $(sum(status))")
println("Censored: $(n - sum(status))")
println("Time range: [$(round(minimum(obs_times), digits=3)), $(round(maximum(obs_times), digits=3))]")

# Create multistate model data
surv_data = DataFrame(
    id = 1:n,
    tstart = zeros(n),
    tstop = obs_times,
    statefrom = ones(Int, n),
    stateto = ifelse.(status .== 1, 2, 1),
    obstype = ones(Int, n)
)

# Save to CSV for R to read
CSV.write(joinpath(@__DIR__, "_surv_data.csv"), DataFrame(time = obs_times, status = status))

# Define model with spline hazard
println("\n=== Fitting Models ===")
h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2;
             degree = 3,
             knots = [1.0, 2.0, 3.0],
             boundaryknots = [0.0, 5.0],
             natural_spline = true)
model = multistatemodel(h12; data=surv_data)

# Fit with each smoothing method
println("\nFitting with PIJCV method...")
result_pijcv = select_smoothing_parameters(model, SplinePenalty(); 
                                           method = :pijcv, verbose = false)
println("  PIJCV λ = $(round(result_pijcv.lambda[1], digits=4)), EDF = $(round(result_pijcv.edf.total, digits=2))")

println("\nFitting with EFS method...")
result_efs = select_smoothing_parameters(model, SplinePenalty(); 
                                         method = :efs, verbose = false)
println("  EFS λ = $(round(result_efs.lambda[1], digits=4)), EDF = $(round(result_efs.edf.total, digits=2))")

println("\nFitting with LOOCV method...")
result_loocv = select_smoothing_parameters(model, SplinePenalty(); 
                                           method = :loocv, verbose = false)
println("  LOOCV λ = $(round(result_loocv.lambda[1], digits=4)), EDF = $(round(result_loocv.edf.total, digits=2))")

# Evaluation grid
println("\n=== Computing Curves ===")
eval_times = collect(range(0.01, max_time, length=200))

# Function to evaluate hazard at a grid of times
function evaluate_curves(model, beta, eval_times)
    haz = model.hazards[1]
    
    # Use hazard_fn and cumhaz_fn (the callable functions)
    hazard_vals = [haz.hazard_fn(t, beta, ()) for t in eval_times]
    cumhaz_vals = [haz.cumhaz_fn(0.0, t, beta, ()) for t in eval_times]
    survival_vals = exp.(-cumhaz_vals)
    
    return (hazard = hazard_vals, cumhaz = cumhaz_vals, survival = survival_vals)
end

# Compute curves for each method
curves_pijcv = evaluate_curves(model, result_pijcv.beta, eval_times)
curves_efs = evaluate_curves(model, result_efs.beta, eval_times)
curves_loocv = evaluate_curves(model, result_loocv.beta, eval_times)

# Save Julia results for R plotting
julia_results = DataFrame(
    time = repeat(eval_times, 3),
    hazard = vcat(curves_pijcv.hazard, curves_efs.hazard, curves_loocv.hazard),
    cumhaz = vcat(curves_pijcv.cumhaz, curves_efs.cumhaz, curves_loocv.cumhaz),
    survival = vcat(curves_pijcv.survival, curves_efs.survival, curves_loocv.survival),
    method = repeat(["Julia PIJCV", "Julia EFS", "Julia LOOCV"], inner=length(eval_times))
)
CSV.write(joinpath(@__DIR__, "_julia_curves.csv"), julia_results)

# Save summary stats
julia_summary = DataFrame(
    method = ["PIJCV", "EFS", "LOOCV"],
    lambda = [result_pijcv.lambda[1], result_efs.lambda[1], result_loocv.lambda[1]],
    edf = [result_pijcv.edf.total, result_efs.edf.total, result_loocv.edf.total]
)
CSV.write(joinpath(@__DIR__, "_julia_summary.csv"), julia_summary)

println("Julia curves computed and saved to CSV files.")
println("  - _surv_data.csv")
println("  - _julia_curves.csv")
println("  - _julia_summary.csv")
