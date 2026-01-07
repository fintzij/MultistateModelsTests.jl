# Generate simple benchmark fixture with all smoothing methods
using Pkg
Pkg.activate(".")

using MultistateModels
using Random
using JSON
using DataFrames
using LinearAlgebra
using Printf

# Configuration
n = 100
true_shape = 1.5
true_rate = 0.3
max_time = 5.0
seed = 12345

Random.seed!(seed)

# Simulate Weibull survival data
E = -log.(rand(n))
event_times = (E ./ true_rate) .^ (1 / true_shape)
obs_times = min.(event_times, max_time)
status = Int.(event_times .<= max_time)

println("=== Simple Survival Data ===")
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

# Define model with spline hazard - use explicit knots for stability
h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2;
             degree = 3,
             knots = [1.0, 2.0, 3.0],
             boundaryknots = [0.0, 5.0],
             natural_spline = true)
model = multistatemodel(h12; data=surv_data)

# Fit with each smoothing method
results = Dict{String, Any}()

println("\nFitting with PIJCV method...")
@time result_pijcv = select_smoothing_parameters(model, SplinePenalty(); 
                                                  method = :pijcv, verbose = true)
results["PIJCV"] = Dict(
    "lambda" => result_pijcv.lambda[1],
    "log_lambda" => log(result_pijcv.lambda[1]),
    "edf" => result_pijcv.edf.total,
    "coefficients" => result_pijcv.beta
)
println("  PIJCV λ = $(round(result_pijcv.lambda[1], digits=2)), EDF = $(round(result_pijcv.edf.total, digits=2))")

println("\nFitting with PERF method...")
@time result_perf = select_smoothing_parameters(model, SplinePenalty(); 
                                                 method = :perf, verbose = true)
results["PERF"] = Dict(
    "lambda" => result_perf.lambda[1],
    "log_lambda" => log(result_perf.lambda[1]),
    "edf" => result_perf.edf.total,
    "coefficients" => result_perf.beta
)
println("  PERF λ = $(round(result_perf.lambda[1], digits=2)), EDF = $(round(result_perf.edf.total, digits=2))")

println("\nFitting with EFS method...")
@time result_efs = select_smoothing_parameters(model, SplinePenalty(); 
                                                method = :efs, verbose = true)
results["EFS"] = Dict(
    "lambda" => result_efs.lambda[1],
    "log_lambda" => log(result_efs.lambda[1]),
    "edf" => result_efs.edf.total,
    "coefficients" => result_efs.beta
)
println("  EFS λ = $(round(result_efs.lambda[1], digits=2)), EDF = $(round(result_efs.edf.total, digits=2))")

println("\nFitting with LOOCV method (exact, slow)...")
@time result_loocv = select_smoothing_parameters(model, SplinePenalty(); 
                                                  method = :loocv, verbose = true)
results["LOOCV"] = Dict(
    "lambda" => result_loocv.lambda[1],
    "log_lambda" => log(result_loocv.lambda[1]),
    "edf" => result_loocv.edf.total,
    "coefficients" => result_loocv.beta
)
println("  LOOCV λ = $(round(result_loocv.lambda[1], digits=2)), EDF = $(round(result_loocv.edf.total, digits=2))")

# Extract penalty matrix S from the penalty configuration
# This is the raw second-order difference penalty matrix
penalty_matrix_S = result_pijcv.penalty_config.terms[1].S

# Extract spline knots from the model for R to reconstruct the basis
# The hazard uses a natural cubic spline (degree=3)
knots_internal = [1.0, 2.0, 3.0]
knots_boundary = [0.0, 5.0]

# Generate evaluation grid for hazard/survival curves
# This avoids basis reconstruction issues in R
eval_times = collect(range(0.01, max_time, length=200))

# Helper function to evaluate fitted hazard at a grid of times
function evaluate_fitted_hazard(model, beta, eval_times)
    # Get the hazard function
    haz = model.hazards[1]
    
    # Evaluate hazard at each time point
    hazard_vals = Float64[]
    cumhaz_vals = Float64[]
    
    for t in eval_times
        # The hazard function expects (t, pars, covars)
        # pars is a vector of parameters, covars is empty for no-covariate model
        h = haz.hazard(t, beta, ())
        push!(hazard_vals, h)
        
        # Cumulative hazard from 0 to t
        H = haz.cumhaz(0.0, t, beta, ())
        push!(cumhaz_vals, H)
    end
    
    # Survival = exp(-cumhaz)
    survival_vals = exp.(-cumhaz_vals)
    
    return Dict(
        "hazard" => hazard_vals,
        "cumhaz" => cumhaz_vals,
        "survival" => survival_vals
    )
end

# Evaluate fitted curves for each method
println("\nEvaluating fitted curves...")
for (method_name, method_result) in [
    ("PIJCV", result_pijcv),
    ("PERF", result_perf),
    ("EFS", result_efs),
    ("LOOCV", result_loocv)
]
    curves = evaluate_fitted_hazard(model, method_result.beta, eval_times)
    results[method_name]["curves"] = curves
end

# Create output structure
output = Dict(
    "config" => Dict(
        "n" => n,
        "true_shape" => true_shape,
        "true_rate" => true_rate,
        "max_time" => max_time,
        "seed" => seed,
        "n_basis" => size(penalty_matrix_S, 1),
        "degree" => 3,
        "penalty_order" => 2,
        "knots_internal" => knots_internal,
        "knots_boundary" => knots_boundary
    ),
    "data" => Dict(
        "time" => collect(obs_times),
        "status" => collect(status)
    ),
    "eval_times" => eval_times,
    "results" => results,
    "penalty_matrix" => Dict(
        "S" => [collect(row) for row in eachrow(penalty_matrix_S)],
        "dim" => size(penalty_matrix_S, 1),
        "description" => "Second-order difference penalty matrix (D'D)"
    )
)

# Save to JSON
outfile = "MultistateModelsTests/fixtures/simple_benchmark_all_methods.json"
open(outfile, "w") do io
    JSON.print(io, output, 2)
end

println("\nResults saved to: $outfile")

# Print summary table
println("\n" * "="^60)
println("SMOOTHING PARAMETER SUMMARY")
println("="^60)
println("Method      λ           log(λ)      EDF")
println("-"^60)
for method in ["PIJCV", "PERF", "EFS", "LOOCV"]
    res = results[method]
    @printf("%-10s  %8.2f    %6.2f      %5.2f\n", 
            method, res["lambda"], res["log_lambda"], res["edf"])
end
println("="^60)
