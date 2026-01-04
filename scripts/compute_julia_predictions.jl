# Compute Julia spline-predicted prevalence and hazards
# To complete the benchmark comparison

using MultistateModels
using DataFrames
using CSV
using JSON
using Random
using Statistics
using LinearAlgebra

println("="^70)
println("Computing Julia Spline Predictions")
println("="^70)

# Load metadata and data
meta_path = joinpath(@__DIR__, "..", "fixtures", "illness_death_metadata.json")
metadata = JSON.parsefile(meta_path)

data_path = joinpath(@__DIR__, "..", "fixtures", "illness_death_data.csv")
sim_data = CSV.read(data_path, DataFrame)

# Rename columns and add obstype
rename!(sim_data, :from => :statefrom, :to => :stateto, :status => :_status)
sim_data.obstype .= 1

# Reorder columns to match MultistateModels expectation
sim_data = select(sim_data, :id, :tstart, :tstop, :statefrom, :stateto, :obstype, :_status)

# True parameters
true_shape_12 = metadata["true_params"]["h12"]["shape"]
true_rate_12 = metadata["true_params"]["h12"]["rate"]
true_shape_13 = metadata["true_params"]["h13"]["shape"]
true_rate_13 = metadata["true_params"]["h13"]["rate"]
true_shape_23 = metadata["true_params"]["h23"]["shape"]
true_rate_23 = metadata["true_params"]["h23"]["rate"]

eval_times = metadata["eval_times"]

# Load Julia fit results
results_path = joinpath(@__DIR__, "..", "fixtures", "illness_death_results.json")
julia_fit_results = JSON.parsefile(results_path)

interior_knots = Float64.(julia_fit_results["julia_fit"]["interior_knots"])

println("\nInterior knots: $(round.(interior_knots, digits=2))")

# Recreate the spline model and set fitted parameters
h12_sp = Hazard(@formula(0 ~ 1), "sp", 1, 2; knots=interior_knots)
h13_sp = Hazard(@formula(0 ~ 1), "sp", 1, 3; knots=interior_knots)
h23_sp = Hazard(@formula(0 ~ 1), "sp", 2, 3; knots=interior_knots)

model_sp = multistatemodel(h12_sp, h13_sp, h23_sp; data=sim_data, surrogate=:markov)

# Set fitted parameters using FLAT scale (log scale for hazard evaluation)
fitted_params_flat = (
    h12 = Float64.(julia_fit_results["julia_fit"]["spline_coeffs_flat_h12"]),
    h13 = Float64.(julia_fit_results["julia_fit"]["spline_coeffs_flat_h13"]),
    h23 = Float64.(julia_fit_results["julia_fit"]["spline_coeffs_flat_h23"])
)
# Natural scale for display
fitted_params = (
    h12 = julia_fit_results["julia_fit"]["spline_coeffs_h12"],
    h13 = julia_fit_results["julia_fit"]["spline_coeffs_h13"],
    h23 = julia_fit_results["julia_fit"]["spline_coeffs_h23"]
)
set_parameters!(model_sp, fitted_params_flat)

println("Fitted coefficients (natural scale for display):")
println("  h12: $(round.(fitted_params.h12, digits=4))")
println("  h13: $(round.(fitted_params.h13, digits=4))")
println("  h23: $(round.(fitted_params.h23, digits=4))")

println("\nFlat coefficients (log scale for hazard evaluation):")
println("  h12: $(round.(fitted_params_flat.h12, digits=4))")

# ============================================================================
# Compute Julia Spline Hazard Predictions
# ============================================================================

println("\n" * "="^70)
println("Computing Hazard Predictions")
println("="^70)

# Access the hazard objects from the model
# The model has hazards indexed by transition ID
# We need to compute hazards at each time point

# Create a dummy subject to compute hazards
dummy_row = DataFrame(
    id = [1],
    tstart = [0.0],
    tstop = [20.0],
    statefrom = [1],
    stateto = [1],
    _status = [0]
)

# Get hazard values at eval_times
# Using the internal hazard function
h12_julia = Float64[]
h13_julia = Float64[]
h23_julia = Float64[]

# True hazard functions
h12_true(t) = true_shape_12 * true_rate_12 * t^(true_shape_12 - 1)
h13_true(t) = true_shape_13 * true_rate_13 * t^(true_shape_13 - 1)
h23_true(t) = true_shape_23 * true_rate_23 * t^(true_shape_23 - 1)

h12_true_vec = h12_true.(eval_times)
h13_true_vec = h13_true.(eval_times)
h23_true_vec = h23_true.(eval_times)

# Get the hazard functions from the model
# model_sp.hazards is a vector of hazard objects keyed by transition index
# We can call them directly: hazard(t, pars, covars)

# Find which hazard corresponds to which transition
# The model stores hazards in order of (from, to) transitions
haz_12 = model_sp.hazards[1]  # 1→2
haz_13 = model_sp.hazards[2]  # 1→3
haz_23 = model_sp.hazards[3]  # 2→3

# Get parameters for each hazard - use FLAT (log scale) params
params_12 = fitted_params_flat.h12
params_13 = fitted_params_flat.h13
params_23 = fitted_params_flat.h23

# Debug: Check parameter values and what the hazard should be
println("\nDebug: Flat parameter values (log scale for splines):")
println("  params_12: $(round.(params_12, digits=4))")
println("  At t=10, true h12 = $(round(h12_true(10.0), digits=4))")

# Test hazard evaluation
test_h12 = haz_12(10.0, params_12, Float64[])
println("  At t=10, Julia spline h12 = $(round(test_h12, digits=4))")

println("\nComputing spline hazards at eval times...")

for t in eval_times
    # Evaluate hazard at time t with no covariates
    push!(h12_julia, haz_12(t, params_12, Float64[]))
    push!(h13_julia, haz_13(t, params_13, Float64[]))
    push!(h23_julia, haz_23(t, params_23, Float64[]))
end

# Compute RMSE
rmse(a, b) = sqrt(mean((a .- b).^2))

println("\nJulia Spline Hazard RMSE (vs True):")
println("  h12: $(round(rmse(h12_julia, h12_true_vec), digits=5))")
println("  h13: $(round(rmse(h13_julia, h13_true_vec), digits=5))")
println("  h23: $(round(rmse(h23_julia, h23_true_vec), digits=5))")

# ============================================================================
# Compute Julia Spline Prevalence via Product Integral
# ============================================================================

println("\n" * "="^70)
println("Computing Julia Spline State Prevalence")
println("="^70)

# Simple linear interpolation function (avoid Interpolations.jl dependency)
function lininterp(xs, ys, x)
    # Find bracketing indices
    if x <= xs[1]
        return ys[1]
    elseif x >= xs[end]
        return ys[end]
    end
    
    i = searchsortedfirst(xs, x)
    if i == 1
        return ys[1]
    end
    
    # Linear interpolation
    t = (x - xs[i-1]) / (xs[i] - xs[i-1])
    return ys[i-1] + t * (ys[i] - ys[i-1])
end

h12_interp(t) = lininterp(eval_times, h12_julia, t)
h13_interp(t) = lininterp(eval_times, h13_julia, t)
h23_interp(t) = lininterp(eval_times, h23_julia, t)

function compute_tpm_julia(t; dt=0.01)
    """Compute transition probability matrix using Julia spline hazards."""
    if t <= 0
        return Matrix{Float64}(I, 3, 3)
    end
    
    n_steps = max(1, Int(ceil(t / dt)))
    actual_dt = t / n_steps
    
    P = Matrix{Float64}(I, 3, 3)
    
    for i in 1:n_steps
        s = (i - 0.5) * actual_dt  # midpoint
        s = max(0.5, min(s, 20.0))  # clamp to eval_times range
        
        # Hazards at midpoint from Julia splines
        h12 = h12_interp(s)
        h13 = h13_interp(s)
        h23 = h23_interp(s)
        
        # Generator matrix
        Q = [-(h12 + h13)  h12  h13;
             0.0           -h23 h23;
             0.0           0.0  0.0]
        
        # Matrix exponential for small dt
        dP = exp(Q * actual_dt)
        P = P * dP
    end
    
    return P
end

# Compute prevalence at eval times
julia_prev_1 = Float64[]
julia_prev_2 = Float64[]
julia_prev_3 = Float64[]

println("\nComputing state prevalence...")
for t in eval_times
    P = compute_tpm_julia(t)
    push!(julia_prev_1, P[1, 1])
    push!(julia_prev_2, P[1, 2])
    push!(julia_prev_3, P[1, 3])
end

# True prevalence (from results file)
true_prev_1 = julia_fit_results["true"]["prevalence_healthy"]
true_prev_2 = julia_fit_results["true"]["prevalence_illness"]
true_prev_3 = julia_fit_results["true"]["prevalence_death"]

println("\nJulia Spline Prevalence RMSE (vs True):")
println("  P(Healthy): $(round(rmse(julia_prev_1, true_prev_1), digits=5))")
println("  P(Illness): $(round(rmse(julia_prev_2, true_prev_2), digits=5))")
println("  P(Death):   $(round(rmse(julia_prev_3, true_prev_3), digits=5))")

# ============================================================================
# Save Complete Results
# ============================================================================

println("\n" * "="^70)
println("Saving Complete Julia Predictions")
println("="^70)

julia_predictions = Dict(
    "eval_times" => eval_times,
    "hazards" => Dict(
        "h12_julia" => h12_julia,
        "h13_julia" => h13_julia,
        "h23_julia" => h23_julia,
        "h12_true" => h12_true_vec,
        "h13_true" => h13_true_vec,
        "h23_true" => h23_true_vec
    ),
    "prevalence" => Dict(
        "julia_healthy" => julia_prev_1,
        "julia_illness" => julia_prev_2,
        "julia_death" => julia_prev_3,
        "true_healthy" => true_prev_1,
        "true_illness" => true_prev_2,
        "true_death" => true_prev_3
    ),
    "rmse" => Dict(
        "hazard_h12" => rmse(h12_julia, h12_true_vec),
        "hazard_h13" => rmse(h13_julia, h13_true_vec),
        "hazard_h23" => rmse(h23_julia, h23_true_vec),
        "prev_healthy" => rmse(julia_prev_1, true_prev_1),
        "prev_illness" => rmse(julia_prev_2, true_prev_2),
        "prev_death" => rmse(julia_prev_3, true_prev_3)
    )
)

pred_path = joinpath(@__DIR__, "..", "fixtures", "illness_death_julia_predictions.json")
open(pred_path, "w") do f
    JSON.print(f, julia_predictions, 2)
end
println("Saved: $pred_path")

# Print sample values
println("\n" * "="^70)
println("Sample Values")
println("="^70)

println("\n--- Hazards at t=10 ---")
t_idx = findfirst(x -> x ≈ 10.0, eval_times)
println("  h12: Julia=$(round(h12_julia[t_idx], digits=4)), True=$(round(h12_true_vec[t_idx], digits=4))")
println("  h13: Julia=$(round(h13_julia[t_idx], digits=4)), True=$(round(h13_true_vec[t_idx], digits=4))")
println("  h23: Julia=$(round(h23_julia[t_idx], digits=4)), True=$(round(h23_true_vec[t_idx], digits=4))")

println("\n--- P(Illness) over time ---")
println("Time\tJulia\t\tTrue")
for i in [1, 10, 20, 30, 40]
    if i <= length(eval_times)
        t = eval_times[i]
        println("$(round(t, digits=1))\t$(round(julia_prev_2[i], digits=4))\t\t$(round(true_prev_2[i], digits=4))")
    end
end

println("\nDone!")
