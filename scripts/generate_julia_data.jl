# Generate survival data using MultistateModels.jl and export for R comparison
# This tests spline fitting on data from our own package
#
# PARAMETERIZATION NOTES:
# - MultistateModels.jl Weibull: h(t) = shape * rate * t^(shape-1) where scale is a RATE
# - Storage: [log(shape), log(rate)]
# - This is NOT flexsurv's scale parameterization
# - Cumulative hazard: H(t) = rate * t^shape

using MultistateModels
using DataFrames
using CSV
using JSON
using Random
using Statistics

println("="^60)
println("Generating Weibull Survival Data with MultistateModels.jl")
println("="^60)

Random.seed!(12345)

# True parameters (our rate parameterization)
# h(t) = shape * rate * t^(shape-1)
# H(t) = rate * t^shape
true_shape = 1.5  # Weibull shape (κ)
true_rate = 0.02  # Weibull rate (λ) - small rate = longer survival
n_subj = 500
max_time = 30.0

println("\nTrue hazard (our parameterization):")
println("  shape = $true_shape")
println("  rate = $true_rate")
println("  Formula: h(t) = shape * rate * t^(shape-1) = $true_shape * $true_rate * t^$(true_shape-1)")

# Theoretical median: solve H(t) = ln(2) => rate * t^shape = ln(2) => t = (ln(2)/rate)^(1/shape)
theoretical_median = (log(2) / true_rate)^(1/true_shape)
println("  Theoretical median: $(round(theoretical_median, digits=3))")

# Create template data for simulation
template_rows = []
for subj in 1:n_subj
    push!(template_rows, (
        id = subj,
        tstart = 0.0,
        tstop = max_time,
        statefrom = 1,
        stateto = 1,  # Will be updated by simulation
        obstype = 1   # Exact observation
    ))
end
template_data = DataFrame(template_rows)

# Create Weibull hazard model
h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)

# Build model
model_sim = multistatemodel(h12; data=template_data, surrogate=:markov, initialize=false)

# Set true parameters: [log(shape), log(rate)]
println("\nSetting parameters...")
println("  Estimation scale: [log(shape), log(rate)] = [$(log(true_shape)), $(log(true_rate))]")

set_parameters!(model_sim, (h12 = [log(true_shape), log(true_rate)],))

# Verify what we get back
p_est = get_parameters(model_sim; scale=:estimation)
p_nat = get_parameters(model_sim; scale=:natural)
println("  Retrieved (estimation): $p_est")
println("  Retrieved (natural): shape=$(p_nat.h12[1]), rate=$(p_nat.h12[2])")

# Simulate data
println("\nSimulating survival times...")
obstype_map = Dict(1 => 1)  # Transition 1→2 gets exact observation
sim_result = simulate(model_sim; paths=false, data=true, nsim=1, autotmax=false,
                     obstype_by_transition=obstype_map)
sim_data = sim_result[1, 1]

# Analyze simulated data
n_events = sum(sim_data.stateto .== 2)
n_censored = sum(sim_data.stateto .== 1)
event_times = sim_data.tstop[sim_data.stateto .== 2]

println("\nSimulated data summary:")
println("  Total subjects: $n_subj")
println("  Events: $n_events")
println("  Censored: $n_censored")
if length(event_times) > 0
    println("  Event time range: [$(round(minimum(event_times), digits=3)), $(round(maximum(event_times), digits=3))]")
    println("  Median event time: $(round(median(event_times), digits=3))")
    println("  Expected median: $(round(theoretical_median, digits=3))")
end

# Theoretical median for Weibull with rate param: (ln(2)/rate)^(1/shape)
println("  Theoretical median: $(round(theoretical_median, digits=3))")

# Export data for R
println("\n" * "="^60)
println("Exporting data for R")
println("="^60)

# Prepare export dataframe
export_df = DataFrame(
    id = sim_data.id,
    time = sim_data.tstop,
    status = Int.(sim_data.stateto .== 2)  # 1 = event, 0 = censored
)

# Save as CSV
csv_path = joinpath(@__DIR__, "..", "fixtures", "julia_generated_data.csv")
CSV.write(csv_path, export_df)
println("Saved CSV: $csv_path")

# Also save metadata as JSON
metadata = Dict(
    "true_shape" => true_shape,
    "true_rate" => true_rate,
    "hazard_type" => "weibull_rate",
    "hazard_formula" => "h(t) = shape * rate * t^(shape-1)",
    "cumhaz_formula" => "H(t) = rate * t^shape",
    "n_subjects" => n_subj,
    "n_events" => n_events,
    "n_censored" => n_censored,
    "max_time" => max_time,
    "seed" => 12345,
    "parameters_estimation_scale" => [log(true_shape), log(true_rate)],
    "theoretical_median" => theoretical_median
)
json_path = joinpath(@__DIR__, "..", "fixtures", "julia_generated_data.json")
open(json_path, "w") do f
    JSON.print(f, metadata, 2)
end
println("Saved JSON: $json_path")

# Now fit a spline model on this data
println("\n" * "="^60)
println("Fitting Spline Hazard in Julia")
println("="^60)

# Use the simulated data directly
fit_data = sim_data

# Create spline hazard with interior knots
time_range = extrema(fit_data.tstop[fit_data.tstop .> 0])
interior_knots = range(time_range[1] + 0.5, time_range[2] - 0.5, length=5)
println("\nSpline configuration:")
println("  Interior knots: $interior_knots")

h12_sp = Hazard(@formula(0 ~ 1), "sp", 1, 2; knots=collect(interior_knots))
model_sp = multistatemodel(h12_sp; data=fit_data, surrogate=:markov)

# Fit the model
println("\nFitting spline model...")
fitted_sp = fit(model_sp;
    proposal=:markov,
    verbose=false,
    maxiter=100,
    tol=1e-4,
    ess_target_initial=100,
    max_ess=1000,
    compute_vcov=false)

println("Fitting complete!")

# Get fitted parameters
p_sp = get_parameters(fitted_sp; scale=:natural)
println("\nFitted spline coefficients: $(round.(p_sp.h12, digits=4))")

# Evaluate true hazard at grid of time points
eval_times = range(0.5, min(max_time, maximum(fit_data.tstop)), length=50)

# True Weibull hazard (our rate parameterization)
true_hazard(t) = true_shape * true_rate * t^(true_shape - 1)

# Compute true hazards for reference
true_hazards = [true_hazard(t) for t in eval_times]

println("\n" * "="^60)
println("True Hazard Values (for R comparison)")
println("="^60)
println("  Mean true hazard: $(round(mean(true_hazards), digits=6))")
println("  Min true hazard: $(round(minimum(true_hazards), digits=6))")
println("  Max true hazard: $(round(maximum(true_hazards), digits=6))")

# Save results for comparison
results = Dict(
    "eval_times" => collect(eval_times),
    "true_hazards" => true_hazards,
    "spline_coefficients" => collect(p_sp.h12),
    "interior_knots" => collect(interior_knots),
    "true_shape" => true_shape,
    "true_rate" => true_rate
)
results_path = joinpath(@__DIR__, "..", "fixtures", "julia_spline_results.json")
open(results_path, "w") do f
    JSON.print(f, results, 2)
end
println("\nSaved Julia results: $results_path")

# Print sample hazard values for inspection
println("\n" * "="^60)
println("Sample True Hazard Values")
println("="^60)
println("Time\t\tTrue h(t)")
for i in [1, 10, 20, 30, 40, 50]
    if i <= length(eval_times)
        t = eval_times[i]
        h_t = true_hazards[i]
        println("$(round(t, digits=2))\t\t$(round(h_t, digits=6))")
    end
end

println("\n" * "="^60)
println("Next: Run R script to fit mgcv on julia_generated_data.csv")
println("="^60)
