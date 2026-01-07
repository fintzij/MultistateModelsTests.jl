using MultistateModels, DataFrames, CSV, Random, StatsBase, LinearAlgebra, JSON
import MultistateModels: compute_hazard, tpm_ode

println("="^60)
println("Regenerating illness-death benchmark fixtures...")
println("="^60)

fixtures_dir = @__DIR__
dat = CSV.read(joinpath(fixtures_dir, "illness_death_data.csv"), DataFrame)
meta = JSON.parsefile(joinpath(fixtures_dir, "illness_death_metadata.json"))
println("Loaded $(nrow(dat)) rows, $(meta["n_subjects"]) subjects")

sim_data = DataFrame(
    id = dat.id, tstart = dat.tstart, tstop = dat.tstop, statefrom = dat.from,
    stateto = ifelse.(dat.status .== 1, dat.to, dat.from),
    obstype = ifelse.(dat.status .== 1, 1, 2)
)

event_times = dat.tstop[(dat.status .== 1)]
interior_knots = StatsBase.quantile(event_times, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
boundary_knots = [minimum(event_times) * 0.99, maximum(event_times) * 1.01]
println("Knots: $(length(interior_knots)) interior")

h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; knots=interior_knots, boundaryknots=boundary_knots)
h13 = Hazard(@formula(0 ~ 1), "sp", 1, 3; knots=interior_knots, boundaryknots=boundary_knots)
h23 = Hazard(@formula(0 ~ 1), "sp", 2, 3; knots=interior_knots, boundaryknots=boundary_knots)
model_sp = multistatemodel(h12, h13, h23; data=sim_data)
penalty = SplinePenalty(:all; order=2)

println("\nSelecting λ with PIJCV...")
flush(stdout)
t_select = @elapsed smoothing_result = select_smoothing_parameters(model_sp, penalty; method=:pijcv, verbose=true)
optimal_lambda = smoothing_result.lambda[1]
println("Optimal λ = $(round(optimal_lambda, digits=2)) ($(round(t_select, digits=1))s)")

println("\nFitting final model with optimal penalty config...")
t_fit = @elapsed fitted = fit(model_sp; penalty_config=smoothing_result.penalty_config, verbose=false, compute_vcov=false)
println("Fit complete ($(round(t_fit, digits=2))s)")

params = get_parameters(fitted)
coefs = Dict("h12" => params.h12, "h13" => params.h13, "h23" => params.h23)

eval_times = meta["eval_times"]
h12_vals = [compute_hazard(fitted.model.hazards[1], t, nothing, fitted.parameters) for t in eval_times]
h13_vals = [compute_hazard(fitted.model.hazards[2], t, nothing, fitted.parameters) for t in eval_times]
h23_vals = [compute_hazard(fitted.model.hazards[3], t, nothing, fitted.parameters) for t in eval_times]

true_h12 = [meta["true_params"]["h12"]["shape"] * meta["true_params"]["h12"]["rate"] * t^(meta["true_params"]["h12"]["shape"]-1) for t in eval_times]
true_h13 = [meta["true_params"]["h13"]["shape"] * meta["true_params"]["h13"]["rate"] * t^(meta["true_params"]["h13"]["shape"]-1) for t in eval_times]
true_h23 = [meta["true_params"]["h23"]["shape"] * meta["true_params"]["h23"]["rate"] * t^(meta["true_params"]["h23"]["shape"]-1) for t in eval_times]

println("Computing transition probabilities...")
tpm_values = Dict("p11" => Float64[], "p12" => Float64[], "p13" => Float64[])
for t in eval_times
    P_t = tpm_ode(fitted.model, fitted.parameters, 0.0, t; tol=1e-8)
    push!(tpm_values["p11"], P_t[1,1]); push!(tpm_values["p12"], P_t[1,2]); push!(tpm_values["p13"], P_t[1,3])
end

function true_tpm(t_end, p; n_steps=1000)
    t_end == 0.0 && return Matrix{Float64}(I, 3, 3)
    dt = t_end / n_steps; P = Matrix{Float64}(I, 3, 3)
    for i in 1:n_steps
        t = (i - 0.5) * dt
        h12 = p["h12"]["shape"] * p["h12"]["rate"] * t^(p["h12"]["shape"]-1)
        h13 = p["h13"]["shape"] * p["h13"]["rate"] * t^(p["h13"]["shape"]-1)
        h23 = p["h23"]["shape"] * p["h23"]["rate"] * t^(p["h23"]["shape"]-1)
        Q = [-(h12+h13) h12 h13; 0.0 -h23 h23; 0.0 0.0 0.0]
        P = P * exp(Q * dt)
    end
    return P
end
true_tpm_values = Dict("p11" => Float64[], "p12" => Float64[], "p13" => Float64[])
for t in eval_times
    P_true = true_tpm(t, meta["true_params"])
    push!(true_tpm_values["p11"], P_true[1,1]); push!(true_tpm_values["p12"], P_true[1,2]); push!(true_tpm_values["p13"], P_true[1,3])
end

function compute_cif(h_vals, p11_vals, times)
    cif = zeros(length(times))
    for i in 2:length(times)
        dt = times[i] - times[i-1]
        cif[i] = cif[i-1] + 0.5 * dt * (p11_vals[i-1] * h_vals[i-1] + p11_vals[i] * h_vals[i])
    end
    return cif
end
ci_illness_julia = compute_cif(h12_vals, tpm_values["p11"], eval_times)
ci_death_julia = compute_cif(h13_vals, tpm_values["p11"], eval_times)
ci_illness_true = compute_cif(true_h12, true_tpm_values["p11"], eval_times)
ci_death_true = compute_cif(true_h13, true_tpm_values["p11"], eval_times)

println("\nWriting fixtures...")
results = Dict(
    "julia_fit" => Dict("fit_time_seconds" => t_fit, "selection_time_seconds" => t_select,
        "optimal_lambda" => optimal_lambda, "interior_knots" => interior_knots,
        "boundary_knots" => boundary_knots, "coefficients" => coefs, "method" => "pijcv"),
    "true" => Dict("hazard_h12" => true_h12, "hazard_h13" => true_h13, "hazard_h23" => true_h23,
        "prevalence_healthy" => true_tpm_values["p11"], "prevalence_illness" => true_tpm_values["p12"],
        "prevalence_death" => true_tpm_values["p13"], "ci_illness" => ci_illness_true, "ci_death_direct" => ci_death_true),
    "julia" => Dict("hazard_h12" => h12_vals, "hazard_h13" => h13_vals, "hazard_h23" => h23_vals,
        "prevalence_healthy" => tpm_values["p11"], "prevalence_illness" => tpm_values["p12"],
        "prevalence_death" => tpm_values["p13"], "ci_illness" => ci_illness_julia, "ci_death_direct" => ci_death_julia)
)
open(joinpath(fixtures_dir, "illness_death_results.json"), "w") do f; JSON.print(f, results, 2); end

predictions = Dict("eval_times" => eval_times,
    "hazards" => Dict("h12_julia" => h12_vals, "h13_julia" => h13_vals, "h23_julia" => h23_vals,
        "h12_true" => true_h12, "h13_true" => true_h13, "h23_true" => true_h23),
    "transition_probs" => Dict("p11_julia" => tpm_values["p11"], "p12_julia" => tpm_values["p12"],
        "p13_julia" => tpm_values["p13"], "p11_true" => true_tpm_values["p11"],
        "p12_true" => true_tpm_values["p12"], "p13_true" => true_tpm_values["p13"]),
    "cumulative_incidence" => Dict("ci_illness_julia" => ci_illness_julia, "ci_death_direct_julia" => ci_death_julia,
        "ci_illness_true" => ci_illness_true, "ci_death_direct_true" => ci_death_true)
)
open(joinpath(fixtures_dir, "illness_death_julia_predictions.json"), "w") do f; JSON.print(f, predictions, 2); end

rmse_h12 = sqrt(sum((h12_vals .- true_h12).^2)/length(h12_vals))
rmse_h13 = sqrt(sum((h13_vals .- true_h13).^2)/length(h13_vals))
rmse_h23 = sqrt(sum((h23_vals .- true_h23).^2)/length(h23_vals))

println("\n" * "="^60)
println("DONE! Fixtures written to $fixtures_dir")
println("Hazard RMSE: h12=$(round(rmse_h12, sigdigits=3)), h13=$(round(rmse_h13, sigdigits=3)), h23=$(round(rmse_h23, sigdigits=3))")
println("="^60)
