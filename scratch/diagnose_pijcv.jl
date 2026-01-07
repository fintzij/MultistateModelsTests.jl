using MultistateModels, DataFrames, Random, LinearAlgebra, Statistics
import MultistateModels: Hazard, @formula, multistatemodel, fit, simulate
import MultistateModels: get_parameters, set_parameters!
import MultistateModels: extract_paths, ExactData, build_penalty_config, SplinePenalty
import MultistateModels: compute_subject_gradients, compute_subject_hessians

Random.seed!(12345)

n_subj = 300
max_time = 50.0
h_wei = Hazard(@formula(0 ~ 1), "wei", 1, 2)
sim_df = DataFrame(
    id = 1:n_subj,
    tstart = zeros(n_subj),
    tstop = fill(max_time, n_subj),
    statefrom = ones(Int, n_subj),
    stateto = fill(2, n_subj),
    obstype = ones(Int, n_subj)
)
model_wei = multistatemodel(h_wei; data=sim_df)
set_parameters!(model_wei, (h12 = [1.2, 0.02],))

sim_data = simulate(model_wei)[1]
event_times = sim_data.tstop[sim_data.statefrom .!= sim_data.stateto]
println("Event time range: ", round.(extrema(event_times), digits=2))

# Use more appropriate knots
interior_knots = quantile(event_times, [0.2, 0.4, 0.6, 0.8])
println("Interior knots: ", round.(interior_knots, digits=2))

h_sp = Hazard(@formula(0 ~ 1), "sp", 1, 2; knots=interior_knots)
model_sp = multistatemodel(h_sp; data=sim_data)

# Fit unpenalized
println("\nFitting unpenalized spline model...")
initial_fit = fit(model_sp; verbose=false, compute_vcov=false)
beta_init = collect(get_parameters(initial_fit; scale=:flat))

println("\nInitial coefficients:")
println("  β = ", round.(beta_init, digits=3))
println("  Any Inf: ", any(isinf.(beta_init)))

# Compute subject-level Hessians
samplepaths = extract_paths(model_sp)
subject_hessians = compute_subject_hessians(beta_init, model_sp, samplepaths)

println("\nTotal Hessian (sum of subject Hessians):")
H_total = sum(subject_hessians)
eigs = eigvals(Symmetric(H_total))
println("  eigenvalues: ", round.(eigs, digits=2))
println("  any Inf/NaN: ", any(isinf.(H_total)) || any(isnan.(H_total)))
println("  positive definite: ", all(eigs .> 0))

# Check gradients
subject_grads = compute_subject_gradients(beta_init, model_sp, samplepaths)
println("\nGradients:")
println("  Sum per param: ", round.(vec(sum(subject_grads, dims=2)), digits=4))
println("  Any Inf/NaN: ", any(isinf.(subject_grads)) || any(isnan.(subject_grads)))

# Key insight: check if the -Inf coefficient causes issues
println("\n" * "="^60)
println("DIAGNOSIS: The -Inf coefficient")
println("="^60)
idx_inf = findfirst(isinf, beta_init)
if !isnothing(idx_inf)
    println("Parameter $idx_inf is -Inf")
    println("This means exp(β[$idx_inf]) = 0, so that basis function contributes nothing")
    println("The gradient w.r.t. this parameter should be ~0 (no sensitivity)")
    println("The Hessian row/col for this parameter should be ~0")
    println("\nHessian row $idx_inf: ", round.(H_total[idx_inf, :], digits=6))
    println("Gradient sum for param $idx_inf: ", round(sum(subject_grads[idx_inf, :]), digits=6))
end
