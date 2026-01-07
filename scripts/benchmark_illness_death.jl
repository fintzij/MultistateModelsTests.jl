# Benchmark: MultistateModels.jl vs mgcv for Illness-Death Model
# 
# Model structure:
#   State 1 (Healthy) → State 2 (Illness) → State 3 (Death)
#                    ↘ State 3 (Death)
#
# Compare: Cumulative incidence curves and state prevalence
#
# PARAMETERIZATION (our rate formulation):
#   h(t) = shape * rate * t^(shape-1)
#   H(t) = rate * t^shape

using MultistateModels
using DataFrames
using CSV
using JSON
using Random
using Statistics
using LinearAlgebra

println("="^70)
println("Illness-Death Model Benchmark: MultistateModels.jl vs mgcv")
println("="^70)

Random.seed!(42)

# ============================================================================
# Model Configuration
# ============================================================================

n_subj = 1000
max_time = 20.0
eval_times = collect(0.5:0.5:max_time)  # Times for prediction comparison

# True hazard parameters (Weibull rate parameterization)
# h12: Healthy → Illness (increasing hazard)
true_shape_12 = 1.3
true_rate_12 = 0.04

# h13: Healthy → Death (increasing hazard, lower than illness)
true_shape_13 = 1.2
true_rate_13 = 0.015

# h23: Illness → Death (higher hazard after illness)
true_shape_23 = 1.4
true_rate_23 = 0.08

println("\nTrue hazard parameters (h(t) = shape * rate * t^(shape-1)):")
println("  h12 (Healthy→Illness): shape=$(true_shape_12), rate=$(true_rate_12)")
println("  h13 (Healthy→Death):   shape=$(true_shape_13), rate=$(true_rate_13)")
println("  h23 (Illness→Death):   shape=$(true_shape_23), rate=$(true_rate_23)")

# ============================================================================
# Simulate Data
# ============================================================================

println("\n" * "="^70)
println("Simulating Illness-Death Data")
println("="^70)

# Create template data
template_rows = []
for subj in 1:n_subj
    push!(template_rows, (
        id = subj,
        tstart = 0.0,
        tstop = max_time,
        statefrom = 1,
        stateto = 1,
        obstype = 1  # Exact observation
    ))
end
template_data = DataFrame(template_rows)

# Create hazards
h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
h13 = Hazard(@formula(0 ~ 1), "wei", 1, 3)
h23 = Hazard(@formula(0 ~ 1), "wei", 2, 3)

# Build model
model_sim = multistatemodel(h12, h13, h23; data=template_data, surrogate=:markov, initialize=false)

# Set true parameters: [log(shape), log(rate)]
true_params = (
    h12 = [log(true_shape_12), log(true_rate_12)],
    h13 = [log(true_shape_13), log(true_rate_13)],
    h23 = [log(true_shape_23), log(true_rate_23)]
)
set_parameters!(model_sim, true_params)

# Verify parameters
p_nat = get_parameters(model_sim; scale=:natural)
println("\nParameters set (natural scale):")
println("  h12: shape=$(round(p_nat.h12[1], digits=4)), rate=$(round(p_nat.h12[2], digits=4))")
println("  h13: shape=$(round(p_nat.h13[1], digits=4)), rate=$(round(p_nat.h13[2], digits=4))")
println("  h23: shape=$(round(p_nat.h23[1], digits=4)), rate=$(round(p_nat.h23[2], digits=4))")

# Simulate
println("\nSimulating paths...")
obstype_map = Dict(1 => 1, 2 => 1, 3 => 1)  # All transitions get exact observation
sim_result = simulate(model_sim; paths=false, data=true, nsim=1, autotmax=false,
                     obstype_by_transition=obstype_map)
sim_data = sim_result[1, 1]

# Analyze simulated data
println("\nSimulation summary:")
println("  Total subjects: $n_subj")

# Count final states
final_states = combine(groupby(sim_data, :id)) do df
    last_row = df[end, :]
    (final_state = last_row.stateto,)
end

state_counts = combine(groupby(final_states, :final_state), nrow => :count)
for row in eachrow(state_counts)
    state_name = row.final_state == 1 ? "Healthy" : (row.final_state == 2 ? "Illness" : "Death")
    println("  Final state $(row.final_state) ($state_name): $(row.count)")
end

# Count transitions
trans_12 = sum((sim_data.statefrom .== 1) .& (sim_data.stateto .== 2))
trans_13 = sum((sim_data.statefrom .== 1) .& (sim_data.stateto .== 3))
trans_23 = sum((sim_data.statefrom .== 2) .& (sim_data.stateto .== 3))
println("\nTransition counts:")
println("  1→2 (Healthy→Illness): $trans_12")
println("  1→3 (Healthy→Death): $trans_13")
println("  2→3 (Illness→Death): $trans_23")

# ============================================================================
# Export Data for R
# ============================================================================

println("\n" * "="^70)
println("Exporting Data for R")
println("="^70)

# Convert to long format for R (one row per transition)
# R's mstate/survival expects: id, tstart, tstop, from, to, status
export_df = DataFrame(
    id = sim_data.id,
    tstart = sim_data.tstart,
    tstop = sim_data.tstop,
    from = sim_data.statefrom,
    to = sim_data.stateto,
    status = Int.(sim_data.statefrom .!= sim_data.stateto)  # 1 if transition occurred
)

# Save data
csv_path = joinpath(@__DIR__, "..", "fixtures", "illness_death_data.csv")
CSV.write(csv_path, export_df)
println("Saved: $csv_path")

# Save metadata
metadata = Dict(
    "n_subjects" => n_subj,
    "max_time" => max_time,
    "true_params" => Dict(
        "h12" => Dict("shape" => true_shape_12, "rate" => true_rate_12),
        "h13" => Dict("shape" => true_shape_13, "rate" => true_rate_13),
        "h23" => Dict("shape" => true_shape_23, "rate" => true_rate_23)
    ),
    "hazard_formula" => "h(t) = shape * rate * t^(shape-1)",
    "trans_12" => trans_12,
    "trans_13" => trans_13,
    "trans_23" => trans_23,
    "eval_times" => eval_times,
    "seed" => 42
)
json_path = joinpath(@__DIR__, "..", "fixtures", "illness_death_metadata.json")
open(json_path, "w") do f
    JSON.print(f, metadata, 2)
end
println("Saved: $json_path")

# ============================================================================
# Fit Spline Model in Julia
# ============================================================================

println("\n" * "="^70)
println("Fitting Spline Model in Julia")
println("="^70)

# Determine knot locations based on quantiles of observed event times
# Use 9 interior knots at 0.1, 0.2, ..., 0.9 decile quantiles
event_times = sim_data.tstop[(sim_data.statefrom .!= sim_data.stateto) .& (sim_data.tstop .> 0)]
knot_quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
interior_knots = quantile(event_times, knot_quantiles)
println("\nSpline configuration:")
println("  Event times range: $(round(minimum(event_times), digits=2)) - $(round(maximum(event_times), digits=2))")
println("  Interior knots (9 at decile quantiles):")
for (q, k) in zip(knot_quantiles, interior_knots)
    println("    Q$(Int(q*100)): $(round(k, digits=3))")
end

# Create spline hazards
h12_sp = Hazard(@formula(0 ~ 1), "sp", 1, 2; knots=collect(interior_knots))
h13_sp = Hazard(@formula(0 ~ 1), "sp", 1, 3; knots=collect(interior_knots))
h23_sp = Hazard(@formula(0 ~ 1), "sp", 2, 3; knots=collect(interior_knots))

# For exact data (obstype=1), don't use surrogate - this enables automatic smoothing selection
model_sp = multistatemodel(h12_sp, h13_sp, h23_sp; data=sim_data)

# Set up penalization for spline smoothing
# SplinePenalty() applies 2nd order difference penalty to all spline hazards
# Using share_lambda=true for a shared smoothing parameter across hazards
penalty_spec = SplinePenalty(:all; order=2, share_lambda=true)

# Note: Automatic smoothing selection via PIJCV is available but requires 
# positive definite Hessians at the initial estimate. For this benchmark,
# we use a fixed lambda value. In practice, lambda can be tuned via:
#   1. select_smoothing_parameters() for exact data
#   2. Cross-validation on held-out data
#   3. AIC/BIC selection
# A reasonable starting point is lambda ~ 10-100 for well-scaled problems.

# Fit the model with penalized likelihood
println("\nFitting spline model with penalized likelihood...")
t_start = time()
fitted_sp = fit(model_sp;
    verbose=true,
    penalty=penalty_spec,
    lambda_init=100.0)  # Higher lambda = more smoothing
t_elapsed = time() - t_start
println("Fitting complete in $(round(t_elapsed, digits=1)) seconds")
t_elapsed = time() - t_start
println("Fitting complete in $(round(t_elapsed, digits=1)) seconds")

# Get fitted parameters
p_sp = get_parameters(fitted_sp; scale=:natural)
p_sp_flat = get_parameters(fitted_sp; scale=:flat)
println("\nFitted spline coefficients (natural scale):")
println("  h12: $(round.(p_sp.h12, digits=4))")
println("  h13: $(round.(p_sp.h13, digits=4))")
println("  h23: $(round.(p_sp.h23, digits=4))")

# Get number of baseline parameters per hazard
nbasis_12 = fitted_sp.hazards[1].npar_baseline
nbasis_13 = fitted_sp.hazards[2].npar_baseline
nbasis_23 = fitted_sp.hazards[3].npar_baseline

# Extract flat parameters for each hazard
flat_12 = p_sp_flat[1:nbasis_12]
flat_13 = p_sp_flat[(nbasis_12+1):(nbasis_12+nbasis_13)]
flat_23 = p_sp_flat[(nbasis_12+nbasis_13+1):(nbasis_12+nbasis_13+nbasis_23)]

println("\nFitted spline coefficients (flat/log scale for hazard evaluation):")
println("  h12: $(round.(flat_12, digits=4))")
println("  h13: $(round.(flat_13, digits=4))")
println("  h23: $(round.(flat_23, digits=4))")

# ============================================================================
# Compute Cumulative Incidence and State Prevalence
# ============================================================================

println("\n" * "="^70)
println("Computing Cumulative Incidence and State Prevalence")
println("="^70)

# True hazard functions
h12_true(t) = true_shape_12 * true_rate_12 * t^(true_shape_12 - 1)
h13_true(t) = true_shape_13 * true_rate_13 * t^(true_shape_13 - 1)
h23_true(t) = true_shape_23 * true_rate_23 * t^(true_shape_23 - 1)

# True cumulative hazard functions
H12_true(t) = true_rate_12 * t^true_shape_12
H13_true(t) = true_rate_13 * t^true_shape_13
H23_true(t) = true_rate_23 * t^true_shape_23

# Compute transition probability matrix via numerical integration
# P(t) = exp(Q*t) where Q is the generator matrix
# For illness-death: Q = [-q1, q12, q13; 0, -q23, q23; 0, 0, 0]

function compute_tpm_true(t; dt=0.01)
    """Compute transition probability matrix at time t using product integral."""
    if t <= 0
        return Matrix{Float64}(I, 3, 3)
    end
    
    n_steps = max(1, Int(ceil(t / dt)))
    actual_dt = t / n_steps
    
    P = Matrix{Float64}(I, 3, 3)
    
    for i in 1:n_steps
        s = (i - 0.5) * actual_dt  # midpoint
        
        # Hazards at midpoint
        h12 = h12_true(s)
        h13 = h13_true(s)
        h23 = h23_true(s)
        
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

# Compute cumulative incidence for true model
# CI_j(t) = P(T ≤ t, final state = j | start in state 1)
# For illness-death starting in state 1:
#   CI_2(t) = P12(t) + P13(t) where we sum paths through state 2 to state 3
#   But P12(t) is "in state 2 at time t", not "ever visited state 2"

# For cumulative incidence, we need:
#   F_12(t) = integral from 0 to t of P11(s) * h12(s) ds  (probability of 1→2 by time t)
#   F_13(t) = integral from 0 to t of P11(s) * h13(s) ds  (probability of direct 1→3 by time t)
#   Total death = 1 - P11(t) - P12(t) but we want to separate direct vs through illness

function compute_ci_prevalence_true(times; dt=0.01)
    """Compute cumulative incidence and state prevalence at specified times."""
    n_times = length(times)
    
    # State prevalence: P(in state j at time t | start in state 1)
    prev_1 = zeros(n_times)  # P11(t)
    prev_2 = zeros(n_times)  # P12(t)
    prev_3 = zeros(n_times)  # P13(t) = 1 - P11 - P12
    
    # Cumulative incidence: P(ever reach state j by time t | start in state 1)
    # For illness-death:
    #   CI illness = P(ever in state 2 by t) = 1 - exp(-integral h12)  [if no competing risk]
    #   But with competing risk of death, it's more complex
    
    # Cause-specific cumulative incidence
    ci_illness = zeros(n_times)  # P(first event is 1→2 by time t)
    ci_death_direct = zeros(n_times)  # P(first event is 1→3 by time t)
    ci_death_total = zeros(n_times)  # P(in state 3 by time t)
    
    for (i, t) in enumerate(times)
        P = compute_tpm_true(t; dt=dt)
        
        # State prevalence (rows are from states, cols are to states)
        prev_1[i] = P[1, 1]
        prev_2[i] = P[1, 2]
        prev_3[i] = P[1, 3]
        
        # CI for death is simply prevalence in state 3 (absorbing)
        ci_death_total[i] = prev_3[i]
        
        # Cumulative incidence via numerical integration
        # CI_12(t) = integral_0^t P11(s) * h12(s) ds
        n_steps = max(1, Int(ceil(t / dt)))
        actual_dt = t / n_steps
        
        ci_12 = 0.0
        ci_13 = 0.0
        for j in 1:n_steps
            s = (j - 0.5) * actual_dt
            P_s = compute_tpm_true(s; dt=dt)
            ci_12 += P_s[1, 1] * h12_true(s) * actual_dt
            ci_13 += P_s[1, 1] * h13_true(s) * actual_dt
        end
        ci_illness[i] = ci_12
        ci_death_direct[i] = ci_13
    end
    
    return (
        prevalence_healthy = prev_1,
        prevalence_illness = prev_2,
        prevalence_death = prev_3,
        ci_illness = ci_illness,
        ci_death_direct = ci_death_direct,
        ci_death_total = ci_death_total
    )
end

println("\nComputing true cumulative incidence and prevalence...")
true_results = compute_ci_prevalence_true(eval_times)

# ============================================================================
# Compute Model-Based Predictions (Julia Spline Fit)
# ============================================================================

println("Computing Julia spline model predictions...")

# Use the fitted model's totalhazards to compute transition probabilities
# We need to extract hazard functions from fitted model

# For now, compute empirical Kaplan-Meier style estimates from data
# as a sanity check

function compute_empirical_prevalence(data, times, n_subj)
    """Compute empirical state prevalence from simulated data."""
    prev_1 = zeros(length(times))
    prev_2 = zeros(length(times))
    prev_3 = zeros(length(times))
    
    for (i, t) in enumerate(times)
        for subj in 1:n_subj
            subj_data = filter(row -> row.id == subj, data)
            
            # Find state at time t
            state = 1  # Start in state 1
            for row in eachrow(subj_data)
                if row.tstart <= t < row.tstop
                    state = row.statefrom
                    break
                elseif row.tstop <= t
                    state = row.stateto
                end
            end
            
            if state == 1
                prev_1[i] += 1
            elseif state == 2
                prev_2[i] += 1
            else
                prev_3[i] += 1
            end
        end
        prev_1[i] /= n_subj
        prev_2[i] /= n_subj
        prev_3[i] /= n_subj
    end
    
    return (prevalence_healthy=prev_1, prevalence_illness=prev_2, prevalence_death=prev_3)
end

empirical = compute_empirical_prevalence(sim_data, eval_times, n_subj)

# ============================================================================
# Save Results for R Comparison
# ============================================================================

println("\n" * "="^70)
println("Saving Results")
println("="^70)

results = Dict(
    "eval_times" => eval_times,
    "true" => Dict(
        "prevalence_healthy" => true_results.prevalence_healthy,
        "prevalence_illness" => true_results.prevalence_illness,
        "prevalence_death" => true_results.prevalence_death,
        "ci_illness" => true_results.ci_illness,
        "ci_death_direct" => true_results.ci_death_direct,
        "ci_death_total" => true_results.ci_death_total
    ),
    "empirical" => Dict(
        "prevalence_healthy" => empirical.prevalence_healthy,
        "prevalence_illness" => empirical.prevalence_illness,
        "prevalence_death" => empirical.prevalence_death
    ),
    "julia_fit" => Dict(
        "spline_coeffs_h12" => collect(p_sp.h12),
        "spline_coeffs_h13" => collect(p_sp.h13),
        "spline_coeffs_h23" => collect(p_sp.h23),
        "spline_coeffs_flat_h12" => collect(flat_12),
        "spline_coeffs_flat_h13" => collect(flat_13),
        "spline_coeffs_flat_h23" => collect(flat_23),
        "interior_knots" => collect(interior_knots),
        "fit_time_seconds" => t_elapsed
    )
)

results_path = joinpath(@__DIR__, "..", "fixtures", "illness_death_results.json")
open(results_path, "w") do f
    JSON.print(f, results, 2)
end
println("Saved: $results_path")

# Print summary comparison
println("\n" * "="^70)
println("Summary: True vs Empirical Prevalence")
println("="^70)
println("\nTime\tTrue P1\t\tEmp P1\t\tTrue P2\t\tEmp P2\t\tTrue P3\t\tEmp P3")
for i in [1, 10, 20, 30, 40]
    if i <= length(eval_times)
        t = eval_times[i]
        println("$(round(t, digits=1))\t$(round(true_results.prevalence_healthy[i], digits=3))\t\t$(round(empirical.prevalence_healthy[i], digits=3))\t\t$(round(true_results.prevalence_illness[i], digits=3))\t\t$(round(empirical.prevalence_illness[i], digits=3))\t\t$(round(true_results.prevalence_death[i], digits=3))\t\t$(round(empirical.prevalence_death[i], digits=3))")
    end
end

# RMSE between true and empirical
rmse_p1 = sqrt(mean((true_results.prevalence_healthy .- empirical.prevalence_healthy).^2))
rmse_p2 = sqrt(mean((true_results.prevalence_illness .- empirical.prevalence_illness).^2))
rmse_p3 = sqrt(mean((true_results.prevalence_death .- empirical.prevalence_death).^2))
println("\nRMSE (True vs Empirical):")
println("  P(Healthy): $(round(rmse_p1, digits=4))")
println("  P(Illness): $(round(rmse_p2, digits=4))")
println("  P(Death):   $(round(rmse_p3, digits=4))")

println("\n" * "="^70)
println("Next: Run R script to fit mgcv and compare")
println("="^70)
