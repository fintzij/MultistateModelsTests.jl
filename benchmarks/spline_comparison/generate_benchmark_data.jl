# =============================================================================
# Penalized Spline Benchmark: Data Generation and Julia Fit
# =============================================================================
#
# This script:
# 1. Simulates illness-death data with known time-varying hazards
# 2. Fits the model using MultistateModels.jl with automatic smoothing selection
# 3. Exports data and results for comparison with R packages (mgcv, flexsurv)
#
# True hazard functions:
#   h12(t) = 0.3 * t^0.5 (increasing, concave)
#   h13(t) = 0.1 + 0.02*t (linear increase)
#   h23(t) = 0.4 * exp(-0.1*t) (decreasing exponential)
#
# Run with: julia --project=../../.. generate_benchmark_data.jl
# =============================================================================

using MultistateModels
using DataFrames
using CSV
using JSON
using Random
using Statistics

const OUTPUT_DIR = @__DIR__
const SEED = 20260104

# =============================================================================
# True Hazard Functions
# =============================================================================

"""True hazard for transition 1→2: h12(t) = 0.3 * t^0.5"""
true_h12(t) = t > 0 ? 0.3 * sqrt(t) : 0.3 * sqrt(0.01)

"""True hazard for transition 1→3: h13(t) = 0.1 + 0.02*t"""
true_h13(t) = 0.1 + 0.02 * t

"""True hazard for transition 2→3: h23(t) = 0.4 * exp(-0.1*t)"""
true_h23(t) = 0.4 * exp(-0.1 * t)

# Cumulative hazards (analytical)
true_H12(t) = t > 0 ? 0.3 * (2/3) * t^1.5 : 0.0
true_H13(t) = 0.1 * t + 0.01 * t^2
true_H23(t) = t > 0 ? 4.0 * (1 - exp(-0.1 * t)) : 0.0

# =============================================================================
# Simulation Functions
# =============================================================================

"""
    simulate_illness_death_subject(max_time::Float64; rng=Random.GLOBAL_RNG)

Simulate one subject's path through the illness-death model using rejection sampling.
"""
function simulate_illness_death_subject(max_time::Float64; rng=Random.GLOBAL_RNG)
    records = NamedTuple[]
    current_state = 1
    current_time = 0.0
    
    while current_state != 3 && current_time < max_time
        if current_state == 1
            # Competing risks: 1→2 or 1→3
            # Find next event time via inverse CDF
            u = rand(rng)
            
            # Numerically solve for time when S(t) = u
            # S(t) = exp(-H12(t) - H13(t))
            t_event = find_event_time(current_time, max_time, u, 
                                       t -> true_H12(t - current_time) + true_H13(t - current_time))
            
            if t_event >= max_time
                # Right-censored in state 1
                push!(records, (tstart=current_time, tstop=max_time, statefrom=1, stateto=1, obstype=2))
                break
            end
            
            # Determine which transition: prob of 1→2 given event at t_event
            h12_t = true_h12(t_event - current_time)
            h13_t = true_h13(t_event - current_time)
            prob_12 = h12_t / (h12_t + h13_t)
            
            if rand(rng) < prob_12
                push!(records, (tstart=current_time, tstop=t_event, statefrom=1, stateto=2, obstype=1))
                current_state = 2
            else
                push!(records, (tstart=current_time, tstop=t_event, statefrom=1, stateto=3, obstype=1))
                current_state = 3
            end
            current_time = t_event
            
        elseif current_state == 2
            # Single transition: 2→3
            u = rand(rng)
            t_sojourn_start = current_time  # Time when entered state 2
            t_event = find_event_time(current_time, max_time, u, 
                                       t -> true_H23(t - t_sojourn_start))
            
            if t_event >= max_time
                # Right-censored in state 2
                push!(records, (tstart=current_time, tstop=max_time, statefrom=2, stateto=2, obstype=2))
                break
            end
            
            push!(records, (tstart=current_time, tstop=t_event, statefrom=2, stateto=3, obstype=1))
            current_state = 3
            current_time = t_event
        end
    end
    
    return records
end

"""Find event time t where S(t) = u, i.e., H(t) = -log(u)"""
function find_event_time(t_start::Float64, t_max::Float64, u::Float64, H_fn)
    target = -log(u)
    
    # Binary search for t where H(t) = target
    t_lo, t_hi = t_start, t_max
    for _ in 1:100
        t_mid = (t_lo + t_hi) / 2
        if H_fn(t_mid) < target
            t_lo = t_mid
        else
            t_hi = t_mid
        end
        if t_hi - t_lo < 1e-6
            break
        end
    end
    
    return (t_lo + t_hi) / 2
end

"""
    simulate_dataset(n_subjects::Int, max_time::Float64; seed=12345) -> DataFrame

Simulate a full illness-death dataset.
"""
function simulate_dataset(n_subjects::Int, max_time::Float64; seed=12345)
    rng = Random.MersenneTwister(seed)
    
    all_records = NamedTuple[]
    for id in 1:n_subjects
        records = simulate_illness_death_subject(max_time; rng=rng)
        for r in records
            push!(all_records, (id=id, r...))
        end
    end
    
    return DataFrame(all_records)
end

# =============================================================================
# Main Benchmark
# =============================================================================

function run_benchmark(; n_subjects=200, max_time=5.0, verbose=true)
    println("="^70)
    println("Penalized Spline Benchmark: mgcv vs flexsurv vs MultistateModels.jl")
    println("="^70)
    
    # Generate data
    println("\n--- Generating Data ---")
    Random.seed!(SEED)
    data = simulate_dataset(n_subjects, max_time; seed=SEED)
    
    # Count transitions
    n_12 = count(r -> r.statefrom == 1 && r.stateto == 2 && r.obstype == 1, eachrow(data))
    n_13 = count(r -> r.statefrom == 1 && r.stateto == 3 && r.obstype == 1, eachrow(data))
    n_23 = count(r -> r.statefrom == 2 && r.stateto == 3 && r.obstype == 1, eachrow(data))
    
    println("  N subjects: $n_subjects")
    println("  Max time: $max_time")
    println("  Transitions 1→2: $n_12")
    println("  Transitions 1→3: $n_13")
    println("  Transitions 2→3: $n_23")
    
    # Export data for R
    CSV.write(joinpath(OUTPUT_DIR, "benchmark_data.csv"), data)
    
    # Define evaluation grid
    eval_times = collect(0.1:0.1:max_time)
    
    # Compute true hazards at evaluation points
    h12_true = true_h12.(eval_times)
    h13_true = true_h13.(eval_times)
    h23_true = true_h23.(eval_times)
    
    # Export metadata for R
    metadata = Dict(
        "n_subjects" => n_subjects,
        "max_time" => max_time,
        "seed" => SEED,
        "n_12" => n_12,
        "n_13" => n_13,
        "n_23" => n_23,
        "eval_times" => eval_times,
        "true_hazards" => Dict(
            "h12" => h12_true,
            "h13" => h13_true,
            "h23" => h23_true
        ),
        "true_hazard_formulas" => Dict(
            "h12" => "0.3 * sqrt(t)",
            "h13" => "0.1 + 0.02 * t",
            "h23" => "0.4 * exp(-0.1 * t)"
        )
    )
    open(joinpath(OUTPUT_DIR, "benchmark_metadata.json"), "w") do f
        JSON.print(f, metadata, 2)
    end
    
    # =========================================================================
    # Fit with MultistateModels.jl
    # =========================================================================
    println("\n--- Fitting with MultistateModels.jl ---")
    
    # Define spline hazards (cubic B-splines)
    # Use interior knots to match mgcv k=8 approximately (8 basis = degree+1 + interior_knots)
    interior_knots = [1.0, 2.0, 3.0, 4.0]  # 4 interior knots
    h12 = Hazard(@formula(0 ~ 1), :sp, 1, 2; degree=3, knots=interior_knots)
    h13 = Hazard(@formula(0 ~ 1), :sp, 1, 3; degree=3, knots=interior_knots)
    h23 = Hazard(@formula(0 ~ 1), :sp, 2, 3; degree=3, knots=interior_knots)
    
    model = multistatemodel(h12, h13, h23; data=data)
    
    # Warm-up JIT
    println("  Warming up JIT...")
    try
        penalty_warmup = SplinePenalty()
        _ = select_smoothing_parameters(model, penalty_warmup; 
                                        method=:pijcv, max_outer_iter=2, verbose=false)
    catch e
        @warn "Warm-up failed, continuing..." exception=e
    end
    
    # Time the fit with smoothing selection
    println("  Running smoothing parameter selection (PIJCV)...")
    penalty = SplinePenalty()
    
    t_julia_start = time()
    result = select_smoothing_parameters(model, penalty; 
                                         method=:pijcv, 
                                         verbose=verbose,
                                         max_outer_iter=20)
    t_julia = time() - t_julia_start
    
    println("\n  Results:")
    println("    Converged: $(result.converged)")
    println("    Lambda: $(round.(result.lambda, sigdigits=4))")
    println("    N outer iterations: $(result.n_outer_iter)")
    println("    Time: $(round(t_julia, digits=2)) seconds")
    
    # Extract fitted hazards
    beta_hat = result.beta
    
    # Compute parameter offsets for each hazard
    # Each hazard has npar_total parameters, stored contiguously in flat vector
    hazards = model.hazards
    n_hazards = length(hazards)
    param_offsets = Vector{UnitRange{Int}}(undef, n_hazards)
    offset = 1
    for i in 1:n_hazards
        n_pars = hazards[i].npar_total
        param_offsets[i] = offset:(offset + n_pars - 1)
        offset += n_pars
    end
    
    println("  Parameter structure:")
    for i in 1:n_hazards
        println("    Hazard $i ($(hazards[i].hazname)): params $(param_offsets[i]) ($(length(param_offsets[i])) params)")
    end
    
    h12_julia = Float64[]
    h13_julia = Float64[]
    h23_julia = Float64[]
    
    # Evaluate hazards at each time point
    # The callable interface: hazard(t, pars, covars) where pars is the slice for that hazard
    for t in eval_times
        # Pass empty NamedTuple for covariates since we have no covariates
        push!(h12_julia, hazards[1](t, beta_hat[param_offsets[1]], NamedTuple()))
        push!(h13_julia, hazards[2](t, beta_hat[param_offsets[2]], NamedTuple()))
        push!(h23_julia, hazards[3](t, beta_hat[param_offsets[3]], NamedTuple()))
    end
    
    # Compute RMSE
    rmse(a, b) = sqrt(mean((a .- b).^2))
    println("\n  Hazard RMSE vs True:")
    println("    h12: $(round(rmse(h12_julia, h12_true), sigdigits=4))")
    println("    h13: $(round(rmse(h13_julia, h13_true), sigdigits=4))")
    println("    h23: $(round(rmse(h23_julia, h23_true), sigdigits=4))")
    
    # Compute effective degrees of freedom
    # EDF = tr(F) where F is the influence matrix
    # For a rough approximation, use n_params for each hazard
    # (true EDF would require computing tr((X'X + λS)^{-1} X'X))
    n_basis = [hazards[i].npar_baseline for i in 1:n_hazards]
    edf_approx = n_basis  # Placeholder - would need more computation for true EDF
    
    # Export Julia results
    julia_results = Dict(
        "hazards" => Dict(
            "h12" => h12_julia,
            "h13" => h13_julia,
            "h23" => h23_julia
        ),
        "lambda" => result.lambda,
        "beta" => result.beta,
        "converged" => result.converged,
        "n_outer_iter" => result.n_outer_iter,
        "time_seconds" => t_julia,
        "criterion" => result.criterion,
        "rmse" => Dict(
            "h12" => rmse(h12_julia, h12_true),
            "h13" => rmse(h13_julia, h13_true),
            "h23" => rmse(h23_julia, h23_true)
        )
    )
    
    open(joinpath(OUTPUT_DIR, "julia_results.json"), "w") do f
        JSON.print(f, julia_results, 2)
    end
    
    println("\n--- Data and results exported to: $OUTPUT_DIR ---")
    println("  benchmark_data.csv")
    println("  benchmark_metadata.json")
    println("  julia_results.json")
    println("\nRun the R script next to compare with mgcv and flexsurv.")
    
    return (data=data, result=result, julia_results=julia_results)
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmark(n_subjects=200, max_time=5.0, verbose=true)
end
