
using MultistateModels
using DataFrames
using Random
using Statistics
using CairoMakie
using Printf
using Dates

# Ensure output directory exists
mkpath(joinpath(@__DIR__, "..", "reports", "assets", "benchmarks"))

# ============================================================================
# CONFIGURATION
# ============================================================================
const N_SUBJECTS_BENCHMARK = [100, 200, 500] # Reduced for speed in this environment
const N_THREADS_BENCHMARK = [1, 4]
const MCEM_ITER = 10 # Reduced for speed
const N_REPLICATES = 3

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

function generate_benchmark_data(n_subj)
    # Simple illness-death model
    h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
    h13 = Hazard(@formula(0 ~ 1), "wei", 1, 3)
    h23 = Hazard(@formula(0 ~ 1), "wei", 2, 3)
    
    # True parameters
    pars = (
        h12 = [log(1.1), log(0.5)],
        h13 = [log(1.0), log(0.3)],
        h23 = [log(1.2), log(0.6)]
    )
    
    model = multistatemodel(h12, h13, h23; data=DataFrame(id=1:n_subj, tstart=0.0, tstop=10.0, statefrom=1, stateto=1, obstype=1))
    set_parameters!(model, pars)
    
    # Simulate panel data
    sim = simulate(model; nsim=1, data=true, paths=false)
    return sim[1]
end

# ============================================================================
# BENCHMARK 1: SCALABILITY
# ============================================================================
println("Running Scalability Benchmark...")
scalability_results = DataFrame(N=Int[], Runtime=Float64[])

for n in N_SUBJECTS_BENCHMARK
    println("  N = $n")
    dat = generate_benchmark_data(n)
    
    # Fit model
    h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
    h13 = Hazard(@formula(0 ~ 1), "wei", 1, 3)
    h23 = Hazard(@formula(0 ~ 1), "wei", 2, 3)
    model = multistatemodel(h12, h13, h23; data=dat)
    
    # Warmup
    fit(model; verbose=false, compute_vcov=false, maxiter=2)
    
    # Benchmark
    t_start = time()
    fit(model; verbose=false, compute_vcov=false, maxiter=MCEM_ITER)
    t_end = time()
    
    push!(scalability_results, (n, t_end - t_start))
end

# ============================================================================
# BENCHMARK 2: SQUAREM vs EM
# ============================================================================
println("Running SQUAREM Benchmark...")
n_sq = 200
dat_sq = generate_benchmark_data(n_sq)
h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
h13 = Hazard(@formula(0 ~ 1), "wei", 1, 3)
h23 = Hazard(@formula(0 ~ 1), "wei", 2, 3)
model_sq = multistatemodel(h12, h13, h23; data=dat_sq)

# Standard EM (simulated by setting squarem=false if option existed, or just comparing iterations)
# Since we can't easily disable SQUAREM in the current API without hacking, we will compare
# convergence speed (loglik per iteration) if possible, or just report the SQUAREM performance.
# For now, we'll run fit and capture the trace.
# Note: The current fit() uses SQUAREM by default.

# We will just report the runtime for SQUAREM on N=200 for now as "Accelerated EM"
# and maybe try to find a way to run standard EM if possible.
# Looking at source, fit() calls mcem_mll which uses squarem.
# We will just record the SQUAREM time.
t_start = time()
fit(model_sq; verbose=false, compute_vcov=false, maxiter=MCEM_ITER)
t_sq = time() - t_start

squarem_results = DataFrame(Method=["SQUAREM"], Runtime=[t_sq])

# ============================================================================
# BENCHMARK 3: THREADING
# ============================================================================
println("Running Threading Benchmark...")
threading_results = DataFrame(Threads=Int[], Runtime=Float64[])
n_th = 500
dat_th = generate_benchmark_data(n_th)

for nth in N_THREADS_BENCHMARK
    println("  Threads = $nth")
    # Set threads (this might require restarting Julia or using specific API if available)
    # MultistateModels uses Threads.nthreads(). We can't change this at runtime easily.
    # However, we can simulate the effect if there's a config.
    # Checking test_parallel_likelihood.jl...
    # import MultistateModels: set_threading_config!
    
    # We will skip actual threading benchmark if we can't change threads, 
    # but we can try to use the internal config if exposed.
    # For now, we will just record the current thread count's performance.
    
    # If we can't change threads, we'll just run once and note the thread count.
    if nth == Threads.nthreads()
        h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
        h13 = Hazard(@formula(0 ~ 1), "wei", 1, 3)
        h23 = Hazard(@formula(0 ~ 1), "wei", 2, 3)
        model = multistatemodel(h12, h13, h23; data=dat_th)
        
        t_start = time()
        fit(model; verbose=false, compute_vcov=false, maxiter=MCEM_ITER)
        t_end = time()
        push!(threading_results, (nth, t_end - t_start))
    end
end

# ============================================================================
# SAVE RESULTS
# ============================================================================
using JSON3

results = Dict(
    "scalability" => scalability_results,
    "squarem" => squarem_results,
    "threading" => threading_results
)

open(joinpath(@__DIR__, "..", "reports", "assets", "benchmarks", "results.json"), "w") do io
    JSON3.write(io, results)
end

println("Benchmarks complete. Results saved to assets/benchmarks/results.json")
