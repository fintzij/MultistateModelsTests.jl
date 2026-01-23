# Minimal PIJCV test: 2-state model (Healthy -> Death) with exact data
using MultistateModels, DataFrames, Random, LinearAlgebra, Statistics
import MultistateModels: Hazard, @formula, multistatemodel, fit, simulate
import MultistateModels: get_parameters, get_parameters_flat, set_parameters!
import MultistateModels: extract_paths, ExactData, build_penalty_config, SplinePenalty
import MultistateModels: select_smoothing_parameters, compute_subject_hessians

Random.seed!(12345)

println("="^70)
println("MINIMAL PIJCV TEST: 2-state model with exact data")
println("="^70)

# Step 1: Create and simulate a simple 2-state model
println("\n1. Setting up 2-state model (Healthy -> Death)...")

n_subj = 100
max_time = 10.0

# True Weibull hazard: h(t) = shape * rate * t^(shape-1)
true_shape = 1.5
true_rate = 0.1
println("   True hazard: Weibull(shape=$true_shape, rate=$true_rate)")

# Create Weibull hazard for simulation
h_wei = Hazard(@formula(0 ~ 1), "wei", 1, 2)

# Create simple data for simulation
sim_df = DataFrame(
    id = 1:n_subj,
    tstart = zeros(n_subj),
    tstop = fill(max_time, n_subj),
    statefrom = ones(Int, n_subj),
    stateto = fill(2, n_subj),
    obstype = ones(Int, n_subj)
)

model_wei = multistatemodel(h_wei; data=sim_df)
set_parameters!(model_wei, (h12 = [true_shape, true_rate],))

# Simulate
println("   Simulating $n_subj subjects...")
sim_data = simulate(model_wei)[1]

# Count events
n_events = sum(sim_data.statefrom .!= sim_data.stateto)
println("   Events: $n_events / $n_subj")

# Get event times for knot placement
event_times = sim_data.tstop[sim_data.statefrom .!= sim_data.stateto]
println("   Event time range: $(round(minimum(event_times), digits=2)) - $(round(maximum(event_times), digits=2))")

# Step 2: Fit spline model
println("\n2. Setting up spline model...")

# Use 4 interior knots at quintiles
knot_quantiles = [0.2, 0.4, 0.6, 0.8]
interior_knots = quantile(event_times, knot_quantiles)
println("   Interior knots: ", round.(interior_knots, digits=2))

h_sp = Hazard(@formula(0 ~ 1), "sp", 1, 2; knots=collect(interior_knots))
model_sp = multistatemodel(h_sp; data=sim_data)

# Get number of spline coefficients
n_basis = model_sp.hazards[1].npar_baseline
println("   Number of basis functions: $n_basis")

# Fit unpenalized first
println("\n3. Fitting unpenalized model...")
initial_fit = fit(model_sp; verbose=false, vcov_type=:none)
beta_init = collect(get_parameters(initial_fit; scale=:flat))
println("   Initial coefficients: ", round.(beta_init, digits=3))

# Step 3: Test PIJCV
println("\n4. Testing PIJCV components...")

# Build penalty config
penalty_spec = SplinePenalty(:all; order=2, share_lambda=false)
penalty_config = build_penalty_config(model_sp, penalty_spec; lambda_init=1.0)
println("   Penalty config: n_lambda=$(penalty_config.n_lambda), n_terms=$(length(penalty_config.terms))")

# Get sample paths and compute Hessians
samplepaths = extract_paths(model_sp)
exact_data = ExactData(model_sp, samplepaths)

println("   Computing subject Hessians...")
subject_hessians = compute_subject_hessians(beta_init, model_sp, samplepaths)

# Aggregate
H_unpenalized = zeros(length(beta_init), length(beta_init))
for H_i in subject_hessians
    H_unpenalized .+= H_i
end

# Check eigenvalues
eigenvalues = eigvals(Symmetric(H_unpenalized))
println("   H_unpenalized eigenvalues: min=$(round(minimum(eigenvalues), digits=2)), max=$(round(maximum(eigenvalues), digits=2))")
println("   Negative eigenvalues: $(sum(eigenvalues .< 0))")

# Test Cholesky with different lambda values
println("\n5. Testing Cholesky factorization with different lambda...")
for log_lam in [-2.0, 0.0, 2.0, 4.0, 6.0]
    lam = exp(log_lam)
    H_pen = copy(H_unpenalized)
    for term in penalty_config.terms
        idx = term.hazard_indices
        H_pen[idx, idx] .+= lam * term.S
    end
    
    eigs = eigvals(Symmetric(H_pen))
    is_pd = try
        cholesky(Symmetric(H_pen))
        true
    catch
        false
    end
    println("   log(lam)=$(round(log_lam, digits=1)) (lam=$(round(lam, digits=2))): min_eig=$(round(minimum(eigs), digits=2)), PD=$is_pd")
end

# Step 4: Run PIJCV
println("\n6. Running PIJCV smoothing parameter selection...")
smoothing_result = select_smoothing_parameters(
    model_sp, exact_data, penalty_config, beta_init;
    method=:pijcv,
    scope=:all,
    verbose=true
)

println("\n   PIJCV Results:")
println("   Optimal lambda: ", round.(smoothing_result.lambda, digits=4))
println("   Optimal log(lambda): ", round.(log.(smoothing_result.lambda), digits=3))
println("   Method used: ", smoothing_result.method_used)
println("   Converged: ", smoothing_result.converged)
println("   Final criterion: ", round(smoothing_result.criterion, digits=4))

# Step 5: Final fit with optimal lambda
if smoothing_result.converged && smoothing_result.criterion < 1e9
    println("\n7. Final fit with optimal smoothing...")
    fitted_sp = fit(model_sp;
        verbose=false,
        penalty=penalty_spec,
        lambda_init=smoothing_result.lambda[1])
    
    final_params = get_parameters(fitted_sp; scale=:flat)
    println("   Final coefficients: ", round.(collect(final_params), digits=3))
else
    println("\n!! PIJCV did not converge or returned invalid criterion")
    println("   This suggests the Hessian is not positive definite")
end

println("\n" * "="^70)
println("DONE")
println("="^70)
