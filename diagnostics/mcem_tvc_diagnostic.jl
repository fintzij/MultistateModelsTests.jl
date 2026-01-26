using MultistateModels
using MultistateModels: get_parameters_flat
using DataFrames
using Random
using Printf
using Statistics

println("="^80)
println("MCEM + TVC DIAGNOSTIC")
println("="^80)

true_params = Dict(
    "scale12" => 5.0,            # Weibull scale (natural scale)
    "scale23" => 5.0,
    "shape12" => 1.2,            # Weibull shape
    "shape23" => 1.3,
    "beta12" => 0.5,
    "beta23" => 0.5
)

Random.seed!(42)
n_subj = 100
max_time = 15.0
change_time = 1.0 # Reduced from 5.0 to ensure we see events with x=1

# Weibull with TVC
h12 = Hazard(@formula(0 ~ x), "wei", 1, 2)
h23 = Hazard(@formula(0 ~ x), "wei", 2, 3)

# Create TVC template
template_rows = DataFrame[]
for id in 1:n_subj
    push!(template_rows, DataFrame(
        id=id, tstart=0.0, tstop=change_time,
        statefrom=1, stateto=1, obstype=1, x=0.0
    ))
    push!(template_rows, DataFrame(
        id=id, tstart=change_time, tstop=max_time,
        statefrom=1, stateto=3, obstype=1, x=1.0
    ))
end
template = vcat(template_rows...)

model_sim = multistatemodel(h12, h23; data=template, initialize=false)
set_parameters!(model_sim, (
    h12 = [true_params["shape12"], true_params["scale12"], true_params["beta12"]],
    h23 = [true_params["shape23"], true_params["scale23"], true_params["beta23"]]
))

# Simulate exact data
sim_result = simulate(model_sim; paths=false, data=true, nsim=1)
exact_data_raw = sim_result[1, 1]

# Post-process exact data to split at change_time for TVC fit
exact_rows = DataFrame[]
for row in eachrow(exact_data_raw)
    # If interval crosses change_time, split it
    if row.tstart < change_time && row.tstop > change_time
        # Part 1: tstart to change_time (x=0)
        push!(exact_rows, DataFrame(
            id=row.id, tstart=row.tstart, tstop=change_time,
            statefrom=row.statefrom, stateto=row.statefrom, # No transition yet
            obstype=1, x=0.0
        ))
        # Part 2: change_time to tstop (x=1)
        push!(exact_rows, DataFrame(
            id=row.id, tstart=change_time, tstop=row.tstop,
            statefrom=row.statefrom, stateto=row.stateto, # Transition happens here
            obstype=1, x=1.0
        ))
    else
        # No split needed
        x_val = row.tstart >= change_time ? 1.0 : 0.0
        push!(exact_rows, DataFrame(
            id=row.id, tstart=row.tstart, tstop=row.tstop,
            statefrom=row.statefrom, stateto=row.stateto,
            obstype=1, x=x_val
        ))
    end
end
exact_data = vcat(exact_rows...)

println("\nExact data sample (processed):")
println(first(exact_data, 10))
println("Exact data x distribution:")
println(combine(groupby(exact_data, :x), nrow => :count))

# Fit exact data (gold standard)
println("\n1. EXACT DATA FIT (gold standard):")
model_exact = multistatemodel(h12, h23; data=exact_data)
fitted_exact = fit(model_exact; verbose=false)
params_exact = get_parameters_flat(fitted_exact)
@printf "   h12: shape=%.3f (true=%.1f), scale=%.3f (true=%.1f), beta=%.3f (true=%.1f)\n" params_exact[1] true_params["shape12"] params_exact[2] true_params["scale12"] params_exact[3] true_params["beta12"]
@printf "   h23: shape=%.3f (true=%.1f), scale=%.3f (true=%.1f), beta=%.3f (true=%.1f)\n" params_exact[4] true_params["shape23"] params_exact[5] true_params["scale23"] params_exact[6] true_params["beta23"]

# Create panel data with TVC
obs_times = [0.0, 3.0, 5.0, 6.0, 9.0, 12.0]
panel_rows = DataFrame[]
for subj_id in unique(exact_data.id)
    subj_data = filter(r -> r.id == subj_id, exact_data)
    
    for i in 1:(length(obs_times)-1)
        t0, t1 = obs_times[i], obs_times[i+1]
        
        # Find state at t0 and t1
        state_t0 = 1
        state_t1 = 1
        for row in eachrow(subj_data)
            if row.tstop <= t0 && row.stateto != row.statefrom
                state_t0 = row.stateto
            end
            if row.tstop <= t1 && row.stateto != row.statefrom
                state_t1 = max(state_t1, row.stateto)
            end
        end
        
        x_val = t0 < change_time ? 0.0 : 1.0
        
        if state_t0 != 3  # Only include if not absorbed
            push!(panel_rows, DataFrame(
                id = subj_id, tstart = t0, tstop = t1,
                statefrom = state_t0, stateto = state_t1, obstype = 2, x = x_val
            ))
        end
    end
end
panel_data = vcat(panel_rows...)

println("\n2. PANEL DATA (for MCEM):")
println("   Total panel observations: ", nrow(panel_data))
println("   Panel data sample:")
println(first(panel_data, 15))

# Fit surrogate manually to inspect
println("\n2b. SURROGATE FIT INSPECTION:")
model_temp = multistatemodel(h12, h23; data=panel_data, initialize=false)
surrogate = MultistateModels.fit_surrogate(model_temp; verbose=true)
println("Surrogate parameters:")
println(surrogate.parameters)

# Fit with MCEM
println("\n3. MCEM FIT:")
model_mcem = multistatemodel(h12, h23; data=panel_data, initialize=false)

# Manually set a reasonable surrogate to avoid numerical issues with stiff rates
println("   Manually setting reasonable surrogate parameters...")
surrogate_pars = (
    h12 = [-1.6, 0.0], 
    h23 = [-1.6, 0.0]
)
surrogate = MultistateModels.fit_surrogate(model_mcem; surrogate_parameters = surrogate_pars)
model_mcem.surrogate = surrogate

# Ensure surrogate is set (this will trigger MLE fit which is unstable)
# set_surrogate!(model_mcem; verbose=true)

fitted_mcem = fit(model_mcem; 
    verbose=true,
    vcov_type=:none,
    MaxIter=25,
    MaxESS=300
)
params_mcem = get_parameters_flat(fitted_mcem)
@printf "\n   h12: shape=%.3f (true=%.1f), log_scale=%.3f (true=%.3f), beta=%.3f (true=%.1f)\n" params_mcem[1] true_params["shape12"] params_mcem[2] true_params["log_scale12"] params_mcem[3] true_params["beta12"]
@printf "   h23: shape=%.3f (true=%.1f), log_scale=%.3f (true=%.3f), beta=%.3f (true=%.1f)\n" params_mcem[4] true_params["shape23"] params_mcem[5] true_params["log_scale23"] params_mcem[6] true_params["beta23"]

# Report errors
println("\n" * "="^80)
println("ERROR SUMMARY:")
println("="^80)
beta12_exact_err = 100*abs(params_exact[3] - true_params["beta12"])/true_params["beta12"]
beta12_mcem_err = 100*abs(params_mcem[3] - true_params["beta12"])/true_params["beta12"]
beta23_exact_err = 100*abs(params_exact[6] - true_params["beta23"])/true_params["beta23"]
beta23_mcem_err = 100*abs(params_mcem[6] - true_params["beta23"])/true_params["beta23"]

@printf "Beta12: Exact=%.1f%%, MCEM=%.1f%%\n" beta12_exact_err beta12_mcem_err
@printf "Beta23: Exact=%.1f%%, MCEM=%.1f%%\n" beta23_exact_err beta23_mcem_err
