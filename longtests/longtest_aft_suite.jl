# =============================================================================
# AFT Comprehensive Test Suite
# =============================================================================
#
# Rigorous validation of Accelerated Failure Time (AFT) models.
# Covers:
# 1. Simulation & Inference (Parameter Recovery)
# 2. Competing Risks with Differential Acceleration
# 3. Exact vs Panel Data
# 4. Covariate Scenarios: None, Time-Fixed (TFC), Time-Varying (TVC)
# 5. Families: Weibull, Gompertz
#
# =============================================================================

using MultistateModels
using DataFrames
using Random
using Test
using Printf
using Statistics

# Include shared configuration and helpers
include("longtest_config.jl")
include("longtest_helpers.jl")
include("../src/LongTestResults.jl")

# =============================================================================
# Scenario Definitions
# =============================================================================

struct AFTScenario
    name::String
    family::String          # "wei", "gom"
    covariate_type::Symbol  # :none, :tfc, :tvc
    data_type::Symbol       # :exact, :panel
    competing_risks::Bool   # true = 1->2 (accel) & 1->3 (decel)
end

const AFT_SCENARIOS = [
    # 1. Basic Weibull AFT (Exact, TFC)
    AFTScenario("wei_aft_exact_tfc", "wei", :tfc, :exact, false),
    
    # 2. Basic Weibull AFT (Panel, TFC)
    AFTScenario("wei_aft_panel_tfc", "wei", :tfc, :panel, false),
    
    # 3. Time-Varying Covariates (Exact, TVC)
    AFTScenario("wei_aft_exact_tvc", "wei", :tvc, :exact, false),
    
    # 4. Time-Varying Covariates (Panel, TVC)
    AFTScenario("wei_aft_panel_tvc", "wei", :tvc, :panel, false),
    
    # 5. Competing Risks Differential Acceleration (Exact, TFC)
    # Hazard 1: beta > 0 (Deceleration), Hazard 2: beta < 0 (Acceleration)
    AFTScenario("wei_aft_competing_exact", "wei", :tfc, :exact, true),
    
    # 6. Gompertz AFT (Exact, TFC) - Verify different family
    AFTScenario("gom_aft_exact_tfc", "gom", :tfc, :exact, false)
]

# =============================================================================
# Helper Functions
# =============================================================================

"""
    setup_aft_model(scenario::AFTScenario)

Create the model structure and true parameters for a given scenario.
"""
function setup_aft_model(scenario::AFTScenario)
    n_subj = N_SUBJECTS
    
    # 1. Create Template Data
    if scenario.covariate_type == :none
        template = create_baseline_template(n_subj)
    elseif scenario.covariate_type == :tfc
        template = create_tfc_template(n_subj)
    elseif scenario.covariate_type == :tvc
        template = create_tvc_template(n_subj)
    end
    
    # 2. Define Hazards
    if scenario.competing_risks
        # Competing Risks: 1->2 and 1->3
        # Both AFT
        h12 = Hazard(@formula(0 ~ x), scenario.family, 1, 2; linpred_effect=:aft)
        h13 = Hazard(@formula(0 ~ x), scenario.family, 1, 3; linpred_effect=:aft)
        hazards = (h12, h13)
        
        # True Parameters
        # h12: Deceleration (beta > 0)
        # h13: Acceleration (beta < 0)
        if scenario.family == "wei"
            # shape, scale, beta (natural scale since v0.3.0)
            true_params = (
                h12 = [1.2, 0.5, 0.5],   # shape=1.2, scale=0.5, beta=0.5
                h13 = [1.2, 0.5, -0.5]   # shape=1.2, scale=0.5, beta=-0.5
            )
        else # gom
            # shape, rate, beta (natural scale since v0.3.0)
            true_params = (
                h12 = [0.1, 0.5, 0.5],
                h13 = [0.1, 0.5, -0.5]
            )
        end
    else
        # Single Transition: 1->2
        if scenario.covariate_type == :none
            h12 = Hazard(@formula(0 ~ 1), scenario.family, 1, 2; linpred_effect=:aft)
            if scenario.family == "wei"
                true_params = (h12 = [1.2, 0.5],)  # natural scale since v0.3.0
            else
                true_params = (h12 = [0.1, 0.5],)  # natural scale since v0.3.0
            end
        else
            h12 = Hazard(@formula(0 ~ x), scenario.family, 1, 2; linpred_effect=:aft)
            if scenario.family == "wei"
                true_params = (h12 = [1.2, 0.5, 0.5],)  # natural scale since v0.3.0
            else
                true_params = (h12 = [0.1, 0.5, 0.5],)  # natural scale since v0.3.0
            end
        end
        hazards = (h12,)
    end
    
    # 3. Initialize Model
    model = multistatemodel(hazards...; data=template)
    set_parameters!(model, true_params)
    
    return model, true_params, hazards
end

"""
    run_aft_scenario(scenario::AFTScenario)

Execute a single AFT test scenario.
"""
function run_aft_scenario(scenario::AFTScenario)
    println("\nRunning Scenario: $(scenario.name)")
    println("-"^60)
    
    # Setup
    model, true_params, hazards = setup_aft_model(scenario)
    
    # Simulate Data
    # For panel data, we first simulate exact, then censor
    sim_data = simulate(model; paths=false, data=true, nsim=1)[1, 1]
    
    if scenario.data_type == :panel
        # Convert to panel data
        # Use standard panel times from config
        panel_data = make_panel_data(sim_data, PANEL_TIMES, scenario.covariate_type)
        
        # Re-initialize model with panel data
        # Note: For MCEM, we need surrogate=:markov
        model_fit = multistatemodel(hazards...; data=panel_data, surrogate=:markov)
    else
        # Exact data
        model_fit = multistatemodel(hazards...; data=sim_data)
    end
    
    # Fit Model
    println("  Fitting model...")
    if scenario.data_type == :panel
        fitted = fit(model_fit; 
            verbose=false, 
            method=:MCEM,
            tol=MCEM_TOL,
            ess_target_initial=MCEM_ESS_INITIAL,
            max_ess=MCEM_ESS_MAX,
            maxiter=MCEM_MAX_ITER
        )
    else
        fitted = fit(model_fit; verbose=false)
    end
    
    # Evaluate Results
    est_flat = get_parameters_flat(fitted)
    
    # Flatten true params in same order
    true_flat = Float64[]
    for haz in fitted.hazards
        append!(true_flat, true_params[haz.hazname])
    end
    
    # Get parameter names
    par_names = get_parnames(fitted)
    if !isempty(par_names) && par_names[1] isa Vector
        par_names = vcat(par_names...)
    end
    
    # Print Comparison
    println(rpad("  Parameter", 20), rpad("True", 10), rpad("Est", 10), rpad("Err%", 10), "Status")
    println("  " * "-"^60)
    
    all_passed = true
    max_err = 0.0
    
    for (i, name) in enumerate(par_names)
        t_val = true_flat[i]
        e_val = est_flat[i]
        
        # Error calculation
        if abs(t_val) < SMALL_PARAM_THRESHOLD
            # Absolute error for small params
            err = abs(e_val - t_val)
            rel_err_disp = NaN
            passed = err <= SHAPE_ABS_TOL # Use looser tolerance for shape/small params
        else
            # Relative error
            err = abs((e_val - t_val) / t_val)
            rel_err_disp = err * 100
            passed = err <= PARAM_REL_TOL
        end
        
        status = passed ? "✓" : "✗"
        all_passed = all_passed && passed
        
        @printf "  %-20s %-10.4f %-10.4f %-10.1f %s\n" name t_val e_val (isnan(rel_err_disp) ? 0.0 : rel_err_disp) status
    end
    
    println("  " * "-"^60)
    println("  Result: " * (all_passed ? "PASS" : "FAIL"))
    
    return all_passed
end

"""
    make_panel_data(exact_data, times, cov_type)

Helper to convert exact data to panel format.
Handles TVC logic correctly (sampling x at interval start).
"""
function make_panel_data(exact_data, times, cov_type)
    panel_rows = DataFrame[]
    
    for subj_id in unique(exact_data.id)
        subj_data = filter(r -> r.id == subj_id, exact_data)
        
        # Find transition times
        # Assuming 1->2 or 1->3
        trans_row = filter(r -> r.stateto != 1, subj_data)
        
        if isempty(trans_row)
            trans_time = Inf
            final_state = 1
        else
            trans_time = first(trans_row).tstop
            final_state = first(trans_row).stateto
        end
        
        for i in 1:(length(times)-1)
            t0 = times[i]
            t1 = times[i+1]
            
            # Determine state at t0 and t1
            s0 = trans_time <= t0 ? final_state : 1
            s1 = trans_time <= t1 ? final_state : 1
            
            # Only include if not already absorbed at t0
            if s0 == 1
                # Covariate handling
                if cov_type == :tvc
                    # Sample x at t0
                    x_val = t0 < TVC_CHANGEPOINT ? 0.0 : 1.0
                elseif cov_type == :tfc
                    # Take x from first row of subject
                    x_val = subj_data[1, :x]
                else
                    x_val = nothing
                end
                
                row = DataFrame(
                    id = subj_id,
                    tstart = t0,
                    tstop = t1,
                    statefrom = s0,
                    stateto = s1,
                    obstype = 2
                )
                
                if !isnothing(x_val)
                    row[!, :x] .= x_val
                end
                
                push!(panel_rows, row)
            end
        end
    end
    panel_df = vcat(panel_rows...)
    
    # Re-index IDs to be consecutive
    # MultistateModels requires IDs 1..N
    old_ids = sort(unique(panel_df.id))
    id_map = Dict(id => i for (i, id) in enumerate(old_ids))
    panel_df.id = [id_map[id] for id in panel_df.id]
    
    return panel_df
end

# =============================================================================
# Main Runner
# =============================================================================

function run_aft_suite()
    println("="^80)
    println("AFT COMPREHENSIVE TEST SUITE")
    println("="^80)
    println("Date: $(Dates.now())")
    println("Scenarios: $(length(AFT_SCENARIOS))")
    
    Random.seed!(RNG_SEED)
    
    results = Dict{String, Bool}()
    
    for scenario in AFT_SCENARIOS
        try
            passed = run_aft_scenario(scenario)
            results[scenario.name] = passed
        catch e
            println("  ERROR: Scenario $(scenario.name) crashed!")
            showerror(stdout, e, catch_backtrace())
            results[scenario.name] = false
        end
    end
    
    println("\n" * "="^80)
    println("SUMMARY")
    println("="^80)
    for (name, passed) in results
        println(rpad(name, 30), passed ? "PASS" : "FAIL")
    end
    
    return all(values(results))
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_aft_suite()
end

