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
# 6. Markov vs PhaseType Proposal Comparison (for Weibull AFT panel tests)
#
# =============================================================================

using MultistateModels
using DataFrames
using Random
using Test
using Printf
using Statistics

# Import PhaseTypeProposal for proposal comparisons
import MultistateModels: PhaseTypeProposal

# Include shared configuration and helpers
include("longtest_config.jl")
include("longtest_helpers.jl")
include("../src/LongTestResults.jl")

# Tolerance for Markov vs PhaseType proposal comparison (30% relative)
const PROPOSAL_COMPARISON_TOL = 0.30

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
    AFTScenario("gom_aft_exact_tfc", "gom", :tfc, :exact, false),
    
    # =========================================================================
    # Additional Weibull AFT Tests (No Covariates)
    # =========================================================================
    # Weibull has 2 baseline parameters: shape, scale
    # Natural scale: [shape, scale] for nocov
    # =========================================================================
    
    # 7. Weibull AFT - Exact data, no covariates
    AFTScenario("wei_aft_exact_nocov", "wei", :none, :exact, false),
    
    # 8. Weibull AFT - Panel data, no covariates
    AFTScenario("wei_aft_panel_nocov", "wei", :none, :panel, false),
    
    # =========================================================================
    # Additional Gompertz AFT Tests (Complete Coverage)
    # =========================================================================
    # Gompertz has 2 baseline parameters: shape (can be negative), rate
    # Natural scale: [shape, rate] for nocov, [shape, rate, beta] for covariates
    # Shape ~0.1-0.15 gives moderate increasing hazard
    # =========================================================================
    
    # 9. Gompertz AFT - Exact data, no covariates  
    AFTScenario("gom_aft_exact_nocov", "gom", :none, :exact, false),
    
    # 10. Gompertz AFT - Exact data, time-varying covariate
    AFTScenario("gom_aft_exact_tvc", "gom", :tvc, :exact, false),
    
    # 11. Gompertz AFT - Panel data, no covariates
    AFTScenario("gom_aft_panel_nocov", "gom", :none, :panel, false),
    
    # 12. Gompertz AFT - Panel data, time-fixed covariate
    AFTScenario("gom_aft_panel_tfc", "gom", :tfc, :panel, false),
    
    # 13. Gompertz AFT - Panel data, time-varying covariate
    AFTScenario("gom_aft_panel_tvc", "gom", :tvc, :panel, false),
    
    # =========================================================================
    # Exponential AFT Tests (Parameter Recovery)
    # =========================================================================
    # Exponential has 1 baseline parameter: rate
    # Natural scale: [rate] for nocov, [rate, beta] for covariates
    # =========================================================================
    
    # 14. Exponential AFT - Exact data, no covariates
    AFTScenario("exp_aft_exact_nocov", "exp", :none, :exact, false),
    
    # 15. Exponential AFT - Exact data, time-fixed covariate
    AFTScenario("exp_aft_exact_tfc", "exp", :tfc, :exact, false),
    
    # 16. Exponential AFT - Exact data, time-varying covariate  
    AFTScenario("exp_aft_exact_tvc", "exp", :tvc, :exact, false),
    
    # 17. Exponential AFT - Panel data, no covariates
    AFTScenario("exp_aft_panel_nocov", "exp", :none, :panel, false),
    
    # 18. Exponential AFT - Panel data, time-fixed covariate
    AFTScenario("exp_aft_panel_tfc", "exp", :tfc, :panel, false),
    
    # 19. Exponential AFT - Panel data, time-varying covariate
    AFTScenario("exp_aft_panel_tvc", "exp", :tvc, :panel, false)
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
        elseif scenario.family == "exp"
            # rate, beta (natural scale)
            true_params = (
                h12 = [0.25, 0.5],    # rate=0.25, beta=0.5
                h13 = [0.25, -0.5]    # rate=0.25, beta=-0.5
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
            elseif scenario.family == "exp"
                true_params = (h12 = [0.25],)  # rate=0.25 (natural scale)
            else # gom
                true_params = (h12 = [0.1, 0.5],)  # natural scale since v0.3.0
            end
        else
            h12 = Hazard(@formula(0 ~ x), scenario.family, 1, 2; linpred_effect=:aft)
            if scenario.family == "wei"
                true_params = (h12 = [1.2, 0.5, 0.5],)  # natural scale since v0.3.0
            elseif scenario.family == "exp"
                true_params = (h12 = [0.25, 0.5],)  # rate=0.25, beta=0.5 (natural scale)
            else # gom
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

For Weibull AFT panel tests (wei_aft_panel_*), also fits with PhaseTypeProposal
and compares parameter estimates to verify proposal-independence of results.
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
    
    # Fit Model (default/Markov proposal)
    println("  Fitting model...")
    if scenario.data_type == :panel
        fitted = fit(model_fit; 
            verbose=false, 
            method=:MCEM,
            proposal=:markov,
            tol=MCEM_TOL,
            ess_target_initial=MCEM_ESS_INITIAL,
            max_ess=MCEM_ESS_MAX,
            maxiter=MCEM_MAX_ITER
        )
    else
        fitted = fit(model_fit; verbose=false)
    end
    
    # ==========================================================================
    # Markov vs PhaseType Proposal Comparison (Weibull AFT panel tests only)
    # ==========================================================================
    fitted_pt = nothing
    est_flat_pt = nothing
    is_wei_aft_panel = scenario.family == "wei" && 
                       scenario.data_type == :panel && 
                       !scenario.competing_risks
    
    if is_wei_aft_panel
        println("  Fitting with PhaseTypeProposal(n_phases=3)...")
        
        # Re-create model for PhaseType fit (fresh hazard objects)
        if scenario.covariate_type == :none
            h12_pt = Hazard(@formula(0 ~ 1), scenario.family, 1, 2; linpred_effect=:aft)
        else
            h12_pt = Hazard(@formula(0 ~ x), scenario.family, 1, 2; linpred_effect=:aft)
        end
        model_fit_pt = multistatemodel(h12_pt; data=panel_data, surrogate=:markov)
        
        fitted_pt = fit(model_fit_pt;
            verbose=false,
            method=:MCEM,
            proposal=PhaseTypeProposal(n_phases=3),
            tol=MCEM_TOL,
            ess_target_initial=MCEM_ESS_INITIAL,
            max_ess=MCEM_ESS_MAX,
            maxiter=MCEM_MAX_ITER
        )
        est_flat_pt = get_parameters_flat(fitted_pt)
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
    
    # Determine if this is a TVC scenario (harder estimation)
    is_tvc = scenario.covariate_type == :tvc
    
    for (i, name) in enumerate(par_names)
        t_val = true_flat[i]
        e_val = est_flat[i]
        
        # Check if this is a covariate (beta) parameter
        is_covariate_param = occursin("_x", String(name)) || occursin("_beta", String(name))
        
        # Error calculation with tolerance selection based on parameter type
        if abs(t_val) < SMALL_PARAM_THRESHOLD
            # Absolute error for small params (typically shape parameters)
            err = abs(e_val - t_val)
            rel_err_disp = NaN
            passed = err <= SHAPE_ABS_TOL
        elseif is_covariate_param
            # For covariate params, use absolute tolerance (beta can be near 0)
            err = abs(e_val - t_val)
            rel_err_disp = (t_val != 0) ? abs((e_val - t_val) / t_val) * 100 : NaN
            # Use relaxed tolerance for TVC scenarios (harder estimation)
            tol = is_tvc ? MCEM_TVC_BETA_ABS_TOL : BETA_ABS_TOL
            passed = err <= tol
        else
            # Relative error for baseline parameters
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
    
    # ==========================================================================
    # Compare Markov vs PhaseType Proposal (Weibull AFT panel tests)
    # ==========================================================================
    proposal_agreement = true
    if is_wei_aft_panel && !isnothing(fitted_pt)
        println("\n  Markov vs PhaseType Proposal Comparison:")
        println("  " * "-"^70)
        println("  ", rpad("Parameter", 18), rpad("Markov", 12), rpad("PhaseType", 12), rpad("Rel Diff", 12), "Status")
        println("  " * "-"^70)
        
        for (i, name) in enumerate(par_names)
            m_val = est_flat[i]
            pt_val = est_flat_pt[i]
            
            # Compute relative difference (use absolute for small values)
            if abs(m_val) > 1e-10
                rel_diff = abs(pt_val - m_val) / abs(m_val)
            else
                rel_diff = abs(pt_val - m_val)
            end
            
            param_agrees = rel_diff < PROPOSAL_COMPARISON_TOL
            proposal_agreement = proposal_agreement && param_agrees
            status = param_agrees ? "✓" : "✗"
            
            @printf "  %-18s %-12.4f %-12.4f %-12.1f%% %s\n" name m_val pt_val (rel_diff * 100) status
        end
        
        println("  " * "-"^70)
        println("  Proposal Agreement: " * (proposal_agreement ? "PASS" : "FAIL"))
        
        # Assert proposal agreement
        @test proposal_agreement
    end
    
    return all_passed && proposal_agreement
end

"""
    make_panel_data(exact_data, times, cov_type; verbose=false)

Helper to convert exact data to panel format.
Handles TVC logic correctly (sampling x at interval start).

# Censoring Behavior
Subjects who reach an absorbing state (state > 1) before the first panel observation
time do not contribute any data rows. This is correct behavior for standard survival
analysis—subjects only contribute data while at risk. However, this creates a form
of informative censoring where fast progressors are excluded.

The function logs the number of dropped subjects when verbose=true. If more than
5% of subjects are dropped, a warning is emitted.
"""
function make_panel_data(exact_data, times, cov_type; verbose::Bool=false)
    panel_rows = DataFrame[]
    n_simulated = length(unique(exact_data.id))
    
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
    
    # Sample size validation (ACTION-1 from longtest review)
    n_retained = length(unique(panel_df.id))
    n_dropped = n_simulated - n_retained
    drop_rate = n_dropped / n_simulated
    
    if verbose || drop_rate > 0.05
        if n_dropped > 0
            @info "make_panel_data: $n_dropped/$n_simulated subjects dropped ($(round(100*drop_rate, digits=1))% reached absorbing state before first panel time)"
        end
        if drop_rate > 0.05
            @warn "High dropout rate ($(round(100*drop_rate, digits=1))%): this may introduce selection bias favoring slow progressors"
        end
    end
    
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

# Run the suite when this file is included
# (PROGRAM_FILE check doesn't work when using include())
run_aft_suite()
