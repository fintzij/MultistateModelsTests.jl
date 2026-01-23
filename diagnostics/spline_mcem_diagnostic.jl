# =============================================================================
# Diagnostic Script: Spline MCEM Boundary Coefficient Bug
# =============================================================================
#
# This script investigates why unpenalized spline hazards fitted via MCEM 
# on panel data produce catastrophically wrong estimates for the rightmost 
# B-spline coefficient (coef_4).
#
# Investigation Areas:
# 1. Extrapolation behavior when t > upper boundary knot
# 2. B-spline basis function values at sampled transition times
# 3. Comparison of transition time distributions: true paths vs sampled paths
# 4. Hessian condition numbers for identifiability analysis
#
# =============================================================================

using Test
using MultistateModels
using DataFrames
using Random
using LinearAlgebra
using Statistics
using BSplineKit
using Printf

import MultistateModels: Hazard, @formula, multistatemodel, fit, set_parameters!, 
    simulate, get_parameters_flat, get_parameters, _Hazard, RuntimeSplineHazard,
    draw_samplepath, build_hazmat_book, build_tpm_book, ForwardFiltering!,
    MarkovSurrogate, _build_markov_surrogate, compute_hazmat!

# =============================================================================
# Configuration
# =============================================================================

const N_SUBJECTS = 500
const MAX_TIME = 15.0
const RNG_SEED = 20260115

# Spline settings (matching longtest_spline_suite.jl)
const SPLINE_DEGREE = 3
const N_INTERIOR_KNOTS = 2
const BOUNDARY_KNOTS = [0.0, MAX_TIME]

# True spline coefficients
const TRUE_COEFS_H12 = [0.08, 0.10, 0.14, 0.18]
const TRUE_COEFS_H23 = [0.06, 0.08, 0.11, 0.14]

# Panel observation times
const PANEL_TIMES = [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0]

# =============================================================================
# Diagnostic 1: Spline Extrapolation Behavior
# =============================================================================

"""
Test what happens when spline is evaluated at times beyond boundary knots.
"""
function diagnose_extrapolation()
    println("\n" * "="^80)
    println("DIAGNOSTIC 1: Spline Extrapolation Behavior")
    println("="^80)
    
    # Build spline basis (matching production code)
    knots = collect(range(BOUNDARY_KNOTS[1] + (BOUNDARY_KNOTS[2] - BOUNDARY_KNOTS[1]) / (N_INTERIOR_KNOTS + 1),
                         BOUNDARY_KNOTS[2] - (BOUNDARY_KNOTS[2] - BOUNDARY_KNOTS[1]) / (N_INTERIOR_KNOTS + 1),
                         length=N_INTERIOR_KNOTS))
    allknots = unique(sort([BOUNDARY_KNOTS[1]; knots; BOUNDARY_KNOTS[2]]))
    
    println("\nKnot configuration:")
    println("  Interior knots: $knots")
    println("  All knots: $allknots")
    println("  Boundary knots: $(BOUNDARY_KNOTS)")
    
    # Build B-spline basis with recombination (constant extrapolation uses Derivative(1) BC)
    B = BSplineBasis(BSplineOrder(SPLINE_DEGREE + 1), copy(allknots))
    B_recombined = RecombinedBSplineBasis(B, Derivative(1))
    
    println("\nBasis info:")
    println("  Original basis functions: $(length(B))")
    println("  Recombined basis functions: $(length(B_recombined))")
    
    # Evaluate basis at various times including extrapolation region
    test_times = [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 14.5, 15.0, 15.5, 16.0, 17.5, 20.0]
    
    println("\nBasis function values at various times:")
    println("-"^100)
    @printf("%8s |", "t")
    for i in 1:length(B_recombined)
        @printf(" B_%d     ", i)
    end
    @printf("| Sum\n")
    println("-"^100)
    
    for t in test_times
        marker = t > BOUNDARY_KNOTS[2] ? " *EXTRAP*" : ""
        @printf("%8.2f |", t)
        
        basis_vals = zeros(length(B_recombined))
        for (i, bf) in enumerate(B_recombined)
            # Direct evaluation at t
            val = t >= BSplineKit.boundaries(B_recombined)[1] && t <= BSplineKit.boundaries(B_recombined)[2] ? bf(t) : 0.0
            basis_vals[i] = val
            @printf(" %.5f", val)
        end
        @printf(" | %.5f%s\n", sum(basis_vals), marker)
    end
    
    # Test spline evaluation with coefficients
    coefs = TRUE_COEFS_H12
    spline = Spline(B_recombined, coefs)
    spline_ext = SplineExtrapolation(spline, BSplineKit.SplineExtrapolations.Flat())
    
    println("\nSpline hazard h(t) = B(t)'β with coefficients β = $coefs:")
    println("-"^60)
    @printf("%8s | %12s | %s\n", "t", "h(t)", "Status")
    println("-"^60)
    
    for t in test_times
        h = spline_ext(t)
        status = t > BOUNDARY_KNOTS[2] ? "EXTRAPOLATED" : "in support"
        @printf("%8.2f | %12.6f | %s\n", t, h, status)
    end
    
    # Identify which coefficient dominates at boundary
    println("\nContribution of each coefficient at t = $(BOUNDARY_KNOTS[2]):")
    t_boundary = BOUNDARY_KNOTS[2]
    basis_at_boundary = [bf(t_boundary) for bf in B_recombined]
    contributions = basis_at_boundary .* coefs
    total = sum(contributions)
    
    for i in 1:length(coefs)
        pct = 100.0 * contributions[i] / total
        @printf("  coef_%d: β_%d=%.4f, B_%d(%.1f)=%.6f, contribution=%.6f (%.1f%%)\n",
                i, i, coefs[i], i, t_boundary, basis_at_boundary[i], contributions[i], pct)
    end
    
    println("\n=> At t=$(BOUNDARY_KNOTS[2]), rightmost basis function dominates.")
    println("   For constant extrapolation, h(t>15) = h(15) = constant.")
    
    return nothing
end

# =============================================================================
# Diagnostic 2: Compare Exact vs Panel Fits
# =============================================================================

"""
Build hazard specification and knots.
"""
function build_spline_hazards()
    knots = collect(range(BOUNDARY_KNOTS[1] + (BOUNDARY_KNOTS[2] - BOUNDARY_KNOTS[1]) / (N_INTERIOR_KNOTS + 1),
                         BOUNDARY_KNOTS[2] - (BOUNDARY_KNOTS[2] - BOUNDARY_KNOTS[1]) / (N_INTERIOR_KNOTS + 1),
                         length=N_INTERIOR_KNOTS))
    
    h12 = Hazard(@formula(0 ~ 1), :sp, 1, 2;
        degree=SPLINE_DEGREE, knots=knots, boundaryknots=BOUNDARY_KNOTS, extrapolation="constant")
    h23 = Hazard(@formula(0 ~ 1), :sp, 2, 3;
        degree=SPLINE_DEGREE, knots=knots, boundaryknots=BOUNDARY_KNOTS, extrapolation="constant")
    
    true_params = (h12 = TRUE_COEFS_H12, h23 = TRUE_COEFS_H23)
    
    return h12, h23, true_params, knots
end

"""
Generate exact observation data from spline model.
"""
function generate_exact_data()
    Random.seed!(RNG_SEED)
    
    # Simple template: one row per subject, starting in state 1
    template = DataFrame(
        id = 1:N_SUBJECTS,
        tstart = zeros(N_SUBJECTS),
        tstop = fill(MAX_TIME, N_SUBJECTS),
        statefrom = ones(Int, N_SUBJECTS),
        stateto = ones(Int, N_SUBJECTS),
        obstype = ones(Int, N_SUBJECTS)
    )
    
    h12, h23, true_params, knots = build_spline_hazards()
    
    model = multistatemodel(h12, h23; data=template)
    set_parameters!(model, true_params)
    
    sim_data = simulate(model; data=true, paths=false, nsim=1)[1]
    
    return sim_data, h12, h23, true_params
end

"""
Generate panel observation data from spline model.
"""
function generate_panel_data()
    Random.seed!(RNG_SEED)  # Same seed for comparison
    
    nobs = length(PANEL_TIMES) - 1
    
    template = DataFrame(
        id = repeat(1:N_SUBJECTS, inner=nobs),
        tstart = repeat(PANEL_TIMES[1:end-1], N_SUBJECTS),
        tstop = repeat(PANEL_TIMES[2:end], N_SUBJECTS),
        statefrom = ones(Int, N_SUBJECTS * nobs),
        stateto = ones(Int, N_SUBJECTS * nobs),
        obstype = fill(2, N_SUBJECTS * nobs)
    )
    
    h12, h23, true_params, knots = build_spline_hazards()
    
    model = multistatemodel(h12, h23; data=template)
    set_parameters!(model, true_params)
    
    # Simulate with panel observation for 1->2, exact for 2->3
    obstype_map = Dict(1 => 2, 2 => 1)
    sim_result = simulate(model; paths=false, data=true, nsim=1, autotmax=false,
                         obstype_by_transition=obstype_map)
    
    return sim_result[1, 1], h12, h23, true_params
end

"""
Compare exact and panel fitting.
"""
function diagnose_exact_vs_panel()
    println("\n" * "="^80)
    println("DIAGNOSTIC 2: Exact vs Panel Fitting Comparison")
    println("="^80)
    
    # Generate data
    println("\nGenerating exact data...")
    exact_data, h12, h23, true_params = generate_exact_data()
    println("  Exact data rows: $(nrow(exact_data))")
    
    println("\nGenerating panel data...")
    panel_data, _, _, _ = generate_panel_data()
    println("  Panel data rows: $(nrow(panel_data))")
    
    # Analyze transition times in exact data
    transitions_12 = filter(r -> r.statefrom == 1 && r.stateto == 2 && r.obstype == 1, exact_data)
    transitions_23 = filter(r -> r.statefrom == 2 && r.stateto == 3 && r.obstype == 1, exact_data)
    
    println("\nExact data transition time statistics:")
    if nrow(transitions_12) > 0
        times_12 = transitions_12.tstop .- transitions_12.tstart
        println("  1→2: n=$(nrow(transitions_12)), mean=$(round(mean(times_12), digits=2)), max=$(round(maximum(times_12), digits=2))")
        # How many are near the boundary?
        near_boundary = sum(times_12 .> 0.9 * BOUNDARY_KNOTS[2])
        println("      Near boundary (t > $(0.9 * BOUNDARY_KNOTS[2])): $near_boundary ($(round(100*near_boundary/length(times_12), digits=1))%)")
    end
    
    if nrow(transitions_23) > 0
        times_23 = transitions_23.tstop .- transitions_23.tstart
        println("  2→3: n=$(nrow(transitions_23)), mean=$(round(mean(times_23), digits=2)), max=$(round(maximum(times_23), digits=2))")
        near_boundary = sum(times_23 .> 0.9 * BOUNDARY_KNOTS[2])
        println("      Near boundary (t > $(0.9 * BOUNDARY_KNOTS[2])): $near_boundary ($(round(100*near_boundary/length(times_23), digits=1))%)")
    end
    
    # Fit exact model
    println("\nFitting exact data model...")
    model_exact = multistatemodel(h12, h23; data=exact_data)
    fitted_exact = fit(model_exact; verbose=false, vcov_type=:ij, penalty=:none)
    
    # Fit panel model with Markov surrogate
    println("Fitting panel data model (Markov surrogate)...")
    model_panel = multistatemodel(h12, h23; data=panel_data, surrogate=:markov)
    fitted_panel = fit(model_panel;
        verbose=false, vcov_type=:ij, method=:MCEM, penalty=:none,
        tol=0.05, ess_target_initial=100, max_ess=1000, maxiter=20)
    
    # Compare estimates
    params_exact = get_parameters(fitted_exact)
    params_panel = get_parameters(fitted_panel)
    
    println("\n" * "-"^80)
    println("Parameter Comparison:")
    println("-"^80)
    @printf("%-15s | %10s | %10s | %10s | %10s | %10s\n", 
            "Parameter", "True", "Exact", "Panel", "Exact Err%", "Panel Err%")
    println("-"^80)
    
    for (i, (name, true_val)) in enumerate(zip(["h12_coef_$j" for j in 1:4], TRUE_COEFS_H12))
        est_exact = params_exact.h12[i]
        est_panel = params_panel.h12[i]
        err_exact = 100 * abs(est_exact - true_val) / true_val
        err_panel = 100 * abs(est_panel - true_val) / true_val
        flag = err_panel > 100 ? " ***" : ""
        @printf("%-15s | %10.4f | %10.4f | %10.4f | %10.1f | %10.1f%s\n",
                name, true_val, est_exact, est_panel, err_exact, err_panel, flag)
    end
    
    println("-"^80)
    
    for (i, (name, true_val)) in enumerate(zip(["h23_coef_$j" for j in 1:4], TRUE_COEFS_H23))
        est_exact = params_exact.h23[i]
        est_panel = params_panel.h23[i]
        err_exact = 100 * abs(est_exact - true_val) / true_val
        err_panel = 100 * abs(est_panel - true_val) / true_val
        flag = err_panel > 100 ? " ***" : ""
        @printf("%-15s | %10.4f | %10.4f | %10.4f | %10.1f | %10.1f%s\n",
                name, true_val, est_exact, est_panel, err_exact, err_panel, flag)
    end
    
    println("\n*** indicates > 100% error")
    
    return fitted_exact, fitted_panel, exact_data, panel_data
end

# =============================================================================
# Diagnostic 3: Sampled Path Transition Time Distribution
# =============================================================================

"""
Analyze transition times in MCEM sampled paths.
"""
function diagnose_sampled_paths(fitted_panel, panel_data)
    println("\n" * "="^80)
    println("DIAGNOSTIC 3: MCEM Sampled Path Analysis")
    println("="^80)
    
    # Check if we have proposed paths
    if !hasproperty(fitted_panel, :ProposedPaths) || isnothing(fitted_panel.ProposedPaths)
        println("\n  No proposed paths stored in fitted model.")
        println("  Re-running fit to capture paths...")
        
        # Re-fit to get paths (need to access internal state)
        knots = collect(range(BOUNDARY_KNOTS[1] + (BOUNDARY_KNOTS[2] - BOUNDARY_KNOTS[1]) / (N_INTERIOR_KNOTS + 1),
                             BOUNDARY_KNOTS[2] - (BOUNDARY_KNOTS[2] - BOUNDARY_KNOTS[1]) / (N_INTERIOR_KNOTS + 1),
                             length=N_INTERIOR_KNOTS))
        
        h12 = Hazard(@formula(0 ~ 1), :sp, 1, 2;
            degree=SPLINE_DEGREE, knots=knots, boundaryknots=BOUNDARY_KNOTS, extrapolation="constant")
        h23 = Hazard(@formula(0 ~ 1), :sp, 2, 3;
            degree=SPLINE_DEGREE, knots=knots, boundaryknots=BOUNDARY_KNOTS, extrapolation="constant")
        
        model = multistatemodel(h12, h23; data=panel_data, surrogate=:markov)
        
        # This would require exposing internal MCEM state
        println("  (Path capture requires internal access - skipping detailed analysis)")
        return nothing
    end
    
    paths = fitted_panel.ProposedPaths
    
    # Collect all transition times from sampled paths
    trans_times_12 = Float64[]
    trans_times_23 = Float64[]
    
    for subj_paths in paths
        for path in subj_paths
            for i in 1:(length(path.states) - 1)
                if path.states[i] == 1 && path.states[i+1] == 2
                    push!(trans_times_12, path.times[i+1] - path.times[i])
                elseif path.states[i] == 2 && path.states[i+1] == 3
                    push!(trans_times_23, path.times[i+1] - path.times[i])
                end
            end
        end
    end
    
    println("\nSampled path transition time statistics:")
    if !isempty(trans_times_12)
        println("  1→2: n=$(length(trans_times_12)), mean=$(round(mean(trans_times_12), digits=2)), max=$(round(maximum(trans_times_12), digits=2))")
        near_boundary = sum(trans_times_12 .> 0.9 * BOUNDARY_KNOTS[2])
        beyond_boundary = sum(trans_times_12 .> BOUNDARY_KNOTS[2])
        println("      Near boundary (t > $(0.9 * BOUNDARY_KNOTS[2])): $near_boundary ($(round(100*near_boundary/length(trans_times_12), digits=1))%)")
        println("      Beyond boundary (t > $(BOUNDARY_KNOTS[2])): $beyond_boundary ($(round(100*beyond_boundary/length(trans_times_12), digits=1))%)")
    end
    
    return trans_times_12, trans_times_23
end

# =============================================================================
# Diagnostic 4: Direct Spline Evaluation at Sampled Times
# =============================================================================

"""
Evaluate spline basis contributions at various times to understand weighting.
"""
function diagnose_basis_weighting()
    println("\n" * "="^80)
    println("DIAGNOSTIC 4: B-spline Basis Function Analysis")
    println("="^80)
    
    # Build recombined basis (matching production code)
    knots = collect(range(BOUNDARY_KNOTS[1] + (BOUNDARY_KNOTS[2] - BOUNDARY_KNOTS[1]) / (N_INTERIOR_KNOTS + 1),
                         BOUNDARY_KNOTS[2] - (BOUNDARY_KNOTS[2] - BOUNDARY_KNOTS[1]) / (N_INTERIOR_KNOTS + 1),
                         length=N_INTERIOR_KNOTS))
    allknots = unique(sort([BOUNDARY_KNOTS[1]; knots; BOUNDARY_KNOTS[2]]))
    
    B = BSplineBasis(BSplineOrder(SPLINE_DEGREE + 1), copy(allknots))
    B_recombined = RecombinedBSplineBasis(B, Derivative(1))
    
    nbasis = length(B_recombined)
    
    # Analyze effective "support" of each basis function
    println("\nBasis function support analysis:")
    println("(Where each basis function has significant weight)")
    
    fine_grid = range(0.0, BOUNDARY_KNOTS[2], length=1000)
    
    for i in 1:nbasis
        bf = B_recombined[i]
        vals = [bf(t) for t in fine_grid]
        
        # Find where this basis has > 10% of its max
        max_val = maximum(vals)
        threshold = 0.1 * max_val
        significant = findall(v -> v > threshold, vals)
        
        if !isempty(significant)
            t_start = fine_grid[significant[1]]
            t_end = fine_grid[significant[end]]
            println("  B_$i: significant on [$(round(t_start, digits=2)), $(round(t_end, digits=2))], max=$(round(max_val, digits=4))")
        end
    end
    
    # Compute cumulative contribution: ∫ B_i(t) dt over [0, T]
    println("\nCumulative exposure (∫ B_i(t) dt over [0, $MAX_TIME]):")
    
    dt = fine_grid[2] - fine_grid[1]
    for i in 1:nbasis
        bf = B_recombined[i]
        integral = sum(bf(t) for t in fine_grid) * dt
        println("  B_$i: $(round(integral, digits=4))")
    end
    
    # Key insight: if transitions are sampled late, B_4 dominates
    println("\nCritical insight:")
    println("  If MCEM samples transitions predominantly at late times (near $MAX_TIME),")
    println("  then B_4 (rightmost basis) will have disproportionate influence in the")
    println("  complete-data likelihood, causing coef_4 to be overestimated.")
    
    return nothing
end

# =============================================================================
# Run All Diagnostics
# =============================================================================

function run_all_diagnostics()
    println("\n" * "="^80)
    println("SPLINE MCEM BOUNDARY COEFFICIENT BUG DIAGNOSTIC")
    println("="^80)
    println("\nThis script investigates why unpenalized spline hazards fitted via MCEM")
    println("produce catastrophically wrong estimates for the rightmost coefficient.")
    println("\nBug summary: coef_4 overestimated by 300-1000% in panel data MCEM fits.")
    
    # Run diagnostics
    diagnose_extrapolation()
    fitted_exact, fitted_panel, exact_data, panel_data = diagnose_exact_vs_panel()
    diagnose_sampled_paths(fitted_panel, panel_data)
    diagnose_basis_weighting()
    
    println("\n" * "="^80)
    println("DIAGNOSTIC SUMMARY")
    println("="^80)
    println("\nKey findings will appear above. Look for:")
    println("  1. How spline behaves at extrapolated times (t > $MAX_TIME)")
    println("  2. Large differences between exact and panel estimates for coef_4")
    println("  3. Distribution of transition times in sampled paths")
    println("  4. Which basis functions dominate at late times")
    
    return fitted_exact, fitted_panel, exact_data, panel_data
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_all_diagnostics()
end
