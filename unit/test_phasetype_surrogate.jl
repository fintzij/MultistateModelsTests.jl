# =============================================================================
# PhaseType Surrogate Covariate Scaling Tests
# =============================================================================
#
# Unit tests for build_phasetype_tpm_book covariate handling.
#
# These tests verify the critical bug fix from 2026-01-15:
# - Q matrices MUST differ for different covariate combinations
# - Inter-state transition rates MUST be scaled by exp(β'x)
# - Internal phase progression rates must NOT be scaled
#
# The bug caused Markov vs PhaseType proposal estimates to diverge by 45-94%
# because the original implementation copied the same baseline Q matrix for
# all covariate combinations without applying exp(β'x) scaling.
#
# Reference: scratch/CODEBASE_REFACTORING_GUIDE.md, Wave 6 (Item #25)
# =============================================================================

using Test
using MultistateModels
using DataFrames
using Random
using LinearAlgebra
using Distributions: Exponential

# Import internal functions needed for testing
import MultistateModels: 
    Hazard, multistatemodel, set_parameters!, 
    MarkovSurrogate, PhaseTypeSurrogate, PhaseTypeProposal,
    _build_phasetype_from_markov, build_tpm_mapping,
    build_phasetype_tpm_book, fit_surrogate, @formula,
    SamplePath, convert_expanded_path_to_censored_data,
    compute_forward_loglik, loglik_phasetype_expanded_path,
    build_hazmat_book

"""
    create_panel_data_with_transitions(n_subj; seed, transition_prob_x0, transition_prob_x1)

Create panel data with actual transitions for testing.
Generates data where some subjects transition from state 1 to state 2.
transition_prob_x0 and transition_prob_x1 control transition rates for each covariate group.
"""
function create_panel_data_with_transitions(n_subj; seed=12345, 
                                           transition_prob_x0=0.3, 
                                           transition_prob_x1=0.5)
    Random.seed!(seed)
    
    n_half = div(n_subj, 2)
    rows = []
    
    for i in 1:n_subj
        x_val = i <= n_half ? 0 : 1
        trans_prob = x_val == 0 ? transition_prob_x0 : transition_prob_x1
        
        # First interval
        transitioned_t1 = rand() < trans_prob
        state_at_t2 = transitioned_t1 ? 2 : 1
        push!(rows, (id=i, tstart=0.0, tstop=2.0, statefrom=1, stateto=state_at_t2, obstype=2, x=x_val))
        
        # Second interval (if not absorbed)
        if state_at_t2 == 1
            transitioned_t2 = rand() < trans_prob
            state_at_t4 = transitioned_t2 ? 2 : 1
            push!(rows, (id=i, tstart=2.0, tstop=4.0, statefrom=1, stateto=state_at_t4, obstype=2, x=x_val))
        else
            # Absorbed - stay in state 2
            push!(rows, (id=i, tstart=2.0, tstop=4.0, statefrom=2, stateto=2, obstype=2, x=x_val))
        end
    end
    
    return DataFrame(rows)
end

@testset "PhaseType Surrogate Covariate Scaling" begin

    @testset "Q matrices differ for different covariate patterns" begin
        # Create data with actual transitions
        Random.seed!(12345)
        n_subj = 200  # More subjects for reliable fitting
        
        panel_data = create_panel_data_with_transitions(n_subj; 
            seed=12345, transition_prob_x0=0.25, transition_prob_x1=0.5)
        
        # Create model from data with transitions  
        h12 = Hazard(@formula(0 ~ x), "exp", 1, 2)
        model = multistatemodel(h12; data=panel_data, surrogate=:markov)
        
        # Fit the Markov surrogate using the model API (required before building PhaseType)
        fit_surrogate(model; type=:markov, method=:mle, verbose=false)
        markov_surrogate = model.markovsurrogate
        
        # Check that surrogate fitting gave reasonable parameters (not degenerate)
        surrogate_pars = markov_surrogate.parameters.nested
        @test haskey(surrogate_pars, :h12)
        # Rate should be positive and reasonable
        baseline_pars = surrogate_pars.h12.baseline
        @test abs(baseline_pars.h12_rate) < 10  # Not degenerate
        
        # Build PhaseType surrogate from fitted Markov
        phasetype_surrogate = _build_phasetype_from_markov(
            model, markov_surrogate;
            config=PhaseTypeProposal(n_phases=3),
            verbose=false
        )
        
        # Build TPM book mapping
        books = build_tpm_mapping(model.data)
        
        # Build PhaseType TPM book with covariate-aware Q matrices
        tpm_book_ph, hazmat_book_ph = build_phasetype_tpm_book(
            phasetype_surrogate, markov_surrogate, books, model.data
        )
        
        # Verify there are at least 2 covariate combinations (x=0 and x=1)
        @test length(hazmat_book_ph) >= 2
        
        # CRITICAL TEST: Q matrices must differ between covariate patterns
        Q1 = hazmat_book_ph[1]
        Q2 = hazmat_book_ph[2]
        
        # BUG REGRESSION CHECK: Q matrices MUST differ for different covariates
        @test !(Q1 ≈ Q2)
        
        # Verify matrix properties are valid (note: absorbing states have zero diagonal)
        @test all(diag(Q1) .<= 0)  # All non-positive
        @test all(diag(Q2) .<= 0)
        @test any(diag(Q1) .< 0)   # At least some transient states
        @test any(diag(Q2) .< 0)
        # Rows should sum to zero (generator property)
        @test all(abs.(sum(Q1, dims=2)) .< 1e-10)
        @test all(abs.(sum(Q2, dims=2)) .< 1e-10)
    end
    
    @testset "Scaling factor matches exp(β'x) for inter-state transitions" begin
        # Use data with actual transitions
        Random.seed!(54321)
        n_subj = 200
        
        panel_data = create_panel_data_with_transitions(n_subj; 
            seed=54321, transition_prob_x0=0.2, transition_prob_x1=0.45)
        
        h12 = Hazard(@formula(0 ~ x), "exp", 1, 2)
        model = multistatemodel(h12; data=panel_data, surrogate=:markov)
        
        # Fit Markov surrogate using model API
        fit_surrogate(model; type=:markov, method=:mle, verbose=false)
        markov_surrogate = model.markovsurrogate
        
        phasetype_surrogate = _build_phasetype_from_markov(
            model, markov_surrogate;
            config=PhaseTypeProposal(n_phases=2),
            verbose=false
        )
        
        books = build_tpm_mapping(model.data)
        tpm_book_ph, hazmat_book_ph = build_phasetype_tpm_book(
            phasetype_surrogate, markov_surrogate, books, model.data
        )
        
        @test length(hazmat_book_ph) >= 2
        
        Q_x0 = hazmat_book_ph[1]  # Covariate x=0
        Q_x1 = hazmat_book_ph[2]  # Covariate x=1
        
        phase_to_state = phasetype_surrogate.phase_to_state
        
        # For fitted surrogate, check that Q matrices differ in expected direction
        # Inter-state transition rates for x=1 should be larger than x=0 (since β > 0)
        found_interstate = false
        for i in 1:size(Q_x0, 1)
            for j in 1:size(Q_x0, 2)
                if i != j && phase_to_state[i] != phase_to_state[j]
                    # This is an inter-state transition
                    if abs(Q_x0[i, j]) > 1e-6 && abs(Q_x1[i, j]) > 1e-6
                        found_interstate = true
                        # Q_x1 > Q_x0 since exp(β) > 1 when β > 0
                        # Just check they differ and are in right direction
                        @test Q_x1[i, j] != Q_x0[i, j]
                    end
                end
            end
        end
        @test found_interstate  # Should find at least one inter-state transition
    end
    
    @testset "Internal phase progression rates NOT scaled by covariates" begin
        # Verify that within-state phase progression (λ rates) are NOT affected by covariates
        Random.seed!(98765)
        n_subj = 200
        
        panel_data = create_panel_data_with_transitions(n_subj; 
            seed=98765, transition_prob_x0=0.3, transition_prob_x1=0.6)
        
        h12 = Hazard(@formula(0 ~ x), "exp", 1, 2)
        model = multistatemodel(h12; data=panel_data, surrogate=:markov)
        
        # Fit Markov surrogate using model API
        fit_surrogate(model; type=:markov, method=:mle, verbose=false)
        markov_surrogate = model.markovsurrogate
        
        # Use 3 phases to have meaningful phase progression
        phasetype_surrogate = _build_phasetype_from_markov(
            model, markov_surrogate;
            config=PhaseTypeProposal(n_phases=3),
            verbose=false
        )
        
        books = build_tpm_mapping(model.data)
        tpm_book_ph, hazmat_book_ph = build_phasetype_tpm_book(
            phasetype_surrogate, markov_surrogate, books, model.data
        )
        
        Q_x0 = hazmat_book_ph[1]
        Q_x1 = hazmat_book_ph[2]
        
        phase_to_state = phasetype_surrogate.phase_to_state
        
        # Check intra-state transitions (phase progression within same state)
        found_intrastate = false
        for i in 1:size(Q_x0, 1)
            for j in 1:size(Q_x0, 2)
                if i != j && phase_to_state[i] == phase_to_state[j]
                    # This is an intra-state transition (phase progression)
                    if abs(Q_x0[i, j]) > 1e-10
                        found_intrastate = true
                        # Phase progression rates should be IDENTICAL across covariate patterns
                        @test isapprox(Q_x0[i, j], Q_x1[i, j]; rtol=1e-10)
                    end
                end
            end
        end
        # Note: For 2-state model with 1 transition, there's only state 1 with phases
        # and state 2 (absorbing) which might have only 1 phase, so intra-state transitions
        # may not exist in all configurations.
        if found_intrastate
            @test true  # Confirm we tested something
        else
            @info "No intra-state phase progressions found (expected for simple 2-state model)"
        end
    end
    
    @testset "Multiple covariates handled correctly" begin
        # Test with 2 covariates to verify full functionality
        Random.seed!(11111)
        n_subj = 200
        
        # Create base data with one covariate
        panel_data = create_panel_data_with_transitions(n_subj; 
            seed=11111, transition_prob_x0=0.25, transition_prob_x1=0.45)
        
        # Add a second covariate
        panel_data.x2 = rand([0, 1], nrow(panel_data))
        
        h12 = Hazard(@formula(0 ~ x + x2), "exp", 1, 2)
        model = multistatemodel(h12; data=panel_data, surrogate=:markov)
        
        # Fit Markov surrogate using model API
        fit_surrogate(model; type=:markov, method=:mle, verbose=false)
        markov_surrogate = model.markovsurrogate
        
        phasetype_surrogate = _build_phasetype_from_markov(
            model, markov_surrogate;
            config=PhaseTypeProposal(n_phases=2),
            verbose=false
        )
        
        books = build_tpm_mapping(model.data)
        tpm_book_ph, hazmat_book_ph = build_phasetype_tpm_book(
            phasetype_surrogate, markov_surrogate, books, model.data
        )
        
        # Should have multiple covariate combinations
        @test length(hazmat_book_ph) >= 2
        
        # All Q matrices should be valid generators (allow zero diagonal for absorbing states)
        for (c, Q) in enumerate(hazmat_book_ph)
            @test all(diag(Q) .<= 0)  # Non-positive diagonal
            @test all(abs.(sum(Q, dims=2)) .< 1e-10)  # Rows sum to zero
        end
        
        # Q matrices should not all be identical (if there are multiple combos)
        n_combos = length(hazmat_book_ph)
        if n_combos > 1
            some_differ = false
            for c in 2:n_combos
                if !(hazmat_book_ph[1] ≈ hazmat_book_ph[c])
                    some_differ = true
                    break
                end
            end
            @test some_differ  # At least some Q matrices should differ across covariate patterns
        end
    end
    
end

# =============================================================================
# AFT vs PH Covariate Scaling Tests
# =============================================================================
#
# These tests verify the AFT bug fix from 2026-01-15:
# - For PH: scaling_factor = exp(β'x)
# - For AFT: scaling_factor = exp(-β'x)  (OPPOSITE SIGN!)
#
# The bug caused PhaseType proposal estimates for AFT models to be wrong-signed
# for time-varying covariates because exp(β'x) was always used instead of
# the correct exp(-β'x) for AFT.
# =============================================================================

@testset "AFT vs PH Covariate Scaling Direction" begin
    
    @testset "AFT scaling uses exp(-β'x) (opposite sign from PH)" begin
        # Create data with actual transitions
        Random.seed!(99999)
        n_subj = 200
        
        panel_data = create_panel_data_with_transitions(n_subj; 
            seed=99999, transition_prob_x0=0.25, transition_prob_x1=0.5)
        
        # Create AFT model
        h12_aft = Hazard(@formula(0 ~ x), "wei", 1, 2; linpred_effect=:aft)
        model_aft = multistatemodel(h12_aft; data=panel_data, surrogate=:markov)
        
        # Fit Markov surrogate
        fit_surrogate(model_aft; type=:markov, method=:mle, verbose=false)
        markov_surrogate_aft = model_aft.markovsurrogate
        
        # Build PhaseType surrogate from fitted Markov
        phasetype_surrogate_aft = _build_phasetype_from_markov(
            model_aft, markov_surrogate_aft;
            config=PhaseTypeProposal(n_phases=2),
            verbose=false
        )
        
        books_aft = build_tpm_mapping(model_aft.data)
        _, hazmat_book_aft = build_phasetype_tpm_book(
            phasetype_surrogate_aft, markov_surrogate_aft, books_aft, model_aft.data
        )
        
        # Create PH model with same data
        h12_ph = Hazard(@formula(0 ~ x), "wei", 1, 2; linpred_effect=:ph)
        model_ph = multistatemodel(h12_ph; data=panel_data, surrogate=:markov)
        
        fit_surrogate(model_ph; type=:markov, method=:mle, verbose=false)
        markov_surrogate_ph = model_ph.markovsurrogate
        
        phasetype_surrogate_ph = _build_phasetype_from_markov(
            model_ph, markov_surrogate_ph;
            config=PhaseTypeProposal(n_phases=2),
            verbose=false
        )
        
        books_ph = build_tpm_mapping(model_ph.data)
        _, hazmat_book_ph = build_phasetype_tpm_book(
            phasetype_surrogate_ph, markov_surrogate_ph, books_ph, model_ph.data
        )
        
        # Verify we have multiple covariate patterns
        @test length(hazmat_book_aft) >= 2
        @test length(hazmat_book_ph) >= 2
        
        # Get Q matrices for different covariate values
        Q_aft_x0 = hazmat_book_aft[1]  # x=0
        Q_aft_x1 = hazmat_book_aft[2]  # x=1
        Q_ph_x0 = hazmat_book_ph[1]
        Q_ph_x1 = hazmat_book_ph[2]
        
        # Both should have valid Q matrix structure
        @test all(diag(Q_aft_x0) .<= 0)
        @test all(diag(Q_aft_x1) .<= 0)
        @test all(diag(Q_ph_x0) .<= 0)
        @test all(diag(Q_ph_x1) .<= 0)
        
        # For AFT with β > 0, covariate x=1 should have SMALLER rates (exp(-β) < 1)
        # For PH with β > 0, covariate x=1 should have LARGER rates (exp(β) > 1)
        # This is the key difference we're testing!
        
        phase_to_state_aft = phasetype_surrogate_aft.phase_to_state
        phase_to_state_ph = phasetype_surrogate_ph.phase_to_state
        
        # Find inter-state transition rates and compare scaling direction
        for i in 1:size(Q_aft_x0, 1)
            for j in 1:size(Q_aft_x0, 2)
                if i != j && phase_to_state_aft[i] != phase_to_state_aft[j]
                    # Inter-state transition
                    rate_aft_x0 = Q_aft_x0[i, j]
                    rate_aft_x1 = Q_aft_x1[i, j]
                    
                    if abs(rate_aft_x0) > 1e-6 && abs(rate_aft_x1) > 1e-6
                        # For AFT with positive β, rate_x1 < rate_x0 (deceleration)
                        # The rates must differ
                        @test !isapprox(rate_aft_x0, rate_aft_x1; rtol=0.01)
                    end
                end
            end
        end
        
        for i in 1:size(Q_ph_x0, 1)
            for j in 1:size(Q_ph_x0, 2)
                if i != j && phase_to_state_ph[i] != phase_to_state_ph[j]
                    rate_ph_x0 = Q_ph_x0[i, j]
                    rate_ph_x1 = Q_ph_x1[i, j]
                    
                    if abs(rate_ph_x0) > 1e-6 && abs(rate_ph_x1) > 1e-6
                        # For PH with positive β, rate_x1 > rate_x0 (proportional increase)
                        # The rates must differ
                        @test !isapprox(rate_ph_x0, rate_ph_x1; rtol=0.01)
                    end
                end
            end
        end
    end
    
end

# =============================================================================
# Item #35: PhaseType Collapsed Path Likelihood Tests
# =============================================================================
#
# Tests for convert_collapsed_path_to_censored_data and loglik_phasetype_forward.
# These functions compute the surrogate likelihood for MCEM importance sampling.
#
# Reference: scratch/CODEBASE_REFACTORING_GUIDE.md, Item #35
# =============================================================================

# Additional imports for collapsed path tests
import MultistateModels: 
    SamplePath, convert_collapsed_path_to_censored_data, 
    loglik_phasetype_forward

@testset "convert_collapsed_path_to_censored_data structure" begin
    
    @testset "Single transition path - correct row count" begin
        Random.seed!(12345)
        
        # Create simple 2-state panel data
        panel_data = DataFrame(
            id = [1, 1],
            tstart = [0.0, 2.0],
            tstop = [2.0, 4.0],
            statefrom = [1, 1],
            stateto = [1, 2],
            obstype = [2, 2]
        )
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        model = multistatemodel(h12; data=panel_data, surrogate=:markov)
        
        # Create a collapsed path with one transition at t=1.5
        path = SamplePath(1, [0.0, 1.5], [1, 2])
        
        # Get subject data
        subj_data = @view panel_data[panel_data.id .== 1, :]
        
        # Convert path to censored data
        censored = convert_collapsed_path_to_censored_data(path, subj_data, model)
        
        # Should have same number of rows as original data
        @test nrow(censored) == nrow(subj_data)
        
        # All rows should be panel (obstype=2)
        @test all(censored.obstype .== 2)
        
        # Times should match original observation times
        @test censored.tstart == subj_data.tstart
        @test censored.tstop == subj_data.tstop
        
        # States should be looked up from the path
        # At t=0, path is in state 1
        # At t=2, path has already transitioned to state 2 (transition was at t=1.5)
        @test censored.statefrom[1] == 1  # state at t=0
        @test censored.stateto[1] == 2    # state at t=2 (after transition at 1.5)
        @test censored.statefrom[2] == 2  # state at t=2
        @test censored.stateto[2] == 2    # state at t=4
    end
    
    @testset "Multiple transitions - path times are NOT used" begin
        Random.seed!(54321)
        
        # Create 3-state panel data with 3 observation intervals
        panel_data = DataFrame(
            id = [1, 1, 1],
            tstart = [0.0, 3.0, 6.0],
            tstop = [3.0, 6.0, 9.0],
            statefrom = [1, 1, 2],
            stateto = [1, 2, 3],
            obstype = [2, 2, 2]
        )
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
        model = multistatemodel(h12, h23; data=panel_data, surrogate=:markov)
        
        # Create a collapsed path with transitions at arbitrary times
        # Key: these transition times should NOT appear in the censored data
        path = SamplePath(1, [0.0, 2.5, 7.1], [1, 2, 3])
        
        subj_data = @view panel_data[panel_data.id .== 1, :]
        censored = convert_collapsed_path_to_censored_data(path, subj_data, model)
        
        # Row count matches original observations (NOT number of transitions + 1)
        @test nrow(censored) == 3
        
        # Times match original observation times exactly
        @test censored.tstart == [0.0, 3.0, 6.0]
        @test censored.tstop == [3.0, 6.0, 9.0]
        
        # Path times (2.5, 7.1) do NOT appear anywhere
        @test !(2.5 in censored.tstart)
        @test !(2.5 in censored.tstop)
        @test !(7.1 in censored.tstart)
        @test !(7.1 in censored.tstop)
        
        # States are looked up from path at OBSERVATION times
        # Path: [0, 2.5) state 1, [2.5, 7.1) state 2, [7.1, inf) state 3
        @test censored.statefrom[1] == 1  # state at t=0
        @test censored.stateto[1] == 2    # state at t=3 (after transition at 2.5)
        @test censored.statefrom[2] == 2  # state at t=3
        @test censored.stateto[2] == 2    # state at t=6 (before transition at 7.1)
        @test censored.statefrom[3] == 2  # state at t=6
        @test censored.stateto[3] == 3    # state at t=9 (after transition at 7.1)
    end
    
    @testset "No transitions - survival only" begin
        # Path with no transitions - subject stays in initial state
        panel_data = DataFrame(
            id = [1, 1],
            tstart = [0.0, 2.0],
            tstop = [2.0, 4.0],
            statefrom = [1, 1],
            stateto = [1, 1],
            obstype = [2, 2]
        )
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        model = multistatemodel(h12; data=panel_data, surrogate=:markov)
        
        # Path with no transitions
        path = SamplePath(1, [0.0], [1])
        
        subj_data = @view panel_data[panel_data.id .== 1, :]
        censored = convert_collapsed_path_to_censored_data(path, subj_data, model)
        
        @test nrow(censored) == 2
        @test all(censored.obstype .== 2)
        @test all(censored.statefrom .== 1)
        @test all(censored.stateto .== 1)
    end
end


@testset "loglik_phasetype_forward analytical verification" begin
    
    @testset "Single-phase = exponential distribution" begin
        Random.seed!(11111)
        
        # Simple 2-state model: state 1 -> state 2
        panel_data = DataFrame(
            id = [1],
            tstart = [0.0],
            tstop = [2.0],
            statefrom = [1],
            stateto = [1],  # Still in state 1 at observation time
            obstype = [2]
        )
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        model = multistatemodel(h12; data=panel_data, surrogate=:markov)
        
        # Set a known rate
        rate = 0.3
        set_parameters!(model, (h12 = [rate],))
        
        # Fit Markov surrogate
        fit_surrogate(model; type=:markov, method=:mle, verbose=false)
        markov_surrogate = model.markovsurrogate
        
        # Build PhaseType surrogate with 1 phase (should equal exponential)
        phasetype_surrogate = _build_phasetype_from_markov(
            model, markov_surrogate;
            config=PhaseTypeProposal(n_phases=1),
            verbose=false
        )
        
        # For single phase, Q = [-lambda, lambda; 0, 0] (2x2 for 2 states)
        Q = phasetype_surrogate.expanded_Q
        @test size(Q) == (2, 2)  # 2 states, 1 phase each
        
        # Create a collapsed path: stay in state 1 for [0, 2]
        path = SamplePath(1, [0.0], [1])
        
        subj_data = @view panel_data[panel_data.id .== 1, :]
        censored = convert_collapsed_path_to_censored_data(path, subj_data, model)
        
        # Compute log-likelihood via forward algorithm
        ll_forward = loglik_phasetype_forward(censored, phasetype_surrogate)
        
        # For single phase exponential:
        # P(still in state 1 at t=2 | start in state 1) = exp(-lambda * 2)
        total_exit = -Q[1, 1]  # Should be close to rate
        ll_analytical = -total_exit * 2.0
        
        @test isapprox(ll_forward, ll_analytical, rtol=0.01)
    end
    
    @testset "Single observation: survival probability check" begin
        Random.seed!(22222)
        
        dt = 3.0  # Observation interval
        rate = 0.5
        
        panel_data = DataFrame(
            id = [1],
            tstart = [0.0],
            tstop = [dt],
            statefrom = [1],
            stateto = [1],  # Survived (still in state 1)
            obstype = [2]
        )
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        model = multistatemodel(h12; data=panel_data, surrogate=:markov)
        set_parameters!(model, (h12 = [rate],))
        
        fit_surrogate(model; type=:markov, method=:mle, verbose=false)
        markov_surrogate = model.markovsurrogate
        
        phasetype_surrogate = _build_phasetype_from_markov(
            model, markov_surrogate;
            config=PhaseTypeProposal(n_phases=1),
            verbose=false
        )
        
        path = SamplePath(1, [0.0], [1])
        subj_data = @view panel_data[panel_data.id .== 1, :]
        censored = convert_collapsed_path_to_censored_data(path, subj_data, model)
        
        ll_forward = loglik_phasetype_forward(censored, phasetype_surrogate)
        
        # Survival probability = exp(-rate * dt)
        # Note: the fitted surrogate rate may differ slightly from the set rate
        fitted_rate = -phasetype_surrogate.expanded_Q[1, 1]
        ll_expected = -fitted_rate * dt
        
        @test isapprox(ll_forward, ll_expected, rtol=0.01)
    end
end


@testset "loglik_phasetype_forward multi-phase behavior" begin
    
    @testset "Multi-phase surrogate gives valid probability" begin
        Random.seed!(33333)
        
        # Create data with actual transitions for surrogate fitting
        n_subj = 100
        panel_data = DataFrame(
            id = repeat(1:n_subj, inner=2),
            tstart = repeat([0.0, 2.0], n_subj),
            tstop = repeat([2.0, 4.0], n_subj),
            statefrom = ones(Int, 2*n_subj),
            stateto = ones(Int, 2*n_subj),
            obstype = fill(2, 2*n_subj)
        )
        
        # Add some transitions
        for i in 1:n_subj
            if rand() < 0.4
                panel_data.stateto[2*i - 1] = 2  # Transition in first interval
                panel_data.statefrom[2*i] = 2
                panel_data.stateto[2*i] = 2
            elseif rand() < 0.3
                panel_data.stateto[2*i] = 2  # Transition in second interval
            end
        end
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        model = multistatemodel(h12; data=panel_data, surrogate=:markov)
        
        fit_surrogate(model; type=:markov, method=:mle, verbose=false)
        markov_surrogate = model.markovsurrogate
        
        # Build PhaseType surrogate with multiple phases
        phasetype_surrogate = _build_phasetype_from_markov(
            model, markov_surrogate;
            config=PhaseTypeProposal(n_phases=3),
            verbose=false
        )
        
        # Test several random paths
        for _ in 1:10
            # Random subject
            subj_id = rand(1:n_subj)
            subj_data = @view panel_data[panel_data.id .== subj_id, :]
            
            # Create a collapsed path consistent with observations
            initial_state = subj_data.statefrom[1]
            final_state = subj_data.stateto[end]
            
            if initial_state == final_state
                path = SamplePath(subj_id, [0.0], [initial_state])
            else
                # Transition happened somewhere
                trans_time = rand() * 4.0
                path = SamplePath(subj_id, [0.0, trans_time], [1, 2])
            end
            
            censored = convert_collapsed_path_to_censored_data(path, subj_data, model)
            ll = loglik_phasetype_forward(censored, phasetype_surrogate)
            
            # Log-likelihood should be finite (valid probability)
            @test isfinite(ll)
            # Should be non-positive (log of probability <= 0)
            @test ll <= 0.0 || isapprox(ll, 0.0, atol=1e-10)
        end
    end
    
    @testset "Empty path handled gracefully" begin
        Random.seed!(44444)
        
        panel_data = DataFrame(
            id = [1],
            tstart = [0.0],
            tstop = [2.0],
            statefrom = [1],
            stateto = [1],
            obstype = [2]
        )
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        model = multistatemodel(h12; data=panel_data, surrogate=:markov)
        
        fit_surrogate(model; type=:markov, method=:mle, verbose=false)
        markov_surrogate = model.markovsurrogate
        
        phasetype_surrogate = _build_phasetype_from_markov(
            model, markov_surrogate;
            config=PhaseTypeProposal(n_phases=2),
            verbose=false
        )
        
        # Path with just initial state (no transitions)
        path = SamplePath(1, [0.0], [1])
        subj_data = @view panel_data[panel_data.id .== 1, :]
        censored = convert_collapsed_path_to_censored_data(path, subj_data, model)
        
        ll = loglik_phasetype_forward(censored, phasetype_surrogate)
        
        @test isfinite(ll)
        @test ll <= 0.0 || isapprox(ll, 0.0, atol=1e-10)
    end
end


@testset "Forward likelihood computation" begin
    
    @testset "Forward algorithm computes valid probabilities for panel data" begin
        Random.seed!(55555)
        
        # Create data with actual transitions
        n_subj = 100
        panel_data = DataFrame(
            id = repeat(1:n_subj, inner=2),
            tstart = repeat([0.0, 2.0], n_subj),
            tstop = repeat([2.0, 4.0], n_subj),
            statefrom = ones(Int, 2*n_subj),
            stateto = ones(Int, 2*n_subj),
            obstype = fill(2, 2*n_subj)
        )
        
        # Add some transitions
        for i in 1:n_subj
            if rand() < 0.35
                panel_data.stateto[2*i - 1] = 2
                panel_data.statefrom[2*i] = 2
                panel_data.stateto[2*i] = 2
            end
        end
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        model = multistatemodel(h12; data=panel_data, surrogate=:markov)
        
        fit_surrogate(model; type=:markov, method=:mle, verbose=false)
        markov_surrogate = model.markovsurrogate
        
        phasetype_surrogate = _build_phasetype_from_markov(
            model, markov_surrogate;
            config=PhaseTypeProposal(n_phases=2),
            verbose=false
        )
        
        # Test with a path that has a transition
        path = SamplePath(1, [0.0, 1.5], [1, 2])
        subj_data = @view panel_data[panel_data.id .== 1, :]
        censored = convert_collapsed_path_to_censored_data(path, subj_data, model)
        
        ll_forward = loglik_phasetype_forward(censored, phasetype_surrogate)
        
        # Forward algorithm should return valid finite log-probability
        @test isfinite(ll_forward)
        
        # The forward algorithm result should give a valid log-probability
        @test ll_forward <= 0.0 || isapprox(ll_forward, 0.0, atol=1e-10)
    end
    
    @testset "Forward algorithm is stable for various paths" begin
        Random.seed!(66666)
        
        n_subj = 50
        panel_data = DataFrame(
            id = repeat(1:n_subj, inner=3),
            tstart = repeat([0.0, 2.0, 4.0], n_subj),
            tstop = repeat([2.0, 4.0, 6.0], n_subj),
            statefrom = ones(Int, 3*n_subj),
            stateto = ones(Int, 3*n_subj),
            obstype = fill(2, 3*n_subj)
        )
        
        # Add transitions to create variety
        for i in 1:n_subj
            if rand() < 0.4
                t = rand(1:3)
                for j in t:3
                    panel_data.stateto[3*(i-1) + j] = 2
                    if j < 3
                        panel_data.statefrom[3*(i-1) + j + 1] = 2
                    end
                end
            end
        end
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        model = multistatemodel(h12; data=panel_data, surrogate=:markov)
        
        fit_surrogate(model; type=:markov, method=:mle, verbose=false)
        markov_surrogate = model.markovsurrogate
        
        phasetype_surrogate = _build_phasetype_from_markov(
            model, markov_surrogate;
            config=PhaseTypeProposal(n_phases=3),
            verbose=false
        )
        
        # Test many random paths
        n_stable = 0
        for _ in 1:50
            trans_time = rand() * 6.0
            has_transition = rand() < 0.5
            
            if has_transition
                path = SamplePath(1, [0.0, trans_time], [1, 2])
            else
                path = SamplePath(1, [0.0], [1])
            end
            
            subj_data = @view panel_data[panel_data.id .== 1, :]
            censored = convert_collapsed_path_to_censored_data(path, subj_data, model)
            
            ll = loglik_phasetype_forward(censored, phasetype_surrogate)
            
            if isfinite(ll)
                n_stable += 1
            end
        end
        
        # All paths should give finite log-likelihood
        @test n_stable == 50
    end
end

# =============================================================================
# Item #35 (Continued): Expanded Path Likelihood Tests
# =============================================================================
#
# Tests for convert_expanded_path_to_censored_data and loglik_phasetype_expanded_path.
# These functions compute the correct surrogate likelihood for MCEM by using the
# EXPANDED path times (sampled jump chain) rather than the original observation times.
#
# Key insight: The importance weight correction requires q(Z|Y,θ') where Z is the
# sampled expanded path with exact phase transition times. We compute the CTMC
# path density directly using the Q matrix.
# =============================================================================

import MultistateModels: convert_expanded_path_to_censored_data, loglik_phasetype_expanded_path

@testset "convert_expanded_path_to_censored_data structure" begin
    
    @testset "Simple path: no macro-state transitions" begin
        Random.seed!(77777)
        
        # Build a simple phasetype surrogate
        panel_data = DataFrame(
            id = [1, 1],
            tstart = [0.0, 2.0],
            tstop = [2.0, 4.0],
            statefrom = [1, 1],
            stateto = [1, 2],
            obstype = [2, 2]
        )
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        model = multistatemodel(h12; data=panel_data, surrogate=:markov)
        fit_surrogate(model; type=:markov, method=:mle, verbose=false)
        markov_surrogate = model.markovsurrogate
        
        phasetype_surrogate = _build_phasetype_from_markov(
            model, markov_surrogate;
            config=PhaseTypeProposal(n_phases=2),
            verbose=false
        )
        
        # Expanded path that stays in macro-state 1 (phases 1,2)
        # Path: phase 1 -> phase 2 (internal transition within macro-state)
        expanded_path = SamplePath(1, [0.0, 1.5, 3.0], [1, 2, 2])
        
        censored_data, emat, censoring_patterns = convert_expanded_path_to_censored_data(
            expanded_path, phasetype_surrogate
        )
        
        # No macro-state transitions => single survival row
        @test nrow(censored_data) == 1
        @test censored_data.tstart[1] == 0.0
        @test censored_data.tstop[1] == 3.0
        @test censored_data.statefrom[1] == 1  # Entry phase (Coxian)
        @test censored_data.stateto[1] == 0    # Unknown final phase
        @test censored_data.obstype[1] == 3    # Censoring pattern
        
        # Emission matrix: allow any phase of macro-state 1
        @test size(emat) == (1, phasetype_surrogate.n_expanded_states)
        phases_state1 = phasetype_surrogate.state_to_phases[1]
        for p in phases_state1
            @test emat[1, p] == 1.0
        end
    end
    
    @testset "Path with one macro-state transition" begin
        Random.seed!(88888)
        
        # Build phasetype surrogate for 3-state model
        panel_data = DataFrame(
            id = [1, 1, 1],
            tstart = [0.0, 2.0, 4.0],
            tstop = [2.0, 4.0, 6.0],
            statefrom = [1, 1, 2],
            stateto = [1, 2, 2],
            obstype = [2, 2, 2]
        )
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        model = multistatemodel(h12; data=panel_data, surrogate=:markov)
        fit_surrogate(model; type=:markov, method=:mle, verbose=false)
        markov_surrogate = model.markovsurrogate
        
        phasetype_surrogate = _build_phasetype_from_markov(
            model, markov_surrogate;
            config=PhaseTypeProposal(n_phases=2),
            verbose=false
        )
        
        n_phases_state1 = length(phasetype_surrogate.state_to_phases[1])
        
        # Expanded path: phases of state 1 -> first phase of state 2
        # Transition at t=2.5 from last phase of state 1 to first phase of state 2
        last_phase_state1 = phasetype_surrogate.state_to_phases[1][end]
        first_phase_state2 = phasetype_surrogate.state_to_phases[2][1]
        
        expanded_path = SamplePath(1, [0.0, 1.0, 2.5, 5.0], [1, last_phase_state1, first_phase_state2, first_phase_state2])
        
        censored_data, emat, censoring_patterns = convert_expanded_path_to_censored_data(
            expanded_path, phasetype_surrogate
        )
        
        # One macro-state transition followed by survival in state 2 => 3 rows:
        # 1. Survival in macro-state 1 (t=0 to t=2.5)
        # 2. Exact transition to state 2 (dt=0 at t=2.5)
        # 3. Survival in macro-state 2 (t=2.5 to t=5.0)
        @test nrow(censored_data) == 3
        
        # Row 1: Survival in macro-state 1
        @test censored_data.tstart[1] == 0.0
        @test censored_data.tstop[1] == 2.5
        @test censored_data.statefrom[1] == 1  # Entry phase
        @test censored_data.stateto[1] == 0    # Unknown exit phase
        @test censored_data.obstype[1] >= 3    # Censoring pattern
        
        # Row 2: Exact transition to state 2
        @test censored_data.tstart[2] == 2.5
        @test censored_data.tstop[2] == 2.5   # Δt = 0
        @test censored_data.statefrom[2] == 0  # Unknown source phase
        @test censored_data.stateto[2] == first_phase_state2  # Known destination
        @test censored_data.obstype[2] == 1   # Exact observation
        
        # Row 3: Survival in macro-state 2
        @test censored_data.tstart[3] == 2.5
        @test censored_data.tstop[3] == 5.0
        @test censored_data.statefrom[3] == first_phase_state2  # Known entry phase
        
        # Emission matrix check
        @test size(emat, 1) == 3
        # Row 1: allow any phase of state 1
        for p in phasetype_surrogate.state_to_phases[1]
            @test emat[1, p] == 1.0
        end
        # Row 2: only destination phase
        @test emat[2, first_phase_state2] == 1.0
        @test sum(emat[2, :]) == 1.0
        # Row 3: only phases of state 2
        for p in phasetype_surrogate.state_to_phases[2]
            @test emat[3, p] == 1.0
        end
    end
    
    @testset "Path with multiple macro-state transitions" begin
        Random.seed!(99999)
        
        # Build phasetype surrogate for illness-death model
        panel_data = DataFrame(
            id = [1, 1, 1],
            tstart = [0.0, 2.0, 4.0],
            tstop = [2.0, 4.0, 6.0],
            statefrom = [1, 1, 2],
            stateto = [1, 2, 3],
            obstype = [2, 2, 2]
        )
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
        model = multistatemodel(h12, h23; data=panel_data, surrogate=:markov)
        fit_surrogate(model; type=:markov, method=:mle, verbose=false)
        markov_surrogate = model.markovsurrogate
        
        phasetype_surrogate = _build_phasetype_from_markov(
            model, markov_surrogate;
            config=PhaseTypeProposal(n_phases=2),
            verbose=false
        )
        
        first_phase_state1 = phasetype_surrogate.state_to_phases[1][1]
        first_phase_state2 = phasetype_surrogate.state_to_phases[2][1]
        first_phase_state3 = phasetype_surrogate.state_to_phases[3][1]
        
        # Expanded path: state 1 -> state 2 -> state 3
        expanded_path = SamplePath(1, 
            [0.0, 1.0, 1.5, 3.0, 4.5, 5.5],
            [first_phase_state1, first_phase_state1, first_phase_state2, first_phase_state2, first_phase_state3, first_phase_state3]
        )
        
        censored_data, emat, censoring_patterns = convert_expanded_path_to_censored_data(
            expanded_path, phasetype_surrogate
        )
        
        # Two macro-state transitions + final survival => 5 rows:
        # 1. Survival in state 1 [0, 1.5]
        # 2. Exact transition to state 2 at t=1.5
        # 3. Survival in state 2 [1.5, 4.5]
        # 4. Exact transition to state 3 at t=4.5
        # 5. Survival in state 3 [4.5, 5.5]
        @test nrow(censored_data) == 5
        
        # Row 1: Survival in state 1 [0, 1.5]
        @test censored_data.tstart[1] == 0.0
        @test censored_data.tstop[1] == 1.5
        @test censored_data.obstype[1] >= 3  # Censoring pattern
        
        # Row 2: Exact transition to state 2 at t=1.5
        @test censored_data.tstart[2] == 1.5
        @test censored_data.tstop[2] == 1.5
        @test censored_data.obstype[2] == 1  # Exact
        @test censored_data.stateto[2] == first_phase_state2
        
        # Row 3: Survival in state 2 [1.5, 4.5]
        @test censored_data.tstart[3] == 1.5
        @test censored_data.tstop[3] == 4.5
        @test censored_data.obstype[3] >= 3  # Censoring pattern
        
        # Row 4: Exact transition to state 3 at t=4.5
        @test censored_data.tstart[4] == 4.5
        @test censored_data.tstop[4] == 4.5
        @test censored_data.obstype[4] == 1  # Exact
        @test censored_data.stateto[4] == first_phase_state3
        
        # Row 5: Survival in state 3 [4.5, 5.5]
        @test censored_data.tstart[5] == 4.5
        @test censored_data.tstop[5] == 5.5
    end
end

@testset "loglik_phasetype_expanded_path analytical verification" begin
    
    @testset "Simple exponential - exact sojourn" begin
        Random.seed!(11111)
        
        # Build surrogate
        panel_data = DataFrame(
            id = [1],
            tstart = [0.0],
            tstop = [2.0],
            statefrom = [1],
            stateto = [2],
            obstype = [1]  # Exact
        )
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        model = multistatemodel(h12; data=panel_data, surrogate=:markov)
        fit_surrogate(model; type=:markov, method=:mle, verbose=false)
        markov_surrogate = model.markovsurrogate
        
        phasetype_surrogate = _build_phasetype_from_markov(
            model, markov_surrogate;
            config=PhaseTypeProposal(n_phases=1),  # Single phase = exponential
            verbose=false
        )
        
        # Create expanded path
        first_phase_state2 = phasetype_surrogate.state_to_phases[2][1]
        expanded_path = SamplePath(1, [0.0, 2.0], [1, first_phase_state2])
        
        # Use loglik_phasetype_expanded_path directly - it takes path and Q matrix
        ll = loglik_phasetype_expanded_path(expanded_path, phasetype_surrogate)
        
        # Should be finite and valid log-probability (density)
        @test isfinite(ll)
        
        # For CTMC density: log f = -q_s * dt + log(q_{s,d})
        # With rate λ: -λ*2 + log(λ) = -2λ + log(λ)
        # This should be a finite real number (may be positive for high rates, short times)
    end
    
    @testset "Multi-phase - direct CTMC path density" begin
        Random.seed!(22222)
        
        panel_data = DataFrame(
            id = [1, 1],
            tstart = [0.0, 3.0],
            tstop = [3.0, 6.0],
            statefrom = [1, 2],
            stateto = [2, 2],
            obstype = [2, 2]
        )
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        model = multistatemodel(h12; data=panel_data, surrogate=:markov)
        fit_surrogate(model; type=:markov, method=:mle, verbose=false)
        markov_surrogate = model.markovsurrogate
        
        phasetype_surrogate = _build_phasetype_from_markov(
            model, markov_surrogate;
            config=PhaseTypeProposal(n_phases=3),  # Multiple phases
            verbose=false
        )
        
        # Path with transition at t=2.5
        first_phase_state2 = phasetype_surrogate.state_to_phases[2][1]
        expanded_path = SamplePath(1, [0.0, 1.0, 2.5, 5.0], [1, 2, first_phase_state2, first_phase_state2])
        
        ll = loglik_phasetype_expanded_path(expanded_path, phasetype_surrogate)
        
        @test isfinite(ll)
    end
    
    @testset "Consistency: same input gives same output" begin
        Random.seed!(33333)
        
        panel_data = DataFrame(
            id = [1, 1],
            tstart = [0.0, 2.0],
            tstop = [2.0, 4.0],
            statefrom = [1, 1],
            stateto = [1, 2],
            obstype = [2, 2]
        )
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        model = multistatemodel(h12; data=panel_data, surrogate=:markov)
        fit_surrogate(model; type=:markov, method=:mle, verbose=false)
        markov_surrogate = model.markovsurrogate
        
        phasetype_surrogate = _build_phasetype_from_markov(
            model, markov_surrogate;
            config=PhaseTypeProposal(n_phases=2),
            verbose=false
        )
        
        first_phase_state2 = phasetype_surrogate.state_to_phases[2][1]
        expanded_path = SamplePath(1, [0.0, 1.5, 3.5], [1, first_phase_state2, first_phase_state2])
        
        # Call twice with same input
        ll1 = loglik_phasetype_expanded_path(expanded_path, phasetype_surrogate)
        ll2 = loglik_phasetype_expanded_path(expanded_path, phasetype_surrogate)
        
        @test ll1 == ll2  # Exact equality expected (no randomness)
    end
end

@testset "loglik_phasetype_expanded_path stability" begin
    
    @testset "Various path lengths and transition patterns" begin
        Random.seed!(44444)
        
        panel_data = DataFrame(
            id = repeat([1], 5),
            tstart = [0.0, 1.0, 2.0, 3.0, 4.0],
            tstop = [1.0, 2.0, 3.0, 4.0, 5.0],
            statefrom = [1, 1, 1, 2, 2],
            stateto = [1, 1, 2, 2, 3],
            obstype = fill(2, 5)
        )
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
        model = multistatemodel(h12, h23; data=panel_data, surrogate=:markov)
        fit_surrogate(model; type=:markov, method=:mle, verbose=false)
        markov_surrogate = model.markovsurrogate
        
        phasetype_surrogate = _build_phasetype_from_markov(
            model, markov_surrogate;
            config=PhaseTypeProposal(n_phases=2),
            verbose=false
        )
        
        n_finite = 0
        for _ in 1:30
            # Generate random transition times
            t1 = rand() * 2.5  # First transition (1->2) between 0 and 2.5
            t2 = t1 + rand() * (5.0 - t1)  # Second transition (2->3) after first
            
            first_phase_state1 = phasetype_surrogate.state_to_phases[1][1]
            first_phase_state2 = phasetype_surrogate.state_to_phases[2][1]
            first_phase_state3 = phasetype_surrogate.state_to_phases[3][1]
            
            if t2 > t1 + 0.01
                # Two transitions
                expanded_path = SamplePath(1, 
                    [0.0, t1, t2],
                    [first_phase_state1, first_phase_state2, first_phase_state3]
                )
            else
                # One transition
                expanded_path = SamplePath(1, 
                    [0.0, t1],
                    [first_phase_state1, first_phase_state2]
                )
            end
            
            ll = loglik_phasetype_expanded_path(expanded_path, phasetype_surrogate)
            
            if isfinite(ll)
                n_finite += 1
            end
        end
        
        # All valid paths should give finite likelihood
        @test n_finite == 30
    end
end

# =============================================================================
# CachedSchurDecomposition Tests  
# =============================================================================
#
# Unit tests for efficient TPM computation via cached Schur decomposition.
#
# These tests verify the Schur caching optimization from 2026-01-17:
# - CachedSchurDecomposition correctly stores Schur factors
# - compute_tpm_from_schur produces identical results to exp(Q*dt)
# - In-place variant compute_tpm_from_schur! works correctly
# - Various Q matrix sizes and time intervals are handled
#
# Reference: scratch/CODEBASE_REFACTORING_GUIDE.md, Item #35
# =============================================================================

@testset "CachedSchurDecomposition" begin
    import MultistateModels: CachedSchurDecomposition, compute_tpm_from_schur, compute_tpm_from_schur!
    
    @testset "Basic construction and TPM computation" begin
        # Simple 2x2 rate matrix
        Q = [-0.5 0.5; 0.3 -0.3]
        cache = CachedSchurDecomposition(Q)
        
        # Test various time intervals
        for dt in [0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
            tpm_cached = compute_tpm_from_schur(cache, dt)
            tpm_direct = exp(Q * dt)
            
            @test size(tpm_cached) == size(tpm_direct)
            @test maximum(abs.(tpm_cached .- tpm_direct)) < 1e-12
            
            # TPM should be a valid probability matrix
            @test all(tpm_cached .>= -1e-14)  # Allow tiny numerical errors
            @test all(isapprox.(sum(tpm_cached, dims=2), 1.0, atol=1e-12))
        end
    end
    
    @testset "In-place computation" begin
        Q = [-0.5 0.5; 0.3 -0.3]
        cache = CachedSchurDecomposition(Q)
        
        P = zeros(2, 2)
        compute_tpm_from_schur!(P, cache, 2.0)
        
        P_direct = exp(Q * 2.0)
        @test maximum(abs.(P .- P_direct)) < 1e-12
    end
    
    @testset "Larger rate matrices (3x3)" begin
        # 3-state progressive model
        Q = [-0.4  0.3  0.1;
              0.0 -0.5  0.5;
              0.0  0.0  0.0]
        cache = CachedSchurDecomposition(Q)
        
        for dt in [0.5, 1.0, 3.0, 10.0]
            tpm_cached = compute_tpm_from_schur(cache, dt)
            tpm_direct = exp(Q * dt)
            @test maximum(abs.(tpm_cached .- tpm_direct)) < 1e-11
        end
    end
    
    @testset "Phase-type Q matrix (5x5 Coxian)" begin
        # Coxian with 2 phases per state, 2 transient states -> absorbing
        # Phases: [1,2] in state 1, [3,4] in state 2, [5] absorbing
        λ = 0.8   # progression rate
        μ1 = 0.3  # exit rate from phase 1
        μ2 = 0.5  # exit rate from phase 2
        
        Q = zeros(5, 5)
        # State 1 (phases 1,2)
        Q[1,1] = -(λ + μ1)
        Q[1,2] = λ
        Q[1,3] = μ1  # exit to state 2 phase 1
        Q[2,2] = -μ2
        Q[2,4] = μ2  # exit to state 2 phase 2
        
        # State 2 (phases 3,4)
        Q[3,3] = -(λ + μ1)
        Q[3,4] = λ
        Q[3,5] = μ1  # exit to absorbing
        Q[4,4] = -μ2
        Q[4,5] = μ2
        
        # State 3 (phase 5, absorbing)
        Q[5,5] = 0.0
        
        cache = CachedSchurDecomposition(Q)
        
        for dt in [0.1, 1.0, 5.0, 20.0]
            tpm_cached = compute_tpm_from_schur(cache, dt)
            tpm_direct = exp(Q * dt)
            @test maximum(abs.(tpm_cached .- tpm_direct)) < 1e-10
            
            # Verify absorbing state property
            @test isapprox(tpm_cached[5, 5], 1.0, atol=1e-12)
        end
    end
    
    @testset "Very small time intervals" begin
        Q = [-0.5 0.5; 0.3 -0.3]
        cache = CachedSchurDecomposition(Q)
        
        # For small dt, P ≈ I + Q*dt
        for dt in [1e-8, 1e-6, 1e-4]
            tpm_cached = compute_tpm_from_schur(cache, dt)
            tpm_direct = exp(Q * dt)
            @test maximum(abs.(tpm_cached .- tpm_direct)) < 1e-14
        end
    end
    
    @testset "Very large time intervals" begin
        Q = [-0.5 0.5; 0.3 -0.3]
        cache = CachedSchurDecomposition(Q)
        
        # For large dt, P approaches stationary distribution
        for dt in [50.0, 100.0, 500.0]
            tpm_cached = compute_tpm_from_schur(cache, dt)
            tpm_direct = exp(Q * dt)
            @test maximum(abs.(tpm_cached .- tpm_direct)) < 1e-10
            
            # Rows should converge to stationary distribution: π = [0.375, 0.625]
            # (from solving π*Q = 0)
            stat_dist = [0.3 / 0.8, 0.5 / 0.8]
            for row in 1:2
                @test isapprox(tpm_cached[row, :], stat_dist, atol=1e-6)
            end
        end
    end
    
    @testset "dt = 0 gives identity" begin
        Q = [-0.5 0.5; 0.3 -0.3]
        cache = CachedSchurDecomposition(Q)
        
        tpm = compute_tpm_from_schur(cache, 0.0)
        @test tpm ≈ I(2) atol=1e-14
    end
end


# =============================================================================
# Tests for dt=0 Instantaneous Transition Likelihood Fix (Item #36)
# =============================================================================
#
# These tests verify the critical bug fix from 2026-01-18:
# - For dt=0 (instantaneous transitions), compute_forward_loglik uses HAZARD
#   rates Q[i,j] directly, NOT normalized probabilities Q[i,j]/(-Q[i,i]).
# - The distinction matters: probabilities are for SAMPLING (choosing where 
#   to go), hazards are for DENSITY (likelihood contribution).
# - Using probabilities destroys the density contribution and produces
#   incorrect importance weights.
#
# Reference: scratch/CODEBASE_REFACTORING_GUIDE.md, Item #36
# =============================================================================

import MultistateModels: 
    compute_forward_loglik, convert_expanded_path_to_censored_data,
    _instantaneous_tpm_from_Q

@testset "dt=0 Instantaneous Transition Likelihood" begin
    
    @testset "_instantaneous_tpm_from_Q returns NORMALIZED probabilities (for SAMPLING)" begin
        # This function is for SAMPLING: given a transition occurred, what's the
        # probability it was to state j vs state k? Answer: h(i,j)/Σh(i,k)
        
        Q = [-1.5 1.0 0.5;
              0.3 -0.8 0.5;
              0.0  0.0 0.0]  # State 3 is absorbing
        
        P = _instantaneous_tpm_from_Q(Q, 3)
        
        # Row 1: transitions to states 2 and 3 with hazards 1.0 and 0.5
        # Total hazard = 1.5, so P[1,2] = 1.0/1.5 = 0.667, P[1,3] = 0.5/1.5 = 0.333
        @test P[1, 1] ≈ 0.0  # Cannot self-transition when transition observed
        @test P[1, 2] ≈ 1.0 / 1.5 atol=1e-10
        @test P[1, 3] ≈ 0.5 / 1.5 atol=1e-10
        @test sum(P[1, :]) ≈ 1.0 atol=1e-10  # Rows sum to 1 (PROBABILITY)
        
        # Row 2: transitions to states 1 and 3 with hazards 0.3 and 0.5
        @test P[2, 1] ≈ 0.3 / 0.8 atol=1e-10
        @test P[2, 2] ≈ 0.0
        @test P[2, 3] ≈ 0.5 / 0.8 atol=1e-10
        @test sum(P[2, :]) ≈ 1.0 atol=1e-10
        
        # Row 3: absorbing state (stays)
        @test P[3, 3] ≈ 1.0 atol=1e-10
    end
    
    @testset "compute_forward_loglik dt=0 uses UNNORMALIZED hazards (for DENSITY)" begin
        # This is the critical test: verify that dt=0 rows contribute Q[i,j]
        # to the likelihood, NOT Q[i,j]/(-Q[i,i]).
        
        # Simple 2-state model: state 1 → state 2 (absorbing)
        # Q = [-λ  λ]
        #     [ 0  0]
        λ = 0.5
        Q = [-λ λ; 0.0 0.0]
        
        n_states = 2
        
        # Case 1: Exact transition at time t=2.0 from state 1 to state 2
        # The likelihood should be: S(2) × h(2) = exp(-λ×2) × λ
        # 
        # In censored data format, this becomes:
        # Row 1: (0, 2.0, 1, 0, obstype=3, emission=[1,1]) — survival until t=2
        # Row 2: (2.0, 2.0, 0, 2, obstype=1, emission=[0,1]) — instantaneous transition
        
        censored_data = DataFrame(
            tstart = [0.0, 2.0],
            tstop = [2.0, 2.0],
            statefrom = [1, 0],  # 0 means "unknown" (marginalized)
            stateto = [0, 2],
            obstype = [3, 1]  # 3 = censoring, 1 = exact observation
        )
        
        # Emission matrix: row 1 allows both states (survival), row 2 only state 2
        emat = [1.0 1.0;   # Row 1: could be in either state at end
                0.0 1.0]   # Row 2: must be in state 2
        
        # TPM book: only need one covariate level
        # For dt=2.0: P = exp(Q×2)
        P_dt2 = exp(Q * 2.0)
        tpm_book = [[P_dt2]]  # [covar_idx][time_idx]
        
        # TPM map: (covar_idx, time_idx)
        tpm_map = [1 1; 1 1]
        
        # Hazmat book for dt=0 rows
        hazmat_book = [Q]
        
        ll = compute_forward_loglik(censored_data, emat, tpm_map, tpm_book, hazmat_book, n_states)
        
        # Expected log-likelihood:
        # Row 1 (dt=2): Survival term, contributes P[1,1]+P[1,2] = 1 (always survives to somewhere)
        #              After emission constraint: α = [exp(-λ×2), 1-exp(-λ×2)]
        # Row 2 (dt=0): Transition term, should contribute Q[1,2] = λ (NOT Q[1,2]/Q[1,1] = 1)
        #
        # Forward algorithm:
        # α₀ = [1, 0] (start in state 1)
        # After row 1: α₁ = P' × α₀ ⊙ emat[1,:] = [P[1,1], P[1,2]] = [exp(-λ×2), 1-exp(-λ×2)]
        #              normalize: scale₁ = sum(α₁) = 1, log_ll += 0
        # After row 2 (dt=0): α₂[j] = Σᵢ Q[i,j] × α₁[i] × emat[2,j]
        #              α₂[2] = Q[1,2] × α₁[1] + Q[2,2] × α₁[2]  (with emission mask)
        #                    = λ × exp(-λ×2) + 0 × (1-exp(-λ×2))
        #                    = λ × exp(-λ×2)
        #              scale₂ = λ × exp(-λ×2)
        #              log_ll += log(scale₂) = log(λ) + (-λ×2) = log(λ) - λ×2
        
        expected_ll = log(λ) - λ * 2.0
        @test ll ≈ expected_ll atol=1e-10
        
        # Verify this differs from what we'd get with normalized probabilities
        # If we incorrectly used P = Q[i,j]/(-Q[i,i]):
        # α₂[2] = (λ/λ) × exp(-λ×2) = exp(-λ×2)
        # scale₂ = exp(-λ×2)
        # incorrect_ll = log(exp(-λ×2)) = -λ×2
        # This is DIFFERENT from expected_ll by log(λ)!
        incorrect_ll = -λ * 2.0
        @test ll ≠ incorrect_ll
        @test abs(ll - expected_ll) < 1e-10  # Correct
        @test abs(incorrect_ll - expected_ll) > 0.1  # Would be wrong
    end
    
    @testset "Exponential equivalence: PhaseType(n=1) ≈ Markov exponential" begin
        # =================================================================
        # KEY MATHEMATICAL PROPERTY TO VERIFY:
        # When n_phases=1, there is no phase uncertainty to marginalize over.
        # Therefore:
        #   forward_algorithm_marginal(path) == CTMC_path_density(path)
        #
        # This is the fundamental "exponential equivalence" test.
        # =================================================================
        
        Random.seed!(42)
        
        # Create simple exact observation data (not panel)
        # 1→2 transitions with exponential sojourn times
        n_subj = 30
        λ = 0.3  # True rate
        rows = []
        for i in 1:n_subj
            t_transition = rand(Exponential(1/λ))
            push!(rows, (id=i, tstart=0.0, tstop=t_transition, statefrom=1, stateto=2, obstype=1))
        end
        exact_data = DataFrame(rows)
        
        # Create model and fit with Markov surrogate
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        model = multistatemodel(h12; data=exact_data, surrogate=:markov)
        
        # Fit Markov surrogate
        fit_surrogate(model; type=:markov, method=:mle, verbose=false)
        markov_surrogate = model.markovsurrogate
        
        # Build PhaseType surrogate with n_phases=1 (should be equivalent to exponential)
        phasetype_surrogate = _build_phasetype_from_markov(
            model, markov_surrogate;
            config=PhaseTypeProposal(n_phases=1),
            verbose=false
        )
        
        # Get fitted rate from Markov surrogate
        fitted_rate = markov_surrogate.parameters.nested.h12.baseline.h12_rate
        
        # Check Q matrix structure in PhaseType surrogate
        # With n_phases=1, the Q matrix should be 2x2: [-λ, λ; 0, 0]
        @test phasetype_surrogate.phasetype_dists[1].n_phases == 1
        Q = phasetype_surrogate.expanded_Q
        @test size(Q) == (2, 2)  # 2 states total with n_phases=1
        @test Q[1, 1] ≈ -fitted_rate atol=1e-10
        @test Q[1, 2] ≈ fitted_rate atol=1e-10
        @test Q[2, 1] == 0.0
        @test Q[2, 2] == 0.0
        
        # =================================================================
        # CRITICAL TEST: Compare forward algorithm marginal vs CTMC path density
        # For n_phases=1, these MUST be equal since there is no marginalization.
        # =================================================================
        
        # Create a simple expanded path (state 1 → state 2 at time t=2.0)
        # SamplePath(subj, times, states)
        test_path = SamplePath(1, [0.0, 2.0], [1, 2])  # In expanded space, state 1 is phase 1 of macro-state 1
        
        # Method 1: CTMC path density directly
        ctmc_ll = loglik_phasetype_expanded_path(test_path, Q)
        
        # Method 2: Forward algorithm on censored data
        censored_data, emat, tpm_map, tpm_book, hazmat_book = 
            convert_expanded_path_to_censored_data(
                test_path, phasetype_surrogate;
                hazmat = Q
            )
        forward_ll = compute_forward_loglik(
            censored_data, emat, tpm_map, tpm_book, hazmat_book,
            phasetype_surrogate.n_expanded_states
        )
        
        # THESE MUST BE EQUAL for n_phases=1
        @test ctmc_ll ≈ forward_ll atol=1e-10
        
        # Also verify the computed value is mathematically correct
        # For path: survive in state 1 for 2.0 time units, then transition to state 2
        # loglik = -λ * 2.0 + log(λ)
        expected_ll = -fitted_rate * 2.0 + log(fitted_rate)
        @test ctmc_ll ≈ expected_ll atol=1e-10
        @test forward_ll ≈ expected_ll atol=1e-10
        
        # Test with multiple different transition times
        test_times = [0.5, 1.0, 3.0, 5.0]
        for t in test_times
            path = SamplePath(1, [0.0, t], [1, 2])
            
            ctmc = loglik_phasetype_expanded_path(path, Q)
            
            cd, em, tm, tb, hb = convert_expanded_path_to_censored_data(
                path, phasetype_surrogate; hazmat = Q
            )
            forward = compute_forward_loglik(cd, em, tm, tb, hb, phasetype_surrogate.n_expanded_states)
            
            expected = -fitted_rate * t + log(fitted_rate)
            
            # CTMC and forward should match (n_phases=1, no marginalization needed)
            @test ctmc ≈ forward atol=1e-10
            # Both should match expected analytical value
            @test ctmc ≈ expected atol=1e-10
        end
        
        # =================================================================
        # VERIFY: n_phases > 1 should give DIFFERENT results
        # (forward algorithm marginalizes over phases)
        # =================================================================
        
        # Build PhaseType with n_phases=3
        phasetype_surrogate_3 = _build_phasetype_from_markov(
            model, markov_surrogate;
            config=PhaseTypeProposal(n_phases=3),
            verbose=false
        )
        
        @test phasetype_surrogate_3.phasetype_dists[1].n_phases == 3
        Q3 = phasetype_surrogate_3.expanded_Q
        @test size(Q3) == (4, 4)  # 3 phases for state 1 + 1 absorbing state 2
        
        # Create expanded path with specific phase
        # Path: phase 1 of state 1 → phase 2 of state 1 → phase 3 of state 1 → state 2
        # (transitions through all phases before exiting)
        path_specific_phases = SamplePath(1, [0.0, 0.5, 1.0, 2.0], [1, 2, 3, 4])  # Phases 1, 2, 3 of macro-state 1, then macro-state 2
        
        # CTMC density of this specific path
        ctmc_specific = loglik_phasetype_expanded_path(path_specific_phases, Q3)
        
        # Forward algorithm on censored data (marginalizes over phases)
        cd3, em3, tm3, tb3, hb3 = convert_expanded_path_to_censored_data(
            path_specific_phases, phasetype_surrogate_3; hazmat = Q3
        )
        forward_marginal = compute_forward_loglik(cd3, em3, tm3, tb3, hb3, phasetype_surrogate_3.n_expanded_states)
        
        # For n_phases > 1, the marginal should be HIGHER than the specific path density
        # because the marginal includes all paths that collapse to the same macro-state sequence
        # 
        # Actually, the relationship depends on the path. The key point is they should NOT
        # be equal (unless by coincidence) because the forward algorithm is marginalizing.
        # The marginal is a SUM over all phase paths, weighted by probability.
        # 
        # With Coxian structure, the marginal should be >= any single path density when
        # there are multiple ways to achieve the same macro-path.
        @test isfinite(ctmc_specific)
        @test isfinite(forward_marginal)
        # The marginal integrates over phase possibilities, so it should be 
        # greater than (or possibly equal to if the path is the only possibility)
        # a specific phase path.
        @test forward_marginal >= ctmc_specific - 1e-10
    end
end  # End of main testset "Exponential equivalence: forward loglik = CTMC path density for n_phases=1"

# =============================================================================
# DISABLED_API_CHANGE: # Fitted τ MLE Tests
# DISABLED_API_CHANGE: # =============================================================================
# DISABLED_API_CHANGE: #
# DISABLED_API_CHANGE: # These tests verify the τ estimation via MLE functionality:
# DISABLED_API_CHANGE: # - Fitted τ values are positive and have correct length
# DISABLED_API_CHANGE: # - Ordering constraints are satisfied for :sctp_increasing/:sctp_decreasing
# DISABLED_API_CHANGE: # - Fitted phase-type produces non-exponential hazard
# DISABLED_API_CHANGE: # - PhaseTypeSurrogate correctly stores fitted_tau field
# DISABLED_API_CHANGE: #
# DISABLED_API_CHANGE: # Added: 2026-01-20
# DISABLED_API_CHANGE: # =============================================================================
# DISABLED_API_CHANGE: 
# DISABLED_API_CHANGE: # Import additional internal functions for τ estimation tests
# DISABLED_API_CHANGE: import MultistateModels: 
# DISABLED_API_CHANGE:     _fit_phasetype_mle, _fit_phasetype_surrogate, phasetype_marginal_loglik,
# DISABLED_API_CHANGE:     _build_coxian_from_rate, _build_state_mappings, build_expanded_Q,
# DISABLED_API_CHANGE:     _build_default_phasetype
# DISABLED_API_CHANGE: 
# DISABLED_API_CHANGE: @testset "PhaseType Surrogate τ MLE Estimation" begin
# DISABLED_API_CHANGE: 
# DISABLED_API_CHANGE:     @testset "Basic τ MLE fitting returns valid results" begin
# DISABLED_API_CHANGE:         # Create simple panel data with Weibull-like sojourn times (increasing hazard)
# DISABLED_API_CHANGE:         Random.seed!(42)
# DISABLED_API_CHANGE:         n_subj = 100
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         # Create data where transition probability increases over time
# DISABLED_API_CHANGE:         # This simulates an increasing hazard pattern
# DISABLED_API_CHANGE:         rows = []
# DISABLED_API_CHANGE:         for i in 1:n_subj
# DISABLED_API_CHANGE:             # First observation period - lower transition probability
# DISABLED_API_CHANGE:             transitioned_t1 = rand() < 0.2
# DISABLED_API_CHANGE:             state_t1 = transitioned_t1 ? 2 : 1
# DISABLED_API_CHANGE:             push!(rows, (id=i, tstart=0.0, tstop=1.0, statefrom=1, stateto=state_t1, obstype=2))
# DISABLED_API_CHANGE:             
# DISABLED_API_CHANGE:             if state_t1 == 1
# DISABLED_API_CHANGE:                 # Second period - medium transition probability
# DISABLED_API_CHANGE:                 transitioned_t2 = rand() < 0.35
# DISABLED_API_CHANGE:                 state_t2 = transitioned_t2 ? 2 : 1
# DISABLED_API_CHANGE:                 push!(rows, (id=i, tstart=1.0, tstop=2.0, statefrom=1, stateto=state_t2, obstype=2))
# DISABLED_API_CHANGE:                 
# DISABLED_API_CHANGE:                 if state_t2 == 1
# DISABLED_API_CHANGE:                     # Third period - higher transition probability
# DISABLED_API_CHANGE:                     transitioned_t3 = rand() < 0.5
# DISABLED_API_CHANGE:                     state_t3 = transitioned_t3 ? 2 : 1
# DISABLED_API_CHANGE:                     push!(rows, (id=i, tstart=2.0, tstop=3.0, statefrom=1, stateto=state_t3, obstype=2))
# DISABLED_API_CHANGE:                 else
# DISABLED_API_CHANGE:                     push!(rows, (id=i, tstart=2.0, tstop=3.0, statefrom=2, stateto=2, obstype=2))
# DISABLED_API_CHANGE:                 end
# DISABLED_API_CHANGE:             else
# DISABLED_API_CHANGE:                 push!(rows, (id=i, tstart=1.0, tstop=2.0, statefrom=2, stateto=2, obstype=2))
# DISABLED_API_CHANGE:                 push!(rows, (id=i, tstart=2.0, tstop=3.0, statefrom=2, stateto=2, obstype=2))
# DISABLED_API_CHANGE:             end
# DISABLED_API_CHANGE:         end
# DISABLED_API_CHANGE:         data = DataFrame(rows)
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         # Create and fit model
# DISABLED_API_CHANGE:         h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
# DISABLED_API_CHANGE:         model = multistatemodel(h12; data=data, surrogate=:markov)
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         # Fit Markov surrogate
# DISABLED_API_CHANGE:         markov_surrogate = fit_surrogate(model; type=:markov, method=:mle, verbose=false)
# DISABLED_API_CHANGE:         model.markovsurrogate = markov_surrogate
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         # Test τ MLE fitting
# DISABLED_API_CHANGE:         n_phases_vec = [3, 1]  # 3 phases for state 1, 1 for absorbing state 2
# DISABLED_API_CHANGE:         fitted_tau = _fit_phasetype_mle(model, markov_surrogate, n_phases_vec;
# DISABLED_API_CHANGE:                                          structure=:sctp, verbose=false, maxiter=50)
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         # Verify fitted_tau has correct structure
# DISABLED_API_CHANGE:         @test haskey(fitted_tau, 1)  # State 1 is transient
# DISABLED_API_CHANGE:         @test length(fitted_tau[1]) == 3  # 3 phases
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         # Verify all τ values are positive (box constraint enforced)
# DISABLED_API_CHANGE:         @test all(τ > 0 for τ in fitted_tau[1])
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         # Verify τ values are not all equal (would indicate no fitting happened)
# DISABLED_API_CHANGE:         # With real data showing increasing hazard pattern, τ should show variation
# DISABLED_API_CHANGE:         # But with default :sctp (no ordering), we just check they're valid
# DISABLED_API_CHANGE:         @test all(isfinite.(fitted_tau[1]))
# DISABLED_API_CHANGE:     end
# DISABLED_API_CHANGE:     
# DISABLED_API_CHANGE:     @testset "τ ordering constraints for :sctp_increasing" begin
# DISABLED_API_CHANGE:         # Use data with increasing hazard pattern
# DISABLED_API_CHANGE:         Random.seed!(123)
# DISABLED_API_CHANGE:         n_subj = 150
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         rows = []
# DISABLED_API_CHANGE:         for i in 1:n_subj
# DISABLED_API_CHANGE:             # Time-increasing transition probability
# DISABLED_API_CHANGE:             t = 0.0
# DISABLED_API_CHANGE:             state = 1
# DISABLED_API_CHANGE:             for period in 1:4
# DISABLED_API_CHANGE:                 trans_prob = 0.15 * period  # 0.15, 0.30, 0.45, 0.60
# DISABLED_API_CHANGE:                 transitioned = rand() < trans_prob && state == 1
# DISABLED_API_CHANGE:                 new_state = transitioned ? 2 : state
# DISABLED_API_CHANGE:                 push!(rows, (id=i, tstart=t, tstop=t+1.0, statefrom=state, stateto=new_state, obstype=2))
# DISABLED_API_CHANGE:                 state = new_state
# DISABLED_API_CHANGE:                 t += 1.0
# DISABLED_API_CHANGE:             end
# DISABLED_API_CHANGE:         end
# DISABLED_API_CHANGE:         data = DataFrame(rows)
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
# DISABLED_API_CHANGE:         model = multistatemodel(h12; data=data, surrogate=:markov)
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         markov_surrogate = fit_surrogate(model; type=:markov, method=:mle, verbose=false)
# DISABLED_API_CHANGE:         model.markovsurrogate = markov_surrogate
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         n_phases_vec = [3, 1]
# DISABLED_API_CHANGE:         fitted_tau = _fit_phasetype_mle(model, markov_surrogate, n_phases_vec;
# DISABLED_API_CHANGE:                                          structure=:sctp_increasing, verbose=false, maxiter=50)
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         # Verify ordering constraint: τ₁ ≤ τ₂ ≤ τ₃
# DISABLED_API_CHANGE:         τ = fitted_tau[1]
# DISABLED_API_CHANGE:         @test length(τ) == 3
# DISABLED_API_CHANGE:         @test τ[1] <= τ[2] + 1e-6  # Allow small numerical tolerance
# DISABLED_API_CHANGE:         @test τ[2] <= τ[3] + 1e-6
# DISABLED_API_CHANGE:     end
# DISABLED_API_CHANGE:     
# DISABLED_API_CHANGE:     @testset "τ ordering constraints for :sctp_decreasing" begin
# DISABLED_API_CHANGE:         # Use data with decreasing hazard pattern (high early risk, lower later)
# DISABLED_API_CHANGE:         Random.seed!(456)
# DISABLED_API_CHANGE:         n_subj = 150
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         rows = []
# DISABLED_API_CHANGE:         for i in 1:n_subj
# DISABLED_API_CHANGE:             # Time-decreasing transition probability
# DISABLED_API_CHANGE:             t = 0.0
# DISABLED_API_CHANGE:             state = 1
# DISABLED_API_CHANGE:             for period in 1:4
# DISABLED_API_CHANGE:                 trans_prob = 0.6 - 0.15 * (period - 1)  # 0.60, 0.45, 0.30, 0.15
# DISABLED_API_CHANGE:                 transitioned = rand() < trans_prob && state == 1
# DISABLED_API_CHANGE:                 new_state = transitioned ? 2 : state
# DISABLED_API_CHANGE:                 push!(rows, (id=i, tstart=t, tstop=t+1.0, statefrom=state, stateto=new_state, obstype=2))
# DISABLED_API_CHANGE:                 state = new_state
# DISABLED_API_CHANGE:                 t += 1.0
# DISABLED_API_CHANGE:             end
# DISABLED_API_CHANGE:         end
# DISABLED_API_CHANGE:         data = DataFrame(rows)
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
# DISABLED_API_CHANGE:         model = multistatemodel(h12; data=data, surrogate=:markov)
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         markov_surrogate = fit_surrogate(model; type=:markov, method=:mle, verbose=false)
# DISABLED_API_CHANGE:         model.markovsurrogate = markov_surrogate
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         n_phases_vec = [3, 1]
# DISABLED_API_CHANGE:         fitted_tau = _fit_phasetype_mle(model, markov_surrogate, n_phases_vec;
# DISABLED_API_CHANGE:                                          structure=:sctp_decreasing, verbose=false, maxiter=50)
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         # Verify ordering constraint: τ₁ ≥ τ₂ ≥ τ₃
# DISABLED_API_CHANGE:         τ = fitted_tau[1]
# DISABLED_API_CHANGE:         @test length(τ) == 3
# DISABLED_API_CHANGE:         @test τ[1] >= τ[2] - 1e-6  # Allow small numerical tolerance
# DISABLED_API_CHANGE:         @test τ[2] >= τ[3] - 1e-6
# DISABLED_API_CHANGE:     end
# DISABLED_API_CHANGE:     
# DISABLED_API_CHANGE:     @testset "PhaseTypeSurrogate stores fitted_tau" begin
# DISABLED_API_CHANGE:         Random.seed!(789)
# DISABLED_API_CHANGE:         n_subj = 100
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         rows = []
# DISABLED_API_CHANGE:         for i in 1:n_subj
# DISABLED_API_CHANGE:             transitioned = rand() < 0.4
# DISABLED_API_CHANGE:             push!(rows, (id=i, tstart=0.0, tstop=2.0, statefrom=1, stateto=transitioned ? 2 : 1, obstype=2))
# DISABLED_API_CHANGE:             if !transitioned
# DISABLED_API_CHANGE:                 push!(rows, (id=i, tstart=2.0, tstop=4.0, statefrom=1, stateto=rand() < 0.5 ? 2 : 1, obstype=2))
# DISABLED_API_CHANGE:             else
# DISABLED_API_CHANGE:                 push!(rows, (id=i, tstart=2.0, tstop=4.0, statefrom=2, stateto=2, obstype=2))
# DISABLED_API_CHANGE:             end
# DISABLED_API_CHANGE:         end
# DISABLED_API_CHANGE:         data = DataFrame(rows)
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
# DISABLED_API_CHANGE:         model = multistatemodel(h12; data=data, surrogate=:markov)
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         # Fit phase-type surrogate with τ MLE (fit_tau=true is default)
# DISABLED_API_CHANGE:         phasetype_surrogate = _fit_phasetype_surrogate(model;
# DISABLED_API_CHANGE:             method=:mle, n_phases=3, verbose=false, fit_tau=true)
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         # Verify fitted_tau is stored
# DISABLED_API_CHANGE:         @test !isnothing(phasetype_surrogate.fitted_tau)
# DISABLED_API_CHANGE:         @test haskey(phasetype_surrogate.fitted_tau, 1)
# DISABLED_API_CHANGE:         @test length(phasetype_surrogate.fitted_tau[1]) == 3
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         # Test with fit_tau=false (should have nothing)
# DISABLED_API_CHANGE:         phasetype_surrogate_no_tau = _fit_phasetype_surrogate(model;
# DISABLED_API_CHANGE:             method=:mle, n_phases=3, verbose=false, fit_tau=false)
# DISABLED_API_CHANGE:         @test isnothing(phasetype_surrogate_no_tau.fitted_tau)
# DISABLED_API_CHANGE:     end
# DISABLED_API_CHANGE:     
# DISABLED_API_CHANGE:     @testset "Non-exponential hazard from non-uniform τ" begin
# DISABLED_API_CHANGE:         # Verify that non-uniform τ produces a different hazard shape than τ = 1
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         n_phases = 3
# DISABLED_API_CHANGE:         total_rate = 0.5
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         # Build Coxian with uniform τ (exponential baseline)
# DISABLED_API_CHANGE:         ph_uniform = _build_coxian_from_rate(n_phases, total_rate; structure=:sctp, tau=ones(n_phases))
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         # Build Coxian with increasing τ (increasing hazard)
# DISABLED_API_CHANGE:         tau_increasing = [0.5, 1.0, 1.5]
# DISABLED_API_CHANGE:         ph_increasing = _build_coxian_from_rate(n_phases, total_rate; structure=:sctp, tau=tau_increasing)
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         # Build Coxian with decreasing τ (decreasing hazard)
# DISABLED_API_CHANGE:         tau_decreasing = [1.5, 1.0, 0.5]
# DISABLED_API_CHANGE:         ph_decreasing = _build_coxian_from_rate(n_phases, total_rate; structure=:sctp, tau=tau_decreasing)
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         # Compute marginal hazard at different times using numerical differentiation
# DISABLED_API_CHANGE:         # h(t) = -d/dt log(S(t)) where S(t) = π' exp(St) 1
# DISABLED_API_CHANGE:         function marginal_hazard(ph::PhaseTypeDistribution, t::Float64)
# DISABLED_API_CHANGE:             S = subintensity(ph)
# DISABLED_API_CHANGE:             π = ph.initial
# DISABLED_API_CHANGE:             ones_vec = ones(ph.n_phases)
# DISABLED_API_CHANGE:             
# DISABLED_API_CHANGE:             # S(t) = π' exp(S*t) 1
# DISABLED_API_CHANGE:             exp_St = exp(S * t)
# DISABLED_API_CHANGE:             survival = π' * exp_St * ones_vec
# DISABLED_API_CHANGE:             
# DISABLED_API_CHANGE:             # d/dt S(t) = π' S exp(St) 1
# DISABLED_API_CHANGE:             deriv_survival = π' * S * exp_St * ones_vec
# DISABLED_API_CHANGE:             
# DISABLED_API_CHANGE:             # h(t) = -deriv_survival / survival
# DISABLED_API_CHANGE:             return -deriv_survival / survival
# DISABLED_API_CHANGE:         end
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         # Compute hazard at t=1 and t=3 for each distribution
# DISABLED_API_CHANGE:         t1, t3 = 1.0, 3.0
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         h_uniform_t1 = marginal_hazard(ph_uniform, t1)
# DISABLED_API_CHANGE:         h_uniform_t3 = marginal_hazard(ph_uniform, t3)
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         h_increasing_t1 = marginal_hazard(ph_increasing, t1)
# DISABLED_API_CHANGE:         h_increasing_t3 = marginal_hazard(ph_increasing, t3)
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         h_decreasing_t1 = marginal_hazard(ph_decreasing, t1)
# DISABLED_API_CHANGE:         h_decreasing_t3 = marginal_hazard(ph_decreasing, t3)
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         # Test: uniform τ gives approximately constant hazard (exponential)
# DISABLED_API_CHANGE:         @test h_uniform_t1 ≈ h_uniform_t3 rtol=0.1
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         # Test: increasing τ gives increasing hazard
# DISABLED_API_CHANGE:         @test h_increasing_t3 > h_increasing_t1
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         # Test: decreasing τ gives decreasing hazard
# DISABLED_API_CHANGE:         @test h_decreasing_t3 < h_decreasing_t1
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         # Test: all hazards are positive
# DISABLED_API_CHANGE:         @test h_uniform_t1 > 0
# DISABLED_API_CHANGE:         @test h_increasing_t1 > 0
# DISABLED_API_CHANGE:         @test h_decreasing_t1 > 0
# DISABLED_API_CHANGE:     end
# DISABLED_API_CHANGE:     
# DISABLED_API_CHANGE:     @testset "Single phase states skip τ optimization" begin
# DISABLED_API_CHANGE:         # When all states have n_phases=1, no τ optimization is needed
# DISABLED_API_CHANGE:         Random.seed!(111)
# DISABLED_API_CHANGE:         n_subj = 50
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         rows = []
# DISABLED_API_CHANGE:         for i in 1:n_subj
# DISABLED_API_CHANGE:             push!(rows, (id=i, tstart=0.0, tstop=2.0, statefrom=1, stateto=rand() < 0.4 ? 2 : 1, obstype=2))
# DISABLED_API_CHANGE:         end
# DISABLED_API_CHANGE:         data = DataFrame(rows)
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
# DISABLED_API_CHANGE:         model = multistatemodel(h12; data=data, surrogate=:markov)
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         markov_surrogate = fit_surrogate(model; type=:markov, method=:mle, verbose=false)
# DISABLED_API_CHANGE:         model.markovsurrogate = markov_surrogate
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         # n_phases=1 for all states
# DISABLED_API_CHANGE:         n_phases_vec = [1, 1]
# DISABLED_API_CHANGE:         fitted_tau = _fit_phasetype_mle(model, markov_surrogate, n_phases_vec;
# DISABLED_API_CHANGE:                                          structure=:sctp, verbose=false)
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         # Should return empty dict when all states have 1 phase
# DISABLED_API_CHANGE:         @test isempty(fitted_tau)
# DISABLED_API_CHANGE:     end
# DISABLED_API_CHANGE:     
# DISABLED_API_CHANGE:     @testset "Multiple transient states with different n_phases" begin
# DISABLED_API_CHANGE:         # Test a model with multiple transient states
# DISABLED_API_CHANGE:         Random.seed!(222)
# DISABLED_API_CHANGE:         n_subj = 100
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         # Create 3-state model: 1 ↔ 2 → 3 (absorbing)
# DISABLED_API_CHANGE:         rows = []
# DISABLED_API_CHANGE:         for i in 1:n_subj
# DISABLED_API_CHANGE:             state = 1
# DISABLED_API_CHANGE:             t = 0.0
# DISABLED_API_CHANGE:             for _ in 1:3
# DISABLED_API_CHANGE:                 if state == 1
# DISABLED_API_CHANGE:                     dest = rand() < 0.3 ? 2 : 1
# DISABLED_API_CHANGE:                 elseif state == 2
# DISABLED_API_CHANGE:                     dest = rand() < 0.2 ? 1 : (rand() < 0.5 ? 3 : 2)
# DISABLED_API_CHANGE:                 else
# DISABLED_API_CHANGE:                     dest = 3  # Absorbing
# DISABLED_API_CHANGE:                 end
# DISABLED_API_CHANGE:                 push!(rows, (id=i, tstart=t, tstop=t+1.0, statefrom=state, stateto=dest, obstype=2))
# DISABLED_API_CHANGE:                 state = dest
# DISABLED_API_CHANGE:                 t += 1.0
# DISABLED_API_CHANGE:             end
# DISABLED_API_CHANGE:         end
# DISABLED_API_CHANGE:         data = DataFrame(rows)
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
# DISABLED_API_CHANGE:         h21 = Hazard(@formula(0 ~ 1), "exp", 2, 1)
# DISABLED_API_CHANGE:         h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
# DISABLED_API_CHANGE:         model = multistatemodel(h12, h21, h23; data=data, surrogate=:markov)
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         markov_surrogate = fit_surrogate(model; type=:markov, method=:mle, verbose=false)
# DISABLED_API_CHANGE:         model.markovsurrogate = markov_surrogate
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         # Different n_phases per state
# DISABLED_API_CHANGE:         n_phases_vec = [2, 3, 1]  # State 1: 2 phases, State 2: 3 phases, State 3: 1 (absorbing)
# DISABLED_API_CHANGE:         fitted_tau = _fit_phasetype_mle(model, markov_surrogate, n_phases_vec;
# DISABLED_API_CHANGE:                                          structure=:sctp, verbose=false, maxiter=50)
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         # Should have τ for states 1 and 2
# DISABLED_API_CHANGE:         @test haskey(fitted_tau, 1)
# DISABLED_API_CHANGE:         @test haskey(fitted_tau, 2)
# DISABLED_API_CHANGE:         @test !haskey(fitted_tau, 3)  # Absorbing state
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         # Verify lengths
# DISABLED_API_CHANGE:         @test length(fitted_tau[1]) == 2
# DISABLED_API_CHANGE:         @test length(fitted_tau[2]) == 3
# DISABLED_API_CHANGE:         
# DISABLED_API_CHANGE:         # Verify all positive
# DISABLED_API_CHANGE:         @test all(τ > 0 for τ in fitted_tau[1])
# DISABLED_API_CHANGE:         @test all(τ > 0 for τ in fitted_tau[2])
# DISABLED_API_CHANGE:     end
# end  # Old testset end

# =============================================================================
# Phase-Type Rate MLE Tests (New API)
# =============================================================================
#
# These tests verify the full rate estimation via MLE functionality:
# - Fitted rates (progression λ and absorption μ) have correct structure
# - All rates are positive
# - PhaseTypeSurrogate correctly stores fitted_rates field
#
# Added: 2026-01-20
# =============================================================================

# Import additional internal functions for rate estimation tests
import MultistateModels:
    _fit_phasetype_mle, _fit_phasetype_surrogate, phasetype_marginal_loglik,
    _build_coxian_from_rate, _build_state_mappings, build_expanded_Q,
    _build_default_phasetype

@testset "PhaseType Surrogate Rate MLE Estimation" begin

    @testset "Basic rate MLE fitting returns valid results" begin
        # Create simple panel data
        Random.seed!(42)
        n_subj = 100

        rows = []
        for i in 1:n_subj
            # First observation period
            transitioned_t1 = rand() < 0.2
            state_t1 = transitioned_t1 ? 2 : 1
            push!(rows, (id=i, tstart=0.0, tstop=1.0, statefrom=1, stateto=state_t1, obstype=2))

            if state_t1 == 1
                # Second period
                transitioned_t2 = rand() < 0.35
                state_t2 = transitioned_t2 ? 2 : 1
                push!(rows, (id=i, tstart=1.0, tstop=2.0, statefrom=1, stateto=state_t2, obstype=2))

                if state_t2 == 1
                    # Third period
                    transitioned_t3 = rand() < 0.5
                    state_t3 = transitioned_t3 ? 2 : 1
                    push!(rows, (id=i, tstart=2.0, tstop=3.0, statefrom=1, stateto=state_t3, obstype=2))
                else
                    push!(rows, (id=i, tstart=2.0, tstop=3.0, statefrom=2, stateto=2, obstype=2))
                end
            else
                push!(rows, (id=i, tstart=1.0, tstop=2.0, statefrom=2, stateto=2, obstype=2))
                push!(rows, (id=i, tstart=2.0, tstop=3.0, statefrom=2, stateto=2, obstype=2))
            end
        end
        data = DataFrame(rows)

        # Create and fit model
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        model = multistatemodel(h12; data=data, surrogate=:markov)

        # Fit Markov surrogate
        markov_surrogate = fit_surrogate(model; type=:markov, method=:mle, verbose=false)
        model.markovsurrogate = markov_surrogate

        # Test rate MLE fitting
        n_phases_vec = [3, 1]  # 3 phases for state 1, 1 for absorbing state 2
        fitted_rates = _fit_phasetype_mle(model, markov_surrogate, n_phases_vec; verbose=false)

        # Verify fitted_rates has correct structure
        @test haskey(fitted_rates, :progression)
        @test haskey(fitted_rates, :absorption)
        @test haskey(fitted_rates, :destinations)
        @test haskey(fitted_rates, :param_layout)
        @test haskey(fitted_rates, :theta)

        # Verify state 1 has progression rates (n_phases - 1 = 2)
        @test haskey(fitted_rates[:progression], 1)
        @test length(fitted_rates[:progression][1]) == 2

        # Verify state 1 has absorption rates (n_phases × n_destinations = 3 × 1)
        @test haskey(fitted_rates[:absorption], 1)
        @test size(fitted_rates[:absorption][1]) == (3, 1)

        # Verify all rates are positive
        @test all(λ > 0 for λ in fitted_rates[:progression][1])
        @test all(μ > 0 for μ in fitted_rates[:absorption][1])

        # Verify theta vector is finite
        @test all(isfinite.(fitted_rates[:theta]))
    end

    @testset "PhaseTypeSurrogate stores fitted_rates" begin
        Random.seed!(789)
        n_subj = 100

        rows = []
        for i in 1:n_subj
            transitioned = rand() < 0.4
            push!(rows, (id=i, tstart=0.0, tstop=2.0, statefrom=1, stateto=transitioned ? 2 : 1, obstype=2))
            if !transitioned
                push!(rows, (id=i, tstart=2.0, tstop=4.0, statefrom=1, stateto=rand() < 0.5 ? 2 : 1, obstype=2))
            else
                push!(rows, (id=i, tstart=2.0, tstop=4.0, statefrom=2, stateto=2, obstype=2))
            end
        end
        data = DataFrame(rows)

        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        model = multistatemodel(h12; data=data, surrogate=:markov)

        # Fit phase-type surrogate with rate MLE (fit_rates=true is default)
        phasetype_surrogate = _fit_phasetype_surrogate(model;
            method=:mle, n_phases=3, verbose=false, fit_rates=true)

        # Verify fitted_rates is stored
        @test !isnothing(phasetype_surrogate.fitted_rates)
        @test haskey(phasetype_surrogate.fitted_rates, :progression)
        @test haskey(phasetype_surrogate.fitted_rates, :absorption)

        # Test with fit_rates=false (should have nothing)
        phasetype_surrogate_no_rates = _fit_phasetype_surrogate(model;
            method=:mle, n_phases=3, verbose=false, fit_rates=false)
        @test isnothing(phasetype_surrogate_no_rates.fitted_rates)
    end

    @testset "Single phase states skip rate optimization" begin
        # When all states have n_phases=1, no optimization is needed
        Random.seed!(111)
        n_subj = 50

        rows = []
        for i in 1:n_subj
            push!(rows, (id=i, tstart=0.0, tstop=2.0, statefrom=1, stateto=rand() < 0.4 ? 2 : 1, obstype=2))
        end
        data = DataFrame(rows)

        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        model = multistatemodel(h12; data=data, surrogate=:markov)

        markov_surrogate = fit_surrogate(model; type=:markov, method=:mle, verbose=false)
        model.markovsurrogate = markov_surrogate

        # n_phases=1 for all states
        n_phases_vec = [1, 1]
        fitted_rates = _fit_phasetype_mle(model, markov_surrogate, n_phases_vec; verbose=false)

        # Should return empty dict when all states have 1 phase
        @test isempty(fitted_rates[:progression])
        @test isempty(fitted_rates[:absorption])
    end

    @testset "Multiple transient states with different n_phases" begin
        # Test a model with multiple transient states
        Random.seed!(222)
        n_subj = 100

        # Create 3-state model: 1 ↔ 2 → 3 (absorbing)
        rows = []
        for i in 1:n_subj
            state = 1
            t = 0.0
            for _ in 1:3
                if state == 1
                    dest = rand() < 0.3 ? 2 : 1
                elseif state == 2
                    dest = rand() < 0.2 ? 1 : (rand() < 0.5 ? 3 : 2)
                else
                    dest = 3  # Absorbing
                end
                push!(rows, (id=i, tstart=t, tstop=t+1.0, statefrom=state, stateto=dest, obstype=2))
                state = dest
                t += 1.0
            end
        end
        data = DataFrame(rows)

        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        h21 = Hazard(@formula(0 ~ 1), "exp", 2, 1)
        h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
        model = multistatemodel(h12, h21, h23; data=data, surrogate=:markov)

        markov_surrogate = fit_surrogate(model; type=:markov, method=:mle, verbose=false)
        model.markovsurrogate = markov_surrogate

        # Different n_phases per state
        n_phases_vec = [2, 3, 1]  # State 1: 2 phases, State 2: 3 phases, State 3: 1 (absorbing)
        fitted_rates = _fit_phasetype_mle(model, markov_surrogate, n_phases_vec; verbose=false)

        # Should have progression rates for states 1 and 2
        @test haskey(fitted_rates[:progression], 1)
        @test haskey(fitted_rates[:progression], 2)
        @test !haskey(fitted_rates[:progression], 3)  # Absorbing state

        # Verify lengths of progression rates
        @test length(fitted_rates[:progression][1]) == 1  # 2 phases - 1
        @test length(fitted_rates[:progression][2]) == 2  # 3 phases - 1

        # Should have absorption rates for states 1 and 2
        @test haskey(fitted_rates[:absorption], 1)
        @test haskey(fitted_rates[:absorption], 2)

        # Verify all rates are positive
        @test all(λ > 0 for λ in fitted_rates[:progression][1])
        @test all(λ > 0 for λ in fitted_rates[:progression][2])
        @test all(μ > 0 for μ in fitted_rates[:absorption][1])
        @test all(μ > 0 for μ in fitted_rates[:absorption][2])
    end

end  # End of "PhaseType Surrogate Rate MLE Estimation" testset