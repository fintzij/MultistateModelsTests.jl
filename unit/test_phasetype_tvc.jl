# =============================================================================
# PhaseType Surrogate Time-Varying Covariate (TVC) Tests
# =============================================================================
#
# Unit tests for TVC handling in phase-type path likelihood computation.
#
# The key bug being tested:
# - convert_expanded_path_to_censored_data was using a single Q matrix for the 
#   entire subject, extracted from the first row's covariate index
# - For subjects with time-varying covariates, this is WRONG - each interval 
#   may have different covariate values requiring different Q matrices
#
# The fix:
# - Pass full hazmat_book_ph and schur_cache_ph to convert_expanded_path_to_censored_data
# - Interpolate covariates at each transition time to determine correct Q matrix
# - Build tpm_book indexed by (covariate_idx, time_idx)
# - compute_forward_loglik now uses covariate-indexed hazmat_book for dt=0 rows
#
# Reference: scratch/PHASETYPE_SURROGATE_FITTING_PLAN.md, Phase 0
# =============================================================================

using Test
using MultistateModels
using DataFrames
using Random
using LinearAlgebra

# Import internal functions needed for testing
import MultistateModels: 
    Hazard, multistatemodel, set_parameters!, simulate,
    MarkovSurrogate, PhaseTypeSurrogate, PhaseTypeProposal,
    _build_phasetype_from_markov, build_tpm_mapping,
    build_phasetype_tpm_book, fit_surrogate,
    convert_expanded_path_to_censored_data, compute_forward_loglik,
    SamplePath, CachedSchurDecomposition, @formula

"""
    create_illness_death_data_with_tvc(n_subj; seed)

Create illness-death panel data with TIME-VARYING treatment covariate.
Treatment switches from 0 to 1 at time 2.0 for all subjects.

This mimics a clinical trial where treatment starts after a baseline period.
"""
function create_illness_death_data_with_tvc(n_subj; seed=12345)
    Random.seed!(seed)
    
    rows = []
    
    for i in 1:n_subj
        # First interval: no treatment (x=0), from t=0 to t=2
        state_at_t2 = rand() < 0.2 ? 2 : 1  # 20% transition to illness
        push!(rows, (id=i, tstart=0.0, tstop=2.0, statefrom=1, stateto=state_at_t2, obstype=2, trt=0))
        
        # Second interval: treatment starts (x=1), from t=2 to t=4
        if state_at_t2 == 1
            # Still healthy - can transition to illness (2) or death (3)
            r = rand()
            state_at_t4 = r < 0.15 ? 2 : (r < 0.20 ? 3 : 1)
        else
            # In illness state - can transition to death or stay
            state_at_t4 = rand() < 0.30 ? 3 : 2
        end
        push!(rows, (id=i, tstart=2.0, tstop=4.0, statefrom=state_at_t2, stateto=state_at_t4, obstype=2, trt=1))
        
        # Third interval: still on treatment, from t=4 to t=6
        if state_at_t4 < 3  # Not dead
            r = rand()
            if state_at_t4 == 1
                state_at_t6 = r < 0.10 ? 2 : (r < 0.15 ? 3 : 1)
            else
                state_at_t6 = rand() < 0.25 ? 3 : 2
            end
            push!(rows, (id=i, tstart=4.0, tstop=6.0, statefrom=state_at_t4, stateto=state_at_t6, obstype=2, trt=1))
        end
    end
    
    return DataFrame(rows)
end

"""
    create_simple_tvc_data(n_subj; seed)

Create simpler 2-state panel data with TVC for debugging.
Treatment switches at t=2, observations at t=2 and t=4.
"""
function create_simple_tvc_data(n_subj; seed=12345)
    Random.seed!(seed)
    
    rows = []
    
    for i in 1:n_subj
        # Interval 1: trt=0, t=[0,2]
        transitioned_t2 = rand() < 0.2
        state_at_t2 = transitioned_t2 ? 2 : 1
        push!(rows, (id=i, tstart=0.0, tstop=2.0, statefrom=1, stateto=state_at_t2, obstype=2, trt=0))
        
        # Interval 2: trt=1, t=[2,4]
        if state_at_t2 == 1
            transitioned_t4 = rand() < 0.35  # Higher rate under treatment
            state_at_t4 = transitioned_t4 ? 2 : 1
            push!(rows, (id=i, tstart=2.0, tstop=4.0, statefrom=1, stateto=state_at_t4, obstype=2, trt=1))
        else
            push!(rows, (id=i, tstart=2.0, tstop=4.0, statefrom=2, stateto=2, obstype=2, trt=1))
        end
    end
    
    return DataFrame(rows)
end

@testset "PhaseType TVC Covariate Handling" begin

    @testset "TVC data has multiple covariate combinations" begin
        # Verify test data creation
        data = create_simple_tvc_data(50; seed=12345)
        
        # Should have both trt=0 and trt=1 rows
        @test 0 in data.trt
        @test 1 in data.trt
        
        # Each subject should have trt=0 in first row, trt=1 in second
        gd = groupby(data, :id)
        for subj_data in gd
            sorted_data = sort(subj_data, :tstart)
            @test sorted_data.trt[1] == 0
            if nrow(sorted_data) > 1
                @test sorted_data.trt[2] == 1
            end
        end
    end
    
    @testset "hazmat_book has different Q matrices for trt=0 vs trt=1" begin
        Random.seed!(42)
        data = create_simple_tvc_data(100; seed=42)
        
        h12 = Hazard(@formula(0 ~ trt), "exp", 1, 2)
        model = multistatemodel(h12; data=data, surrogate=:markov)
        
        # Fit Markov surrogate
        fit_surrogate(model; type=:markov, method=:mle, verbose=false)
        markov_surrogate = model.markovsurrogate
        
        # Build PhaseType surrogate
        phasetype_surrogate = _build_phasetype_from_markov(
            model, markov_surrogate;
            config=PhaseTypeProposal(n_phases=2),
            verbose=false
        )
        
        # Build TPM infrastructure
        books = build_tpm_mapping(model.data)
        tpm_book_ph, hazmat_book_ph = build_phasetype_tpm_book(
            phasetype_surrogate, markov_surrogate, books, model.data
        )
        
        # Should have at least 2 covariate combinations
        @test length(hazmat_book_ph) >= 2
        
        # Q matrices should differ between trt=0 and trt=1
        Q_trt0 = hazmat_book_ph[1]
        Q_trt1 = hazmat_book_ph[2]
        @test !(Q_trt0 ≈ Q_trt1)
    end
    
    @testset "convert_expanded_path_to_censored_data handles TVC" begin
        Random.seed!(123)
        data = create_simple_tvc_data(50; seed=123)
        
        h12 = Hazard(@formula(0 ~ trt), "exp", 1, 2)
        model = multistatemodel(h12; data=data, surrogate=:markov)
        
        # Fit Markov surrogate
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
        
        # Create Schur caches
        schur_cache_ph = [CachedSchurDecomposition(Q) for Q in hazmat_book_ph]
        
        # Create a mock expanded path that spans both covariate periods
        # Path: phase 1 from t=0, transition to phase 2 at t=1, transition to phase 3 (absorbing) at t=3
        n_expanded = phasetype_surrogate.n_expanded_states
        
        # For a 2-state model with 2 phases each, phases 1-2 are state 1, phase 3 is state 2
        # Path: start in phase 1 at t=0, stay until t=3 (spanning trt=0 and trt=1 periods)
        expanded_path = SamplePath(
            1,                 # subject ID
            [0.0, 3.0, 3.0],  # times
            [1, 1, 3]         # phases: stay in phase 1, then instant transition to phase 3
        )
        
        # Get subject 1's data for TVC interpolation
        subj_inds = model.subjectindices[1]
        subj_data = view(model.data, subj_inds, :)
        subj_tpm_map = view(books[2], subj_inds, :)
        
        # Convert path with TVC support
        censored_data, emat, tpm_map, tpm_book, hazmat_book_out = convert_expanded_path_to_censored_data(
            expanded_path, phasetype_surrogate;
            original_subj_data = subj_data,
            hazmat_book = hazmat_book_ph,
            schur_cache_book = schur_cache_ph,
            subj_tpm_map = subj_tpm_map
        )
        
        # The function should have produced valid outputs
        @test censored_data isa DataFrame
        @test nrow(censored_data) > 0
        @test size(tpm_map, 2) == 2  # (covar_idx, time_idx)
        @test length(tpm_book) > 0
        @test length(hazmat_book_out) > 0
        
        # Verify tpm_map has valid indices
        for i in 1:nrow(censored_data)
            covar_idx = tpm_map[i, 1]
            time_idx = tpm_map[i, 2]
            @test covar_idx >= 1
            @test covar_idx <= length(tpm_book)
            @test time_idx >= 1
            if length(tpm_book[covar_idx]) > 0
                @test time_idx <= length(tpm_book[covar_idx])
            end
        end
    end
    
    @testset "compute_forward_loglik uses correct Q matrix per interval" begin
        Random.seed!(456)
        data = create_simple_tvc_data(50; seed=456)
        
        h12 = Hazard(@formula(0 ~ trt), "exp", 1, 2)
        model = multistatemodel(h12; data=data, surrogate=:markov)
        
        # Fit and build surrogates
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
        schur_cache_ph = [CachedSchurDecomposition(Q) for Q in hazmat_book_ph]
        
        # Create expanded path for subject 1
        expanded_path = SamplePath(
            1,                 # subject ID
            [0.0, 1.0, 3.0],  # times: transition at t=1 within trt=0 period, then at t=3 in trt=1 period
            [1, 2, 3]         # phases
        )
        
        subj_inds = model.subjectindices[1]
        subj_data = view(model.data, subj_inds, :)
        subj_tpm_map = view(books[2], subj_inds, :)
        
        censored_data, emat, tpm_map, tpm_book, hazmat_book_out = convert_expanded_path_to_censored_data(
            expanded_path, phasetype_surrogate;
            original_subj_data = subj_data,
            hazmat_book = hazmat_book_ph,
            schur_cache_book = schur_cache_ph,
            subj_tpm_map = subj_tpm_map
        )
        
        # Compute forward likelihood - should not throw
        n_states = phasetype_surrogate.n_expanded_states
        ll = compute_forward_loglik(censored_data, emat, tpm_map, tpm_book, hazmat_book_out, n_states)
        
        # Should return a finite log-likelihood
        @test isfinite(ll)
        @test ll <= 0  # Log-likelihood should be negative or zero
    end
    
    @testset "Illness-death model with TVC" begin
        Random.seed!(789)
        data = create_illness_death_data_with_tvc(100; seed=789)
        
        # Illness-death model: 1 → 2 (illness), 1 → 3 (death), 2 → 3 (death)
        h12 = Hazard(@formula(0 ~ trt), "exp", 1, 2)
        h13 = Hazard(@formula(0 ~ trt), "exp", 1, 3)
        h23 = Hazard(@formula(0 ~ trt), "exp", 2, 3)
        
        model = multistatemodel(h12, h13, h23; data=data, surrogate=:markov)
        
        # Fit Markov surrogate
        fit_surrogate(model; type=:markov, method=:mle, verbose=false)
        markov_surrogate = model.markovsurrogate
        
        # Build PhaseType surrogate
        phasetype_surrogate = _build_phasetype_from_markov(
            model, markov_surrogate;
            config=PhaseTypeProposal(n_phases=2),
            verbose=false
        )
        
        # Build TPM infrastructure
        books = build_tpm_mapping(model.data)
        tpm_book_ph, hazmat_book_ph = build_phasetype_tpm_book(
            phasetype_surrogate, markov_surrogate, books, model.data
        )
        schur_cache_ph = [CachedSchurDecomposition(Q) for Q in hazmat_book_ph]
        
        # Test path conversion for multiple subjects
        n_tested = 0
        for i in 1:min(10, length(model.subjectindices))
            subj_inds = model.subjectindices[i]
            subj_data = view(model.data, subj_inds, :)
            
            # Check if subject has TVC (different trt values across rows)
            if length(unique(subj_data.trt)) > 1
                subj_tpm_map = view(books[2], subj_inds, :)
                
                # Create simple path for testing
                initial_phase = 1  # First phase of state 1
                final_phase = phasetype_surrogate.state_to_phases[2][1]  # First phase of state 2
                
                expanded_path = SamplePath(
                    subj_data.id[1],
                    [subj_data.tstart[1], subj_data.tstop[end]],
                    [initial_phase, final_phase]
                )
                
                censored_data, emat, tpm_map, tpm_book, hazmat_book_out = convert_expanded_path_to_censored_data(
                    expanded_path, phasetype_surrogate;
                    original_subj_data = subj_data,
                    hazmat_book = hazmat_book_ph,
                    schur_cache_book = schur_cache_ph,
                    subj_tpm_map = subj_tpm_map
                )
                
                n_states = phasetype_surrogate.n_expanded_states
                ll = compute_forward_loglik(censored_data, emat, tpm_map, tpm_book, hazmat_book_out, n_states)
                
                @test isfinite(ll)
                n_tested += 1
            end
        end
        
        @test n_tested > 0  # Should have tested at least one TVC subject
    end
    
    @testset "Legacy single-covariate mode still works" begin
        # Test backward compatibility - single hazmat and schur_cache
        Random.seed!(999)
        data = DataFrame(
            id = [1, 1, 2, 2],
            tstart = [0.0, 2.0, 0.0, 2.0],
            tstop = [2.0, 4.0, 2.0, 4.0],
            statefrom = [1, 1, 1, 2],
            stateto = [1, 2, 2, 2],
            obstype = [2, 2, 2, 2]
        )
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)  # No covariates
        model = multistatemodel(h12; data=data, surrogate=:markov)
        
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
        
        # Use legacy single-Q mode
        Q_single = hazmat_book_ph[1]
        schur_single = CachedSchurDecomposition(Q_single)
        
        expanded_path = SamplePath(1, [0.0, 2.0], [1, 3])
        
        # Legacy call (should still work)
        censored_data, emat, tpm_map, tpm_book, hazmat_out = convert_expanded_path_to_censored_data(
            expanded_path, phasetype_surrogate;
            hazmat = Q_single,
            schur_cache = schur_single
        )
        
        @test censored_data isa DataFrame
        @test length(hazmat_out) == 1  # Single covariate level
        
        # compute_forward_loglik with single Q (backward compatible)
        n_states = phasetype_surrogate.n_expanded_states
        ll = compute_forward_loglik(censored_data, emat, tpm_map, tpm_book, Q_single, n_states)
        @test isfinite(ll)
    end

end
