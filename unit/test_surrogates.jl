# =============================================================================
# Surrogate Fitting Tests
# =============================================================================
#
# Tests for:
# - AbstractSurrogate type hierarchy and is_fitted() interface
# - Markov surrogate Q-matrix validity (negative diagonals, row sum = 0)
# - Phase-type surrogate Q-matrix validity
# - MLE produces non-negative rates
# - MLE log-likelihood >= heuristic log-likelihood
# - `fit_surrogate` always returns lightweight surrogate types
# - `is_surrogate_fitted` function
# - Deprecation warnings
using Test
using MultistateModels
using Random
using DataFrames
using Statistics

@testset "Surrogate Fitting" begin

    # Setup: Panel data for testing
    function create_test_data(; n_subj = 30, seed = 12345)
        Random.seed!(seed)
        dat = DataFrame(
            id = repeat(1:n_subj, inner = 3),
            tstart = repeat([0.0, 1.0, 2.0], n_subj),
            tstop = repeat([1.0, 2.0, 3.0], n_subj),
            statefrom = repeat([1, 1, 1], n_subj),
            stateto = vcat([[rand() < 0.3 ? 2 : 1, rand() < 0.5 ? 2 : 1, 2] for _ in 1:n_subj]...),
            obstype = repeat([2, 2, 2], n_subj)
        )
        return dat
    end
    
    # =========================================================================
    # AbstractSurrogate Type Hierarchy Tests
    # =========================================================================
    
    @testset "AbstractSurrogate type hierarchy" begin
        dat = create_test_data()
        h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
        model = multistatemodel(h12; data = dat)
        
        # MarkovSurrogate inherits from AbstractSurrogate
        surrogate_markov = MultistateModels.fit_surrogate(model; method = :mle, verbose = false)
        @test surrogate_markov isa AbstractSurrogate
        @test surrogate_markov isa MarkovSurrogate
        
        # PhaseTypeSurrogate inherits from AbstractSurrogate
        surrogate_pt = MultistateModels.fit_surrogate(model; type = :phasetype, n_phases = 2, verbose = false)
        @test surrogate_pt isa AbstractSurrogate
        @test surrogate_pt isa PhaseTypeSurrogate
        
        # is_fitted works on both types via AbstractSurrogate dispatch
        @test is_fitted(surrogate_markov)
        @test is_fitted(surrogate_pt)
    end
    
    @testset "is_fitted generic function" begin
        dat = create_test_data()
        h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
        
        # Unfitted MarkovSurrogate (from model creation with fit_surrogate=false)
        model_unfitted = multistatemodel(h12; data = dat, surrogate = :markov, fit_surrogate = false)
        @test !is_fitted(model_unfitted.surrogate)
        
        # Fitted MarkovSurrogate
        model_fitted = multistatemodel(h12; data = dat, surrogate = :markov, fit_surrogate = true)
        @test is_fitted(model_fitted.surrogate)
        
        # PhaseTypeSurrogate is always fitted (built from fitted Markov)
        model = multistatemodel(h12; data = dat)
        surrogate_pt = MultistateModels.fit_surrogate(model; type = :phasetype, verbose = false)
        @test is_fitted(surrogate_pt)
        @test surrogate_pt.fitted == true  # Direct field access
    end
    
    # =========================================================================
    # fit_surrogate Return Type Consistency Tests
    # =========================================================================
    
    @testset "fit_surrogate always returns lightweight types" begin
        dat = create_test_data()
        h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
        model = multistatemodel(h12; data = dat)
        
        # All Markov variations return MarkovSurrogate (never MultistateModelFitted)
        s1 = MultistateModels.fit_surrogate(model; verbose = false)
        s2 = MultistateModels.fit_surrogate(model; method = :mle, verbose = false)
        s3 = MultistateModels.fit_surrogate(model; method = :heuristic, verbose = false)
        s4 = MultistateModels.fit_surrogate(model; type = :markov, verbose = false)
        s5 = MultistateModels.fit_surrogate(model; type = :markov, method = :mle, verbose = false)
        
        for s in [s1, s2, s3, s4, s5]
            @test s isa MarkovSurrogate
            @test !(s isa MultistateModels.MultistateModelFitted)
        end
        
        # Phase-type returns PhaseTypeSurrogate
        s_pt = MultistateModels.fit_surrogate(model; type = :phasetype, verbose = false)
        @test s_pt isa PhaseTypeSurrogate
    end
    
    @testset "fit_surrogate with surrogate_parameters" begin
        dat = create_test_data()
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        model = multistatemodel(h12; data = dat)
        
        # Providing fixed parameters should still return MarkovSurrogate
        # v0.3.0+: Parameters are on natural scale (rates directly)
        fixed_params = (h12 = [0.3],)  # rate = 0.3
        surrogate = MultistateModels.fit_surrogate(model; 
            surrogate_parameters = fixed_params, verbose = false)
        
        @test surrogate isa MarkovSurrogate
        @test is_fitted(surrogate)
        
        # Verify the parameter was set correctly (access via nested structure)
        rate = surrogate.parameters.nested.h12.baseline.h12_rate
        @test isapprox(rate, 0.3, rtol = 1e-6)
    end
    
    # =========================================================================
    # Original Tests (retained)
    # =========================================================================
    
    @testset "fit_surrogate default behavior" begin
        dat = create_test_data()
        h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
        
        # Default: fit_surrogate=true - surrogate should be fitted during model creation
        model_default = multistatemodel(h12; data = dat, surrogate = :markov)
        @test is_surrogate_fitted(model_default)
        @test model_default.surrogate.fitted == true
        
        # Explicit fit_surrogate=false - surrogate exists but not fitted
        model_deferred = multistatemodel(h12; data = dat, surrogate = :markov, fit_surrogate = false)
        @test !is_surrogate_fitted(model_deferred)
        @test model_deferred.surrogate.fitted == false
        
        # No surrogate - is_surrogate_fitted returns false
        model_none = multistatemodel(h12; data = dat)
        @test !is_surrogate_fitted(model_none)
    end
    
    @testset "initialize_surrogate! marks as fitted" begin
        dat = create_test_data()
        h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
        
        # Start with unfitted surrogate
        model = multistatemodel(h12; data = dat, surrogate = :markov, fit_surrogate = false)
        @test !is_surrogate_fitted(model)
        
        # initialize_surrogate! should fit and mark as fitted
        initialize_surrogate!(model; verbose = false)
        @test is_surrogate_fitted(model)
    end
    
    @testset "Markov MLE produces valid rates" begin
        dat = create_test_data()
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        model = multistatemodel(h12; data = dat)
        
        surrogate = MultistateModels._fit_markov_surrogate(model; method = :mle, verbose = false)
        
        # Rates must be positive (access via flat parameters)
        rates = surrogate.parameters.flat
        @test all(r > 0 for r in rates)
        
        # Verify rate is in plausible range for test data (30 subjects, ~30% transition rate)
        # With 3 observations per subject at times 0-1, 1-2, 2-3, rate should be ~0.1-0.5
        rate_12 = surrogate.parameters.nested.h12.baseline.h12_rate
        @test rate_12 > 0.05  # Lower bound based on data generation
        @test rate_12 < 1.0   # Upper bound: not implausibly high
        
        # Surrogate should be marked as fitted
        @test surrogate.fitted == true
    end
    
    @testset "Phase-Type Q-matrix validity" begin
        dat = create_test_data()
        h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
        model = multistatemodel(h12; data = dat)
        
        surrogate = MultistateModels.fit_surrogate(model; type = :phasetype, 
            n_phases = Dict(1=>2), verbose = false)
        
        Q = surrogate.expanded_Q
        
        # Diagonals must be non-positive
        for i in 1:size(Q, 1)
            @test Q[i, i] <= 0.0
        end
        
        # Rows must sum to 0 (generator property)
        for i in 1:size(Q, 1)
            @test isapprox(sum(Q[i, :]), 0.0, atol = 1e-10)
        end
        
    end
    
    @testset "_build_coxian_from_rate structure variants" begin
        # Test that SCTP and erlang structures produce valid distributions
        # with correct means and distinct hazard shapes
        
        using LinearAlgebra
        
        n_phases = 5
        total_rate = 0.25
        target_mean = 1.0 / total_rate
        
        # :sctp now includes eigenvalue ordering for identifiability
        for structure in [:sctp, :erlang]
            ph = MultistateModels._build_coxian_from_rate(n_phases, total_rate; structure=structure)
            
            # Basic validity: correct dimensions
            @test size(ph.Q) == (n_phases + 1, n_phases + 1)
            @test length(ph.initial) == n_phases
            
            # Initial distribution starts in first phase
            @test ph.initial[1] == 1.0
            @test all(ph.initial[2:end] .== 0.0)
            
            # Q-matrix validity: rows sum to zero
            for i in 1:n_phases
                @test isapprox(sum(ph.Q[i, :]), 0.0, atol=1e-12)
            end
            
            # Q-matrix validity: absorbing row is zeros
            @test all(ph.Q[end, :] .== 0.0)
            
            # Correct mean (critical for proper scaling)
            S = ph.Q[1:n_phases, 1:n_phases]
            α = ph.initial
            e = ones(n_phases)
            mean_time = -(α' * (S \ e))
            @test isapprox(mean_time, target_mean, rtol=1e-10)
        end
        
        # Verify distinct hazard behavior between :sctp and :erlang
        function hazard_at(ph, t)
            S = ph.Q[1:ph.n_phases, 1:ph.n_phases]
            α = ph.initial
            e = ones(ph.n_phases)
            P_t = exp(S * t)
            surv = α' * P_t * e
            dsdt = α' * S * P_t * e
            return -dsdt / surv
        end
        
        ph_sctp = MultistateModels._build_coxian_from_rate(n_phases, total_rate; structure=:sctp)
        ph_erl = MultistateModels._build_coxian_from_rate(n_phases, total_rate; structure=:erlang)
        
        # SCTP: constant hazard (exponential when τ's are uniform)
        @test isapprox(hazard_at(ph_sctp, 1.0), hazard_at(ph_sctp, 10.0), rtol=1e-10)
        
        # Erlang: strongly increasing hazard
        @test hazard_at(ph_erl, 1.0) < hazard_at(ph_erl, 10.0)
    end
    
    @testset "_build_coxian_from_rate invalid structure" begin
        # Test that invalid structure throws an error
        @test_throws ArgumentError MultistateModels._build_coxian_from_rate(3, 0.25; structure=:invalid_structure)
    end
    
    @testset "MLE >= heuristic log-likelihood" begin
        dat = create_test_data(n_subj = 100)
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        model = multistatemodel(h12; data = dat)
        
        surrogate_mle = MultistateModels._fit_markov_surrogate(model; method = :mle, verbose = false)
        surrogate_heur = MultistateModels._fit_markov_surrogate(model; method = :heuristic, verbose = false)
        
        ll_mle = MultistateModels.compute_markov_marginal_loglik(model, surrogate_mle)
        ll_heur = MultistateModels.compute_markov_marginal_loglik(model, surrogate_heur)
        
        # MLE is optimal, so its log-likelihood should be >= heuristic
        @test ll_mle >= ll_heur - 1e-6
        
        # Verify the actual log-likelihoods are finite and reasonable in magnitude
        # For n=100 subjects with 3 observations each, expect total ll in [-500, -10] range
        @test isfinite(ll_mle)
        @test isfinite(ll_heur)
        @test ll_mle > -1000.0  # Not implausibly negative
        @test ll_mle < -5.0     # Not implausibly close to zero for 100 subjects
    end
    
    @testset "initialize_surrogate! auto-fits unfitted surrogate" begin
        dat = create_test_data(n_subj = 20)
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        
        # Create model with unfitted surrogate
        model = multistatemodel(h12; data = dat, surrogate = :markov, fit_surrogate = false)
        @test !is_surrogate_fitted(model)
        
        # initialize_surrogate! should fit the surrogate (this is what fit() calls internally)
        initialize_surrogate!(model; verbose = false)
        
        # After fitting, the model's surrogate should be fitted
        @test is_surrogate_fitted(model)
        @test model.surrogate.fitted == true
        
        # Rates should be positive and in plausible range (access via flat params)
        rates = model.surrogate.parameters.flat
        rate_12 = rates[1]  # First (and only) rate for exp hazard
        @test rate_12 > 0.05  # Lower bound
        @test rate_12 < 1.0   # Upper bound
    end
    
    @testset "Input validation" begin
        dat = create_test_data()
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        model = multistatemodel(h12; data = dat)
        
        @test_throws ArgumentError MultistateModels._validate_surrogate_inputs(:invalid, :mle)
        @test_throws ArgumentError MultistateModels._validate_surrogate_inputs(:markov, :invalid)
    end
    
    # =========================================================================
    # BIC-Based Surrogate Selection Tests
    # =========================================================================
    
    @testset "select_surrogate basic functionality" begin
        dat = create_test_data(n_subj = 50)
        h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
        model = multistatemodel(h12; data = dat)
        
        # select_surrogate returns a valid surrogate
        surrogate = select_surrogate(model; verbose = false)
        @test surrogate isa AbstractSurrogate
        @test surrogate isa Union{MarkovSurrogate, PhaseTypeSurrogate}
        @test is_fitted(surrogate)
    end
    
    @testset "select_surrogate with return_all" begin
        dat = create_test_data(n_subj = 50)
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        model = multistatemodel(h12; data = dat)
        
        # Return all results for inspection
        result = select_surrogate(model; verbose = false, return_all = true)
        
        @test haskey(result, :best)
        @test haskey(result, :comparison)
        @test haskey(result, :selected)
        
        @test result.best isa AbstractSurrogate
        @test result.comparison isa DataFrame
        @test result.selected isa Symbol
        
        # Comparison DataFrame should have expected columns
        @test "candidate" in names(result.comparison)
        @test "bic" in names(result.comparison)
        @test "loglik" in names(result.comparison)
        @test "n_params" in names(result.comparison)
        @test "selected" in names(result.comparison)
        
        # Exactly one candidate should be selected
        @test sum(result.comparison.selected) == 1
    end
    
    @testset "select_surrogate custom candidates" begin
        dat = create_test_data(n_subj = 30)
        h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
        model = multistatemodel(h12; data = dat)
        
        # Test with custom candidates
        surrogate = select_surrogate(model; 
            candidates = [:markov, :phasetype_2],
            verbose = false)
        @test surrogate isa AbstractSurrogate
        
        # Test with tuple format
        surrogate2 = select_surrogate(model; 
            candidates = [:markov, (:phasetype, 3)],
            verbose = false)
        @test surrogate2 isa AbstractSurrogate
    end
    
    @testset "compute_surrogate_bic for MarkovSurrogate" begin
        dat = create_test_data(n_subj = 50)
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        model = multistatemodel(h12; data = dat)
        
        surrogate = fit_surrogate(model; type = :markov, verbose = false)
        bic_val, loglik, n_params = compute_surrogate_bic(model, surrogate)
        
        # BIC should be finite
        @test isfinite(bic_val)
        @test isfinite(loglik)
        
        # Parameter count: 1 transition = 1 parameter
        @test n_params == 1
        
        # Manual BIC calculation should match
        n_subjects = 50
        expected_bic = -2.0 * loglik + log(n_subjects) * n_params
        @test isapprox(bic_val, expected_bic, rtol = 1e-10)
    end
    
    @testset "compute_surrogate_bic for PhaseTypeSurrogate" begin
        dat = create_test_data(n_subj = 50)
        h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
        model = multistatemodel(h12; data = dat)
        
        surrogate = fit_surrogate(model; type = :phasetype, n_phases = 2, verbose = false)
        bic_val, loglik, n_params = compute_surrogate_bic(model, surrogate)
        
        # BIC should be finite
        @test isfinite(bic_val)
        @test isfinite(loglik)
        
        # Phase-type with 2 phases, 1 destination: 
        # Parameters = (2-1) progression + 2*1 absorption = 3
        @test n_params == 3
        
        # Manual BIC calculation should match
        n_subjects = 50
        expected_bic = -2.0 * loglik + log(n_subjects) * n_params
        @test isapprox(bic_val, expected_bic, rtol = 1e-10)
    end
    
    @testset "BIC favors parsimony for exponential data" begin
        # For truly exponential data, Markov should have lower BIC
        # because it has fewer parameters and exponential is the true model
        Random.seed!(42)
        
        # Generate data with exponential sojourns (Markov is correct model)
        n_subj = 100
        dat = DataFrame(
            id = repeat(1:n_subj, inner = 3),
            tstart = repeat([0.0, 1.0, 2.0], n_subj),
            tstop = repeat([1.0, 2.0, 3.0], n_subj),
            statefrom = repeat([1, 1, 1], n_subj),
            stateto = vcat([[rand() < 0.25 ? 2 : 1, rand() < 0.25 ? 2 : 1, 2] for _ in 1:n_subj]...),
            obstype = repeat([2, 2, 2], n_subj)
        )
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        model = multistatemodel(h12; data = dat)
        
        result = select_surrogate(model; 
            candidates = [:markov, :phasetype_2],
            verbose = false, 
            return_all = true)
        
        # Extract BIC values
        markov_row = filter(row -> row.candidate == :markov, result.comparison)
        phasetype_row = filter(row -> row.candidate == :phasetype_2, result.comparison)
        
        markov_bic = markov_row.bic[1]
        phasetype_bic = phasetype_row.bic[1]
        
        # Markov should have lower (or nearly equal) BIC for exponential data
        # Phase-type has 3 params vs 1 param, so it needs significantly better fit to win
        # Allow some tolerance since MLE can have variance
        @test markov_bic <= phasetype_bic + 5.0  # Markov shouldn't be much worse
    end
    
    @testset "_parse_surrogate_candidates" begin
        # Test parsing of various candidate formats
        parsed = MultistateModels._parse_surrogate_candidates([:markov])
        @test parsed == [(:markov, nothing)]
        
        parsed = MultistateModels._parse_surrogate_candidates([:phasetype_2])
        @test parsed == [(:phasetype, 2)]
        
        parsed = MultistateModels._parse_surrogate_candidates([:phasetype_3])
        @test parsed == [(:phasetype, 3)]
        
        parsed = MultistateModels._parse_surrogate_candidates([(:phasetype, 4)])
        @test parsed == [(:phasetype, 4)]
        
        # Invalid formats should throw
        @test_throws ArgumentError MultistateModels._parse_surrogate_candidates([:invalid])
        @test_throws ArgumentError MultistateModels._parse_surrogate_candidates([:phasetype_abc])
        @test_throws ArgumentError MultistateModels._parse_surrogate_candidates([(:phasetype, -1)])
    end
    
    @testset "initialize_surrogate! with type=:auto" begin
        dat = create_test_data(n_subj = 30)
        h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
        
        # Create model with unfitted surrogate
        model = multistatemodel(h12; data = dat, surrogate = :markov, fit_surrogate = false)
        
        # Initialize with :auto should use BIC-based selection
        initialize_surrogate!(model; type = :auto, verbose = false)
        
        # Should be fitted with some surrogate
        @test is_surrogate_fitted(model)
        @test model.surrogate isa AbstractSurrogate
    end
    
    @testset "multistatemodel surrogate=:auto uses BIC selection" begin
        dat = create_test_data(n_subj = 30)
        h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
        
        # Create model with :auto surrogate selection
        model = multistatemodel(h12; data = dat, surrogate = :auto, verbose = false)
        
        # Should have a fitted surrogate
        @test is_surrogate_fitted(model)
        @test model.surrogate isa AbstractSurrogate
    end

end
