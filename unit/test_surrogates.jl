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
        @test !is_fitted(model_unfitted.markovsurrogate)
        
        # Fitted MarkovSurrogate
        model_fitted = multistatemodel(h12; data = dat, surrogate = :markov, fit_surrogate = true)
        @test is_fitted(model_fitted.markovsurrogate)
        
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
        @test model_default.markovsurrogate.fitted == true
        
        # Explicit fit_surrogate=false - surrogate exists but not fitted
        model_deferred = multistatemodel(h12; data = dat, surrogate = :markov, fit_surrogate = false)
        @test !is_surrogate_fitted(model_deferred)
        @test model_deferred.markovsurrogate.fitted == false
        
        # No surrogate - is_surrogate_fitted returns false
        model_none = multistatemodel(h12; data = dat)
        @test !is_surrogate_fitted(model_none)
    end
    
    @testset "set_surrogate! marks as fitted" begin
        dat = create_test_data()
        h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
        
        # Start with unfitted surrogate
        model = multistatemodel(h12; data = dat, surrogate = :markov, fit_surrogate = false)
        @test !is_surrogate_fitted(model)
        
        # set_surrogate! should fit and mark as fitted
        set_surrogate!(model; verbose = false)
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
        
        # PhaseTypeSurrogate should have fitted=true
        @test surrogate.fitted == true
        @test is_fitted(surrogate)
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
    
    @testset "set_surrogate! auto-fits unfitted surrogate" begin
        dat = create_test_data(n_subj = 20)
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        
        # Create model with unfitted surrogate
        model = multistatemodel(h12; data = dat, surrogate = :markov, fit_surrogate = false)
        @test !is_surrogate_fitted(model)
        
        # set_surrogate! should fit the surrogate (this is what fit() calls internally)
        MultistateModels.set_surrogate!(model; verbose = false)
        
        # After fitting, the model's surrogate should be fitted
        @test is_surrogate_fitted(model)
        @test model.markovsurrogate.fitted == true
        
        # Rates should be positive and in plausible range (access via flat params)
        rates = model.markovsurrogate.parameters.flat
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
    
end
