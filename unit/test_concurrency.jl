# =============================================================================
# Concurrency Stress Tests for MultistateModels.jl
# =============================================================================
#
# Tests parallel/concurrent usage patterns to document expected behavior
# and verify thread-safety characteristics.
#
# SUMMARY OF CONCURRENT USAGE PATTERNS:
# =====================================
#
# SAFE (Supported):
# - Parallel simulate() calls on the SAME model (read-only operations)
# - Parallel fit() calls on DEEPCOPIED models (independent state)
# - Parallel likelihood evaluations on different models
# - Concurrent read operations (get_parameters, get_vcov, etc.)
#
# UNSAFE (NOT Supported):
# - Parallel fit() calls on the SAME model (mutates shared state)
# - Concurrent set_parameters! on the same model
# - Any operation that mutates model state during parallel reads
#
# =============================================================================

using Test
using MultistateModels
using DataFrames
using Random

@testset "Concurrency Tests" begin
    
    # =========================================================================
    # Test Setup: Create a simple model for testing
    # =========================================================================
    
    function create_test_model()
        # Small dataset for fast tests
        Random.seed!(12345)
        n_subjects = 30
        
        data = DataFrame(
            id = repeat(1:n_subjects, inner=2),
            tstart = repeat([0.0, 1.0], n_subjects),
            tstop = repeat([1.0, 2.0], n_subjects),
            statefrom = repeat([1, 1], n_subjects),
            stateto = repeat([1, 2], n_subjects),
            obstype = repeat([1, 1], n_subjects),
            x = rand(2*n_subjects)
        )
        # Make half progress to state 2
        for i in 1:n_subjects
            if rand() < 0.5
                data[(i-1)*2 + 2, :stateto] = 2
            end
        end
        
        h12 = Hazard(@formula(0 ~ 1 + x), :wei, 1, 2)
        
        model = multistatemodel(h12; data=data)
        return model
    end
    
    # =========================================================================
    # SAFE: Parallel simulate() on same model
    # =========================================================================
    
    @testset "Parallel simulate() on same model (SAFE)" begin
        model = create_test_model()
        fitted = fit(model; verbose=false)
        
        # Parallel simulation should work - simulate is read-only
        n_simulations = 10
        results = Vector{Any}(undef, n_simulations)
        
        Threads.@threads for i in 1:n_simulations
            # Each thread simulates from the same fitted model
            # This should be safe because simulate doesn't mutate the model
            Random.seed!(1000 + i)
            paths = simulate(fitted; nsim=5, data=true)
            results[i] = paths
        end
        
        # Verify all simulations completed
        @test all(!isnothing, results)
        @test all(r -> r isa DataFrame, results)
        
        # Verify simulations are different (different seeds)
        if n_simulations >= 2
            @test results[1] != results[2]  # Different seeds should give different results
        end
    end
    
    # =========================================================================
    # SAFE: Parallel fit() on deepcopied models
    # =========================================================================
    
    @testset "Parallel fit() on deepcopied models (SAFE)" begin
        model_template = create_test_model()
        
        # Create independent copies for each parallel fit
        n_fits = 4
        models = [deepcopy(model_template) for _ in 1:n_fits]
        
        # Fit in parallel - each model is independent
        fitted_models = Vector{Any}(undef, n_fits)
        
        Threads.@threads for i in 1:n_fits
            # Fitting deepcopied models should be safe
            fitted_models[i] = fit(models[i]; verbose=false, vcov_type=:none)
        end
        
        # Verify all fits completed
        @test all(!isnothing, fitted_models)
        @test all(m -> m isa MultistateModelFitted, fitted_models)
        
        # Parameters should be similar (same data, same model)
        params = [get_parameters(m) for m in fitted_models]
        for i in 2:n_fits
            # Allow small numerical differences
            @test isapprox(params[1].h12, params[i].h12, rtol=1e-3)
        end
    end
    
    # =========================================================================
    # SAFE: Concurrent read operations
    # =========================================================================
    
    @testset "Concurrent read operations (SAFE)" begin
        model = create_test_model()
        fitted = fit(model; verbose=false)
        
        # Multiple threads reading fitted model state
        n_reads = 20
        results = Vector{Any}(undef, n_reads)
        
        Threads.@threads for i in 1:n_reads
            op = mod(i, 4)
            if op == 0
                results[i] = get_parameters(fitted)
            elseif op == 1
                results[i] = get_loglik(fitted)
            elseif op == 2
                results[i] = get_vcov(fitted)
            else
                results[i] = aic(fitted)
            end
        end
        
        # All reads should succeed
        @test all(!isnothing, results)
        
        # Same values should be returned (no race conditions)
        params_results = [results[i] for i in 1:n_reads if mod(i, 4) == 0]
        if length(params_results) >= 2
            @test all(p -> p == params_results[1], params_results)
        end
    end
    
    # =========================================================================
    # Documentation: UNSAFE patterns (not tested but documented)
    # =========================================================================
    
    @testset "Documentation of UNSAFE patterns" begin
        # These patterns are UNSAFE and should NOT be used:
        #
        # 1. Parallel fit() on SAME model:
        #    model = create_test_model()
        #    Threads.@threads for i in 1:4
        #        fit(model)  # UNSAFE: mutates shared state
        #    end
        #
        # 2. Concurrent set_parameters! + other operations:
        #    Threads.@threads for i in 1:4
        #        set_parameters!(model, new_params)  # UNSAFE: race condition
        #    end
        #
        # 3. Modifying model during simulation:
        #    Threads.@spawn begin
        #        while running
        #            set_parameters!(model, new_params)  # UNSAFE
        #        end
        #    end
        #    simulate(model; nsim=1000)  # May see inconsistent state
        #
        # SAFE ALTERNATIVE: Always deepcopy() before parallel modification
        
        @test true  # Documentation test always passes
    end
    
    # =========================================================================
    # Stress test: Many parallel simulations
    # =========================================================================
    
    @testset "Stress: many parallel simulations" begin
        model = create_test_model()
        fitted = fit(model; verbose=false)
        
        # Stress test with many parallel simulations
        n_stress = 50
        results = Vector{Int}(undef, n_stress)
        
        Threads.@threads for i in 1:n_stress
            Random.seed!(2000 + i)
            paths = simulate(fitted; nsim=3, data=true)
            results[i] = nrow(paths)
        end
        
        # All should complete with valid results
        @test all(r -> r > 0, results)
    end
    
end  # @testset "Concurrency Tests"
