# =============================================================================
# Test suite for Sampling Importance Resampling (SIR) functions
# =============================================================================
#
# Tests for:
# - sir_pool_size: Pool size computation
# - resample_multinomial: Standard SIR resampling
# - resample_lhs: Latin Hypercube Sampling resampling
# - get_sir_subsample_indices: Dispatcher
# - mcem_mll_sir: MCEM marginal log-likelihood with SIR
# - mcem_ase_sir: MCEM asymptotic standard error with SIR

using Test
using MultistateModels
using Random
using Statistics
using LinearAlgebra

@testset "SIR Tests" begin

    @testset "sir_pool_size" begin
        # Basic formula: c * m * log(m)
        @test MultistateModels.sir_pool_size(50, 2.0, 8192) == ceil(Int, 2.0 * 50 * log(50))
        @test MultistateModels.sir_pool_size(100, 2.0, 8192) == ceil(Int, 2.0 * 100 * log(100))
        
        # Max pool cap
        @test MultistateModels.sir_pool_size(1000, 2.0, 500) == 500  # Capped at 500
        
        # Large ESS should be capped
        large_pool = MultistateModels.sir_pool_size(1000, 2.0, 8192)
        @test large_pool == 8192  # ceil(2*1000*log(1000)) ≈ 13816 -> capped
        
        # Different constants
        @test MultistateModels.sir_pool_size(50, 1.0, 8192) == ceil(Int, 1.0 * 50 * log(50))
        @test MultistateModels.sir_pool_size(50, 3.0, 8192) == ceil(Int, 3.0 * 50 * log(50))
        
        # Edge cases
        @test MultistateModels.sir_pool_size(1, 2.0, 8192) == 1  # log(1) = 0, handled
        @test MultistateModels.sir_pool_size(2, 2.0, 8192) == ceil(Int, 2.0 * 2 * log(2))
    end

    @testset "resample_multinomial" begin
        Random.seed!(12345)
        
        # Output length correct
        weights = [0.1, 0.2, 0.3, 0.4]
        indices = MultistateModels.resample_multinomial(weights, 100)
        @test length(indices) == 100
        
        # Indices in valid range
        @test all(1 .<= indices .<= 4)
        
        # Higher weights selected more often (statistical test)
        counts = zeros(Int, 4)
        for idx in indices
            counts[idx] += 1
        end
        @test counts[4] > counts[1]  # Very likely with 100 samples
        
        # Deterministic with seed
        Random.seed!(999)
        idx1 = MultistateModels.resample_multinomial(weights, 50)
        Random.seed!(999)
        idx2 = MultistateModels.resample_multinomial(weights, 50)
        @test idx1 == idx2
        
        # Uniform weights
        Random.seed!(54321)
        uniform_weights = fill(0.25, 4)
        indices_uniform = MultistateModels.resample_multinomial(uniform_weights, 1000)
        counts_uniform = [count(==(i), indices_uniform) for i in 1:4]
        # Each should be ~250, allow 30% tolerance for statistical variation
        @test all(175 .<= counts_uniform .<= 325)
    end

    @testset "resample_lhs" begin
        Random.seed!(12345)
        
        # Output length correct
        weights = [0.1, 0.2, 0.3, 0.4]
        indices = MultistateModels.resample_lhs(weights, 100)
        @test length(indices) == 100
        
        # Indices in valid range
        @test all(1 .<= indices .<= 4)
        
        # Higher weights still selected more often
        counts = zeros(Int, 4)
        for idx in indices
            counts[idx] += 1
        end
        @test counts[4] > counts[1]
        
        # Edge cases
        @test isempty(MultistateModels.resample_lhs(weights, 0))
        
        # Single sample
        single = MultistateModels.resample_lhs(weights, 1)
        @test length(single) == 1
        @test 1 <= single[1] <= 4
    end

    @testset "LHS vs multinomial variance reduction" begin
        # LHS should have lower variance than multinomial resampling
        # when estimating means
        Random.seed!(42)
        
        n_reps = 200
        n_resample = 50
        weights_test = normalize([1.0, 2.0, 3.0, 4.0, 5.0], 1)
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        means_lhs = Float64[]
        means_sir = Float64[]
        
        for _ in 1:n_reps
            idx_lhs = MultistateModels.resample_lhs(weights_test, n_resample)
            idx_sir = MultistateModels.resample_multinomial(weights_test, n_resample)
            push!(means_lhs, mean(values[idx_lhs]))
            push!(means_sir, mean(values[idx_sir]))
        end
        
        # LHS should have lower or similar variance (with some tolerance)
        var_ratio = var(means_lhs) / var(means_sir)
        @test var_ratio <= 1.5  # Allow some slack for sampling variability
    end

    @testset "get_sir_subsample_indices" begin
        Random.seed!(12345)
        weights = normalize(rand(100), 1)
        
        # Dispatches correctly
        idx_sir = MultistateModels.get_sir_subsample_indices(weights, 50, :sir)
        idx_lhs = MultistateModels.get_sir_subsample_indices(weights, 50, :lhs)
        
        @test length(idx_sir) == 50
        @test length(idx_lhs) == 50
        @test all(1 .<= idx_sir .<= 100)
        @test all(1 .<= idx_lhs .<= 100)
        
        # Unknown method throws
        @test_throws ErrorException MultistateModels.get_sir_subsample_indices(weights, 50, :unknown)
    end

    @testset "mcem_mll_sir" begin
        # Test simple average on subsample
        logliks = [
            [-1.0, -2.0, -3.0, -4.0, -5.0],
            [-10.0, -20.0, -30.0, -40.0, -50.0]
        ]
        sir_indices = [
            [1, 3, 5],  # Select indices 1, 3, 5 for subject 1
            [2, 4]      # Select indices 2, 4 for subject 2
        ]
        SubjectWeights = [1.0, 1.0]
        
        # Subject 1: mean([-1, -3, -5]) = -3
        # Subject 2: mean([-20, -40]) = -30
        # Total: -3 + -30 = -33
        result = MultistateModels.mcem_mll_sir(logliks, sir_indices, SubjectWeights)
        @test isapprox(result, -33.0)
        
        # With subject weights
        SubjectWeights_scaled = [2.0, 0.5]
        result_weighted = MultistateModels.mcem_mll_sir(logliks, sir_indices, SubjectWeights_scaled)
        # Subject 1: -3 * 2 = -6
        # Subject 2: -30 * 0.5 = -15
        # Total: -21
        @test isapprox(result_weighted, -21.0)
        
        # Empty indices handled
        sir_indices_empty = [Int[], Int[]]
        result_empty = MultistateModels.mcem_mll_sir(logliks, sir_indices_empty, SubjectWeights)
        @test result_empty == 0.0
    end

    @testset "mcem_ase_sir" begin
        # Test ASE calculation with simple differences
        loglik_prop = [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [10.0, 20.0, 30.0, 40.0, 50.0]
        ]
        loglik_cur = [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [5.0, 10.0, 15.0, 20.0, 25.0]
        ]
        sir_indices = [
            [1, 2, 3],  # Differences: [1, 2, 3], var = 1.0
            [2, 4]      # Differences: [10, 20], var = 50.0
        ]
        SubjectWeights = [1.0, 1.0]
        
        # Subject 1: var/m = 1.0/3 ≈ 0.333
        # Subject 2: var/m = 50.0/2 = 25.0
        # Total variance: 0.333 + 25.0 = 25.333
        # ASE: sqrt(25.333) ≈ 5.03
        result = MultistateModels.mcem_ase_sir(loglik_prop, loglik_cur, sir_indices, SubjectWeights)
        expected = sqrt(var([1.0, 2.0, 3.0])/3 + var([10.0, 20.0])/2)
        @test isapprox(result, expected; rtol=1e-10)
        
        # Single element in subsample -> variance undefined, returns 0 contribution
        sir_indices_single = [
            [1],
            [2]
        ]
        result_single = MultistateModels.mcem_ase_sir(loglik_prop, loglik_cur, sir_indices_single, SubjectWeights)
        @test result_single == 0.0  # No variance with single element
    end

    @testset "create_sir_subsampled_data" begin
        # Create sample paths
        paths1 = [MultistateModels.SamplePath(1, [0.0, 1.0], [1, 2]) for _ in 1:10]
        paths2 = [MultistateModels.SamplePath(2, [0.0, 2.0], [1, 3]) for _ in 1:8]
        samplepaths = [paths1, paths2]
        
        sir_indices = [
            [1, 3, 5, 7],  # 4 paths from subject 1
            [2, 4, 6]      # 3 paths from subject 2
        ]
        
        result = MultistateModels.create_sir_subsampled_data(samplepaths, sir_indices)
        
        # Check correct number of paths
        @test length(result.paths[1]) == 4
        @test length(result.paths[2]) == 3
        
        # Check uniform weights
        @test result.weights[1] == fill(0.25, 4)
        @test result.weights[2] == fill(1/3, 3)
        
        # Weights sum to 1
        @test sum(result.weights[1]) ≈ 1.0
        @test sum(result.weights[2]) ≈ 1.0
        
        # Empty indices handled
        sir_indices_empty = [Int[], Int[]]
        result_empty = MultistateModels.create_sir_subsampled_data(samplepaths, sir_indices_empty)
        @test isempty(result_empty.paths[1])
        @test isempty(result_empty.paths[2])
    end

end
