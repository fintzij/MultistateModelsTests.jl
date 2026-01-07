# =============================================================================
# Infrastructure and Utilities Tests
# =============================================================================
#
# Tests for threading utilities, simulation strategies, and other infrastructure.
#
# Coverage:
#   1. get_physical_cores() - CPU detection
#   2. recommended_nthreads() - Thread recommendation logic
#   3. Simulation strategy types - CachedTransformStrategy, DirectTransformStrategy
#   4. Jump solver types - HybridJumpSolver, ExponentialJumpSolver, OptimJumpSolver
# =============================================================================

using Test
using MultistateModels
using DataFrames
using StatsModels

# =============================================================================
# Threading Utilities
# =============================================================================

@testset "Threading Utilities" begin
    
    @testset "get_physical_cores()" begin
        cores = get_physical_cores()
        
        # Basic sanity checks
        @test cores isa Int
        @test cores >= 1  # At least 1 core
        @test cores <= Sys.CPU_THREADS  # Can't exceed total threads
        
        # Should be consistent across calls
        @test get_physical_cores() == cores
    end
    
    @testset "recommended_nthreads()" begin
        # Default call (no task_count)
        n = recommended_nthreads()
        
        @test n isa Int
        @test n >= 1
        @test n <= Threads.nthreads()
        @test n <= get_physical_cores()
        
        # With task_count = 1, should return 1
        @test recommended_nthreads(task_count=1) == 1
        
        # With very large task_count, should be bounded by threads/cores
        n_large = recommended_nthreads(task_count=10000)
        @test n_large <= Threads.nthreads()
        @test n_large <= get_physical_cores()
        
        # With task_count = 2
        n_two = recommended_nthreads(task_count=2)
        @test n_two <= 2
        @test n_two >= 1
        
        # Consistency
        @test recommended_nthreads() == recommended_nthreads()
    end
    
    @testset "Thread count bounds" begin
        # recommended_nthreads should never exceed available threads
        available = Threads.nthreads()
        physical = get_physical_cores()
        
        for task_count in [0, 1, 10, 100, 1000]
            n = recommended_nthreads(task_count=task_count)
            @test n >= 1
            @test n <= available
            @test n <= physical || physical == 0
            if task_count > 0
                @test n <= task_count
            end
        end
    end
end

# =============================================================================
# Simulation Strategy Types
# =============================================================================

@testset "Simulation Strategy Types" begin
    
    @testset "CachedTransformStrategy" begin
        strategy = CachedTransformStrategy()
        @test strategy isa CachedTransformStrategy
        @test strategy isa MultistateModels.AbstractTransformStrategy
    end
    
    @testset "DirectTransformStrategy" begin
        strategy = DirectTransformStrategy()
        @test strategy isa DirectTransformStrategy
        @test strategy isa MultistateModels.AbstractTransformStrategy
    end
    
    @testset "Strategy type distinction" begin
        cached = CachedTransformStrategy()
        direct = DirectTransformStrategy()
        
        @test typeof(cached) != typeof(direct)
        @test cached isa MultistateModels.AbstractTransformStrategy
        @test direct isa MultistateModels.AbstractTransformStrategy
    end
end

# =============================================================================
# Jump Solver Types
# =============================================================================

@testset "Jump Solver Types" begin
    
    @testset "OptimJumpSolver" begin
        solver = MultistateModels.OptimJumpSolver()
        @test solver isa MultistateModels.OptimJumpSolver
        @test solver isa MultistateModels.AbstractJumpSolver
    end
    
    @testset "ExponentialJumpSolver" begin
        solver = MultistateModels.ExponentialJumpSolver()
        @test solver isa MultistateModels.ExponentialJumpSolver
        @test solver isa MultistateModels.AbstractJumpSolver
        
        # Has fallback solver
        @test hasproperty(solver, :fallback)
        @test solver.fallback isa MultistateModels.OptimJumpSolver
    end
    
    @testset "HybridJumpSolver" begin
        solver = HybridJumpSolver()
        @test solver isa HybridJumpSolver
        @test solver isa MultistateModels.AbstractJumpSolver
        
        # Has component solvers
        @test hasproperty(solver, :exp_solver)
        @test hasproperty(solver, :itp_solver)
        @test solver.exp_solver isa MultistateModels.ExponentialJumpSolver
        @test solver.itp_solver isa MultistateModels.OptimJumpSolver
    end
    
    @testset "Custom solver construction" begin
        # HybridJumpSolver with custom components
        custom_optim = MultistateModels.OptimJumpSolver()
        custom_exp = MultistateModels.ExponentialJumpSolver(fallback=custom_optim)
        hybrid = HybridJumpSolver(exp_solver=custom_exp, itp_solver=custom_optim)
        
        @test hybrid.exp_solver === custom_exp
        @test hybrid.itp_solver === custom_optim
    end
end

# =============================================================================
# Simulation with Different Strategies
# =============================================================================

@testset "Simulation Strategy Integration" begin
    
    # Create a simple model for testing
    n = 30
    dat = DataFrame(
        id = 1:n,
        tstart = zeros(n),
        tstop = fill(10.0, n),
        statefrom = ones(Int, n),
        stateto = fill(2, n),
        obstype = fill(1, n)
    )
    
    h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    model = multistatemodel(h12; data=dat)
    set_parameters!(model, (h12 = [0.1],))
    
    @testset "simulate() with CachedTransformStrategy" begin
        result = simulate(model; nsim=5, strategy=CachedTransformStrategy())
        @test length(result) == 5
        @test all(r -> r isa DataFrame, result)
    end
    
    @testset "simulate() with DirectTransformStrategy" begin
        result = simulate(model; nsim=5, strategy=DirectTransformStrategy())
        @test length(result) == 5
        @test all(r -> r isa DataFrame, result)
    end
    
    @testset "simulate_paths() with different strategies" begin
        # Using cached strategy
        paths_cached = simulate_paths(model; nsim=3, strategy=CachedTransformStrategy())
        @test length(paths_cached) == 3
        
        # Using direct strategy
        paths_direct = simulate_paths(model; nsim=3, strategy=DirectTransformStrategy())
        @test length(paths_direct) == 3
    end
    
    @testset "Strategies produce equivalent structure" begin
        # Both strategies should produce DataFrames with same columns
        result_cached = simulate(model; nsim=2, strategy=CachedTransformStrategy())
        result_direct = simulate(model; nsim=2, strategy=DirectTransformStrategy())
        
        @test names(result_cached[1]) == names(result_direct[1])
    end
end

# =============================================================================
# HybridJumpSolver Selection Logic
# =============================================================================

@testset "HybridJumpSolver Integration" begin
    
    @testset "Exponential model uses exponential solver path" begin
        # Pure exponential model - should use fast path
        n = 20
        dat = DataFrame(
            id = 1:n,
            tstart = zeros(n),
            tstop = fill(5.0, n),
            statefrom = ones(Int, n),
            stateto = fill(2, n),
            obstype = fill(1, n)
        )
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        model = multistatemodel(h12; data=dat)
        set_parameters!(model, (h12 = [0.2],))
        
        # Simulate with HybridJumpSolver (via default solver selection)
        result = simulate(model; nsim=5)
        @test length(result) == 5
    end
    
    @testset "Non-exponential model uses ITP solver path" begin
        # Weibull model - requires root finding
        n = 20
        dat = DataFrame(
            id = 1:n,
            tstart = zeros(n),
            tstop = fill(5.0, n),
            statefrom = ones(Int, n),
            stateto = fill(2, n),
            obstype = fill(1, n)
        )
        
        h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
        model = multistatemodel(h12; data=dat)
        set_parameters!(model, (h12 = [1.5, 0.1],))
        
        result = simulate(model; nsim=5)
        @test length(result) == 5
    end
    
    @testset "Mixed model uses hybrid selection" begin
        # Model with both exponential and non-exponential hazards
        n = 20
        dat = DataFrame(
            id = 1:n,
            tstart = zeros(n),
            tstop = fill(5.0, n),
            statefrom = ones(Int, n),
            stateto = ones(Int, n),
            obstype = fill(1, n)
        )
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)  # Exponential
        h13 = Hazard(@formula(0 ~ 1), "wei", 1, 3)  # Weibull
        model = multistatemodel(h12, h13; data=dat)
        set_parameters!(model, (h12 = [0.1], h13 = [1.5, 0.05]))
        
        result = simulate(model; nsim=5)
        @test length(result) == 5
    end
end
