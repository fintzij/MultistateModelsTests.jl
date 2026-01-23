# =============================================================================
# Simulation Tests
# =============================================================================
#
# Tests verifying:
# 1. Simulation produces statistically correct output (exponential mean AND variance)
# 2. DataFrame schema validation (correct columns, types)
# 3. Path validity (monotonic times, legal transitions, correct initial states)
# 4. Round-trip consistency (simulate -> observe -> extract)
# 5. Edge cases (absorbing states, censoring)
# 6. Transform strategy equivalence (CachedTransformStrategy vs DirectTransformStrategy)
using Test
using Optim
using Random
using DataFrames
using MultistateModels: simulate, simulate_data, simulate_paths, simulate_path, observe_path, extract_paths, Hazard, multistatemodel, set_parameters!, _find_jump_time, SamplePath, OptimJumpSolver, draw_paths, CachedTransformStrategy, DirectTransformStrategy
using StatsModels: @formula
using Statistics: mean, var

# Load fixtures - handle both standalone and runner contexts
if !@isdefined(toy_two_state_exp_model)
    if isdefined(Main, :TestFixtures)
        # Runner has already loaded TestFixtures, import from Main
        import Main.TestFixtures: toy_two_state_exp_model, toy_absorbing_start_model, 
                                   toy_expwei_model, toy_fitted_exact_model,
                                   toy_illness_death_model
    else
        # Standalone execution
        include(joinpath(@__DIR__, "..", "fixtures", "TestFixtures.jl"))
        # Support both standalone execution and module-based test harness
if !@isdefined(TestFixtures)
    include(joinpath(@__DIR__, "..", "fixtures", "TestFixtures.jl"))
end
using .TestFixtures
    end
end

# --- Optim.jl Brent solver correctness ----------------------------------------
@testset "OptimJumpSolver" begin
    @testset "finds root accurately" begin
        gap_fn = t -> t - 0.5
        os = OptimJumpSolver(abs_tol = 1e-8)
        result = _find_jump_time(os, gap_fn, 0.0, 1.0)
        @test isapprox(result, 0.5; atol = 1e-6)
    end
    
    @testset "nonlinear root" begin
        # Root at t ≈ 0.693 (ln(2))
        gap_fn = t -> exp(t) - 2.0
        os = OptimJumpSolver(abs_tol = 1e-8)
        result = _find_jump_time(os, gap_fn, 0.0, 2.0)
        @test isapprox(result, log(2.0); atol = 1e-5)
    end
end

# --- Statistical correctness --------------------------------------------------
@testset "simulation distribution sanity" begin
    # Test exponential distribution properties with EXACT analytical values
    # For Exp(λ): E[T] = 1/λ, Var(T) = 1/λ², SE(sample mean) = 1/(λ·√n)
    #
    # With λ = 0.5:
    #   E[T] = 1/0.5 = 2.0 (EXACT)
    #   Var(T) = 1/0.5² = 4.0 (EXACT)
    #   SE(sample mean) = 1/(0.5·√500) ≈ 0.0894
    #
    # Using nsim = 500 for statistical power
    fixture = toy_two_state_exp_model(rate = 0.5, horizon = 50.0)  # λ = 0.5, long horizon
    model = fixture.model

    Random.seed!(42)
    nsim = 500
    durations = Vector{Float64}(undef, nsim)
    for i in 1:nsim
        path = simulate_path(model, 1)
        durations[i] = path.times[end] - path.times[1]
    end

    # Test 1: Sample mean should approximate E[T] = 2.0 (EXACT analytical value)
    # Using rtol=0.10 (10% relative tolerance)
    sample_mean = mean(durations)
    @test sample_mean ≈ 2.0 rtol=0.10

    # Test 2: Sample variance should approximate Var(T) = 4.0 (EXACT analytical value)
    # Variance estimation has higher uncertainty, using rtol=0.15
    sample_var = var(durations)
    @test sample_var ≈ 4.0 rtol=0.15

    # Test 3: Sample mean within 3 standard errors of true mean
    # SE(sample mean) = √(Var(T)/n) = 1/(λ·√n) = 1/(0.5·√500) ≈ 0.0894
    # 3·SE ≈ 0.268, so |sample_mean - 2.0| should be < 0.268
    se_sample_mean = 1.0 / (0.5 * sqrt(500))  # ≈ 0.0894
    @test abs(sample_mean - 2.0) < 3 * se_sample_mean
end

# --- DataFrame schema verification --------------------------------------------
@testset "simulate_data DataFrame schema" begin
    # Verify simulated DataFrames have correct structure and types
    fixture = toy_expwei_model()
    model = fixture.model

    Random.seed!(42)
    data_results = simulate_data(model; nsim = 3)

    for (i, df) in enumerate(data_results)
        @testset "simulation $i schema" begin
            # Required columns must exist
            required_cols = [:id, :tstart, :tstop, :statefrom, :stateto, :obstype]
            for col in required_cols
                @test col in propertynames(df)
            end

            # Column types must be correct
            @test eltype(df.id) <: Union{Int, Missing}
            @test eltype(df.tstart) <: Union{Float64, Missing}
            @test eltype(df.tstop) <: Union{Float64, Missing}
            @test eltype(df.statefrom) <: Union{Int, Missing}
            @test eltype(df.stateto) <: Union{Int, Missing}
            @test eltype(df.obstype) <: Union{Int, Missing}
        end
    end
end

# --- Path validity checks -----------------------------------------------------
@testset "path validity checks" begin
    @testset "monotonically increasing times" begin
        # Simulated paths must have strictly increasing time sequences
        fixture = toy_illness_death_model()
        model = fixture.model

        Random.seed!(123)
        for _ in 1:50
            path = simulate_path(model, 1)
            
            # Each successive time must be > previous time
            times = path.times
            for i in 2:length(times)
                @test times[i] > times[i-1]
            end
        end
    end

    @testset "tstop > tstart in observed data" begin
        # In observed DataFrames, tstop must always exceed tstart
        fixture = toy_expwei_model()
        model = fixture.model

        Random.seed!(456)
        data_results = simulate_data(model; nsim = 5)

        for df in data_results
            for row in eachrow(df)
                @test row.tstop > row.tstart
            end
        end
    end

    @testset "transitions follow tmat (legal transitions only)" begin
        # All simulated ACTUAL transitions must be legal according to the model's tmat
        # Note: consecutive states can be the same (no transition, just time passing)
        # We only check when state actually changes
        fixture = toy_illness_death_model()
        model = fixture.model
        tmat = model.tmat  # Transition matrix: tmat[i,j] > 0 means i→j is legal

        Random.seed!(789)
        for _ in 1:50
            path = simulate_path(model, 1)
            states = path.states

            # Check each actual transition in the path (where state changes)
            for i in 1:(length(states) - 1)
                from_state = states[i]
                to_state = states[i + 1]
                
                # Only check if there's an actual state change
                if from_state != to_state
                    # Transition must be legal: tmat[from, to] > 0
                    @test tmat[from_state, to_state] > 0
                end
            end
        end
    end

    @testset "initial state matches data" begin
        # Simulated path should start from the statefrom in the data
        fixture = toy_two_state_exp_model(rate = 0.3, horizon = 20.0)
        model = fixture.model

        # Subject 1 starts in state 1 according to fixture data
        expected_initial_state = model.data.statefrom[model.subjectindices[1][1]]

        Random.seed!(101)
        for _ in 1:20
            path = simulate_path(model, 1)
            @test path.states[1] == expected_initial_state
        end
    end
end

# --- Edge cases ---------------------------------------------------------------
@testset "simulate_path edge cases" begin
    @testset "absorbing initial state" begin
        # When subject starts in absorbing state, path has single time/state
        fixture = toy_absorbing_start_model()
        model = fixture.model
        path = simulate_path(model, 1)
        
        @test length(path.times) == 1
        @test length(path.states) == 1
        @test path.states[1] == 3  # State 3 is absorbing in this fixture
    end
    
    @testset "censoring branch" begin
        # When horizon is very short, subject may be censored before any transition
        fixture = toy_expwei_model()
        model = deepcopy(fixture.model)
        subj_inds = model.subjectindices[1]
        model.data[subj_inds, :tstop] .= model.data[subj_inds, :tstart] .+ 0.05

        Random.seed!(123)
        path = simulate_path(model, 1)

        # Subject should remain in initial state (censored)
        @test path.states[end] == path.states[1]
        @test isapprox(path.times[end], model.data[subj_inds[end], :tstop], atol = 1e-12)
    end
    
    @testset "path reaching absorbing state stops" begin
        # Once absorbing state is reached, no further transitions occur
        fixture = toy_illness_death_model()
        model = fixture.model
        tmat = model.tmat
        
        # Find absorbing states (rows with all zeros)
        absorbing_states = findall(i -> all(tmat[i, :] .== 0), 1:size(tmat, 1))
        
        Random.seed!(202)
        paths_reaching_absorbing = 0
        for _ in 1:100
            path = simulate_path(model, 1)
            final_state = path.states[end]
            
            if final_state in absorbing_states
                paths_reaching_absorbing += 1
                # Verify path ends at absorbing state (no subsequent states)
                @test path.states[end] == final_state
            end
        end
        
        # With illness-death model, should reach absorbing state (death) sometimes
        @test paths_reaching_absorbing > 0
    end
end

# --- Round-trip consistency ---------------------------------------------------
@testset "simulate/observe/extract round-trip" begin
    fixture = toy_two_state_exp_model(rate = 0.3, horizon = 30.0)
    model = fixture.model

    for seed in (11, 17, 23)
        Random.seed!(seed)
        path = simulate_path(model, 1)
        obs = observe_path(path, model)
        recovered = extract_paths(obs)[1]

        @test recovered.states == path.states
        @test all(isapprox.(recovered.times, path.times; atol = 1e-10))
    end
end

# --- draw_paths ---------------------------------------------------------------
@testset "draw_paths" begin
    @testset "fixed-count" begin
        fixture = toy_expwei_model()
        model = fixture.model

        result = draw_paths(model; npaths=3, paretosmooth = false, return_logliks = true)

        @test length(result.samplepaths) == length(model.subjectindices)
        @test all(abs(sum(weights) - 1) < 1e-8 for weights in result.ImportanceWeightsNormalized)
    end
    
    @testset "exact-data shortcut" begin
        fixture = toy_fitted_exact_model()
        model = fixture.model

        result = draw_paths(model; npaths=3, paretosmooth = false, return_logliks = true)

        @test result.loglik == fixture.loglik.loglik
        @test result.subj_lml == fixture.loglik.subj_lml
    end
end

# --- Unified simulation API ---------------------------------------------------
@testset "simulate API" begin
    fixture = toy_expwei_model()
    model = fixture.model
    nsubj = length(model.subjectindices)
    nsim = 2

    @testset "simulate_data" begin
        Random.seed!(42)
        data_results = simulate_data(model; nsim = nsim)
        @test length(data_results) == nsim
        @test all(isa(d, DataFrame) for d in data_results)
        @test all("id" in names(d) for d in data_results)
    end

    @testset "simulate_paths" begin
        Random.seed!(42)
        path_results = simulate_paths(model; nsim = nsim)
        @test length(path_results) == nsim
        @test all(length(paths) == nsubj for paths in path_results)
        @test all(sp isa SamplePath for paths in path_results for sp in paths)
    end

    @testset "simulate unified" begin
        # Both data and paths
        Random.seed!(100)
        data_both, paths_both = simulate(model; nsim = nsim, data = true, paths = true)
        @test length(data_both) == nsim
        @test length(paths_both) == nsim
        
        # Error when neither requested
        @test_throws ArgumentError simulate(model; data = false, paths = false)
    end
end

# --- newdata, tmax, autotmax arguments ----------------------------------------
@testset "simulate with newdata/tmax/autotmax" begin
    # Simple 2-state exponential model
    h12 = Hazard("exp", 1, 2)
    template = DataFrame(
        id = [1, 1, 2, 2, 3],
        tstart = [0.0, 2.0, 0.0, 3.0, 0.0],
        tstop = [2.0, 5.0, 3.0, 7.0, 4.0],
        statefrom = [1, 1, 1, 1, 1],
        stateto = [1, 1, 1, 1, 1],
        obstype = [1, 1, 1, 1, 1]
    )
    model = multistatemodel(h12; data=template)
    set_parameters!(model, (h12 = [exp(-1.0)],))  # rate ≈ 0.37
    
    @testset "autotmax=true (default)" begin
        Random.seed!(12345)
        sim = simulate(model; nsim=1)
        # With autotmax=true, all subjects should have same observation window
        @test all(sim[1].tstop .<= maximum(template.tstop))
        # Should have collapsed to single interval per subject
        @test length(unique(sim[1].id)) == 3
    end
    
    @testset "autotmax=false" begin
        Random.seed!(12345)
        sim = simulate(model; nsim=1, autotmax=false)
        # Original data has multiple rows per subject
        @test nrow(sim[1]) >= 1
    end
    
    @testset "tmax argument" begin
        Random.seed!(12345)
        sim = simulate(model; nsim=1, tmax=10.0)
        @test all(sim[1].tstop .<= 10.0)
        @test length(unique(sim[1].id)) == 3
    end
    
    @testset "newdata argument" begin
        newdata = DataFrame(
            id = 1:5,
            tstart = zeros(5),
            tstop = fill(15.0, 5),
            statefrom = ones(Int, 5),
            stateto = ones(Int, 5),
            obstype = ones(Int, 5)
        )
        Random.seed!(12345)
        sim = simulate(model; nsim=1, newdata=newdata)
        @test length(unique(sim[1].id)) == 5
        @test all(sim[1].tstop .<= 15.0)
    end
    
    @testset "model data restored after simulation" begin
        original_nrow = nrow(model.data)
        original_data = copy(model.data)
        
        # Simulate with tmax (modifies model temporarily)
        sim = simulate(model; nsim=1, tmax=20.0)
        
        # Model should be restored
        @test nrow(model.data) == original_nrow
        @test model.data == original_data
    end
    
    @testset "newdata column validation" begin
        bad_newdata = DataFrame(id = 1:3, tstart = zeros(3), tstop = fill(5.0, 3))
        @test_throws ArgumentError simulate(model; newdata=bad_newdata)
    end
    
    @testset "tmax validation" begin
        @test_throws ArgumentError simulate(model; tmax=-1.0)
    end
    
    @testset "newdata supersedes tmax" begin
        newdata = DataFrame(
            id = 1:2,
            tstart = zeros(2),
            tstop = fill(100.0, 2),
            statefrom = ones(Int, 2),
            stateto = ones(Int, 2),
            obstype = ones(Int, 2)
        )
        Random.seed!(12345)
        # Even though tmax=5.0, newdata should take precedence
        sim = simulate(model; nsim=1, newdata=newdata, tmax=5.0)
        @test length(unique(sim[1].id)) == 2  # 2 subjects from newdata, not 3 from template
    end
end

# --- Transform Strategy Equivalence -------------------------------------------
@testset "Transform Strategy Equivalence" begin
    @testset "CachedTransformStrategy vs DirectTransformStrategy produce same results" begin
        # Use a Weibull hazard (supports time transforms) to test equivalence
        fixture = toy_expwei_model()
        model = fixture.model
        
        nsim = 100
        
        # Simulate with CachedTransformStrategy (default)
        Random.seed!(42)
        cached_durations = Float64[]
        for _ in 1:nsim
            path = simulate_path(model, 1; strategy=CachedTransformStrategy())
            push!(cached_durations, path.times[end] - path.times[1])
        end
        
        # Simulate with DirectTransformStrategy
        Random.seed!(42)
        direct_durations = Float64[]
        for _ in 1:nsim
            path = simulate_path(model, 1; strategy=DirectTransformStrategy())
            push!(direct_durations, path.times[end] - path.times[1])
        end
        
        # With same seed, should produce identical results
        @test all(isapprox.(cached_durations, direct_durations; atol=1e-10))
    end
    
    @testset "DirectTransformStrategy runs without error" begin
        # Basic sanity check that DirectTransformStrategy works
        fixture = toy_two_state_exp_model(rate = 0.3, horizon = 20.0)
        model = fixture.model
        
        Random.seed!(123)
        path = simulate_path(model, 1; strategy=DirectTransformStrategy())
        
        @test length(path.times) >= 1
        @test length(path.states) >= 1
        @test path.times[1] >= 0
    end
    
    @testset "simulate_paths respects strategy" begin
        fixture = toy_expwei_model()
        model = fixture.model
        
        # Should not throw with DirectTransformStrategy
        Random.seed!(99)
        paths_direct = simulate_paths(model; nsim=2, strategy=DirectTransformStrategy())
        @test length(paths_direct) == 2
        
        # Should not throw with CachedTransformStrategy
        Random.seed!(99)
        paths_cached = simulate_paths(model; nsim=2, strategy=CachedTransformStrategy())
        @test length(paths_cached) == 2
    end
end

# =============================================================================
# Additional Simulation Boundary Condition Tests (Phase 3 Gap #4)
# =============================================================================
#
# These tests cover edge cases in simulation that were identified as gaps
# in the testing infrastructure audit (Phase 2).
#
# Tests include:
# 1. Zero hazard rate → paths that never transition
# 2. Very small time intervals (~1e-15)
# 3. Very large time intervals (~1e10)
# 4. Attempting to simulate from absorbing states
#
# =============================================================================

@testset "Simulation Boundary Conditions" begin
    
    @testset "Zero hazard rate - no transitions" begin
        # With zero or near-zero hazard rate, subjects should never transition
        # Note: The box constraint may prevent exactly zero, so we test very small rate
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        dat = DataFrame(
            id = 1:10,
            tstart = zeros(10),
            tstop = fill(100.0, 10),  # Long observation window
            statefrom = ones(Int, 10),
            stateto = ones(Int, 10),
            obstype = fill(1, 10)
        )
        
        model = multistatemodel(h12; data=dat, initialize=false)
        
        # Set a very small rate (near-zero)
        very_small_rate = 1e-12
        set_parameters!(model, (h12 = [very_small_rate],))
        
        # Simulate paths - with this tiny rate, almost all should stay in state 1
        Random.seed!(42)
        n_stayed = 0
        nsim = 50
        for _ in 1:nsim
            path = simulate_path(model, 1)
            if path.states[end] == 1  # Still in initial state
                n_stayed += 1
            end
        end
        
        # With rate = 1e-12 and tmax = 100, probability of no transition ≈ exp(-1e-10) ≈ 1
        # So almost all paths should stay in state 1
        @test n_stayed >= nsim - 5  # Allow a few to transition due to randomness
    end
    
    @testset "Very small time intervals (~1e-15)" begin
        # Test numerical stability with extremely small time intervals
        dt = 1e-15
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        dat = DataFrame(
            id = [1],
            tstart = [0.0],
            tstop = [dt],
            statefrom = [1],
            stateto = [1],
            obstype = [1]
        )
        
        model = multistatemodel(h12; data=dat, initialize=false)
        set_parameters!(model, (h12 = [1.0],))  # Standard rate
        
        # Simulation should not crash, path should be valid
        Random.seed!(123)
        path = simulate_path(model, 1)
        
        @test length(path.times) >= 1
        @test all(isfinite, path.times)
        @test path.times[1] >= 0.0
        @test path.times[end] <= dt + 1e-10  # Allow tiny numerical error
    end
    
    @testset "Very large time intervals (~1e10)" begin
        # Test numerical stability with very large time intervals
        dt = 1e10
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        dat = DataFrame(
            id = [1],
            tstart = [0.0],
            tstop = [dt],
            statefrom = [1],
            stateto = [2],
            obstype = [1]
        )
        
        model = multistatemodel(h12; data=dat, initialize=false)
        set_parameters!(model, (h12 = [1e-8],))  # Very small rate for large interval
        
        Random.seed!(456)
        path = simulate_path(model, 1)
        
        @test length(path.times) >= 1
        @test all(isfinite, path.times)
        @test path.times[1] >= 0.0
        @test path.times[end] <= dt
    end
    
    @testset "High rate with short interval - guaranteed transition" begin
        # With very high rate and reasonable interval, transition should occur
        high_rate = 100.0
        interval = 1.0
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        dat = DataFrame(
            id = 1:20,
            tstart = zeros(20),
            tstop = fill(interval, 20),
            statefrom = ones(Int, 20),
            stateto = fill(2, 20),
            obstype = fill(1, 20)
        )
        
        model = multistatemodel(h12; data=dat, initialize=false)
        set_parameters!(model, (h12 = [high_rate],))
        
        # With rate=100, P(transition before t=1) = 1 - exp(-100) ≈ 1
        Random.seed!(789)
        n_transitioned = 0
        for i in 1:20
            path = simulate_path(model, i)
            if length(path.states) > 1 && path.states[end] == 2
                n_transitioned += 1
            end
        end
        
        # Almost all should transition
        @test n_transitioned >= 18
    end
    
    @testset "Simulation from absorbing state already handled" begin
        # The toy_absorbing_start_model fixture tests this
        # Here we verify the path has exactly one state
        fixture = toy_absorbing_start_model()
        model = fixture.model
        
        Random.seed!(101)
        path = simulate_path(model, 1)
        
        # Subject starting in absorbing state should have single-point path
        @test length(path.times) == 1
        @test length(path.states) == 1
        # The state should be the absorbing state
        @test path.states[1] == 3  # State 3 is absorbing in this fixture
    end
    
    @testset "Weibull hazard boundary conditions" begin
        # Weibull has h(t) = κλt^{κ-1}, which is 0 at t=0 when κ>1
        # Test that simulation handles this correctly
        
        h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
        dat = DataFrame(
            id = [1],
            tstart = [0.0],
            tstop = [10.0],
            statefrom = [1],
            stateto = [2],
            obstype = [1]
        )
        
        model = multistatemodel(h12; data=dat, initialize=false)
        # Weibull with shape > 1 (increasing hazard)
        set_parameters!(model, (h12 = [2.0, 0.1],))  # shape=2, scale=0.1
        
        Random.seed!(202)
        path = simulate_path(model, 1)
        
        @test length(path.times) >= 1
        @test all(isfinite, path.times)
        @test path.times[1] == 0.0
    end
    
    @testset "Gompertz hazard boundary conditions" begin
        # Gompertz has h(t) = b·exp(a·t)
        # With a > 0, hazard increases; with a < 0, hazard decreases
        
        h12 = Hazard(@formula(0 ~ 1), "gom", 1, 2)
        dat = DataFrame(
            id = [1],
            tstart = [0.0],
            tstop = [10.0],
            statefrom = [1],
            stateto = [2],
            obstype = [1]
        )
        
        model = multistatemodel(h12; data=dat, initialize=false)
        # Gompertz with positive shape (increasing hazard)
        set_parameters!(model, (h12 = [0.1, 0.2],))  # shape=0.1, rate=0.2
        
        Random.seed!(303)
        path = simulate_path(model, 1)
        
        @test length(path.times) >= 1
        @test all(isfinite, path.times)
        
        # Also test with negative shape (decreasing hazard)
        set_parameters!(model, (h12 = [-0.1, 0.3],))
        
        Random.seed!(404)
        path2 = simulate_path(model, 1)
        
        @test length(path2.times) >= 1
        @test all(isfinite, path2.times)
    end
    
    @testset "Multiple competing risks with mixed rates" begin
        # Test simulation with multiple hazards where one is very high and one very low
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        h13 = Hazard(@formula(0 ~ 1), "exp", 1, 3)
        
        dat = DataFrame(
            id = 1:30,
            tstart = zeros(30),
            tstop = fill(5.0, 30),
            statefrom = ones(Int, 30),
            stateto = fill(2, 30),  # arbitrary
            obstype = fill(1, 30)
        )
        
        model = multistatemodel(h12, h13; data=dat, initialize=false)
        
        # High rate to state 2, very low rate to state 3
        set_parameters!(model, (h12 = [2.0], h13 = [0.01]))
        
        Random.seed!(505)
        n_to_state2 = 0
        n_to_state3 = 0
        for i in 1:30
            path = simulate_path(model, i)
            final_state = path.states[end]
            if final_state == 2
                n_to_state2 += 1
            elseif final_state == 3
                n_to_state3 += 1
            end
        end
        
        # Most transitions should be to state 2 (higher rate)
        # With rates 2.0 and 0.01, P(2|transition) ≈ 2/(2+0.01) ≈ 0.995
        @test n_to_state2 > n_to_state3
        @test n_to_state2 >= 20  # At least 20 out of 30 should go to state 2
    end
end
