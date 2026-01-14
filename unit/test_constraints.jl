# =============================================================================
# Parameter Constraints Unit Tests
# =============================================================================
#
# Tests for make_constraints() and parse_constraints() functions:
# 1. Basic functionality - creating valid constraints
# 2. Input validation - catching invalid inputs
# 3. Integration with parse_constraints - function generation
# 4. End-to-end integration with fit() - constrained optimization
# 5. Error handling - graceful failures

# Handle both standalone and suite execution
if !@isdefined(TestFixtures)
    using Test
    using MultistateModels
    using DataFrames
    using StatsModels
    using LinearAlgebra
    include(joinpath(@__DIR__, "..", "fixtures", "TestFixtures.jl"))
end
# Support both standalone execution and module-based test harness
if !@isdefined(TestFixtures)
    include(joinpath(@__DIR__, "..", "fixtures", "TestFixtures.jl"))
end
using .TestFixtures
using Random
import Distributions

# Import internal functions for testing
using MultistateModels: make_constraints, parse_constraints

# =============================================================================
# Test Data Generators
# =============================================================================

"""Generate simple two-state reversible data with exact observation times."""
function generate_two_state_exact_data(; n_subj=20, true_rate_12=0.5, true_rate_21=0.3, 
                                        max_time=10.0, seed=12345)
    Random.seed!(seed)
    
    rows = Vector{NamedTuple{(:id, :tstart, :tstop, :statefrom, :stateto, :obstype), 
                              Tuple{Int, Float64, Float64, Int, Int, Int}}}()
    
    for subj in 1:n_subj
        t = 0.0
        state = 1
        obs_id = subj
        
        while t < max_time
            # Get rate for leaving current state
            rate = state == 1 ? true_rate_12 : true_rate_21
            
            # Draw sojourn time
            sojourn = rand(Distributions.Exponential(1.0 / rate))
            
            if t + sojourn >= max_time
                # Right-censored at max_time - use obstype=2 (panel) with statefrom=stateto
                # This indicates we observed the state at the end but don't know what happened after
                push!(rows, (id=obs_id, tstart=t, tstop=max_time, statefrom=state, 
                            stateto=state, obstype=2))
                break
            else
                # Transition occurred - exact observation
                next_state = state == 1 ? 2 : 1
                push!(rows, (id=obs_id, tstart=t, tstop=t + sojourn, statefrom=state, 
                            stateto=next_state, obstype=1))
                t = t + sojourn
                state = next_state
            end
        end
    end
    
    return DataFrame(rows)
end

# =============================================================================
# Basic Functionality Tests
# =============================================================================

@testset "make_constraints basic functionality" begin
    
    @testset "Single constraint with matching lengths" begin
        # Using generic parameter names - actual model names are h12_rate, h21_rate
        cons = make_constraints(
            cons = [:(h12_rate - h21_rate)],
            lcons = [0.0],
            ucons = [0.0]
        )
        
        # Verify returned structure
        @test cons isa NamedTuple
        @test haskey(cons, :cons)
        @test haskey(cons, :lcons)
        @test haskey(cons, :ucons)
        
        # Verify lengths
        @test length(cons.cons) == 1
        @test length(cons.lcons) == 1
        @test length(cons.ucons) == 1
        
        # Verify content
        @test cons.cons[1] == :(h12_rate - h21_rate)
        @test cons.lcons[1] == 0.0
        @test cons.ucons[1] == 0.0
    end
    
    @testset "Multiple constraints" begin
        cons = make_constraints(
            cons = Expr[:(h12_rate - h21_rate), :(h12_rate + 0)],
            lcons = [0.0, -1.0],
            ucons = [0.0, 0.5]
        )
        
        @test length(cons.cons) == 2
        @test length(cons.lcons) == 2
        @test length(cons.ucons) == 2
        
        @test cons.lcons == [0.0, -1.0]
        @test cons.ucons == [0.0, 0.5]
    end
    
    @testset "Empty vectors (no constraints)" begin
        cons = make_constraints(
            cons = Expr[],
            lcons = Float64[],
            ucons = Float64[]
        )
        
        @test length(cons.cons) == 0
        @test length(cons.lcons) == 0
        @test length(cons.ucons) == 0
    end
    
    @testset "Inequality constraints (bounds)" begin
        # Constraint: param >= -2.0 (lower bound)
        # Constraint: param <= 1.0 (upper bound)
        cons = make_constraints(
            cons = Expr[:(h12_rate + 0), :(h12_rate + 0)],
            lcons = [-2.0, -Inf],
            ucons = [Inf, 1.0]
        )
        
        @test cons.lcons[1] == -2.0
        @test cons.ucons[1] == Inf
        @test cons.lcons[2] == -Inf
        @test cons.ucons[2] == 1.0
    end
end

# =============================================================================
# Input Validation Tests
# =============================================================================

@testset "make_constraints input validation" begin
    
    @testset "ArgumentError when cons, lcons, ucons have different lengths" begin
        # cons longer than lcons/ucons (using expressions to ensure Expr type)
        @test_throws ArgumentError make_constraints(
            cons = Expr[:(h12_rate + 0), :(h21_rate + 0)],
            lcons = [0.0],
            ucons = [0.0]
        )
        
        # lcons longer
        @test_throws ArgumentError make_constraints(
            cons = Expr[:(h12_rate + 0)],
            lcons = [0.0, 1.0],
            ucons = [0.0]
        )
        
        # ucons longer
        @test_throws ArgumentError make_constraints(
            cons = Expr[:(h12_rate + 0)],
            lcons = [0.0],
            ucons = [0.0, 1.0]
        )
    end
    
    @testset "Error message includes lengths" begin
        err = try
            make_constraints(
                cons = Expr[:(a + 0), :(b + 0), :(c + 0)],
                lcons = [0.0, 1.0],
                ucons = [0.0]
            )
            nothing
        catch e
            e
        end
        
        @test err isa ArgumentError
        @test occursin("3", err.msg)  # cons length
        @test occursin("2", err.msg)  # lcons length
        @test occursin("1", err.msg)  # ucons length
    end
end

# =============================================================================
# parse_constraints Integration Tests
# =============================================================================

@testset "parse_constraints function generation" begin
    
    @testset "Constraint expressions are correctly parsed" begin
        # Create a simple model to get hazard objects
        dat = DataFrame(
            id = [1, 2],
            tstart = [0.0, 0.0],
            tstop = [1.0, 2.0],
            statefrom = [1, 1],
            stateto = [2, 2],
            obstype = [1, 1]
        )
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        h21 = Hazard(@formula(0 ~ 1), "exp", 2, 1)
        model = multistatemodel(h12, h21; data=dat)
        
        # Parse equality constraint using actual parameter names: h12_rate - h21_rate = 0
        cons_expr = [:(h12_rate - h21_rate)]
        consfun = parse_constraints(deepcopy(cons_expr), model.hazards)
        
        # The generated function should be callable
        @test consfun isa Function
        
        # Test the function output
        res = zeros(1)
        params = [log(0.5), log(0.3)]  # h12_rate, h21_rate (log scale)
        consfun(res, params, nothing)
        
        # Should equal log(0.5) - log(0.3)
        @test res[1] ≈ log(0.5) - log(0.3) rtol=1e-10
    end
    
    @testset "Parameter names are correctly substituted with indices" begin
        dat = DataFrame(
            id = [1],
            tstart = [0.0],
            tstop = [1.0],
            statefrom = [1],
            stateto = [2],
            obstype = [1]
        )
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        h21 = Hazard(@formula(0 ~ 1), "exp", 2, 1)
        model = multistatemodel(h12, h21; data=dat)
        
        # Constraint that uses both parameters with actual names
        cons_expr = [:(2 * h12_rate + h21_rate)]
        consfun = parse_constraints(deepcopy(cons_expr), model.hazards)
        
        res = zeros(1)
        params = [1.0, 2.0]  # h12_rate=1, h21_rate=2
        consfun(res, params, nothing)
        
        @test res[1] ≈ 2 * 1.0 + 2.0  # = 4.0
    end
    
    @testset "Multiple constraints generate correct function" begin
        dat = DataFrame(
            id = [1],
            tstart = [0.0],
            tstop = [1.0],
            statefrom = [1],
            stateto = [2],
            obstype = [1]
        )
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        h21 = Hazard(@formula(0 ~ 1), "exp", 2, 1)
        model = multistatemodel(h12, h21; data=dat)
        
        # Two constraints with actual parameter names
        cons_expr = [:(h12_rate - h21_rate), :(h12_rate + h21_rate)]
        consfun = parse_constraints(deepcopy(cons_expr), model.hazards)
        
        res = zeros(2)
        params = [0.5, 0.3]  # h12_rate, h21_rate
        consfun(res, params, nothing)
        
        @test res[1] ≈ 0.5 - 0.3  # = 0.2
        @test res[2] ≈ 0.5 + 0.3  # = 0.8
    end
end

# =============================================================================
# End-to-End Integration with fit()
# =============================================================================

@testset "Constraints integration with fit()" begin
    
    # NOTE: As of v0.3.0+, all parameters are stored on NATURAL scale.
    # Constraints operate directly on natural-scale parameter values.
    # Box constraints handle non-negativity (rates must be >= 0).
    
    @testset "Equality constraint - rates equal" begin
        # Generate data with equal rates (use larger rates to stay away from bounds)
        dat = generate_two_state_exact_data(n_subj=15, true_rate_12=1.5, true_rate_21=1.5, 
                                            max_time=8.0, seed=42)
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        h21 = Hazard(@formula(0 ~ 1), "exp", 2, 1)
        
        # Equality constraint on natural scale: h12_rate = h21_rate
        cons = make_constraints(
            cons = [:(h12_rate - h21_rate)],
            lcons = [0.0],
            ucons = [0.0]
        )
        
        model = multistatemodel(h12, h21; data=dat)
        # Set initial parameters that satisfy the equality constraint (natural scale)
        set_parameters!(model, (h12 = [1.5], h21 = [1.5]))
        fitted = fit(model; constraints=cons, verbose=false)
        
        # Get fitted parameters on natural scale
        params = get_parameters(fitted; scale=:natural)
        
        # Verify constraint is satisfied: rates should be equal
        @test isapprox(params.h12[1], params.h21[1]; rtol=1e-4)
    end
    
    @testset "Inequality constraint - bounded parameter" begin
        dat = generate_two_state_exact_data(n_subj=15, true_rate_12=1.5, true_rate_21=1.0, 
                                            max_time=8.0, seed=123)
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        h21 = Hazard(@formula(0 ~ 1), "exp", 2, 1)
        
        # Constraint on natural scale: h12_rate <= 2.0
        cons = make_constraints(
            cons = Expr[:(h12_rate + 0)],
            lcons = [0.0],  # Must be non-negative (box constraint already enforces this)
            ucons = [2.0]   # Upper bound on rate
        )
        
        model = multistatemodel(h12, h21; data=dat)
        # Set initial parameters that satisfy the inequality constraint
        set_parameters!(model, (h12 = [1.5], h21 = [1.0]))
        fitted = fit(model; constraints=cons, verbose=false)
        
        params = get_parameters(fitted; scale=:natural)
        
        # Verify constraint: h12_rate <= 2.0
        @test params.h12[1] <= 2.0 + 1e-6
    end
    
    @testset "Constraint preserves valid optimization" begin
        dat = generate_two_state_exact_data(n_subj=20, true_rate_12=1.5, true_rate_21=1.0, 
                                            max_time=10.0, seed=456)
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        h21 = Hazard(@formula(0 ~ 1), "exp", 2, 1)
        
        # Light constraint that should not affect fit much
        # Lower bound: h12_rate >= 0.01 (very weak constraint)
        cons = make_constraints(
            cons = Expr[:(h12_rate + 0)],
            lcons = [0.01],
            ucons = [Inf]
        )
        
        model = multistatemodel(h12, h21; data=dat)
        fitted = fit(model; constraints=cons, verbose=false)
        
        # Should complete without error and have reasonable fit
        params = get_parameters(fitted; scale=:natural)
        @test params.h12[1] > 0.0  # Rate is positive
        @test params.h21[1] > 0.0
    end
    
    @testset "Multiple constraints simultaneously" begin
        dat = generate_two_state_exact_data(n_subj=15, true_rate_12=1.5, true_rate_21=1.5, 
                                            max_time=8.0, seed=789)
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        h21 = Hazard(@formula(0 ~ 1), "exp", 2, 1)
        
        # Equality + bound constraint on natural scale
        # Constraint 1: h12_rate = h21_rate (rates equal)
        # Constraint 2: h12_rate >= 0.1 (rate >= 0.1)
        cons = make_constraints(
            cons = Expr[:(h12_rate - h21_rate), :(h12_rate + 0)],
            lcons = [0.0, 0.1],
            ucons = [0.0, Inf]
        )
        
        model = multistatemodel(h12, h21; data=dat)
        # Set initial parameters that satisfy both constraints
        set_parameters!(model, (h12 = [1.5], h21 = [1.5]))
        fitted = fit(model; constraints=cons, verbose=false)
        
        params = get_parameters(fitted; scale=:natural)
        
        # Equality satisfied
        @test isapprox(params.h12[1], params.h21[1]; rtol=1e-4)
        # Lower bound satisfied
        @test params.h12[1] >= 0.1 - 1e-6
    end
end

# =============================================================================
# Error Handling Tests
# =============================================================================

@testset "Constraint error handling" begin
    
    @testset "Initial values violating constraint throws error" begin
        dat = DataFrame(
            id = [1, 2],
            tstart = [0.0, 0.0],
            tstop = [1.0, 2.0],
            statefrom = [1, 1],
            stateto = [2, 2],
            obstype = [1, 1]
        )
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        h21 = Hazard(@formula(0 ~ 1), "exp", 2, 1)
        
        # Constraint that initial values (default around 0) will violate
        # Requires h12_rate = 10.0 (which default is far from)
        cons = make_constraints(
            cons = Expr[:(h12_rate + 0)],
            lcons = [10.0],
            ucons = [10.0]
        )
        
        model = multistatemodel(h12, h21; data=dat)
        
        # Should throw ArgumentError about violated constraints
        @test_throws ArgumentError fit(model; constraints=cons, verbose=false)
    end
    
    @testset "Vcov IS returned when constraints provided (Item #27)" begin
        # Use larger rates to stay well within bounds
        dat = generate_two_state_exact_data(n_subj=10, true_rate_12=1.5, true_rate_21=1.0, 
                                            max_time=5.0, seed=111)
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        h21 = Hazard(@formula(0 ~ 1), "exp", 2, 1)
        
        # Equality constraint on natural scale
        cons = make_constraints(
            cons = [:(h12_rate - h21_rate)],
            lcons = [0.0],
            ucons = [0.0]
        )
        
        model = multistatemodel(h12, h21; data=dat)
        # Set initial parameters satisfying constraint
        set_parameters!(model, (h12 = [1.25], h21 = [1.25]))
        
        # Fit with constraints
        fitted = fit(model; constraints=cons, verbose=false, compute_vcov=true)
        
        # Item #27: Variance SHOULD be computed even with constraints (reduced Hessian approach)
        vcov = get_vcov(fitted; type=:model)
        @test !isnothing(vcov)
        @test size(vcov) == (2, 2)  # 2 parameters total
        @test issymmetric(vcov)
        
        # IJ variance should also be computable with constraints
        vcov_ij = get_vcov(fitted; type=:ij)
        @test !isnothing(vcov_ij)
    end
end

# =============================================================================
# Constraints with Covariates
# =============================================================================

@testset "Constraints with covariates" begin
    
    @testset "Constraint on covariate coefficient" begin
        Random.seed!(222)
        n_subj = 20
        
        # Generate data with a covariate
        # Use obstype=1 for transitions and obstype=2 for censored (panel observation)
        rows = Vector{NamedTuple{(:id, :tstart, :tstop, :statefrom, :stateto, :obstype, :x), 
                                  Tuple{Int, Float64, Float64, Int, Int, Int, Float64}}}()
        
        for subj in 1:n_subj
            x = rand() > 0.5 ? 1.0 : 0.0
            # Use larger base rate to stay within bounds
            true_rate = 1.5 * exp(0.3 * x)
            sojourn = rand(Distributions.Exponential(1.0 / true_rate))
            if sojourn < 5.0
                # Exact observation of transition
                push!(rows, (id=subj, tstart=0.0, tstop=sojourn, 
                            statefrom=1, stateto=2, obstype=1, x=x))
            else
                # Right-censored: use obstype=2 (panel) with statefrom=stateto
                push!(rows, (id=subj, tstart=0.0, tstop=5.0, 
                            statefrom=1, stateto=1, obstype=2, x=x))
            end
        end
        
        dat = DataFrame(rows)
        
        h12 = Hazard(@formula(0 ~ 1 + x), "exp", 1, 2)
        model = multistatemodel(h12; data=dat)
        
        # Check the actual parameter names used by the model
        # Should be something like h12_rate, h12_x
        parnames = model.hazards[1].parnames
        coef_name = parnames[2]  # The covariate coefficient
        
        # Constraint: coefficient on x is zero (no covariate effect)
        cons = make_constraints(
            cons = Expr[:($(coef_name) + 0)],
            lcons = [0.0],
            ucons = [0.0]
        )
        
        fitted = fit(model; constraints=cons, verbose=false)
        params = get_parameters(fitted; scale=:natural)
        
        # The covariate coefficient should be zero
        # params is a NamedTuple with params.h12 being a vector [rate, coef_x]
        @test isapprox(params.h12[2], 0.0; atol=1e-5)
    end
end

# Print completion message when run standalone
if abspath(PROGRAM_FILE) == @__FILE__
    println("\n✓ All constraint tests completed")
end
