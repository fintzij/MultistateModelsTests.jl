# =============================================================================
# Unit Tests for Parameter Bounds Generation
# =============================================================================
# Tests the bounds infrastructure for box-constrained optimization.
# 
# Key aspects tested:
# 1. Package bounds generation for each hazard family
# 2. User bounds via Dict API
# 3. Constraint combination (intersection rule)
# 4. Error handling for conflicts and invalid inputs
# =============================================================================

using Test
using MultistateModels
using DataFrames
using Random

@testset "Parameter Bounds" begin
    
    # =========================================================================
    # Test data setup
    # =========================================================================
    Random.seed!(12345)
    n = 100
    
    simple_data = DataFrame(
        id = repeat(1:n, inner=2),
        tstart = repeat([0.0], 2n),
        tstop = repeat([1.0], 2n),
        statefrom = repeat([1], 2n),
        stateto = repeat([2], 2n),
        obstype = repeat([1], 2n)
    )
    
    # Adjust times properly for survival data
    for i in 1:n
        idx1 = 2*(i-1) + 1
        idx2 = 2*(i-1) + 2
        event_time = rand() * 5.0 + 0.1
        final_state = rand() < 0.7 ? 2 : 1  # 70% event, 30% censored
        simple_data[idx1, :tstop] = event_time
        simple_data[idx2, :tstart] = event_time
        simple_data[idx2, :tstop] = event_time + 0.01
        simple_data[idx2, :statefrom] = final_state == 2 ? 2 : 1
        simple_data[idx2, :stateto] = final_state == 2 ? 2 : 1
        simple_data[idx2, :obstype] = 2
    end
    
    # Create data with a covariate
    cov_data = copy(simple_data)
    cov_data.x = randn(nrow(cov_data))
    
    # =========================================================================
    # Package bounds by hazard family
    # =========================================================================
    @testset "Package bounds by family" begin
        
        @testset "Exponential hazard" begin
            h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
            model = multistatemodel(h12; data=simple_data)
            
            lb, ub = generate_parameter_bounds(model)
            
            @test length(lb) == 1
            @test length(ub) == 1
            @test lb[1] == MultistateModels.NONNEG_LB  # Non-negativity: 0.0
            @test ub[1] == Inf
        end
        
        @testset "Weibull hazard" begin
            h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
            model = multistatemodel(h12; data=simple_data)
            
            lb, ub = generate_parameter_bounds(model)
            
            @test length(lb) == 2  # shape, scale
            @test all(lb .== MultistateModels.NONNEG_LB)  # Both non-negative
            @test all(ub .== Inf)
        end
        
        @testset "Gompertz hazard" begin
            h12 = Hazard(@formula(0 ~ 1), "gom", 1, 2)
            model = multistatemodel(h12; data=simple_data)
            
            lb, ub = generate_parameter_bounds(model)
            
            @test length(lb) == 2  # shape, rate
            @test lb[1] == -Inf  # Shape can be negative
            @test lb[2] == MultistateModels.NONNEG_LB  # Rate non-negative
            @test all(ub .== Inf)
        end
        
        @testset "Spline hazard" begin
            h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2)
            model = multistatemodel(h12; data=simple_data)
            calibrate_splines!(model; nknots=5)
            
            lb, ub = generate_parameter_bounds(model)
            n_coefs = model.hazards[1].npar_baseline
            
            @test length(lb) == n_coefs
            @test all(lb .== MultistateModels.NONNEG_LB)  # All spline coefs non-negative
            @test all(ub .== Inf)
        end
        
        @testset "Exponential with covariates" begin
            h12 = Hazard(@formula(0 ~ 1 + x), "exp", 1, 2)
            model = multistatemodel(h12; data=cov_data)
            
            lb, ub = generate_parameter_bounds(model)
            
            @test length(lb) == 2  # rate + 1 covariate
            @test lb[1] == MultistateModels.NONNEG_LB  # Rate non-negative
            @test lb[2] == -Inf  # Covariate coef unconstrained
            @test all(ub .== Inf)
        end
    end
    
    # =========================================================================
    # User bounds (Dict API only)
    # =========================================================================
    @testset "User bounds - Dict API" begin
        h12 = Hazard(@formula(0 ~ 1 + x), "exp", 1, 2)
        model = multistatemodel(h12; data=cov_data)
        
        parnames = MultistateModels._get_flat_parnames(model)
        
        @testset "Single parameter by name (lb and ub)" begin
            lb, ub = generate_parameter_bounds(model;
                user_bounds = Dict(parnames[1] => (lb=0.01, ub=100.0)))
            
            @test lb[1] == 0.01
            @test ub[1] == 100.0
            @test lb[2] == -Inf  # Covariate unchanged
            @test ub[2] == Inf
        end
        
        @testset "Only lower bound" begin
            lb, ub = generate_parameter_bounds(model;
                user_bounds = Dict(parnames[1] => (lb=0.01,)))
            
            @test lb[1] == 0.01
            @test ub[1] == Inf  # Upper unchanged
        end
        
        @testset "Only upper bound" begin
            lb, ub = generate_parameter_bounds(model;
                user_bounds = Dict(parnames[2] => (ub=5.0,)))
            
            @test lb[2] == -Inf  # Lower unchanged
            @test ub[2] == 5.0
        end
        
        @testset "Multiple parameters" begin
            lb, ub = generate_parameter_bounds(model;
                user_bounds = Dict(
                    parnames[1] => (lb=0.01, ub=100.0),
                    parnames[2] => (lb=-5.0, ub=5.0)
                ))
            
            @test lb[1] == 0.01
            @test ub[1] == 100.0
            @test lb[2] == -5.0
            @test ub[2] == 5.0
        end
        
        @testset "Unknown parameter name throws" begin
            @test_throws ArgumentError generate_parameter_bounds(model;
                user_bounds = Dict(:nonexistent_param => (lb=0.0,)))
        end
        
        @testset "String keys converted to Symbol" begin
            parname_str = string(parnames[1])
            lb, ub = generate_parameter_bounds(model;
                user_bounds = Dict(parname_str => (ub=100.0,)))
            @test ub[1] == 100.0
        end
    end
    
    # =========================================================================
    # Constraint combination (intersection rule)
    # =========================================================================
    @testset "Constraint combination" begin
        h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
        model = multistatemodel(h12; data=simple_data)
        
        parnames = MultistateModels._get_flat_parnames(model)
        
        @testset "User tightens lower bound" begin
            lb, ub = generate_parameter_bounds(model;
                user_bounds = Dict(
                    parnames[1] => (lb=0.1,),
                    parnames[2] => (lb=0.2,)
                ))
            
            @test lb[1] == 0.1
            @test lb[2] == 0.2
        end
        
        @testset "User adds upper bound" begin
            lb, ub = generate_parameter_bounds(model;
                user_bounds = Dict(
                    parnames[1] => (ub=10.0,),
                    parnames[2] => (ub=20.0,)
                ))
            
            @test ub[1] == 10.0
            @test ub[2] == 20.0
        end
        
        @testset "User tries to loosen package lb (blocked)" begin
            # User tries lb = -1 for shape, but package requires ≥ 0
            lb, ub = generate_parameter_bounds(model;
                user_bounds = Dict(parnames[1] => (lb=-1.0,)))
            
            # Intersection with package bounds should keep non-negative
            @test lb[1] == MultistateModels.NONNEG_LB
        end
        
        @testset "Conflicting bounds throw error" begin
            # User says lb = 10, ub = 5 → conflict
            @test_throws ArgumentError generate_parameter_bounds(model;
                user_bounds = Dict(parnames[1] => (lb=10.0, ub=5.0)))
        end
    end
    
    # =========================================================================
    # Multi-hazard models
    # =========================================================================
    @testset "Multi-hazard models" begin
        # Illness-death model
        illness_death_data = DataFrame(
            id = repeat(1:50, inner=2),
            tstart = repeat([0.0], 100),
            tstop = repeat([1.0], 100),
            statefrom = repeat([1], 100),
            stateto = repeat([2], 100),
            obstype = repeat([1], 100),
            x = randn(100)
        )
        
        for i in 1:50
            idx1 = 2*(i-1) + 1
            idx2 = 2*(i-1) + 2
            t = rand() * 3.0 + 0.1
            illness_death_data[idx1, :tstop] = t
            illness_death_data[idx2, :tstart] = t
            illness_death_data[idx2, :tstop] = t + 0.01
        end
        
        h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)  # 2 baseline params
        h13 = Hazard(@formula(0 ~ 1 + x), "exp", 1, 3)  # 1 baseline + 1 coef
        h23 = Hazard(@formula(0 ~ 1), "gom", 2, 3)  # 2 baseline params
        
        model = multistatemodel(h12, h13, h23; data=illness_death_data)
        
        lb, ub = generate_parameter_bounds(model)
        
        # Total: 2 (wei) + 2 (exp+cov) + 2 (gom) = 6 params
        @test length(lb) == 6
        
        # Weibull: shape ≥ 0, scale ≥ 0
        @test lb[1] == MultistateModels.NONNEG_LB
        @test lb[2] == MultistateModels.NONNEG_LB
        
        # Exponential rate ≥ 0, covariate unconstrained
        @test lb[3] == MultistateModels.NONNEG_LB
        @test lb[4] == -Inf
        
        # Gompertz: shape ∈ ℝ, rate ≥ 0
        @test lb[5] == -Inf
        @test lb[6] == MultistateModels.NONNEG_LB
    end
    
    # =========================================================================
    # Initial value validation
    # =========================================================================
    @testset "Initial value validation" begin
        lb = [0.0, -1.0, 0.0]
        ub = [10.0, 1.0, Inf]
        
        @testset "Valid initial values" begin
            init = [5.0, 0.0, 1.0]
            @test isnothing(MultistateModels.validate_initial_values(init, lb, ub))
        end
        
        @testset "Violates lower bound" begin
            init = [-1.0, 0.0, 1.0]
            @test_throws ArgumentError MultistateModels.validate_initial_values(init, lb, ub)
        end
        
        @testset "Violates upper bound" begin
            init = [5.0, 2.0, 1.0]  # 2.0 > 1.0 upper bound
            @test_throws ArgumentError MultistateModels.validate_initial_values(init, lb, ub)
        end
    end
    
    # =========================================================================
    # Edge cases
    # =========================================================================
    @testset "Edge cases" begin
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        model = multistatemodel(h12; data=simple_data)
        
        @testset "Empty user bounds (nothing)" begin
            lb, ub = generate_parameter_bounds(model)
            @test lb[1] == MultistateModels.NONNEG_LB
            @test ub[1] == Inf
        end
        
        @testset "user_bounds must be Dict" begin
            # Vector API is no longer supported
            @test_throws ArgumentError generate_parameter_bounds(model;
                user_bounds = [0.01])
        end
    end
    
end  # testset "Parameter Bounds"
