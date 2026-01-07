# =============================================================================
# Direct Tests for Hazard Computation (Internal API)
# =============================================================================
#
# These tests validate the internal hazard computation functions by testing
# eval_hazard() and eval_cumhaz() directly with proper parameter structures.
#
# NOTE: The public API functions compute_hazard() and compute_cumulative_hazard()
# in src/hazard/api.jl have a bug where they pass a plain vector to eval_hazard()
# instead of the expected NamedTuple structure. Until that's fixed, these tests
# use the internal API which is what's actually tested elsewhere in the codebase.
#
# Coverage:
#   1. Direct hazard evaluation for each family (Exp, Wei, Gom)
#   2. Cumulative hazard integration correctness
#   3. Covariate effects (PH)
#   4. Parameter transformation correctness
# =============================================================================

using Test
using MultistateModels
using DataFrames
using StatsModels

import MultistateModels: eval_hazard, eval_cumhaz, get_hazard_params, extract_covariates_fast

# =============================================================================
# Test Fixtures
# =============================================================================

function create_test_model(family::String; with_covariate::Bool=false)
    n = 20
    dat = DataFrame(
        id = 1:n,
        tstart = zeros(n),
        tstop = fill(10.0, n),
        statefrom = ones(Int, n),
        stateto = fill(2, n),
        obstype = fill(1, n),
        x = range(-1, 1, length=n)
    )
    
    formula = with_covariate ? @formula(0 ~ x) : @formula(0 ~ 1)
    h12 = Hazard(formula, family, 1, 2)
    return multistatemodel(h12; data=dat)
end

# =============================================================================
# Exponential Hazard Direct Tests
# =============================================================================

@testset "eval_hazard - Exponential" begin
    
    @testset "Constant hazard (intercept only)" begin
        model = create_test_model("exp")
        λ = 0.2
        set_parameters!(model, (h12 = [λ],))
        
        hazard = model.hazards[1]
        pars = get_hazard_params(model.parameters, model.hazards)[1]
        subjdat_row = model.data[1, :]
        covars = extract_covariates_fast(subjdat_row, hazard.covar_names)
        
        # Exponential is constant in time
        for t in [0.0, 1.0, 5.0, 10.0]
            h = eval_hazard(hazard, t, pars, covars)
            @test isapprox(h, λ; rtol=1e-10)
        end
    end
    
    @testset "With covariate (PH effect)" begin
        model = create_test_model("exp"; with_covariate=true)
        λ = 0.2
        β = 0.5
        set_parameters!(model, (h12 = [λ, β],))
        
        hazard = model.hazards[1]
        pars = get_hazard_params(model.parameters, model.hazards)[1]
        
        # Test different subjects with different x values
        for i in [1, 10, 20]
            subjdat_row = model.data[i, :]
            covars = extract_covariates_fast(subjdat_row, hazard.covar_names)
            x = model.data.x[i]
            
            h = eval_hazard(hazard, 1.0, pars, subjdat_row)
            expected = λ * exp(β * x)
            @test isapprox(h, expected; rtol=1e-6)
        end
    end
end

# =============================================================================
# Weibull Hazard Direct Tests
# =============================================================================

@testset "eval_hazard - Weibull" begin
    
    @testset "Shape < 1: decreasing hazard" begin
        model = create_test_model("wei")
        κ = 0.7  # shape
        λ = 0.2  # rate
        set_parameters!(model, (h12 = [κ, λ],))
        
        hazard = model.hazards[1]
        pars = get_hazard_params(model.parameters, model.hazards)[1]
        subjdat_row = model.data[1, :]
        covars = extract_covariates_fast(subjdat_row, hazard.covar_names)
        
        # h(t) = κ λ t^(κ-1)
        times = [0.5, 1.0, 2.0, 5.0]
        hazards = [eval_hazard(hazard, t, pars, covars) for t in times]
        
        # Decreasing for κ < 1
        @test all(diff(hazards) .< 0)
        
        # Verify formula
        for (i, t) in enumerate(times)
            expected = κ * λ * t^(κ - 1)
            @test isapprox(hazards[i], expected; rtol=1e-6)
        end
    end
    
    @testset "Shape > 1: increasing hazard" begin
        model = create_test_model("wei")
        κ = 1.5
        λ = 0.1
        set_parameters!(model, (h12 = [κ, λ],))
        
        hazard = model.hazards[1]
        pars = get_hazard_params(model.parameters, model.hazards)[1]
        subjdat_row = model.data[1, :]
        covars = extract_covariates_fast(subjdat_row, hazard.covar_names)
        
        times = [0.5, 1.0, 2.0, 5.0]
        hazards = [eval_hazard(hazard, t, pars, covars) for t in times]
        
        # Increasing for κ > 1
        @test all(diff(hazards) .> 0)
        
        for (i, t) in enumerate(times)
            expected = κ * λ * t^(κ - 1)
            @test isapprox(hazards[i], expected; rtol=1e-6)
        end
    end
    
    @testset "Shape = 1: reduces to exponential" begin
        model = create_test_model("wei")
        κ = 1.0
        λ = 0.25
        set_parameters!(model, (h12 = [κ, λ],))
        
        hazard = model.hazards[1]
        pars = get_hazard_params(model.parameters, model.hazards)[1]
        subjdat_row = model.data[1, :]
        covars = extract_covariates_fast(subjdat_row, hazard.covar_names)
        
        # Should be constant
        for t in [0.5, 1.0, 2.0, 5.0]
            h = eval_hazard(hazard, t, pars, covars)
            @test isapprox(h, λ; rtol=1e-6)
        end
    end
end

# =============================================================================
# Gompertz Hazard Direct Tests
# =============================================================================

@testset "eval_hazard - Gompertz" begin
    
    @testset "Shape > 0: increasing hazard" begin
        model = create_test_model("gom")
        a = 0.1   # shape
        b = 0.05  # rate
        set_parameters!(model, (h12 = [a, b],))
        
        hazard = model.hazards[1]
        pars = get_hazard_params(model.parameters, model.hazards)[1]
        subjdat_row = model.data[1, :]
        covars = extract_covariates_fast(subjdat_row, hazard.covar_names)
        
        # h(t) = b exp(at)
        times = [0.0, 1.0, 5.0, 10.0]
        hazards = [eval_hazard(hazard, t, pars, covars) for t in times]
        
        # Increasing for a > 0
        @test all(diff(hazards) .> 0)
        
        for (i, t) in enumerate(times)
            expected = b * exp(a * t)
            @test isapprox(hazards[i], expected; rtol=1e-6)
        end
    end
    
    @testset "Shape < 0: decreasing hazard" begin
        model = create_test_model("gom")
        a = -0.1
        b = 0.2
        set_parameters!(model, (h12 = [a, b],))
        
        hazard = model.hazards[1]
        pars = get_hazard_params(model.parameters, model.hazards)[1]
        subjdat_row = model.data[1, :]
        covars = extract_covariates_fast(subjdat_row, hazard.covar_names)
        
        times = [0.0, 1.0, 5.0, 10.0]
        hazards = [eval_hazard(hazard, t, pars, covars) for t in times]
        
        # Decreasing for a < 0
        @test all(diff(hazards) .< 0)
        
        for (i, t) in enumerate(times)
            expected = b * exp(a * t)
            @test isapprox(hazards[i], expected; rtol=1e-6)
        end
    end
    
    @testset "Shape = 0: reduces to exponential" begin
        model = create_test_model("gom")
        a = 0.0
        b = 0.15
        set_parameters!(model, (h12 = [a, b],))
        
        hazard = model.hazards[1]
        pars = get_hazard_params(model.parameters, model.hazards)[1]
        subjdat_row = model.data[1, :]
        covars = extract_covariates_fast(subjdat_row, hazard.covar_names)
        
        for t in [0.0, 1.0, 5.0, 10.0]
            h = eval_hazard(hazard, t, pars, covars)
            @test isapprox(h, b; rtol=1e-6)
        end
    end
end

# =============================================================================
# Cumulative Hazard Direct Tests
# =============================================================================

@testset "eval_cumhaz - Integration Correctness" begin
    
    @testset "Exponential: H(0,t) = λt" begin
        model = create_test_model("exp")
        λ = 0.3
        set_parameters!(model, (h12 = [λ],))
        
        hazard = model.hazards[1]
        pars = get_hazard_params(model.parameters, model.hazards)[1]
        subjdat_row = model.data[1, :]
        covars = extract_covariates_fast(subjdat_row, hazard.covar_names)
        
        for t in [1.0, 2.0, 5.0, 10.0]
            cumhaz = eval_cumhaz(hazard, 0.0, t, pars, covars)
            expected = λ * t
            @test isapprox(cumhaz, expected; rtol=1e-6)
        end
    end
    
    @testset "Weibull: H(0,t) = λt^κ" begin
        model = create_test_model("wei")
        κ = 1.5
        λ = 0.2
        set_parameters!(model, (h12 = [κ, λ],))
        
        hazard = model.hazards[1]
        pars = get_hazard_params(model.parameters, model.hazards)[1]
        subjdat_row = model.data[1, :]
        covars = extract_covariates_fast(subjdat_row, hazard.covar_names)
        
        for t in [1.0, 2.0, 5.0]
            cumhaz = eval_cumhaz(hazard, 0.0, t, pars, covars)
            expected = λ * t^κ
            @test isapprox(cumhaz, expected; rtol=1e-6)
        end
    end
    
    @testset "Gompertz: H(0,t) = (b/a)(exp(at)-1)" begin
        model = create_test_model("gom")
        a = 0.1
        b = 0.05
        set_parameters!(model, (h12 = [a, b],))
        
        hazard = model.hazards[1]
        pars = get_hazard_params(model.parameters, model.hazards)[1]
        subjdat_row = model.data[1, :]
        covars = extract_covariates_fast(subjdat_row, hazard.covar_names)
        
        for t in [1.0, 2.0, 5.0]
            cumhaz = eval_cumhaz(hazard, 0.0, t, pars, covars)
            expected = (b / a) * (exp(a * t) - 1)
            @test isapprox(cumhaz, expected; rtol=1e-4)
        end
    end
    
    @testset "Cumulative hazard with matching start/stop is zero" begin
        model = create_test_model("exp")
        set_parameters!(model, (h12 = [0.1],))
        
        hazard = model.hazards[1]
        pars = get_hazard_params(model.parameters, model.hazards)[1]
        subjdat_row = model.data[1, :]
        covars = extract_covariates_fast(subjdat_row, hazard.covar_names)
        
        cumhaz = eval_cumhaz(hazard, 5.0, 5.0, pars, covars)
        @test isapprox(cumhaz, 0.0; atol=1e-10)
    end
end

# =============================================================================
# Multi-Transition Model Tests
# =============================================================================

@testset "eval_hazard - Multi-transition Models" begin
    
    @testset "Each hazard computes independently" begin
        n = 30
        dat = DataFrame(
            id = 1:n,
            tstart = zeros(n),
            tstop = fill(10.0, n),
            statefrom = ones(Int, n),
            stateto = ones(Int, n),
            obstype = fill(1, n)
        )
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        h13 = Hazard(@formula(0 ~ 1), "wei", 1, 3)
        h23 = Hazard(@formula(0 ~ 1), "gom", 2, 3)
        
        model = multistatemodel(h12, h13, h23; data=dat)
        
        set_parameters!(model, (
            h12 = [0.1],
            h13 = [1.5, 0.05],
            h23 = [0.1, 0.02]
        ))
        
        subjdat_row = model.data[1, :]
        all_pars = get_hazard_params(model.parameters, model.hazards)
        
        # Exponential at t=1
        haz1 = model.hazards[1]
        cov1 = extract_covariates_fast(subjdat_row, haz1.covar_names)
        h1 = eval_hazard(haz1, 1.0, all_pars[1], cov1)
        @test isapprox(h1, 0.1; rtol=1e-6)
        
        # Weibull at t=1: h(1) = κ λ 1^(κ-1) = 1.5 * 0.05 * 1 = 0.075
        haz2 = model.hazards[2]
        cov2 = extract_covariates_fast(subjdat_row, haz2.covar_names)
        h2 = eval_hazard(haz2, 1.0, all_pars[2], cov2)
        @test isapprox(h2, 1.5 * 0.05; rtol=1e-6)
        
        # Gompertz at t=1: h(1) = b exp(a*1) = 0.02 * exp(0.1)
        haz3 = model.hazards[3]
        cov3 = extract_covariates_fast(subjdat_row, haz3.covar_names)
        h3 = eval_hazard(haz3, 1.0, all_pars[3], cov3)
        @test isapprox(h3, 0.02 * exp(0.1); rtol=1e-6)
    end
end

# =============================================================================
# Edge Cases
# =============================================================================

@testset "eval_hazard - Edge Cases" begin
    
    @testset "Weibull at t=0 with κ>1" begin
        model = create_test_model("wei")
        κ = 1.5
        λ = 0.1
        set_parameters!(model, (h12 = [κ, λ],))
        
        hazard = model.hazards[1]
        pars = get_hazard_params(model.parameters, model.hazards)[1]
        subjdat_row = model.data[1, :]
        covars = extract_covariates_fast(subjdat_row, hazard.covar_names)
        
        h = eval_hazard(hazard, 0.0, pars, covars)
        @test h == 0.0
    end
    
    @testset "Very small rate" begin
        model = create_test_model("exp")
        λ = 1e-8
        set_parameters!(model, (h12 = [λ],))
        
        hazard = model.hazards[1]
        pars = get_hazard_params(model.parameters, model.hazards)[1]
        subjdat_row = model.data[1, :]
        covars = extract_covariates_fast(subjdat_row, hazard.covar_names)
        
        h = eval_hazard(hazard, 1.0, pars, covars)
        @test isfinite(h)
        @test h > 0
    end
end
