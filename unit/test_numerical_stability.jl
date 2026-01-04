# =============================================================================
# Numerical Stability Tests for MultistateModels.jl
# =============================================================================
#
# These tests validate numerical stability under extreme parameter values,
# edge cases, and challenging numerical scenarios.
#
# Coverage:
#   1. Extreme parameter values (very small/large rates)
#   2. Large time scales
#   3. Covariate effects at extremes
#   4. Return type correctness (always finite numbers)
# =============================================================================

using Test
using MultistateModels
using DataFrames
using StatsModels

import MultistateModels: eval_hazard, eval_cumhaz, get_hazard_params, extract_covariates_fast

# =============================================================================
# Test Fixtures
# =============================================================================

function create_simple_model(family::String; with_covariate::Bool=false)
    n = 20
    dat = DataFrame(
        id = 1:n,
        tstart = zeros(n),
        tstop = fill(10.0, n),
        statefrom = ones(Int, n),
        stateto = fill(2, n),
        obstype = fill(1, n),
        x = range(-2, 2, length=n)
    )
    
    formula = with_covariate ? @formula(0 ~ x) : @formula(0 ~ 1)
    h12 = Hazard(formula, family, 1, 2)
    return multistatemodel(h12; data=dat)
end

# =============================================================================
# Extreme Parameter Values
# =============================================================================

@testset "Numerical Stability - Extreme Rates" begin
    
    @testset "Very small rates (rare events)" begin
        model = create_simple_model("exp")
        
        for log_rate in [-10.0, -15.0, -18.0]
            set_parameters!(model, (h12 = [log_rate],))
            
            hazard = model.hazards[1]
            pars = get_hazard_params(model.parameters, model.hazards)[1]
            subjdat_row = model.data[1, :]
            covars = extract_covariates_fast(subjdat_row, hazard.covar_names)
            
            h = eval_hazard(hazard, 1.0, pars, covars)
            @test isfinite(h)
            @test h > 0
            @test h ≈ exp(log_rate) rtol=1e-6
        end
    end
    
    @testset "Moderate-large rates" begin
        model = create_simple_model("exp")
        
        for log_rate in [0.0, 1.0, 2.0, 3.0]
            set_parameters!(model, (h12 = [log_rate],))
            
            hazard = model.hazards[1]
            pars = get_hazard_params(model.parameters, model.hazards)[1]
            subjdat_row = model.data[1, :]
            covars = extract_covariates_fast(subjdat_row, hazard.covar_names)
            
            h = eval_hazard(hazard, 1.0, pars, covars)
            @test isfinite(h)
            @test h ≈ exp(log_rate) rtol=1e-6
        end
    end
end

@testset "Numerical Stability - Extreme Weibull Parameters" begin
    
    @testset "Very small shape (heavy early hazard)" begin
        model = create_simple_model("wei")
        κ = 0.1
        λ = 0.1
        set_parameters!(model, (h12 = [log(κ), log(λ)],))
        
        hazard = model.hazards[1]
        pars = get_hazard_params(model.parameters, model.hazards)[1]
        subjdat_row = model.data[1, :]
        covars = extract_covariates_fast(subjdat_row, hazard.covar_names)
        
        # At t > 0, should be finite
        for t in [0.1, 0.5, 1.0]
            h = eval_hazard(hazard, t, pars, covars)
            @test isfinite(h)
            @test h > 0
        end
    end
    
    @testset "Large shape (late hazard concentration)" begin
        model = create_simple_model("wei")
        κ = 5.0
        λ = 0.1
        set_parameters!(model, (h12 = [log(κ), log(λ)],))
        
        hazard = model.hazards[1]
        pars = get_hazard_params(model.parameters, model.hazards)[1]
        subjdat_row = model.data[1, :]
        covars = extract_covariates_fast(subjdat_row, hazard.covar_names)
        
        for t in [0.5, 1.0, 2.0]
            h = eval_hazard(hazard, t, pars, covars)
            @test isfinite(h)
            @test h ≥ 0
        end
    end
end

@testset "Numerical Stability - Extreme Gompertz Parameters" begin
    
    @testset "Large positive shape (rapidly increasing hazard)" begin
        model = create_simple_model("gom")
        a = 1.0  # Large positive shape
        b = 0.01
        set_parameters!(model, (h12 = [a, log(b)],))
        
        hazard = model.hazards[1]
        pars = get_hazard_params(model.parameters, model.hazards)[1]
        subjdat_row = model.data[1, :]
        covars = extract_covariates_fast(subjdat_row, hazard.covar_names)
        
        # At moderate times, should still be finite
        for t in [0.0, 1.0, 2.0]
            h = eval_hazard(hazard, t, pars, covars)
            @test isfinite(h)
            @test h > 0
        end
    end
    
    @testset "Large negative shape (rapidly decreasing hazard)" begin
        model = create_simple_model("gom")
        a = -1.0
        b = 1.0
        set_parameters!(model, (h12 = [a, log(b)],))
        
        hazard = model.hazards[1]
        pars = get_hazard_params(model.parameters, model.hazards)[1]
        subjdat_row = model.data[1, :]
        covars = extract_covariates_fast(subjdat_row, hazard.covar_names)
        
        # Should approach zero but remain finite
        for t in [0.0, 1.0, 5.0]
            h = eval_hazard(hazard, t, pars, covars)
            @test isfinite(h)
            @test h > 0
        end
        
        # Should be decreasing
        h0 = eval_hazard(hazard, 0.0, pars, covars)
        h5 = eval_hazard(hazard, 5.0, pars, covars)
        @test h5 < h0
    end
end

# =============================================================================
# Large Time Scales
# =============================================================================

@testset "Numerical Stability - Large Time Values" begin
    
    @testset "Exponential at large times" begin
        model = create_simple_model("exp")
        λ = 0.1
        set_parameters!(model, (h12 = [log(λ)],))
        
        hazard = model.hazards[1]
        pars = get_hazard_params(model.parameters, model.hazards)[1]
        subjdat_row = model.data[1, :]
        covars = extract_covariates_fast(subjdat_row, hazard.covar_names)
        
        # Hazard should be constant and finite
        for t in [100.0, 500.0, 1000.0]
            h = eval_hazard(hazard, t, pars, covars)
            @test isfinite(h)
            @test h ≈ λ rtol=1e-6
        end
    end
    
    @testset "Weibull at large times" begin
        model = create_simple_model("wei")
        κ = 1.5
        λ = 0.01  # Small rate to keep hazard bounded
        set_parameters!(model, (h12 = [log(κ), log(λ)],))
        
        hazard = model.hazards[1]
        pars = get_hazard_params(model.parameters, model.hazards)[1]
        subjdat_row = model.data[1, :]
        covars = extract_covariates_fast(subjdat_row, hazard.covar_names)
        
        for t in [10.0, 50.0, 100.0]
            h = eval_hazard(hazard, t, pars, covars)
            @test isfinite(h)
            @test h > 0
        end
    end
    
    @testset "Cumulative hazard over large intervals" begin
        model = create_simple_model("exp")
        λ = 0.1
        set_parameters!(model, (h12 = [log(λ)],))
        
        hazard = model.hazards[1]
        pars = get_hazard_params(model.parameters, model.hazards)[1]
        subjdat_row = model.data[1, :]
        covars = extract_covariates_fast(subjdat_row, hazard.covar_names)
        
        # Cumulative hazard should scale linearly
        for t in [10.0, 50.0, 100.0]
            H = eval_cumhaz(hazard, 0.0, t, pars, covars)
            @test isfinite(H)
            @test H ≈ λ * t rtol=1e-6
        end
    end
end

# =============================================================================
# Covariate Effects at Extremes
# =============================================================================

@testset "Numerical Stability - Extreme Covariate Effects" begin
    
    @testset "Large positive covariate coefficient" begin
        model = create_simple_model("exp"; with_covariate=true)
        λ = 0.1
        β = 2.0  # Large positive effect
        set_parameters!(model, (h12 = [log(λ), β],))
        
        hazard = model.hazards[1]
        pars = get_hazard_params(model.parameters, model.hazards)[1]
        
        # Test at extreme covariate values
        for i in [1, 20]  # x ranges from -2 to +2
            subjdat_row = model.data[i, :]
            h = eval_hazard(hazard, 1.0, pars, subjdat_row)
            @test isfinite(h)
            @test h > 0
        end
    end
    
    @testset "Large negative covariate coefficient" begin
        model = create_simple_model("exp"; with_covariate=true)
        λ = 0.1
        β = -2.0  # Large negative effect
        set_parameters!(model, (h12 = [log(λ), β],))
        
        hazard = model.hazards[1]
        pars = get_hazard_params(model.parameters, model.hazards)[1]
        
        for i in [1, 20]
            subjdat_row = model.data[i, :]
            h = eval_hazard(hazard, 1.0, pars, subjdat_row)
            @test isfinite(h)
            @test h > 0
        end
    end
    
    @testset "Covariate effect ordering preserved" begin
        model = create_simple_model("exp"; with_covariate=true)
        λ = 0.1
        β = 1.0
        set_parameters!(model, (h12 = [log(λ), β],))
        
        hazard = model.hazards[1]
        pars = get_hazard_params(model.parameters, model.hazards)[1]
        
        # Hazard should increase with x (β > 0)
        h_low = eval_hazard(hazard, 1.0, pars, model.data[1, :])   # x = -2
        h_high = eval_hazard(hazard, 1.0, pars, model.data[20, :]) # x = +2
        
        @test h_high > h_low
    end
end

# =============================================================================
# Zero Time Behavior
# =============================================================================

@testset "Numerical Stability - Zero Time Edge Cases" begin
    
    @testset "Exponential at t=0" begin
        model = create_simple_model("exp")
        λ = 0.5
        set_parameters!(model, (h12 = [log(λ)],))
        
        hazard = model.hazards[1]
        pars = get_hazard_params(model.parameters, model.hazards)[1]
        subjdat_row = model.data[1, :]
        covars = extract_covariates_fast(subjdat_row, hazard.covar_names)
        
        h = eval_hazard(hazard, 0.0, pars, covars)
        @test isfinite(h)
        @test h ≈ λ
    end
    
    @testset "Gompertz at t=0" begin
        model = create_simple_model("gom")
        a = 0.1
        b = 0.2
        set_parameters!(model, (h12 = [a, log(b)],))
        
        hazard = model.hazards[1]
        pars = get_hazard_params(model.parameters, model.hazards)[1]
        subjdat_row = model.data[1, :]
        covars = extract_covariates_fast(subjdat_row, hazard.covar_names)
        
        h = eval_hazard(hazard, 0.0, pars, covars)
        @test isfinite(h)
        @test h ≈ b  # h(0) = b * exp(0) = b
    end
    
    @testset "Cumulative hazard from 0 to 0" begin
        model = create_simple_model("exp")
        set_parameters!(model, (h12 = [log(0.5)],))
        
        hazard = model.hazards[1]
        pars = get_hazard_params(model.parameters, model.hazards)[1]
        subjdat_row = model.data[1, :]
        covars = extract_covariates_fast(subjdat_row, hazard.covar_names)
        
        H = eval_cumhaz(hazard, 0.0, 0.0, pars, covars)
        @test isfinite(H)
        @test H ≈ 0.0 atol=1e-10
    end
end

# =============================================================================
# Return Type Verification
# =============================================================================

@testset "Numerical Stability - Return Types" begin
    
    @testset "eval_hazard returns Real" begin
        model = create_simple_model("exp")
        set_parameters!(model, (h12 = [log(0.1)],))
        
        hazard = model.hazards[1]
        pars = get_hazard_params(model.parameters, model.hazards)[1]
        subjdat_row = model.data[1, :]
        covars = extract_covariates_fast(subjdat_row, hazard.covar_names)
        
        h = eval_hazard(hazard, 1.0, pars, covars)
        @test isa(h, Real)
        @test isfinite(h)
    end
    
    @testset "eval_cumhaz returns Real" begin
        model = create_simple_model("exp")
        set_parameters!(model, (h12 = [log(0.1)],))
        
        hazard = model.hazards[1]
        pars = get_hazard_params(model.parameters, model.hazards)[1]
        subjdat_row = model.data[1, :]
        covars = extract_covariates_fast(subjdat_row, hazard.covar_names)
        
        H = eval_cumhaz(hazard, 0.0, 1.0, pars, covars)
        @test isa(H, Real)
        @test isfinite(H)
    end
end
