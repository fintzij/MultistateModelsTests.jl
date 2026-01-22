# =============================================================================
# Phase-Type Parameter Round-Trip Tests
# =============================================================================
#
# Test that get_parameters → set_parameters! → get_parameters cycle
# preserves parameter values for phase-type models.
#
# This is H12_P2 from the codebase audit: the mapping between user-facing
# parameters (λ, μ) and internal expanded representation must be consistent.
#
# =============================================================================

using Test
using MultistateModels
using DataFrames
using LinearAlgebra

import MultistateModels: get_parameters_flat, get_parameters_nested, 
    get_parameters_natural, set_parameters!, has_phasetype_expansion

@testset "Phase-Type Parameter Round-Trip" begin
    
    @testset "Simple 2-state phase-type model" begin
        # Create model with phase-type hazard
        h12 = Hazard(@formula(0 ~ 1), "pt", 1, 2; n_phases=2)
        
        dat = DataFrame(
            id = [1, 2, 3],
            tstart = [0.0, 0.0, 0.0],
            tstop = [5.0, 3.0, 7.0],
            statefrom = [1, 1, 1],
            stateto = [2, 2, 2],
            obstype = [1, 1, 1]
        )
        
        model = multistatemodel(h12; data=dat, initialize=true)
        
        # Skip if model doesn't have phase-type expansion
        if !has_phasetype_expansion(model)
            @test_skip "Phase-type expansion not available"
        else
            # Get initial parameters
            params_flat_1 = get_parameters_flat(model)
            params_nested_1 = get_parameters_nested(model)
            
            # Set parameters to known values
            # Phase-type has progression (λ) and absorption (μ) rates
            new_params = copy(params_flat_1)
            new_params .= abs.(randn(length(new_params))) .+ 0.1  # Positive values
            
            # Set using flat representation
            set_parameters!(model, new_params)
            
            # Get parameters back
            params_flat_2 = get_parameters_flat(model)
            
            # Round-trip should preserve values
            @test params_flat_2 ≈ new_params atol=1e-10
            
            # Get nested parameters - verify structure preserved
            params_nested_2 = get_parameters_nested(model)
            @test length(params_nested_2) == length(params_nested_1)
            
            # Additional round-trip via nested representation
            set_parameters!(model, params_nested_2)
            params_flat_3 = get_parameters_flat(model)
            @test params_flat_3 ≈ params_flat_2 atol=1e-10
        end
    end
    
    @testset "Multi-transition phase-type model" begin
        # Illness-death model with phase-type hazards
        h12 = Hazard(@formula(0 ~ 1), "pt", 1, 2; n_phases=2)
        h23 = Hazard(@formula(0 ~ 1), "pt", 2, 3; n_phases=3)
        
        dat = DataFrame(
            id = [1, 1, 2, 2, 3],
            tstart = [0.0, 2.0, 0.0, 3.0, 0.0],
            tstop = [2.0, 5.0, 3.0, 6.0, 4.0],
            statefrom = [1, 2, 1, 2, 1],
            stateto = [2, 3, 2, 3, 2],
            obstype = [1, 1, 1, 1, 1]
        )
        
        model = multistatemodel(h12, h23; data=dat, initialize=true)
        
        if !has_phasetype_expansion(model)
            @test_skip "Phase-type expansion not available"
        else
            # Get initial parameters
            params_flat_1 = get_parameters_flat(model)
            
            # Modify parameters
            new_params = copy(params_flat_1)
            new_params .*= 1.5  # Scale up by 50%
            
            set_parameters!(model, new_params)
            params_flat_2 = get_parameters_flat(model)
            
            # Should match within numerical precision
            @test params_flat_2 ≈ new_params atol=1e-10
            
            # Test idempotence: set_parameters twice should be stable
            set_parameters!(model, params_flat_2)
            params_flat_3 = get_parameters_flat(model)
            @test params_flat_3 ≈ params_flat_2 atol=1e-10
        end
    end
    
    @testset "Phase-type with covariates" begin
        # Phase-type hazard with covariate effects
        h12 = Hazard(@formula(0 ~ 1 + age), "pt", 1, 2; n_phases=2)
        
        dat = DataFrame(
            id = [1, 2, 3, 4],
            tstart = [0.0, 0.0, 0.0, 0.0],
            tstop = [5.0, 3.0, 7.0, 4.0],
            statefrom = [1, 1, 1, 1],
            stateto = [2, 2, 2, 2],
            obstype = [1, 1, 1, 1],
            age = [30.0, 50.0, 40.0, 60.0]
        )
        
        model = multistatemodel(h12; data=dat, initialize=true)
        
        if !has_phasetype_expansion(model)
            @test_skip "Phase-type expansion not available"
        else
            # Get parameters
            params_flat_1 = get_parameters_flat(model)
            params_nested_1 = get_parameters_nested(model)
            
            # Set to new values (baseline + covariate effect)
            new_params = abs.(randn(length(params_flat_1))) .+ 0.1
            set_parameters!(model, new_params)
            
            # Verify round-trip
            params_flat_2 = get_parameters_flat(model)
            @test params_flat_2 ≈ new_params atol=1e-10
            
            # Nested round-trip
            params_nested_2 = get_parameters_nested(model)
            set_parameters!(model, params_nested_2)
            params_flat_3 = get_parameters_flat(model)
            @test params_flat_3 ≈ params_flat_2 atol=1e-10
        end
    end
end

@testset "Natural Parameter Round-Trip" begin
    # Test get_parameters_natural if available
    
    @testset "Parametric hazard natural parameters" begin
        # Weibull has natural parameters [κ, λ]
        h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
        
        dat = DataFrame(
            id = [1, 2, 3],
            tstart = [0.0, 0.0, 0.0],
            tstop = [5.0, 3.0, 7.0],
            statefrom = [1, 1, 1],
            stateto = [2, 2, 2],
            obstype = [1, 1, 1]
        )
        
        model = multistatemodel(h12; data=dat, initialize=true)
        
        # Get natural parameters
        params_natural_1 = get_parameters_natural(model)
        
        # Set via nested representation
        set_parameters!(model, params_natural_1)
        
        # Get back
        params_natural_2 = get_parameters_natural(model)
        
        # Should match
        for (k, v1) in pairs(params_natural_1)
            v2 = params_natural_2[k]
            @test v2 ≈ v1 atol=1e-10
        end
    end
end
