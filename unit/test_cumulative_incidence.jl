# =============================================================================
# Unit Tests for cumulative_incidence API
# =============================================================================
# Tests for Phase 1 of Item #29: cumulative_incidence with newdata argument
# =============================================================================

using Test
using MultistateModels
using DataFrames

@testset "cumulative_incidence API" begin
    
    @testset "Basic functionality with NamedTuple" begin
        # Simple illness-death model with exponential hazards
        h12 = Hazard(@formula(0 ~ 1 + x), "exp", 1, 2)
        h13 = Hazard(@formula(0 ~ 1), "exp", 1, 3)
        
        dat = DataFrame(
            id = [1, 1, 2, 2],
            tstart = [0.0, 0.5, 0.0, 0.3],
            tstop = [0.5, 1.0, 0.3, 0.8],
            statefrom = [1, 1, 1, 1],
            stateto = [1, 2, 1, 3],
            obstype = [2, 1, 2, 1],
            x = [0.5, 0.5, -0.5, -0.5]
        )
        
        model = multistatemodel(h12, h13; data=dat)
        set_parameters!(model, (h12=[0.5, 0.1], h13=[0.3]))
        
        t = [0.25, 0.5, 0.75, 1.0]
        
        # Test with NamedTuple covariate specification
        ci_nt = cumulative_incidence(t, model, (x=0.5,); statefrom=1)
        
        @test ci_nt isa Matrix{Float64}
        @test size(ci_nt) == (4, 2)  # 4 times, 2 transitions (1→2, 1→3)
        @test all(ci_nt .>= 0)
        @test all(ci_nt .<= 1)
        @test all(diff(ci_nt, dims=1) .>= -1e-10)  # Cumulative incidence is non-decreasing
    end
    
    @testset "DataFrameRow method" begin
        h12 = Hazard(@formula(0 ~ 1 + x), "exp", 1, 2)
        h13 = Hazard(@formula(0 ~ 1), "exp", 1, 3)
        
        dat = DataFrame(
            id = [1, 1],
            tstart = [0.0, 0.5],
            tstop = [0.5, 1.0],
            statefrom = [1, 1],
            stateto = [1, 2],
            obstype = [2, 1],
            x = [0.5, 0.5]
        )
        
        model = multistatemodel(h12, h13; data=dat)
        set_parameters!(model, (h12=[0.5, 0.1], h13=[0.3]))
        
        t = [0.25, 0.5, 0.75, 1.0]
        
        # Create DataFrameRow with covariate values
        newdata_row = DataFrame(x=0.5)[1, :]
        ci_row = cumulative_incidence(t, model, newdata_row; statefrom=1)
        
        @test ci_row isa Matrix{Float64}
        @test size(ci_row) == (4, 2)
        @test all(ci_row .>= 0)
    end
    
    @testset "Reference method (covariates at zero)" begin
        h12 = Hazard(@formula(0 ~ 1 + x), "exp", 1, 2)
        h13 = Hazard(@formula(0 ~ 1), "exp", 1, 3)
        
        dat = DataFrame(
            id = [1, 1],
            tstart = [0.0, 0.5],
            tstop = [0.5, 1.0],
            statefrom = [1, 1],
            stateto = [1, 2],
            obstype = [2, 1],
            x = [0.5, 0.5]
        )
        
        model = multistatemodel(h12, h13; data=dat)
        set_parameters!(model, (h12=[0.5, 0.1], h13=[0.3]))
        
        t = [0.25, 0.5, 0.75, 1.0]
        
        ci_ref = cumulative_incidence_at_reference(t, model; statefrom=1)
        
        @test ci_ref isa Matrix{Float64}
        @test size(ci_ref) == (4, 2)
        @test all(ci_ref .>= 0)
    end
    
    @testset "Consistency: x=0 equals reference" begin
        h12 = Hazard(@formula(0 ~ 1 + x), "exp", 1, 2)
        h13 = Hazard(@formula(0 ~ 1), "exp", 1, 3)
        
        dat = DataFrame(
            id = [1, 1],
            tstart = [0.0, 0.5],
            tstop = [0.5, 1.0],
            statefrom = [1, 1],
            stateto = [1, 2],
            obstype = [2, 1],
            x = [0.5, 0.5]
        )
        
        model = multistatemodel(h12, h13; data=dat)
        set_parameters!(model, (h12=[0.5, 0.1], h13=[0.3]))
        
        t = [0.25, 0.5, 0.75, 1.0]
        
        # At x=0 (reference level), both methods should give identical results
        ci_zero = cumulative_incidence(t, model, (x=0.0,); statefrom=1)
        ci_ref = cumulative_incidence_at_reference(t, model; statefrom=1)
        
        @test ci_zero ≈ ci_ref atol=1e-10
    end
    
    @testset "Covariate effect direction" begin
        # Higher hazard rate should lead to higher cumulative incidence
        h12 = Hazard(@formula(0 ~ 1 + x), "exp", 1, 2)
        h13 = Hazard(@formula(0 ~ 1), "exp", 1, 3)
        
        dat = DataFrame(
            id = [1, 1],
            tstart = [0.0, 0.5],
            tstop = [0.5, 1.0],
            statefrom = [1, 1],
            stateto = [1, 2],
            obstype = [2, 1],
            x = [0.5, 0.5]
        )
        
        model = multistatemodel(h12, h13; data=dat)
        # Positive covariate effect on h12
        set_parameters!(model, (h12=[0.5, 0.5], h13=[0.3]))
        
        t = [1.0]
        
        ci_high = cumulative_incidence(t, model, (x=1.0,); statefrom=1)
        ci_low = cumulative_incidence(t, model, (x=-1.0,); statefrom=1)
        
        # Higher x should increase h12 hazard → higher CI for transition 1→2
        @test ci_high[1, 1] > ci_low[1, 1]
    end
    
    @testset "Missing covariate defaults to zero" begin
        h12 = Hazard(@formula(0 ~ 1 + x + y), "exp", 1, 2)
        h13 = Hazard(@formula(0 ~ 1), "exp", 1, 3)
        
        dat = DataFrame(
            id = [1, 1],
            tstart = [0.0, 0.5],
            tstop = [0.5, 1.0],
            statefrom = [1, 1],
            stateto = [1, 2],
            obstype = [2, 1],
            x = [0.5, 0.5],
            y = [0.3, 0.3]
        )
        
        model = multistatemodel(h12, h13; data=dat)
        set_parameters!(model, (h12=[0.5, 0.1, 0.2], h13=[0.3]))
        
        t = [0.5, 1.0]
        
        # Only specify x, y should default to 0
        ci_partial = cumulative_incidence(t, model, (x=0.5,); statefrom=1)
        ci_full = cumulative_incidence(t, model, (x=0.5, y=0.0); statefrom=1)
        
        @test ci_partial ≈ ci_full atol=1e-10
    end
    
    @testset "No covariates model" begin
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        h13 = Hazard(@formula(0 ~ 1), "exp", 1, 3)
        
        dat = DataFrame(
            id = [1, 1],
            tstart = [0.0, 0.5],
            tstop = [0.5, 1.0],
            statefrom = [1, 1],
            stateto = [1, 2],
            obstype = [2, 1]
        )
        
        model = multistatemodel(h12, h13; data=dat)
        set_parameters!(model, (h12=[0.5], h13=[0.3]))
        
        t = [0.5, 1.0]
        
        # Should work with empty NamedTuple
        ci_empty = cumulative_incidence(t, model, NamedTuple(); statefrom=1)
        ci_ref = cumulative_incidence_at_reference(t, model; statefrom=1)
        
        @test ci_empty ≈ ci_ref atol=1e-10
    end
end
