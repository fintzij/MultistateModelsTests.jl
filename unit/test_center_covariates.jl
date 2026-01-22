# =============================================================================
# Unit Tests for center_covariates utility
# =============================================================================
# Tests for Phase 2 of Item #29: center_covariates function
# =============================================================================

using Test
using MultistateModels
using DataFrames

@testset "center_covariates" begin
    
    @testset "Basic centering at mean" begin
        h12 = Hazard(@formula(0 ~ 1 + age + weight), "exp", 1, 2)
        
        dat = DataFrame(
            id = [1, 1, 2, 2],
            tstart = [0.0, 0.5, 0.0, 0.3],
            tstop = [0.5, 1.0, 0.3, 0.8],
            statefrom = [1, 1, 1, 1],
            stateto = [1, 2, 1, 2],
            obstype = [2, 1, 2, 1],
            age = [30.0, 30.0, 50.0, 50.0],
            weight = [70.0, 70.0, 90.0, 90.0]
        )
        
        model = multistatemodel(h12; data=dat)
        
        # Center at mean
        centered = center_covariates(model; centering=:mean)
        
        # Mean age = 40, mean weight = 80
        @test centered[:age] ≈ 40.0
        @test centered[:weight] ≈ 80.0
    end
    
    @testset "Centering at median" begin
        h12 = Hazard(@formula(0 ~ 1 + x), "exp", 1, 2)
        
        dat = DataFrame(
            id = [1, 2, 3, 4, 5],
            tstart = fill(0.0, 5),
            tstop = fill(1.0, 5),
            statefrom = fill(1, 5),
            stateto = fill(2, 5),
            obstype = fill(1, 5),
            x = [1.0, 2.0, 3.0, 4.0, 100.0]  # Outlier at 100
        )
        
        model = multistatemodel(h12; data=dat)
        
        centered = center_covariates(model; centering=:median)
        
        # Median x = 3
        @test centered[:x] ≈ 3.0
    end
    
    @testset "Invalid centering throws ArgumentError" begin
        h12 = Hazard(@formula(0 ~ 1 + age), "exp", 1, 2)
        
        dat = DataFrame(
            id = [1, 1, 2, 2],
            tstart = [0.0, 0.5, 0.0, 0.3],
            tstop = [0.5, 1.0, 0.3, 0.8],
            statefrom = [1, 1, 1, 1],
            stateto = [1, 2, 1, 2],
            obstype = [2, 1, 2, 1],
            age = [30.0, 30.0, 50.0, 50.0]
        )
        
        model = multistatemodel(h12; data=dat)
        
        @test_throws ArgumentError center_covariates(model; centering=:reference)
        @test_throws ArgumentError center_covariates(model; centering=:invalid)
    end
    
    @testset "No covariates returns empty NamedTuple" begin
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        
        dat = DataFrame(
            id = [1, 1],
            tstart = [0.0, 0.5],
            tstop = [0.5, 1.0],
            statefrom = [1, 1],
            stateto = [1, 2],
            obstype = [2, 1]
        )
        
        model = multistatemodel(h12; data=dat)
        
        centered = center_covariates(model; centering=:mean)
        
        @test centered == NamedTuple()
    end
    
    @testset "Multiple hazards with different covariates" begin
        h12 = Hazard(@formula(0 ~ 1 + age), "exp", 1, 2)
        h13 = Hazard(@formula(0 ~ 1 + weight), "exp", 1, 3)
        h23 = Hazard(@formula(0 ~ 1 + age + weight), "exp", 2, 3)
        
        dat = DataFrame(
            id = [1, 1, 2, 2],
            tstart = [0.0, 0.5, 0.0, 0.3],
            tstop = [0.5, 1.0, 0.3, 0.8],
            statefrom = [1, 1, 1, 1],
            stateto = [1, 2, 1, 3],
            obstype = [2, 1, 2, 1],
            age = [30.0, 30.0, 50.0, 50.0],
            weight = [70.0, 70.0, 90.0, 90.0]
        )
        
        model = multistatemodel(h12, h13, h23; data=dat)
        
        centered = center_covariates(model; centering=:mean)
        
        # Both covariates should be included
        @test haskey(centered, :age)
        @test haskey(centered, :weight)
        @test centered[:age] ≈ 40.0
        @test centered[:weight] ≈ 80.0
    end
    
    @testset "Weighted mean by unique subjects" begin
        # Multiple rows per subject - should use first occurrence or subject-level
        h12 = Hazard(@formula(0 ~ 1 + x), "exp", 1, 2)
        
        # Subject 1 has 3 rows, Subject 2 has 1 row
        # The covariate values are constant within subject
        dat = DataFrame(
            id = [1, 1, 1, 2],
            tstart = [0.0, 0.3, 0.6, 0.0],
            tstop = [0.3, 0.6, 1.0, 1.0],
            statefrom = [1, 1, 1, 1],
            stateto = [1, 1, 2, 2],
            obstype = [2, 2, 1, 1],
            x = [10.0, 10.0, 10.0, 20.0]  # Subject 1: 10, Subject 2: 20
        )
        
        model = multistatemodel(h12; data=dat)
        
        centered = center_covariates(model; centering=:mean)
        
        # Mean across unique subjects should be (10 + 20) / 2 = 15
        @test centered[:x] ≈ 15.0
    end
    
    @testset "Invalid centering option throws error" begin
        h12 = Hazard(@formula(0 ~ 1 + x), "exp", 1, 2)
        
        dat = DataFrame(
            id = [1, 1],
            tstart = [0.0, 0.5],
            tstop = [0.5, 1.0],
            statefrom = [1, 1],
            stateto = [1, 2],
            obstype = [2, 1],
            x = [1.0, 1.0]
        )
        
        model = multistatemodel(h12; data=dat)
        
        @test_throws ArgumentError center_covariates(model; centering=:invalid)
    end
end
