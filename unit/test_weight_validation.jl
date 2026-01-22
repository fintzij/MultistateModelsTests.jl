# =============================================================================
# Weight Validation Edge Case Tests
# =============================================================================
#
# Tests for weight validation edge cases including:
# - Zero weights (should be rejected - weights must be positive)
# - Negative weights (should be rejected)
# - Inf/NaN weights (should be rejected)
# - Very small positive weights (should be accepted)
# - Very large weights (should be accepted)
#
# This is from Sprint 11 Workstream D, verifying H3_P1 weight validation.
#
# =============================================================================

using Test
using MultistateModels
using DataFrames

@testset "Weight Validation Edge Cases" begin
    
    # Helper to create basic model data
    function basic_data(nsubj::Int)
        DataFrame(
            id = repeat(1:nsubj, inner=2),
            tstart = repeat([0.0, 1.0], outer=nsubj),
            tstop = repeat([1.0, 2.0], outer=nsubj),
            statefrom = repeat([1, 2], outer=nsubj),
            stateto = repeat([2, 3], outer=nsubj),
            obstype = fill(1, 2*nsubj)
        )
    end
    
    @testset "SubjectWeights Validation" begin
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
        dat = basic_data(5)
        
        @testset "Zero weights rejected" begin
            zero_weights = [1.0, 0.0, 1.0, 1.0, 1.0]
            @test_throws ArgumentError multistatemodel(h12, h23; data=dat, SubjectWeights=zero_weights)
        end
        
        @testset "Negative weights rejected" begin
            neg_weights = [1.0, -0.5, 1.0, 1.0, 1.0]
            @test_throws ArgumentError multistatemodel(h12, h23; data=dat, SubjectWeights=neg_weights)
        end
        
        @testset "Inf weights rejected" begin
            inf_weights = [1.0, Inf, 1.0, 1.0, 1.0]
            @test_throws ArgumentError multistatemodel(h12, h23; data=dat, SubjectWeights=inf_weights)
        end
        
        @testset "NaN weights rejected" begin
            nan_weights = [1.0, NaN, 1.0, 1.0, 1.0]
            @test_throws ArgumentError multistatemodel(h12, h23; data=dat, SubjectWeights=nan_weights)
        end
        
        @testset "Very small positive weights accepted" begin
            tiny_weights = fill(1e-15, 5)
            model = multistatemodel(h12, h23; data=dat, SubjectWeights=tiny_weights)
            @test model.SubjectWeights == tiny_weights
        end
        
        @testset "Very large weights accepted" begin
            large_weights = fill(1e15, 5)
            model = multistatemodel(h12, h23; data=dat, SubjectWeights=large_weights)
            @test model.SubjectWeights == large_weights
        end
        
        @testset "Wrong length rejected" begin
            short_weights = [1.0, 1.0, 1.0]
            @test_throws ArgumentError multistatemodel(h12, h23; data=dat, SubjectWeights=short_weights)
            
            long_weights = fill(1.0, 10)
            @test_throws ArgumentError multistatemodel(h12, h23; data=dat, SubjectWeights=long_weights)
        end
        
        @testset "All valid positive weights accepted" begin
            valid_weights = [0.5, 1.0, 1.5, 2.0, 0.1]
            model = multistatemodel(h12, h23; data=dat, SubjectWeights=valid_weights)
            @test model.SubjectWeights == valid_weights
        end
    end
    
    @testset "ObservationWeights Validation" begin
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
        dat = basic_data(3)
        
        @testset "Zero observation weights rejected" begin
            zero_weights = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
            @test_throws ArgumentError multistatemodel(h12, h23; data=dat, ObservationWeights=zero_weights)
        end
        
        @testset "Negative observation weights rejected" begin
            neg_weights = [1.0, -0.5, 1.0, 1.0, 1.0, 1.0]
            @test_throws ArgumentError multistatemodel(h12, h23; data=dat, ObservationWeights=neg_weights)
        end
        
        @testset "Valid observation weights accepted" begin
            valid_weights = [0.5, 1.0, 1.5, 2.0, 0.1, 3.0]
            model = multistatemodel(h12, h23; data=dat, ObservationWeights=valid_weights)
            @test model.ObservationWeights == valid_weights
        end
    end
    
    @testset "Mutual Exclusivity" begin
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
        dat = basic_data(3)
        
        @testset "Both weights provided - error" begin
            subj_weights = [1.0, 1.0, 1.0]
            obs_weights = fill(1.0, 6)
            @test_throws ArgumentError multistatemodel(h12, h23; data=dat, 
                SubjectWeights=subj_weights, ObservationWeights=obs_weights)
        end
    end
end
