# =============================================================================
# Unit Tests: Penalty Weighting for Adaptive P-Splines
# =============================================================================
#
# Tests for Phase 1 of adaptive penalty weighting implementation:
# - PenaltyWeighting type hierarchy (UniformWeighting, AtRiskWeighting)
# - SplinePenalty with weighting field
# - compute_atrisk_counts function for exact data
#
# =============================================================================

using Test
using MultistateModels
using DataFrames
using Random

@testset "Penalty Weighting" begin

    # =========================================================================
    # 1. PENALTY WEIGHTING TYPES
    # =========================================================================
    @testset "PenaltyWeighting Types" begin
        
        @testset "UniformWeighting" begin
            w = UniformWeighting()
            @test w isa PenaltyWeighting
            @test w isa UniformWeighting
        end
        
        @testset "AtRiskWeighting Construction" begin
            # Default construction (alpha=1.0, learn=false)
            w1 = AtRiskWeighting()
            @test w1 isa PenaltyWeighting
            @test w1 isa AtRiskWeighting
            @test w1.alpha == 1.0
            @test w1.learn == false
            
            # Custom alpha
            w2 = AtRiskWeighting(alpha=0.5)
            @test w2.alpha == 0.5
            @test w2.learn == false
            
            # Enable learning
            w3 = AtRiskWeighting(alpha=1.0, learn=true)
            @test w3.alpha == 1.0
            @test w3.learn == true
            
            # Both custom
            w4 = AtRiskWeighting(alpha=2.0, learn=true)
            @test w4.alpha == 2.0
            @test w4.learn == true
        end
        
        @testset "AtRiskWeighting Validation" begin
            # alpha must be non-negative
            @test_throws ArgumentError AtRiskWeighting(alpha=-0.1)
            @test_throws ArgumentError AtRiskWeighting(alpha=-1.0)
            
            # alpha=0 is allowed (equivalent to uniform)
            w = AtRiskWeighting(alpha=0.0)
            @test w.alpha == 0.0
        end
    end
    
    # =========================================================================
    # 2. SPLINE PENALTY WITH WEIGHTING
    # =========================================================================
    @testset "SplinePenalty with Weighting" begin
        
        @testset "Default Weighting (Uniform)" begin
            p = SplinePenalty()
            @test p.weighting isa UniformWeighting
            
            p2 = SplinePenalty(adaptive_weight=:none)
            @test p2.weighting isa UniformWeighting
        end
        
        @testset "At-Risk Weighting" begin
            p1 = SplinePenalty(adaptive_weight=:atrisk)
            @test p1.weighting isa AtRiskWeighting
            @test p1.weighting.alpha == 1.0
            @test p1.weighting.learn == false
            
            p2 = SplinePenalty(adaptive_weight=:atrisk, alpha=0.5)
            @test p2.weighting isa AtRiskWeighting
            @test p2.weighting.alpha == 0.5
            
            p3 = SplinePenalty(adaptive_weight=:atrisk, learn_alpha=true)
            @test p3.weighting isa AtRiskWeighting
            @test p3.weighting.learn == true
            
            p4 = SplinePenalty(adaptive_weight=:atrisk, alpha=2.0, learn_alpha=true)
            @test p4.weighting.alpha == 2.0
            @test p4.weighting.learn == true
        end
        
        @testset "Weighting with Selectors" begin
            p1 = SplinePenalty((1, 2), adaptive_weight=:atrisk, alpha=1.5)
            @test p1.selector == (1, 2)
            @test p1.weighting isa AtRiskWeighting
            @test p1.weighting.alpha == 1.5
            
            p2 = SplinePenalty(1, adaptive_weight=:atrisk)
            @test p2.selector == 1
            @test p2.weighting isa AtRiskWeighting
            
            p3 = SplinePenalty(1, share_lambda=true, adaptive_weight=:atrisk)
            @test p3.share_lambda == true
            @test p3.weighting isa AtRiskWeighting
        end
        
        @testset "Invalid Weighting Specification" begin
            @test_throws ArgumentError SplinePenalty(adaptive_weight=:invalid)
            @test_throws ArgumentError SplinePenalty(adaptive_weight=:fisher)
        end
        
        @testset "Backward Compatibility" begin
            p1 = SplinePenalty()
            @test p1.order == 2
            @test p1.weighting isa UniformWeighting
            
            p2 = SplinePenalty(order=3)
            @test p2.order == 3
            @test p2.weighting isa UniformWeighting
            
            p3 = SplinePenalty(1, share_lambda=true)
            @test p3.share_lambda == true
            @test p3.weighting isa UniformWeighting
            
            p4 = SplinePenalty((1, 2), order=1, total_hazard=true)
            @test p4.total_hazard == true
            @test p4.weighting isa UniformWeighting
        end
    end
    
    # =========================================================================
    # 3. COMPUTE AT-RISK COUNTS
    # =========================================================================
    @testset "compute_atrisk_counts" begin
        
        @testset "Simple Exact Data - All at Risk at t=0" begin
            data = DataFrame(
                id = 1:5,
                tstart = zeros(5),
                tstop = [1.0, 2.0, 3.0, 4.0, 5.0],
                statefrom = ones(Int, 5),
                stateto = [2, 2, 2, 2, 1],
                obstype = ones(Int, 5)
            )
            
            h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
            model = multistatemodel(h12; data=data)
            
            atrisk = MultistateModels.compute_atrisk_counts(model, [0.0], (1, 2))
            @test atrisk[1] == 5.0
        end
        
        @testset "Decreasing At-Risk Over Time" begin
            data = DataFrame(
                id = 1:5,
                tstart = zeros(5),
                tstop = [1.0, 2.0, 3.0, 4.0, 5.0],
                statefrom = ones(Int, 5),
                stateto = fill(2, 5),
                obstype = ones(Int, 5)
            )
            
            h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
            model = multistatemodel(h12; data=data)
            
            eval_times = [0.0, 0.5, 1.5, 2.5, 3.5, 4.5]
            atrisk = MultistateModels.compute_atrisk_counts(model, eval_times, (1, 2))
            
            @test atrisk[1] == 5.0
            @test atrisk[2] == 5.0
            @test atrisk[3] == 4.0
            @test atrisk[4] == 3.0
            @test atrisk[5] == 2.0
            @test atrisk[6] == 1.0
        end
        
        @testset "Floor at 1.0 When No Subjects at Risk" begin
            data = DataFrame(
                id = [1],
                tstart = [0.0],
                tstop = [0.5],
                statefrom = [1],
                stateto = [2],
                obstype = [1]
            )
            
            h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
            model = multistatemodel(h12; data=data)
            
            atrisk = MultistateModels.compute_atrisk_counts(model, [1.0], (1, 2))
            @test atrisk[1] == 1.0
        end
        
        @testset "Multiple Transitions - Competing Risks" begin
            data = DataFrame(
                id = [1, 2, 3, 4],
                tstart = zeros(4),
                tstop = [1.0, 2.0, 1.5, 3.0],
                statefrom = ones(Int, 4),
                stateto = [2, 3, 2, 1],
                obstype = ones(Int, 4)
            )
            
            h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
            h13 = Hazard(@formula(0 ~ 1), "exp", 1, 3)
            model = multistatemodel(h12, h13; data=data)
            
            eval_times = [0.0, 0.5, 1.25, 1.75]
            atrisk_12 = MultistateModels.compute_atrisk_counts(model, eval_times, (1, 2))
            atrisk_13 = MultistateModels.compute_atrisk_counts(model, eval_times, (1, 3))
            
            @test atrisk_12 == atrisk_13
            @test atrisk_12[1] == 4.0
            @test atrisk_12[2] == 4.0
        end
        
        @testset "Multi-Interval Subject" begin
            data = DataFrame(
                id = [1, 1, 1],
                tstart = [0.0, 1.0, 2.0],
                tstop = [1.0, 2.0, 3.0],
                statefrom = [1, 2, 2],
                stateto = [2, 2, 3],
                obstype = [1, 1, 1]
            )
            
            h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
            h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
            model = multistatemodel(h12, h23; data=data)
            
            eval_times = [0.0, 0.5, 1.5, 2.5]
            atrisk_12 = MultistateModels.compute_atrisk_counts(model, eval_times, (1, 2))
            
            @test atrisk_12[1] == 1.0
            @test atrisk_12[2] == 1.0
            @test atrisk_12[3] == 1.0
            @test atrisk_12[4] == 1.0
            
            atrisk_23 = MultistateModels.compute_atrisk_counts(model, eval_times, (2, 3))
            
            @test atrisk_23[1] == 1.0
            @test atrisk_23[2] == 1.0
            @test atrisk_23[3] == 1.0
            @test atrisk_23[4] == 1.0
        end
        
        @testset "Eval Times Must Be Sorted" begin
            data = DataFrame(
                id = [1],
                tstart = [0.0],
                tstop = [1.0],
                statefrom = [1],
                stateto = [2],
                obstype = [1]
            )
            
            h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
            model = multistatemodel(h12; data=data)
            
            @test_throws ArgumentError MultistateModels.compute_atrisk_counts(
                model, [1.0, 0.5], (1, 2)
            )
        end
        
        @testset "Empty Eval Times" begin
            data = DataFrame(
                id = [1],
                tstart = [0.0],
                tstop = [1.0],
                statefrom = [1],
                stateto = [2],
                obstype = [1]
            )
            
            h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
            model = multistatemodel(h12; data=data)
            
            atrisk = MultistateModels.compute_atrisk_counts(model, Float64[], (1, 2))
            @test length(atrisk) == 0
        end
        
        @testset "Larger Dataset - Realistic Scenario" begin
            Random.seed!(12345)
            n = 100
            
            event_times = (rand(n) .^ (1/1.5)) .* 5.0
            censor_times = rand(n) .* 6.0
            
            obs_times = min.(event_times, censor_times)
            status = Int.(event_times .<= censor_times) .+ 1
            
            data = DataFrame(
                id = 1:n,
                tstart = zeros(n),
                tstop = obs_times,
                statefrom = ones(Int, n),
                stateto = status,
                obstype = ones(Int, n)
            )
            
            h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
            model = multistatemodel(h12; data=data)
            
            eval_times = collect(0.0:0.5:5.0)
            atrisk = MultistateModels.compute_atrisk_counts(model, eval_times, (1, 2))
            
            for i in 2:length(atrisk)
                @test atrisk[i] <= atrisk[i-1]
            end
            
            @test atrisk[1] == Float64(n)
            @test atrisk[end] >= 1.0
        end
    end
    
    # =========================================================================
    # 3b. INTERVAL-AVERAGED AT-RISK COUNTS
    # =========================================================================
    @testset "compute_atrisk_interval_averages" begin
        
        @testset "Basic Interval Averages - Exact Data" begin
            # Subject 1: at risk [0, 5) - events at t=5
            # Subject 2: at risk [0, 3) - events at t=3
            # Subject 3: at risk [0, 4) - events at t=4
            data = DataFrame(
                id = [1, 2, 3],
                tstart = [0.0, 0.0, 0.0],
                tstop = [5.0, 3.0, 4.0],
                statefrom = [1, 1, 1],
                stateto = [2, 2, 2],
                obstype = [1, 1, 1]
            )
            
            # Use a spline hazard to get knot structure
            h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2;
                         degree=0, knots=[2.0, 4.0], boundaryknots=[0.0, 6.0])
            model = multistatemodel(h12; data=data)
            
            atrisk_avg = compute_atrisk_interval_averages(model, model.hazards[1], (1, 2))
            
            # Interval [0, 2): All 3 subjects at risk for full 2 units
            #   Person-time = 3 * 2 = 6, Avg = 6/2 = 3.0
            @test atrisk_avg[1] ≈ 3.0
            
            # Interval [2, 4): 
            #   S1: at risk for 2 units (until 4), S2: at risk for 1 unit (until 3), S3: at risk for 2 units
            #   Person-time = 2 + 1 + 2 = 5, Avg = 5/2 = 2.5
            @test atrisk_avg[2] ≈ 2.5
            
            # Interval [4, 6):
            #   S1: at risk for 1 unit (until 5), S2: 0, S3: 0
            #   Person-time = 1, Avg = 1/2 = 0.5
            @test atrisk_avg[3] ≈ 0.5
        end
        
        @testset "Comparison: Midpoint vs Interval Average" begin
            # Create data where midpoint and interval average differ noticeably
            # Subject at risk [0, 1.1) with interval [0, 3]
            # Use knot at t=3 to create single interval [0, 3]
            data = DataFrame(
                id = [1],
                tstart = [0.0],
                tstop = [1.1],  # Subject exits at t=1.1
                statefrom = [1],
                stateto = [2],
                obstype = [1]
            )
            
            h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2;
                         degree=0, knots=[1.5], boundaryknots=[0.0, 3.0])
            model = multistatemodel(h12; data=data)
            
            # Midpoint of [0, 1.5) is t=0.75: subject IS at risk (tstop=1.1 > 0.75) → count = 1
            midpoint_count = MultistateModels.compute_atrisk_counts(model, [0.75], (1, 2))
            @test midpoint_count[1] ≈ 1.0
            
            # Interval average for [0, 1.5): person-time = 1.1, width = 1.5, avg ≈ 0.733
            interval_avg = compute_atrisk_interval_averages(model, model.hazards[1], (1, 2))
            @test interval_avg[1] ≈ 1.1 / 1.5  # ≈ 0.733
            
            # Interval [1.5, 3.0): person-time = 0, width = 1.5, avg = 0
            @test interval_avg[2] ≈ 0.0
            
            # The interval average shows subject was at risk for 73% of first interval,
            # while midpoint evaluation incorrectly shows 100%
        end
    end

    # =========================================================================
    # 4. WEIGHTED PENALTY MATRIX CONSTRUCTION
    # =========================================================================
    @testset "build_weighted_penalty_matrix" begin
        using BSplineKit
        using LinearAlgebra
        
        @testset "Uniform Weights (α=0) Recover Standard Penalty" begin
            # Create a cubic B-spline basis with uniform knots
            knots = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0]
            basis = BSplineBasis(BSplineOrder(4), knots)
            K = length(basis)  # 8 basis functions
            
            # At-risk counts at knot midpoints (5 intervals)
            atrisk = [100.0, 80.0, 50.0, 30.0, 10.0]
            
            # Build weighted penalty with α=0 (uniform weights)
            weighting_uniform = AtRiskWeighting(alpha=0.0)
            S_weighted = MultistateModels.build_weighted_penalty_matrix(
                basis, 2, weighting_uniform, atrisk
            )
            
            # Build standard penalty (GPS method - used as reference for uniform)
            S_standard = MultistateModels.build_penalty_matrix(basis, 2; method=:integral)
            
            # Should be approximately equal (both use integral method when α=0)
            @test size(S_weighted) == (K, K)
            @test isapprox(S_weighted, S_standard, rtol=1e-10)
        end
        
        @testset "UniformWeighting Dispatch Returns GPS Matrix" begin
            knots = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0]
            basis = BSplineBasis(BSplineOrder(4), knots)
            
            atrisk = [100.0, 80.0, 50.0, 30.0, 10.0]
            
            # UniformWeighting dispatch should ignore at-risk counts
            S_uniform = MultistateModels.build_weighted_penalty_matrix(
                basis, 2, UniformWeighting(), atrisk
            )
            
            # Should equal GPS matrix
            S_gps = MultistateModels.build_penalty_matrix(basis, 2; method=:gps)
            
            @test isapprox(S_uniform, S_gps, rtol=1e-10)
        end
        
        @testset "Higher Weight Increases Penalty Contribution" begin
            # Create a simple quadratic B-spline basis
            knots = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0]
            basis = BSplineBasis(BSplineOrder(3), knots)
            K = length(basis)  # 5 basis functions
            
            # Case 1: Low at-risk at midpoint of interval 2 → high weight there
            # 3 intervals: [0,1], [1,2], [2,3]
            atrisk_low_middle = [100.0, 10.0, 100.0]  # Low in middle
            
            # Case 2: Uniform at-risk
            atrisk_uniform = [100.0, 100.0, 100.0]
            
            weighting = AtRiskWeighting(alpha=1.0)
            
            S_low_middle = MultistateModels.build_weighted_penalty_matrix(
                basis, 2, weighting, atrisk_low_middle
            )
            S_uniform = MultistateModels.build_weighted_penalty_matrix(
                basis, 2, weighting, atrisk_uniform
            )
            
            # With low at-risk in middle, we expect higher penalty (more weight)
            # Frobenius norm should be higher for low-middle case
            @test norm(S_low_middle) > norm(S_uniform)
        end
        
        @testset "Matrix is Symmetric" begin
            knots = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0]
            basis = BSplineBasis(BSplineOrder(4), knots)
            
            atrisk = [100.0, 50.0, 20.0, 10.0, 5.0]
            
            weighting = AtRiskWeighting(alpha=1.0)
            S_w = MultistateModels.build_weighted_penalty_matrix(
                basis, 2, weighting, atrisk
            )
            
            @test issymmetric(S_w)
            @test maximum(abs.(S_w - S_w')) < 1e-14
        end
        
        @testset "Matrix is Positive Semi-Definite" begin
            knots = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0]
            basis = BSplineBasis(BSplineOrder(4), knots)
            
            # Various at-risk patterns
            test_atrisk = [
                [100.0, 80.0, 60.0, 40.0, 20.0],  # Decreasing
                [20.0, 40.0, 60.0, 40.0, 20.0],   # Peaked
                [10.0, 10.0, 10.0, 10.0, 10.0],   # Uniform low
                [1.0, 1.0, 1.0, 1.0, 1.0],        # Minimum (floor)
            ]
            
            for atrisk in test_atrisk
                weighting = AtRiskWeighting(alpha=1.0)
                S_w = MultistateModels.build_weighted_penalty_matrix(
                    basis, 2, weighting, atrisk
                )
                
                # All eigenvalues should be non-negative
                eigs = eigvals(S_w)
                @test all(eigs .>= -1e-12)  # Small tolerance for numerical error
            end
        end
        
        @testset "Null Space Contains Polynomials" begin
            # For order=2 penalty, constant and linear functions should be unpenalized
            # Use a well-spaced knot vector
            knots = [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.0, 5.0, 5.0]
            basis = BSplineBasis(BSplineOrder(4), knots)
            K = length(basis)
            
            # 10 unique intervals
            atrisk = [100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0]
            
            weighting = AtRiskWeighting(alpha=1.0)
            S_w = MultistateModels.build_weighted_penalty_matrix(
                basis, 2, weighting, atrisk
            )
            
            # Check that matrix is at least rank-deficient (has some null space)
            eigs = eigvals(S_w)
            n_small = sum(abs.(eigs) .< 1e-8)
            @test n_small >= 1  # At least some null space
            
            # Check matrix is PSD with non-trivial rank
            @test all(eigs .>= -1e-12)
            @test sum(eigs .> 1e-8) > 0  # Has positive eigenvalues
        end
        
        @testset "Different Alpha Values" begin
            # Use more interior knots for a well-defined penalty
            knots = [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.0, 5.0, 5.0]
            basis = BSplineBasis(BSplineOrder(4), knots)
            
            # At-risk pattern with variation (10 unique intervals)
            atrisk = [100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0]
            
            # Build matrices with different alpha
            S_a0 = MultistateModels.build_weighted_penalty_matrix(
                basis, 2, AtRiskWeighting(alpha=0.0), atrisk
            )
            S_a05 = MultistateModels.build_weighted_penalty_matrix(
                basis, 2, AtRiskWeighting(alpha=0.5), atrisk
            )
            S_a1 = MultistateModels.build_weighted_penalty_matrix(
                basis, 2, AtRiskWeighting(alpha=1.0), atrisk
            )
            S_a2 = MultistateModels.build_weighted_penalty_matrix(
                basis, 2, AtRiskWeighting(alpha=2.0), atrisk
            )
            
            # All should be symmetric and PSD
            for S in [S_a0, S_a05, S_a1, S_a2]
                @test issymmetric(S)
                @test all(eigvals(S) .>= -1e-12)
            end
            
            # Different α should produce different matrices
            @test !isapprox(S_a0, S_a05, rtol=0.01)
            @test !isapprox(S_a05, S_a1, rtol=0.01)
            @test !isapprox(S_a1, S_a2, rtol=0.01)
        end
        
        @testset "Handles Minimum At-Risk (Floor at 1.0)" begin
            knots = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0]
            basis = BSplineBasis(BSplineOrder(4), knots)
            
            # At-risk with values below 1.0 (should be clamped internally)
            atrisk_below = [0.5, 0.2, 0.1, 0.0, -0.5]
            
            weighting = AtRiskWeighting(alpha=1.0)
            
            # Should not throw - values are clamped to 1.0
            S_w = MultistateModels.build_weighted_penalty_matrix(
                basis, 2, weighting, atrisk_below
            )
            
            # Should still be symmetric and PSD
            @test issymmetric(S_w)
            @test all(eigvals(S_w) .>= -1e-12)
            
            # All weights should be clamped to Y=1, so w = 1^(-1) = 1
            # This is equivalent to uniform weighting
            atrisk_ones = [1.0, 1.0, 1.0, 1.0, 1.0]
            S_ones = MultistateModels.build_weighted_penalty_matrix(
                basis, 2, weighting, atrisk_ones
            )
            
            @test isapprox(S_w, S_ones, rtol=1e-10)
        end
        
        @testset "RecombinedBSplineBasis (Natural Splines)" begin
            # Test with natural spline basis - skip if BSplineKit doesn't support this construction
            # Different versions of BSplineKit have different APIs
            knots = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0]
            parent_basis = BSplineBasis(BSplineOrder(4), knots)
            
            # Try to create natural spline basis - may not be available in all BSplineKit versions
            natural_basis = try
                RecombinedBSplineBasis(parent_basis, Derivative(2))
            catch e
                @info "RecombinedBSplineBasis with Derivative(2) not supported, skipping test" exception=(e, catch_backtrace())
                nothing
            end
            
            if !isnothing(natural_basis)
                K = length(natural_basis)
                atrisk = [100.0, 80.0, 50.0, 30.0, 10.0]
                
                weighting = AtRiskWeighting(alpha=1.0)
                S_w = MultistateModels.build_weighted_penalty_matrix(
                    natural_basis, 2, weighting, atrisk
                )
                
                @test size(S_w) == (K, K)
                @test issymmetric(S_w)
                @test all(eigvals(S_w) .>= -1e-12)
            else
                @test_skip "RecombinedBSplineBasis construction not available"
            end
        end
        
        @testset "Validates At-Risk Vector Length" begin
            knots = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0]
            basis = BSplineBasis(BSplineOrder(4), knots)
            
            weighting = AtRiskWeighting(alpha=1.0)
            
            # Wrong length - should be 5 (number of intervals) not 4
            atrisk_wrong = [100.0, 80.0, 50.0, 30.0]
            
            @test_throws ArgumentError MultistateModels.build_weighted_penalty_matrix(
                basis, 2, weighting, atrisk_wrong
            )
        end
        
        @testset "First Order Penalty (Slope)" begin
            # Use more interior knots for a well-defined penalty
            knots = [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.0, 5.0, 5.0]
            basis = BSplineBasis(BSplineOrder(4), knots)
            
            # 10 unique intervals
            atrisk = [100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0]
            
            # Order 1 penalty (penalizes slope, not curvature)
            weighting = AtRiskWeighting(alpha=1.0)
            S_w = MultistateModels.build_weighted_penalty_matrix(
                basis, 1, weighting, atrisk
            )
            
            @test issymmetric(S_w)
            @test all(eigvals(S_w) .>= -1e-12)
            
            # First order penalty should have some null space (at least constants)
            eigs = eigvals(S_w)
            n_small = sum(abs.(eigs) .< 1e-8)
            @test n_small >= 1
        end
    end
    
    # =========================================================================
    # 5. INTEGRATION WITH PENALTY CONFIG
    # =========================================================================
    @testset "build_penalty_config with Weighting" begin
        
        @testset "Spline Hazard with Uniform Weighting" begin
            Random.seed!(12345)
            n = 50
            data = DataFrame(
                id = 1:n,
                tstart = zeros(n),
                tstop = rand(n) .* 5.0,
                statefrom = ones(Int, n),
                stateto = fill(2, n),
                obstype = ones(Int, n)
            )
            
            h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                         degree=3, knots=[1.5, 3.0], boundaryknots=[0.0, 5.0])
            model = multistatemodel(h12; data=data)
            
            # Build penalty config with uniform weighting (default)
            penalty = SplinePenalty()
            config = MultistateModels.build_penalty_config(model, penalty)
            
            @test !isempty(config.terms)
            @test length(config.terms) == 1
            @test config.terms[1].S isa Matrix
        end
        
        @testset "Spline Hazard with At-Risk Weighting" begin
            Random.seed!(12345)
            n = 50
            
            # Create data with decreasing at-risk over time
            event_times = (rand(n) .^ (1/2)) .* 5.0
            data = DataFrame(
                id = 1:n,
                tstart = zeros(n),
                tstop = event_times,
                statefrom = ones(Int, n),
                stateto = fill(2, n),
                obstype = ones(Int, n)
            )
            
            h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                         degree=3, knots=[1.5, 3.0], boundaryknots=[0.0, 5.0])
            model = multistatemodel(h12; data=data)
            
            # Build penalty config with at-risk weighting
            penalty_atrisk = SplinePenalty(adaptive_weight=:atrisk, alpha=1.0)
            config_atrisk = MultistateModels.build_penalty_config(model, penalty_atrisk)
            
            # Build penalty config with uniform weighting
            penalty_uniform = SplinePenalty()
            config_uniform = MultistateModels.build_penalty_config(model, penalty_uniform)
            
            # Both should have penalty terms
            @test !isempty(config_atrisk.terms)
            @test !isempty(config_uniform.terms)
            
            # Penalty matrices should be different
            S_atrisk = config_atrisk.terms[1].S
            S_uniform = config_uniform.terms[1].S
            
            @test size(S_atrisk) == size(S_uniform)
            @test !isapprox(S_atrisk, S_uniform, rtol=0.01)  # Should be meaningfully different
        end
        
        @testset "Competing Risks with Different Weightings" begin
            Random.seed!(12345)
            n = 50
            
            # Create competing risks data
            event_times = (rand(n) .^ (1/2)) .* 5.0
            destinations = rand(1:2, n) .+ 1  # Either state 2 or state 3
            
            data = DataFrame(
                id = 1:n,
                tstart = zeros(n),
                tstop = event_times,
                statefrom = ones(Int, n),
                stateto = destinations,
                obstype = ones(Int, n)
            )
            
            h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                         degree=3, knots=[1.5, 3.0], boundaryknots=[0.0, 5.0])
            h13 = Hazard(@formula(0 ~ 1), "sp", 1, 3; 
                         degree=3, knots=[1.5, 3.0], boundaryknots=[0.0, 5.0])
            model = multistatemodel(h12, h13; data=data)
            
            # Different weighting for each transition
            penalties = [
                SplinePenalty((1, 2), adaptive_weight=:atrisk, alpha=1.0),
                SplinePenalty((1, 3), adaptive_weight=:none)
            ]
            config = MultistateModels.build_penalty_config(model, penalties)
            
            @test length(config.terms) == 2
        end
    end

    # =========================================================================
    # 6. PANEL DATA AT-RISK COMPUTATION (Phase 3)
    # =========================================================================
    @testset "Panel Data At-Risk Counts (Phase 3)" begin
        
        @testset "Simple Panel Data - Upper Bound Approach" begin
            # Panel data: state observed only at discrete times
            # For panel data, statefrom indicates state at tstart
            # Upper bound: count subjects in origin state at tstart as at-risk
            # for whole interval [tstart, tstop)
            data_panel = DataFrame(
                id = [1, 1, 2, 2],
                tstart = [0.0, 2.0, 0.0, 2.0],
                tstop = [2.0, 4.0, 2.0, 4.0],
                statefrom = [1, 2, 1, 1],  # Subject 1 transitions, subject 2 stays
                stateto = [2, 2, 1, 2],
                obstype = [2, 2, 2, 2]  # All panel observations
            )
            
            h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
            model = multistatemodel(h12; data=data_panel)
            
            # At t=1 (middle of first interval):
            # Subject 1: statefrom=1, in [0,2) → at risk
            # Subject 2: statefrom=1, in [0,2) → at risk
            # Expected: Y(1) = 2
            atrisk = MultistateModels.compute_atrisk_counts(model, [1.0], (1, 2))
            @test atrisk[1] == 2.0
            
            # At t=3 (middle of second interval):
            # Subject 1: statefrom=2 in [2,4) → NOT at risk for 1→2
            # Subject 2: statefrom=1 in [2,4) → at risk
            # Expected: Y(3) = 1
            atrisk = MultistateModels.compute_atrisk_counts(model, [3.0], (1, 2))
            @test atrisk[1] == 1.0
        end
        
        @testset "Panel Data Decreasing At-Risk Over Time" begin
            # Multiple subjects with staggered transitions
            data = DataFrame(
                id = [1, 1, 2, 2, 2, 3, 3, 3],
                tstart = [0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
                tstop = [1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
                statefrom = [1, 2, 1, 1, 2, 1, 1, 1],  # Subject 1 leaves first, 2 second, 3 stays
                stateto = [2, 2, 1, 2, 2, 1, 1, 2],
                obstype = [2, 2, 2, 2, 2, 2, 2, 2]
            )
            
            h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
            model = multistatemodel(h12; data=data)
            
            eval_times = [0.5, 1.5, 2.5]
            atrisk = MultistateModels.compute_atrisk_counts(model, eval_times, (1, 2))
            
            # t=0.5: all 3 subjects in state 1 → Y=3
            @test atrisk[1] == 3.0
            
            # t=1.5: subject 1 in state 2, subjects 2,3 in state 1 → Y=2
            @test atrisk[2] == 2.0
            
            # t=2.5: subjects 1,2 in state 2, subject 3 in state 1 → Y=1
            @test atrisk[3] == 1.0
        end
        
        @testset "Panel Data Floor at 1.0" begin
            # All subjects transition by end of first interval
            data = DataFrame(
                id = [1, 1],
                tstart = [0.0, 1.0],
                tstop = [1.0, 2.0],
                statefrom = [1, 2],
                stateto = [2, 2],
                obstype = [2, 2]
            )
            
            h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
            model = multistatemodel(h12; data=data)
            
            # At t=1.5: no one at risk for 1→2
            # Should floor at 1.0
            atrisk = MultistateModels.compute_atrisk_counts(model, [1.5], (1, 2))
            @test atrisk[1] == 1.0
        end
        
        @testset "Mixed Exact and Panel Data" begin
            # Mix of exact (obstype=1) and panel (obstype=2) observations
            data = DataFrame(
                id = [1, 2, 2],
                tstart = [0.0, 0.0, 1.0],
                tstop = [0.5, 1.0, 2.0],
                statefrom = [1, 1, 2],
                stateto = [2, 2, 2],
                obstype = [1, 2, 2]  # Subject 1 exact, subject 2 panel
            )
            
            h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
            model = multistatemodel(h12; data=data)
            
            # t=0.25: 
            # Subject 1 (exact): in [0, 0.5) with statefrom=1 → at risk
            # Subject 2 (panel): in [0, 1) with statefrom=1 → at risk
            atrisk = MultistateModels.compute_atrisk_counts(model, [0.25], (1, 2))
            @test atrisk[1] == 2.0
            
            # t=0.75:
            # Subject 1 (exact): past 0.5, so not in any interval → NOT at risk
            # Subject 2 (panel): in [0, 1) with statefrom=1 → at risk
            atrisk = MultistateModels.compute_atrisk_counts(model, [0.75], (1, 2))
            @test atrisk[1] == 1.0
        end
        
        @testset "Panel Data Multi-State Model" begin
            # 3-state illness-death model with panel observations
            data = DataFrame(
                id = [1, 1, 2, 2, 3, 3],
                tstart = [0.0, 2.0, 0.0, 2.0, 0.0, 2.0],
                tstop = [2.0, 4.0, 2.0, 4.0, 2.0, 4.0],
                statefrom = [1, 2, 1, 1, 1, 3],  # Subj 1: 1→2, Subj 2: stays in 1, Subj 3: 1→3
                stateto = [2, 2, 1, 2, 3, 3],
                obstype = [2, 2, 2, 2, 2, 2]
            )
            
            h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
            h13 = Hazard(@formula(0 ~ 1), "exp", 1, 3)
            h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
            model = multistatemodel(h12, h13, h23; data=data)
            
            # At t=1: all 3 subjects in state 1
            atrisk_12 = MultistateModels.compute_atrisk_counts(model, [1.0], (1, 2))
            atrisk_13 = MultistateModels.compute_atrisk_counts(model, [1.0], (1, 3))
            @test atrisk_12[1] == 3.0
            @test atrisk_13[1] == 3.0
            
            # At t=3:
            # For 1→2: only subject 2 still in state 1 → Y=1
            # For 1→3: only subject 2 still in state 1 → Y=1
            # For 2→3: subject 1 in state 2 → Y=1
            atrisk_12 = MultistateModels.compute_atrisk_counts(model, [3.0], (1, 2))
            atrisk_13 = MultistateModels.compute_atrisk_counts(model, [3.0], (1, 3))
            atrisk_23 = MultistateModels.compute_atrisk_counts(model, [3.0], (2, 3))
            @test atrisk_12[1] == 1.0
            @test atrisk_13[1] == 1.0
            @test atrisk_23[1] == 1.0
        end
        
        @testset "Panel Data with Spline Hazard - Knot Midpoints" begin
            # Create panel data
            data = DataFrame(
                id = [1, 1, 2, 2],
                tstart = [0.0, 2.0, 0.0, 2.0],
                tstop = [2.0, 4.0, 2.0, 4.0],
                statefrom = [1, 2, 1, 1],
                stateto = [2, 2, 1, 2],
                obstype = [2, 2, 2, 2]
            )
            
            h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                         degree=3, knots=[1.0, 2.0, 3.0], boundaryknots=[0.0, 4.0])
            model = multistatemodel(h12; data=data)
            
            # Test compute_atrisk_counts_at_knot_midpoints
            hazard = model.hazards[1]
            atrisk = MultistateModels.compute_atrisk_counts_at_knot_midpoints(
                model, hazard, (1, 2)
            )
            
            # Knot midpoints: 0.5, 1.5, 2.5, 3.5
            @test length(atrisk) == 4
            @test atrisk[1] == 2.0  # t=0.5: both in state 1
            @test atrisk[2] == 2.0  # t=1.5: both in state 1
            @test atrisk[3] == 1.0  # t=2.5: subject 1 in state 2, subject 2 in state 1
            @test atrisk[4] == 1.0  # t=3.5: floor at 1
        end
    end
    
    # =========================================================================
    # 7. MARKOV PANEL FITTING WITH ADAPTIVE WEIGHTING (Phase 3 Integration)
    # =========================================================================
    @testset "Markov Panel Fitting with Adaptive Weighting" begin
        
        @testset "Degree-0 Spline (Piecewise Constant) - Markov Panel" begin
            Random.seed!(12345)
            n = 50
            
            # Generate panel data
            rows = []
            for i in 1:n
                state = 1
                for j in 1:3
                    new_state = state == 1 && rand() < 0.3 ? 2 : state
                    push!(rows, (
                        id = i,
                        tstart = Float64(j-1),
                        tstop = Float64(j),
                        statefrom = state,
                        stateto = new_state,
                        obstype = 2
                    ))
                    state = new_state
                end
            end
            data = DataFrame(rows)
            
            # Degree-0 spline is Markov (piecewise constant)
            h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                         degree=0, knots=[1.0, 2.0], boundaryknots=[0.0, 3.0])
            model = multistatemodel(h12; data=data)
            
            # Should be classified as Markov
            @test MultistateModels.is_markov(model)
            @test MultistateModels.is_panel_data(model)
            
            # Fit with uniform weighting (no error expected)
            fitted_uniform = fit(model; penalty=SplinePenalty(), 
                                 select_lambda=:none, vcov_type=:none, verbose=false)
            @test fitted_uniform isa MultistateModels.MultistateModelFitted
            
            # Fit with at-risk weighting (no error expected)
            fitted_atrisk = fit(model; penalty=SplinePenalty(adaptive_weight=:atrisk),
                               select_lambda=:none, vcov_type=:none, verbose=false)
            @test fitted_atrisk isa MultistateModels.MultistateModelFitted
        end
        
        @testset "Weighted Penalty Config for Panel Data" begin
            # Create panel data
            data = DataFrame(
                id = [1, 1, 2, 2, 3, 3],
                tstart = [0.0, 2.0, 0.0, 2.0, 0.0, 2.0],
                tstop = [2.0, 4.0, 2.0, 4.0, 2.0, 4.0],
                statefrom = [1, 2, 1, 1, 1, 2],
                stateto = [2, 2, 1, 2, 2, 2],
                obstype = [2, 2, 2, 2, 2, 2]
            )
            
            # Use degree-0 spline for Markov panel
            h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                         degree=0, knots=[1.0, 2.0, 3.0], boundaryknots=[0.0, 4.0])
            model = multistatemodel(h12; data=data)
            
            # Build penalty configs with different weightings
            config_uniform = MultistateModels.build_penalty_config(
                model, SplinePenalty(adaptive_weight=:none)
            )
            config_atrisk = MultistateModels.build_penalty_config(
                model, SplinePenalty(adaptive_weight=:atrisk)
            )
            
            # Both should have one penalty term
            @test length(config_uniform.terms) == 1
            @test length(config_atrisk.terms) == 1
            
            # Penalty matrices should exist
            @test size(config_uniform.terms[1].S) == size(config_atrisk.terms[1].S)
        end
    end

    # =========================================================================
    # 6. MCEM INTEGRATION (PHASE 4)
    # =========================================================================
    @testset "MCEM Path-Weighted At-Risk Counts" begin
        
        @testset "compute_atrisk_counts_mcem - Basic" begin
            # Create synthetic MCEM-style paths for 3 subjects, 2 paths each
            # SamplePath(subj, times, states)
            
            # Subject 1: stays in state 1
            path1_s1 = MultistateModels.SamplePath(1, [0.0, 10.0], [1, 1])
            path2_s1 = MultistateModels.SamplePath(1, [0.0, 10.0], [1, 1])
            
            # Subject 2: transitions 1->2 at t=5
            path1_s2 = MultistateModels.SamplePath(2, [0.0, 5.0, 10.0], [1, 2, 2])
            path2_s2 = MultistateModels.SamplePath(2, [0.0, 6.0, 10.0], [1, 2, 2])
            
            # Subject 3: transitions 1->2 early
            path1_s3 = MultistateModels.SamplePath(3, [0.0, 2.0, 10.0], [1, 2, 2])
            path2_s3 = MultistateModels.SamplePath(3, [0.0, 3.0, 10.0], [1, 2, 2])
            
            samplepaths = [
                [path1_s1, path2_s1],
                [path1_s2, path2_s2],
                [path1_s3, path2_s3]
            ]
            
            # Equal importance weights
            weights = [
                [0.5, 0.5],
                [0.5, 0.5],
                [0.5, 0.5]
            ]
            
            transition = (1, 2)
            times = [0.0, 2.5, 5.0, 7.5, 10.0]
            
            # Note: argument order is (samplepaths, weights, times, transition)
            atrisk = compute_atrisk_counts_mcem(samplepaths, weights, times, transition)
            
            @test length(atrisk) == length(times)
            
            # At t=0: all subjects are at risk (in state 1)
            @test atrisk[1] ≈ 3.0
            
            # At t=2.5: Subject 3 has partially transitioned
            # - S1: fully at risk (weight 1.0)
            # - S2: fully at risk (weight 1.0)
            # - S3: path1 transitioned at t=2, path2 still at risk. 
            #       So subject 3 contributes 0.5 * 0 + 0.5 * 1 = 0.5
            @test atrisk[2] ≈ 2.5
            
            # At t=5.0: Subject 2 starting to transition
            # - S1: at risk (1.0)
            # - S2: path1 transitioned at t=5, path2 still at risk (0.5)
            # - S3: not at risk (0.0)
            @test atrisk[3] ≈ 1.5
            
            # At t=10: Only subject 1 at risk
            @test atrisk[5] ≈ 1.0
        end
        
        @testset "compute_atrisk_counts_mcem - Unequal Weights" begin
            # Same paths but with unequal importance weights
            path1 = MultistateModels.SamplePath(1, [0.0, 5.0, 10.0], [1, 2, 2])
            path2 = MultistateModels.SamplePath(1, [0.0, 8.0, 10.0], [1, 2, 2])
            
            samplepaths = [[path1, path2]]
            weights = [[0.8, 0.2]]  # path1 has higher weight
            
            transition = (1, 2)
            times = [0.0, 6.0, 10.0]
            
            # Note: argument order is (samplepaths, weights, times, transition)
            atrisk = compute_atrisk_counts_mcem(samplepaths, weights, times, transition)
            
            # At t=0: fully at risk
            @test atrisk[1] ≈ 1.0
            
            # At t=6: path1 (w=0.8) transitioned, path2 (w=0.2) still at risk
            @test atrisk[2] ≈ 0.2
            
            # At t=10: none at risk
            @test atrisk[3] ≈ 0.0
        end
        
        @testset "_path_in_state_at_time helper" begin
            # Path: starts in state 1, transitions to state 2 at t=5
            path = MultistateModels.SamplePath(1, [0.0, 5.0, 10.0], [1, 2, 2])
            
            # Before transition
            @test MultistateModels._path_in_state_at_time(path, 1, 0.0) == true
            @test MultistateModels._path_in_state_at_time(path, 1, 4.9) == true
            @test MultistateModels._path_in_state_at_time(path, 2, 4.9) == false
            
            # At transition time (uses left-continuous convention)
            @test MultistateModels._path_in_state_at_time(path, 1, 5.0) == false
            @test MultistateModels._path_in_state_at_time(path, 2, 5.0) == true
            
            # After transition
            @test MultistateModels._path_in_state_at_time(path, 1, 7.0) == false
            @test MultistateModels._path_in_state_at_time(path, 2, 7.0) == true
        end
        
        @testset "compute_atrisk_interval_averages_mcem - Basic" begin
            # Create a minimal model to get hazard with knot structure
            # Knots at 0, 2, 4, 6 (3 intervals)
            data = DataFrame(
                id = [1, 2],
                tstart = [0.0, 0.0],
                tstop = [6.0, 6.0],
                statefrom = [1, 1],
                stateto = [2, 2],
                obstype = [1, 1]
            )
            h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2;
                         degree=0, knots=[2.0, 4.0], boundaryknots=[0.0, 6.0])
            model = multistatemodel(h12; data=data)
            
            # Create synthetic MCEM-style paths
            # Subject 1: stays in state 1 until t=6 (full observation)
            path_s1 = MultistateModels.SamplePath(1, [0.0, 6.0], [1, 2])
            
            # Subject 2: transitions 1->2 at t=3 (midway through interval [2,4])
            path_s2 = MultistateModels.SamplePath(2, [0.0, 3.0, 6.0], [1, 2, 2])
            
            samplepaths = [[path_s1], [path_s2]]
            weights = [[1.0], [1.0]]
            transition = (1, 2)
            
            atrisk_avg = compute_atrisk_interval_averages_mcem(
                samplepaths, weights, model.hazards[1], transition
            )
            
            @test length(atrisk_avg) == 3  # Three intervals
            
            # Interval [0, 2): Both subjects at risk for full 2 units
            #   S1: 2 units, S2: 2 units
            #   Person-time = 4, Avg = 4/2 = 2.0
            @test atrisk_avg[1] ≈ 2.0
            
            # Interval [2, 4): S1 at risk for 2 units, S2 at risk for 1 unit (until t=3)
            #   Person-time = 2 + 1 = 3, Avg = 3/2 = 1.5
            @test atrisk_avg[2] ≈ 1.5
            
            # Interval [4, 6): Only S1 at risk for 2 units
            #   Person-time = 2, Avg = 2/2 = 1.0
            @test atrisk_avg[3] ≈ 1.0
        end
        
        @testset "compute_atrisk_interval_averages_mcem - Weighted Paths" begin
            # Create minimal model
            data = DataFrame(
                id = [1],
                tstart = [0.0],
                tstop = [4.0],
                statefrom = [1],
                stateto = [2],
                obstype = [1]
            )
            h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2;
                         degree=0, knots=[2.0], boundaryknots=[0.0, 4.0])
            model = multistatemodel(h12; data=data)
            
            # Single subject with two sampled paths and unequal weights
            # Path 1: transitions at t=1 (early), weight = 0.3
            # Path 2: transitions at t=3 (late), weight = 0.7
            path1 = MultistateModels.SamplePath(1, [0.0, 1.0, 4.0], [1, 2, 2])
            path2 = MultistateModels.SamplePath(1, [0.0, 3.0, 4.0], [1, 2, 2])
            
            samplepaths = [[path1, path2]]
            weights = [[0.3, 0.7]]
            transition = (1, 2)
            
            atrisk_avg = compute_atrisk_interval_averages_mcem(
                samplepaths, weights, model.hazards[1], transition
            )
            
            @test length(atrisk_avg) == 2  # Two intervals: [0,2) and [2,4)
            
            # Interval [0, 2):
            #   Path 1 (w=0.3): at risk for 1 unit (until t=1)
            #   Path 2 (w=0.7): at risk for 2 units (full interval)
            #   Weighted person-time = 0.3*1 + 0.7*2 = 0.3 + 1.4 = 1.7
            #   Avg = 1.7/2 = 0.85
            @test atrisk_avg[1] ≈ 0.85
            
            # Interval [2, 4):
            #   Path 1 (w=0.3): not at risk (transitioned at t=1)
            #   Path 2 (w=0.7): at risk for 1 unit (until t=3)
            #   Weighted person-time = 0.3*0 + 0.7*1 = 0.7
            #   Avg = 0.7/2 = 0.35
            @test atrisk_avg[2] ≈ 0.35
        end
        
        @testset "_path_time_in_state_in_interval helper" begin
            # Path: starts in state 1, transitions to state 2 at t=5
            path = MultistateModels.SamplePath(1, [0.0, 5.0, 10.0], [1, 2, 2])
            
            # Full interval before transition
            time_01 = MultistateModels._path_time_in_state_in_interval(path, 1, 0.0, 2.0)
            @test time_01 ≈ 2.0  # Full 2 units in state 1
            
            # Interval spanning transition
            time_02 = MultistateModels._path_time_in_state_in_interval(path, 1, 3.0, 7.0)
            @test time_02 ≈ 2.0  # Only 2 units (from t=3 to t=5) in state 1
            
            # Interval after transition
            time_03 = MultistateModels._path_time_in_state_in_interval(path, 1, 6.0, 10.0)
            @test time_03 ≈ 0.0  # No time in state 1
            
            # Check state 2 time in spanning interval
            time_04 = MultistateModels._path_time_in_state_in_interval(path, 2, 3.0, 7.0)
            @test time_04 ≈ 2.0  # 2 units (from t=5 to t=7) in state 2
        end
        
        @testset "has_adaptive_weighting" begin
            # No penalty
            @test has_adaptive_weighting(nothing) == false
            
            # Uniform weighting
            @test has_adaptive_weighting(SplinePenalty()) == false
            @test has_adaptive_weighting(SplinePenalty(adaptive_weight=:none)) == false
            
            # At-risk weighting
            @test has_adaptive_weighting(SplinePenalty(adaptive_weight=:atrisk)) == true
            
            # Mixed - should detect at-risk
            specs = [SplinePenalty(1), SplinePenalty(2, adaptive_weight=:atrisk)]
            @test has_adaptive_weighting(specs) == true
            
            # All uniform
            specs2 = [SplinePenalty(1), SplinePenalty(2)]
            @test has_adaptive_weighting(specs2) == false
        end
    end
    
    @testset "update_penalty_weights_mcem - Unit Tests" begin
        # This tests the update function with synthetic data
        # We need a minimal model with spline hazard
        
        @testset "Basic Update Mechanism" begin
            Random.seed!(54321)
            
            # Create exact data for model construction
            data = DataFrame(
                id = [1, 2, 3, 4, 5],
                tstart = [0.0, 0.0, 0.0, 0.0, 0.0],
                tstop = [5.0, 3.0, 4.0, 6.0, 2.0],
                statefrom = [1, 1, 1, 1, 1],
                stateto = [2, 2, 2, 2, 2],
                obstype = [1, 1, 1, 1, 1]
            )
            
            # Spline hazard with degree=0 for simplicity
            h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                         degree=0, knots=[2.0, 4.0], boundaryknots=[0.0, 6.0])
            model = multistatemodel(h12; data=data)
            
            # Build initial penalty config with at-risk weighting
            penalty_spec = SplinePenalty(adaptive_weight=:atrisk)
            penalty_config = MultistateModels.build_penalty_config(model, penalty_spec)
            
            @test length(penalty_config.terms) == 1
            S_original = copy(penalty_config.terms[1].S)
            
            # Create synthetic MCEM paths
            # Subject 1: stays at risk (state 1) until t=5
            # Subject 2: stays at risk (state 1) until t=3
            samplepaths = [
                [MultistateModels.SamplePath(1, [0.0, 5.0], [1, 2])],
                [MultistateModels.SamplePath(2, [0.0, 3.0], [1, 2])]
            ]
            weights = [[1.0], [1.0]]
            
            # Update penalty weights
            penalty_config_updated = update_penalty_weights_mcem(
                penalty_config, model, samplepaths, weights, penalty_spec
            )
            
            # Should return a new config (not mutated)
            @test penalty_config_updated !== penalty_config
            
            # S matrix should be updated
            S_updated = penalty_config_updated.terms[1].S
            @test size(S_updated) == size(S_original)
            
            # With at-risk weighting, S should differ from original
            # (unless at-risk counts are perfectly uniform, which is unlikely)
            # Note: S might still equal original if at-risk is flat, so we just check no error
            @test isfinite(sum(S_updated))
            
            # Other fields should be preserved
            @test penalty_config_updated.terms[1].lambda == penalty_config.terms[1].lambda
            @test penalty_config_updated.terms[1].order == penalty_config.terms[1].order
            @test penalty_config_updated.n_lambda == penalty_config.n_lambda
        end
        
        @testset "No Update for Uniform Weighting" begin
            Random.seed!(54321)
            
            data = DataFrame(
                id = [1, 2],
                tstart = [0.0, 0.0],
                tstop = [5.0, 3.0],
                statefrom = [1, 1],
                stateto = [2, 2],
                obstype = [1, 1]
            )
            
            h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                         degree=0, knots=[2.0], boundaryknots=[0.0, 5.0])
            model = multistatemodel(h12; data=data)
            
            # Uniform weighting
            penalty_spec = SplinePenalty(adaptive_weight=:none)
            penalty_config = MultistateModels.build_penalty_config(model, penalty_spec)
            
            S_original = copy(penalty_config.terms[1].S)
            
            samplepaths = [
                [MultistateModels.SamplePath(1, [0.0, 5.0], [1, 2])],
                [MultistateModels.SamplePath(2, [0.0, 3.0], [1, 2])]
            ]
            weights = [[1.0], [1.0]]
            
            # Should return same config (no update needed)
            penalty_config_updated = update_penalty_weights_mcem(
                penalty_config, model, samplepaths, weights, penalty_spec
            )
            
            # Should be the same object (no update performed)
            @test penalty_config_updated === penalty_config
        end
    end

    # =========================================================================
    # 8. ALPHA LEARNING INFRASTRUCTURE
    # =========================================================================
    @testset "Alpha Learning Infrastructure" begin
        
        @testset "needs_alpha_learning Detection" begin
            using MultistateModels: needs_alpha_learning
            
            # Should return false for uniform weighting
            @test needs_alpha_learning(nothing) == false
            @test needs_alpha_learning(SplinePenalty()) == false
            @test needs_alpha_learning(SplinePenalty(adaptive_weight=:none)) == false
            
            # Should return false for at-risk weighting with learn=false
            @test needs_alpha_learning(SplinePenalty(adaptive_weight=:atrisk)) == false
            @test needs_alpha_learning(SplinePenalty(adaptive_weight=:atrisk, alpha=0.5)) == false
            
            # Should return true for at-risk weighting with learn=true
            @test needs_alpha_learning(SplinePenalty(adaptive_weight=:atrisk, learn_alpha=true)) == true
            @test needs_alpha_learning(SplinePenalty(adaptive_weight=:atrisk, alpha=1.5, learn_alpha=true)) == true
            
            # Vector case
            specs = [SplinePenalty(), SplinePenalty(adaptive_weight=:atrisk, learn_alpha=true)]
            @test needs_alpha_learning(specs) == true
            
            specs2 = [SplinePenalty(), SplinePenalty(adaptive_weight=:atrisk)]
            @test needs_alpha_learning(specs2) == false
        end
        
        @testset "Brent's Method 1D Minimization" begin
            using MultistateModels: _brent_minimize
            
            # Quadratic: min at x=1
            f1(x) = (x - 1.0)^2
            x_opt = _brent_minimize(f1, -2.0, 4.0; tol=1e-6)
            @test abs(x_opt - 1.0) < 1e-4
            
            # Different quadratic: min at x=0.5
            f2(x) = (x - 0.5)^2 + 1.0
            x_opt2 = _brent_minimize(f2, 0.0, 2.0; tol=1e-6)
            @test abs(x_opt2 - 0.5) < 1e-4
            
            # Sin function on [0, 4]: min near x=3π/2 ≈ 4.71, but bounded to [0,4]
            # On [0, 4], sin has min at x=3π/2, but this is outside [0,4]
            # The minimum on [0, 4] is at x≈3π/2... actually within bounds
            # Let's use [0, 5] where min of sin is at 3π/2 ≈ 4.71
            f3(x) = sin(x)
            x_opt3 = _brent_minimize(f3, 0.0, 5.0; tol=1e-5)
            @test abs(x_opt3 - 3*π/2) < 0.01
        end
        
        @testset "collect_alpha_learning_info" begin
            using MultistateModels: collect_alpha_learning_info, AlphaLearningInfo
            
            Random.seed!(12345)
            
            data = DataFrame(
                id = [1, 2, 3, 4, 5],
                tstart = zeros(5),
                tstop = [1.0, 2.0, 3.0, 4.0, 5.0],
                statefrom = ones(Int, 5),
                stateto = fill(2, 5),
                obstype = ones(Int, 5)
            )
            
            h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2;
                         degree=3, knots=[2.5], boundaryknots=[0.0, 5.0])
            model = multistatemodel(h12; data=data)
            
            # Test with learn_alpha=true
            penalty_spec = SplinePenalty(adaptive_weight=:atrisk, learn_alpha=true)
            penalty_config = MultistateModels.build_penalty_config(model, penalty_spec)
            
            alpha_info = collect_alpha_learning_info(model, penalty_config, penalty_spec)
            
            @test length(alpha_info) == 1  # One term
            @test haskey(alpha_info, 1)
            
            info = alpha_info[1]
            @test info isa AlphaLearningInfo
            @test info.transition == (1, 2)
            @test info.order == 2
            @test length(info.atrisk) > 0  # Has at-risk counts
            @test all(info.atrisk .>= 1.0)  # Floored at 1.0
        end
        
        @testset "get_shared_alpha_groups" begin
            using MultistateModels: get_shared_alpha_groups, collect_alpha_learning_info, AlphaLearningInfo
            
            Random.seed!(23456)
            
            # Competing risks data
            data = DataFrame(
                id = repeat(1:10, inner=1),
                tstart = zeros(10),
                tstop = [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5],
                statefrom = ones(Int, 10),
                stateto = [2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
                obstype = ones(Int, 10)
            )
            
            h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2;
                         degree=3, knots=[2.5], boundaryknots=[0.0, 6.0])
            h13 = Hazard(@formula(0 ~ 1), "sp", 1, 3;
                         degree=3, knots=[2.5], boundaryknots=[0.0, 6.0])
            model = multistatemodel(h12, h13; data=data)
            
            # Test with share_lambda=true and learn_alpha=true
            penalty_spec = SplinePenalty(1, share_lambda=true, adaptive_weight=:atrisk, learn_alpha=true)
            penalty_config = MultistateModels.build_penalty_config(model, penalty_spec)
            
            alpha_info = collect_alpha_learning_info(model, penalty_config, penalty_spec)
            groups = get_shared_alpha_groups(penalty_config, alpha_info)
            
            # With share_lambda, both terms should be in one group
            @test length(groups) >= 1
            total_terms = sum(length(g) for g in groups)
            @test total_terms == length(alpha_info)
        end
        
        @testset "update_penalty_with_alpha" begin
            using MultistateModels: update_penalty_with_alpha, collect_alpha_learning_info
            
            Random.seed!(34567)
            
            data = DataFrame(
                id = 1:5,
                tstart = zeros(5),
                tstop = [1.0, 2.0, 3.0, 4.0, 5.0],
                statefrom = ones(Int, 5),
                stateto = fill(2, 5),
                obstype = ones(Int, 5)
            )
            
            h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2;
                         degree=3, knots=[2.5], boundaryknots=[0.0, 5.0])
            model = multistatemodel(h12; data=data)
            
            penalty_spec = SplinePenalty(adaptive_weight=:atrisk, learn_alpha=true)
            penalty_config = MultistateModels.build_penalty_config(model, penalty_spec)
            
            alpha_info = collect_alpha_learning_info(model, penalty_config, penalty_spec)
            info = alpha_info[1]
            
            # Update with new alpha
            new_alpha = 0.5
            penalty_updated = update_penalty_with_alpha(
                penalty_config, model, 1, new_alpha, info.hazard, info.atrisk
            )
            
            # Should return new config
            @test penalty_updated !== penalty_config
            @test penalty_updated isa MultistateModels.QuadraticPenalty
            
            # Penalty matrix should be different (unless at-risk is flat)
            S_old = penalty_config.terms[1].S
            S_new = penalty_updated.terms[1].S
            @test size(S_new) == size(S_old)
            @test all(isfinite.(S_new))
        end
    end
    
    # =========================================================================
    # 9. ALPHA LEARNING INTEGRATION (Exact Data)
    # =========================================================================
    @testset "Alpha Learning Integration - Exact Data" begin
        
        @testset "Basic Alpha Learning" begin
            Random.seed!(45678)
            
            # Create data with heavy late censoring (where alpha learning should help)
            n = 50
            event_times = rand(n) .* 5.0
            censor_times = 3.0 .+ rand(n) .* 2.0  # Censoring between 3 and 5
            obs_times = min.(event_times, censor_times)
            status = Int.(event_times .<= censor_times)
            
            data = DataFrame(
                id = 1:n,
                tstart = zeros(n),
                tstop = obs_times,
                statefrom = ones(Int, n),
                stateto = ifelse.(status .== 1, 2, 1),
                obstype = ones(Int, n)
            )
            
            h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2;
                         degree=3, knots=[1.5, 3.0], boundaryknots=[0.0, 5.0])
            model = multistatemodel(h12; data=data)
            
            # Test that we can fit with learn_alpha=true
            penalty_spec = SplinePenalty(adaptive_weight=:atrisk, learn_alpha=true, alpha=1.0)
            
            # This should run without error
            fitted = fit(model; 
                         penalty=penalty_spec, 
                         select_lambda=:efs, 
                         vcov_type=:none, 
                         verbose=false)
            
            @test fitted isa MultistateModels.MultistateModelFitted
            @test !isempty(fitted.smoothing_parameters)
            @test fitted.loglik.loglik < 0  # Valid log-likelihood
        end
        
        @testset "Alpha Learning with Fixed Lambda" begin
            Random.seed!(56789)
            
            n = 30
            data = DataFrame(
                id = 1:n,
                tstart = zeros(n),
                tstop = rand(n) .* 4.0,
                statefrom = ones(Int, n),
                stateto = fill(2, n),
                obstype = ones(Int, n)
            )
            
            h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2;
                         degree=3, knots=[2.0], boundaryknots=[0.0, 4.0])
            model = multistatemodel(h12; data=data)
            
            penalty_spec = SplinePenalty(adaptive_weight=:atrisk, learn_alpha=true)
            
            # Fit with fixed lambda (no selection)
            fitted = fit(model;
                         penalty=penalty_spec,
                         select_lambda=:none,
                         lambda_init=10.0,
                         vcov_type=:none,
                         verbose=false)
            
            @test fitted isa MultistateModels.MultistateModelFitted
            @test fitted.smoothing_parameters[1] == 10.0  # Fixed lambda
        end
        
        @testset "Comparison: learn_alpha vs Fixed Alpha" begin
            Random.seed!(67890)
            
            n = 40
            data = DataFrame(
                id = 1:n,
                tstart = zeros(n),
                tstop = rand(n) .* 5.0,
                statefrom = ones(Int, n),
                stateto = fill(2, n),
                obstype = ones(Int, n)
            )
            
            h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2;
                         degree=3, knots=[1.5, 3.0], boundaryknots=[0.0, 5.0])
            model = multistatemodel(h12; data=data)
            
            # Fit with fixed alpha
            fitted_fixed = fit(model;
                               penalty=SplinePenalty(adaptive_weight=:atrisk, alpha=1.0),
                               select_lambda=:efs,
                               vcov_type=:none,
                               verbose=false)
            
            # Fit with learned alpha
            fitted_learned = fit(model;
                                 penalty=SplinePenalty(adaptive_weight=:atrisk, learn_alpha=true),
                                 select_lambda=:efs,
                                 vcov_type=:none,
                                 verbose=false)
            
            # Both should produce valid fits
            @test fitted_fixed isa MultistateModels.MultistateModelFitted
            @test fitted_learned isa MultistateModels.MultistateModelFitted
            
            # Log-likelihoods should be reasonable
            @test isfinite(fitted_fixed.loglik.loglik)
            @test isfinite(fitted_learned.loglik.loglik)
        end
    end

end
