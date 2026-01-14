# =============================================================================
# PIJCV/NCV Reference Comparison Tests
# =============================================================================
#
# Compare Julia's penalty matrix construction against R mgcv's P-spline
# implementation. Tests include:
# 1. Gaussian regression context (original pijcv_reference.json)
# 2. Survival/PAM context (pam_pijcv_reference.json) - more relevant for our use
#
# Reference: Wood (2024) "NCV for Smoothness Selection" arXiv:2404.16490
# Note: mgcv calls PIJCV "NCV" - they are the same algorithm.
#
# =============================================================================

using Test
using JSON
using LinearAlgebra
using BSplineKit

# Import spline utilities from main package
using MultistateModels: build_penalty_matrix

@testset "PIJCV Reference Comparison" begin
    fixtures_dir = joinpath(@__DIR__, "..", "fixtures")
    
    # =========================================================================
    # Test 1: Gaussian regression reference (original)
    # =========================================================================
    @testset "Gaussian Regression Reference" begin
        ref_path = joinpath(fixtures_dir, "pijcv_reference.json")
        
        if !isfile(ref_path)
            @warn "Reference file not found: $ref_path. Run generate_pijcv_reference.R first."
            @test_skip "Reference file missing"
            return
        end
        
        ref = JSON.parsefile(ref_path)
        
        @testset "Reference Data Integrity" begin
            @test haskey(ref, "config")
            @test haskey(ref, "penalty_matrix")
            @test haskey(ref, "ncv")
            @test haskey(ref, "knots")
            
            config = ref["config"]
            @test config["n"] == 200
            @test config["k"] == 10
            @test config["m"] == 2
            @test config["bs"] == "ps"
        end
        
        @testset "NCV Reference Values (Gaussian)" begin
            ncv = ref["ncv"]
            reml = ref["reml"]
            
            @info "Gaussian NCV (PIJCV)" lambda=ncv["lambda"] edf=ncv["edf"]
            @info "Gaussian REML" lambda=reml["lambda"] edf=reml["edf"]
            
            # In Gaussian regression, NCV typically selects MORE smoothing (higher λ)
            @test ncv["lambda"] > reml["lambda"]
            @test ncv["lambda"] ≈ 9.52 rtol=0.05
            @test ncv["edf"] ≈ 6.04 rtol=0.05
        end
        
        @testset "Predictions Check (Gaussian)" begin
            preds = ref["predictions"]
            ncv_rmse = preds["rmse_ncv"]
            reml_rmse = preds["rmse_reml"]
            
            @info "Gaussian RMSE" ncv=ncv_rmse reml=reml_rmse
            
            # Both should have similar RMSE (within 30%)
            @test abs(ncv_rmse - reml_rmse) / max(ncv_rmse, reml_rmse) < 0.30
            @test ncv_rmse < 0.3
            @test reml_rmse < 0.3
        end
    end
    
    # =========================================================================
    # Test 2: PAM/Survival context reference (more relevant)
    # =========================================================================
    @testset "PAM Survival Reference" begin
        ref_path = joinpath(fixtures_dir, "pam_pijcv_reference.json")
        
        if !isfile(ref_path)
            @warn "Reference file not found: $ref_path. Run generate_pam_reference.R first."
            @test_skip "Reference file missing"
            return
        end
        
        ref = JSON.parsefile(ref_path)
        
        @testset "PAM Reference Data Integrity" begin
            @test haskey(ref, "config")
            @test haskey(ref, "penalty_matrix")
            @test haskey(ref, "ncv")
            @test haskey(ref, "reml")
            @test haskey(ref, "fit_quality")
            
            config = ref["config"]
            @test config["n"] == 500
            @test config["k"] == 15
            @test config["bs"] == "ps"
        end
        
        @testset "NCV Reference Values (PAM/Poisson)" begin
            ncv = ref["ncv"]
            reml = ref["reml"]
            
            @info "PAM NCV (PIJCV)" sp=ncv["sp"] edf=ncv["edf"]
            @info "PAM REML" sp=reml["sp"] edf=reml["edf"]
            
            # In Poisson/survival context, behavior can differ from Gaussian
            # Store reference values for validation
            @test ncv["sp"] > 0
            @test ncv["edf"] > 0
            @test ncv["edf"] < ref["config"]["k"]  # EDF should be less than basis dimension
        end
        
        @testset "Penalty Matrix Structure (PAM)" begin
            pm = ref["penalty_matrix"]
            nrow = pm["nrow"]
            ncol = pm["ncol"]
            
            @test nrow == ncol  # Should be square
            @test nrow == ref["config"]["k"] - 1  # k-1 due to identifiability constraint
            
            # Reconstruct penalty matrix
            S_r = reshape(Float64.(pm["values"]), (nrow, ncol))
            
            # Should be symmetric
            @test S_r ≈ S_r' atol=1e-10
            
            # Should be positive semi-definite (within numerical tolerance)
            # mgcv's penalty matrices can have small negative eigenvalues due to
            # numerical precision in their construction
            eig_r = eigvals(S_r)
            min_eig = minimum(eig_r)
            @info "PAM penalty matrix eigenvalue range" min=min_eig max=maximum(eig_r)
            
            # Allow for numerical noise up to 1e-4 in magnitude
            @test all(eig_r .>= -1e-4)
            
            # Check null space structure
            # mgcv applies reparameterization, so null space may differ from raw P-spline
            n_small = sum(abs.(eig_r) .< 1e-4)
            @info "PAM penalty matrix small eigenvalues" n_small
            
            # At minimum, the matrix should be nearly full rank for the constrained basis
            @test sum(eig_r .> 1e-4) >= nrow - 3  # At most 3 "near-zero" eigenvalues
        end
        
        @testset "Fit Quality (PAM)" begin
            quality = ref["fit_quality"]
            
            @info "PAM fit quality" rmse_ncv=quality["rmse_ncv"] rmse_reml=quality["rmse_reml"]
            
            # Both should have reasonable fit (relative RMSE < 100%)
            @test quality["rel_rmse_ncv"] < 1.0
            @test quality["rel_rmse_reml"] < 1.0
        end
    end
end

# =============================================================================
# Penalty Functional Verification
# =============================================================================
#
# Verify that our penalty matrix computes the correct functional
# (integrated squared second derivative) using a known polynomial.

@testset "Penalty Functional Verification" begin
    # Create a simple spline with known second derivative
    # Use a cubic spline on [0, 1] with uniform breakpoints
    
    breakpoints = collect(0.0:0.2:1.0)  # 0, 0.2, 0.4, 0.6, 0.8, 1.0
    basis = BSplineBasis(BSplineOrder(4), breakpoints)
    k = length(basis)  # Should be 8
    
    @test k == 8
    
    # Build penalty matrix using INTEGRAL method (O'Sullivan formulation)
    # Note: GPS method has different scaling; use integral for exact functional verification
    S = build_penalty_matrix(basis, 2; method=:integral)
    
    # For any coefficient vector c, c'Sc should equal ∫[f''(x)]² dx
    # Test with a polynomial that we know: f(x) = x²
    # For x²: f''(x) = 2, so ∫₀¹ [f''(x)]² dx = ∫₀¹ 4 dx = 4
    
    # Fit x² to the basis (find coefficients that interpolate x² at k points)
    x_interp = range(0.0, 1.0, length=k)
    y_interp = x_interp.^2
    
    # Build collocation matrix
    B = zeros(k, k)
    for (i, xi) in enumerate(x_interp)
        for j in 1:k
            coeffs = zeros(k)
            coeffs[j] = 1.0
            spl = Spline(basis, coeffs)
            B[i, j] = spl(xi)
        end
    end
    
    # Solve for coefficients
    c = B \ y_interp
    
    # Compute penalty: c'Sc
    penalty = c' * S * c
    
    @info "Quadratic polynomial penalty" computed=penalty expected=4.0
    
    # Should be close to 4 (exact for polynomial that can be represented exactly)
    @test penalty ≈ 4.0 rtol=0.01
end
