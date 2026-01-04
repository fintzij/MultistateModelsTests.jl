# =============================================================================
# Unit tests for spline hazard functionality
# =============================================================================
# 
# These tests verify spline hazard implementation correctness by:
#   1. Numerical integration: H(a,b) ≈ ∫ₐᵇ h(t) dt using QuadGK
#   2. PH covariate effect: h(t|x) = h₀(t) exp(β'x) 
#   3. Survival probability: S(a,b) = exp(-H(a,b))
#   4. Cumulative hazard additivity: H(a,c) = H(a,b) + H(b,c)
#   5. Spline infrastructure: knot placement, coefficient transforms, etc.
# =============================================================================

using Test
using DataFrames
using Distributions
using MultistateModels
using Random
using QuadGK
using LinearAlgebra
using StatsModels

# Import internal functions for testing
import MultistateModels: get_parameters_flat, default_nknots, place_interior_knots,
    ReConstructor, unflatten, flatten,
    SmoothTerm, TensorProductTerm, SmoothTermInfo, SmoothCovariatePenaltyTerm,
    build_penalty_config, expand_smooth_term_columns!, build_tensor_penalty_matrix
import StatsModels: apply_schema, coefnames, modelcols, termvars

@testset "Spline Hazards" begin

    # =========================================================================
    # Test data setup
    # =========================================================================
    
    # Simple test data for 3-state model (transitions 1→2, 2→1, 1→3, 3→1)
    simple_dat = DataFrame(
        id = [1, 1, 2, 2],
        tstart = [0.0, 0.5, 0.0, 0.3],
        tstop = [0.5, 1.0, 0.3, 1.0],
        statefrom = [1, 2, 1, 3],
        stateto = [2, 1, 3, 1],
        obstype = [1, 1, 1, 1],
        x = [0.5, 0.5, -0.3, -0.3]
    )
    
    # Simple 2-state test data for single-transition tests
    two_state_dat = DataFrame(
        id = [1, 1, 2, 2],
        tstart = [0.0, 0.5, 0.0, 0.4],
        tstop = [0.5, 1.0, 0.4, 0.9],
        statefrom = [1, 1, 1, 1],
        stateto = [2, 2, 2, 2],
        obstype = [1, 1, 1, 1],
        x = [0.5, 0.5, -0.3, -0.3]
    )

    # Larger dataset for auto-knot tests
    Random.seed!(42)
    n_subjects = 50
    
    function generate_test_data(n_subjects)
        rows = []
        for subj in 1:n_subjects
            t1 = rand(Uniform(0.3, 0.8))
            push!(rows, (id=subj, tstart=0.0, tstop=t1, statefrom=1, stateto=2, obstype=1, x=randn()))
            t2 = t1 + rand(Uniform(0.2, 0.6))
            push!(rows, (id=subj, tstart=t1, tstop=t2, statefrom=2, stateto=1, obstype=1, x=rows[end].x))
        end
        return DataFrame(rows)
    end
    
    large_dat = generate_test_data(n_subjects)

    # =========================================================================
    # CORE VERIFICATION: Cumulative hazard matches numerical integration
    # =========================================================================
    # This is the fundamental test: H(a,b) = ∫ₐᵇ h(t) dt
    # We verify the implementation computes the correct integral.
    
    @testset "Cumulative hazard vs QuadGK integration" begin
        # Create spline hazard with explicit knots
        h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                     degree=3, knots=[0.3, 0.5, 0.7], 
                     boundaryknots=[0.0, 1.0],
                     natural_spline=true)
        
        model = multistatemodel(h12; data=two_state_dat)
        
        # Test with several different parameter configurations
        Random.seed!(12345)
        for trial in 1:5
            # Set random parameters (log scale)
            npar = model.hazards[1].npar_total
            test_pars = randn(npar) * 0.5
            set_parameters!(model, 1, test_pars)
            
            pars = get_parameters(model, 1, scale=:log)
            haz = model.hazards[1]
            covars = NamedTuple()
            
            # Test multiple intervals
            intervals = [(0.1, 0.4), (0.0, 0.5), (0.2, 0.8), (0.3, 0.9), (0.0, 1.0)]
            
            for (lb, ub) in intervals
                # Analytical cumulative hazard from implementation
                H_impl = MultistateModels.eval_cumhaz(haz, lb, ub, pars, covars)
                
                # Numerical integration with QuadGK
                H_quad, _ = quadgk(t -> MultistateModels.eval_hazard(haz, t, pars, covars), 
                                   lb, ub; rtol=1e-10)
                
                @test isapprox(H_impl, H_quad; rtol=1e-6)
            end
        end
    end
    
    # =========================================================================
    # B-SPLINE ANTIDERIVATIVE VERIFICATION (Machine Precision)
    # =========================================================================
    # The spline cumulative hazard uses B-spline antiderivative (analytical),
    # which should match numerical integration to high precision when both
    # are computed with tight tolerances. This test verifies the antiderivative
    # implementation is correct.
    
    @testset "B-spline antiderivative correctness (machine precision)" begin
        using BSplineKit
        
        # Test that B-spline antiderivative matches high-precision QuadGK
        # for simple polynomial cases where both should agree to ~1e-10
        
        # Create a simple natural cubic spline with uniform coefficients
        # (approximately constant hazard - easy to verify)
        knots = [0.0, 0.25, 0.5, 0.75, 1.0]
        degree = 3
        B = BSplineBasis(BSplineOrder(degree + 1), knots)
        B_natural = RecombinedBSplineBasis(B, Natural())
        
        # Coefficients for approximately constant hazard = 1
        n_coefs = length(B_natural)
        coefs = ones(n_coefs)  # Uniform coefficients
        
        spline = Spline(B_natural, coefs)
        
        # Integrate the spline directly using QuadGK
        for (lb, ub) in [(0.0, 0.5), (0.1, 0.9), (0.25, 0.75)]
            integral_quad, err = quadgk(t -> spline(t), lb, ub; rtol=1e-12)
            
            # BSplineKit provides integral() which uses antiderivative
            antideriv = BSplineKit.integral(spline)
            integral_bspline = antideriv(ub) - antideriv(lb)
            
            # These should match to machine precision since both are exact for polynomials
            @test isapprox(integral_bspline, integral_quad; rtol=1e-10)
        end
        
        # Test with varying coefficients (typical spline hazard)
        Random.seed!(77777)
        coefs_varied = exp.(0.5 .* randn(n_coefs))  # Log-normal for positive hazard
        spline_varied = Spline(B_natural, coefs_varied)
        
        for (lb, ub) in [(0.0, 0.5), (0.1, 0.9), (0.2, 0.8)]
            integral_quad, err = quadgk(t -> spline_varied(t), lb, ub; rtol=1e-12)
            
            antideriv = BSplineKit.integral(spline_varied)
            integral_bspline = antideriv(ub) - antideriv(lb)
            
            @test isapprox(integral_bspline, integral_quad; rtol=1e-10)
        end
    end
    
    @testset "Cumulative hazard with covariates vs QuadGK" begin
        # PH model: h(t|x) = h₀(t) exp(β'x)
        # Verify numerical integration gives same result as analytical cumhaz
        
        h_cov = Hazard(@formula(0 ~ x), "sp", 1, 2; 
                       degree=3, knots=[0.3, 0.6], 
                       boundaryknots=[0.0, 1.0],
                       natural_spline=true)
        
        model = multistatemodel(h_cov; data=two_state_dat)
        haz = model.hazards[1]
        
        # Set parameters: spline coefficients + covariate effect
        Random.seed!(67890)
        nbasis = haz.npar_baseline
        spline_pars = randn(nbasis) * 0.3
        beta = 0.7
        all_pars = vcat(spline_pars, [beta])
        set_parameters!(model, 1, all_pars)
        
        pars = get_parameters(model, 1, scale=:log)
        
        # Test with different covariate values
        for x_val in [-0.5, 0.0, 0.5, 1.0]
            covars = (x = x_val,)
            
            for (lb, ub) in [(0.1, 0.5), (0.2, 0.8), (0.0, 1.0)]
                H_impl = MultistateModels.eval_cumhaz(haz, lb, ub, pars, covars)
                H_quad, _ = quadgk(t -> MultistateModels.eval_hazard(haz, t, pars, covars), 
                                   lb, ub; rtol=1e-10)
                
                @test isapprox(H_impl, H_quad; rtol=1e-6)
            end
        end
    end

    # =========================================================================
    # PH MODEL VERIFICATION: h(t|x) = h₀(t) exp(β'x)
    # =========================================================================
    
    @testset "PH covariate effect verification" begin
        h_cov = Hazard(@formula(0 ~ x), "sp", 1, 2; 
                       degree=3, knots=[0.3, 0.5, 0.7], 
                       boundaryknots=[0.0, 1.0],
                       natural_spline=true)
        
        model = multistatemodel(h_cov; data=two_state_dat)
        haz = model.hazards[1]
        
        # Set known covariate effect
        Random.seed!(11111)
        nbasis = haz.npar_baseline
        spline_pars = randn(nbasis) * 0.3
        beta = 0.5  # Known coefficient
        all_pars = vcat(spline_pars, [beta])
        set_parameters!(model, 1, all_pars)
        pars = get_parameters(model, 1, scale=:log)
        
        # Test: hazard ratio should equal exp(β * Δx)
        x1, x2 = 1.0, -0.5
        covars1 = (x = x1,)
        covars2 = (x = x2,)
        
        for t in [0.2, 0.5, 0.8]
            h1 = MultistateModels.eval_hazard(haz, t, pars, covars1)
            h2 = MultistateModels.eval_hazard(haz, t, pars, covars2)
            
            expected_hr = exp(beta * (x1 - x2))
            actual_hr = h1 / h2
            
            @test isapprox(actual_hr, expected_hr; rtol=1e-10)
        end
        
        # Test: cumulative hazard ratio should also equal exp(β * Δx)
        for (lb, ub) in [(0.1, 0.4), (0.2, 0.7)]
            H1 = MultistateModels.eval_cumhaz(haz, lb, ub, pars, covars1)
            H2 = MultistateModels.eval_cumhaz(haz, lb, ub, pars, covars2)
            
            expected_hr = exp(beta * (x1 - x2))
            actual_hr = H1 / H2
            
            @test isapprox(actual_hr, expected_hr; rtol=1e-10)
        end
    end

    # =========================================================================
    # SURVIVAL PROBABILITY VERIFICATION: S(a,b) = exp(-H(a,b))
    # =========================================================================
    
    @testset "Survival probability correctness" begin
        h_sp = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                      degree=3, knots=[0.4, 0.6], 
                      boundaryknots=[0.0, 1.0],
                      natural_spline=true)
        
        model = multistatemodel(h_sp; data=two_state_dat)
        haz = model.hazards[1]
        
        Random.seed!(22222)
        npar = haz.npar_total
        test_pars = randn(npar) * 0.4
        set_parameters!(model, 1, test_pars)
        
        params = MultistateModels.get_hazard_params(model.parameters, model.hazards)
        pars = params[1]
        subjdat_row = model.data[1, :]
        covars = MultistateModels.extract_covariates_fast(subjdat_row, haz.covar_names)
        
        # Test: S(a,b) = exp(-H(a,b))
        test_intervals = [(0.0, 0.3), (0.0, 0.5), (0.0, 0.8), (0.2, 0.7)]
        
        for (lb, ub) in test_intervals
            H = MultistateModels.eval_cumhaz(haz, lb, ub, pars, covars)
            S = MultistateModels.survprob(lb, ub, params, subjdat_row, 
                                          model.totalhazards[1], model.hazards; 
                                          give_log=false)
            
            @test isapprox(S, exp(-H); rtol=1e-10)
            
            # Log survival should equal -H
            log_S = MultistateModels.survprob(lb, ub, params, subjdat_row, 
                                              model.totalhazards[1], model.hazards; 
                                              give_log=true)
            @test isapprox(log_S, -H; rtol=1e-10)
        end
        
        # Test: survival is monotonically decreasing
        times = [0.2, 0.4, 0.6, 0.8, 1.0]
        surv_vals = [MultistateModels.survprob(0.0, t, params, subjdat_row, 
                                               model.totalhazards[1], model.hazards; 
                                               give_log=false)
                     for t in times]
        
        for i in 1:(length(surv_vals)-1)
            @test surv_vals[i] >= surv_vals[i+1]
        end
    end

    # =========================================================================
    # CUMULATIVE HAZARD ADDITIVITY: H(a,c) = H(a,b) + H(b,c)
    # =========================================================================
    
    @testset "Cumulative hazard additivity" begin
        h_sp = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                      degree=3, knots=[0.3, 0.5, 0.7], 
                      boundaryknots=[0.0, 1.0],
                      natural_spline=true)
        
        model = multistatemodel(h_sp; data=two_state_dat)
        haz = model.hazards[1]
        
        Random.seed!(33333)
        npar = haz.npar_total
        test_pars = randn(npar) * 0.5
        set_parameters!(model, 1, test_pars)
        pars = get_parameters(model, 1, scale=:log)
        covars = NamedTuple()
        
        # Test additivity: H(a,c) = H(a,b) + H(b,c)
        test_cases = [
            (0.1, 0.4, 0.7),
            (0.0, 0.5, 1.0),
            (0.2, 0.3, 0.8)
        ]
        
        for (a, b, c) in test_cases
            H_ac = MultistateModels.eval_cumhaz(haz, a, c, pars, covars)
            H_ab = MultistateModels.eval_cumhaz(haz, a, b, pars, covars)
            H_bc = MultistateModels.eval_cumhaz(haz, b, c, pars, covars)
            
            @test isapprox(H_ac, H_ab + H_bc; rtol=1e-10)
        end
        
        # Test: zero-length interval gives zero
        @test MultistateModels.eval_cumhaz(haz, 0.5, 0.5, pars, covars) == 0.0
    end


    # =========================================================================
    # Automatic knot placement
    # =========================================================================
    
    @testset "Automatic Knot Placement" begin
        h12_auto = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                          degree=3, knots=nothing, natural_spline=true)
        h21_auto = Hazard(@formula(0 ~ 1 + x), "sp", 2, 1; 
                          degree=3, knots=nothing)
        
        auto_model = multistatemodel(h12_auto, h21_auto; data=large_dat)
        
        for (i, haz) in enumerate(auto_model.hazards)
            # Should have interior knots placed automatically
            @test length(haz.knots) > 2
            # Knots should be sorted
            @test issorted(haz.knots)
            # Interior knots should be within boundaries
            interior = haz.knots[2:end-1]
            @test all(interior .> haz.knots[1])
            @test all(interior .< haz.knots[end])
        end
        
        # Verify cumulative hazard matches numerical integration (QuadGK)
        # Note: rtol=1e-3 because spline cumhaz uses a different internal integration
        # method than QuadGK, introducing small numerical differences.
        # Only test hazard 1 which has no covariates; hazard 2 requires covariates.
        haz = auto_model.hazards[1]
        pars = get_parameters(auto_model, 1, scale=:log)
        covars = NamedTuple()
        lb, ub = 0.1, 0.8
        
        H_impl = MultistateModels.eval_cumhaz(haz, lb, ub, pars, covars)
        H_quad, _ = quadgk(t -> MultistateModels.eval_hazard(haz, t, pars, covars), lb, ub; rtol=1e-10)
        
        @test isapprox(H_impl, H_quad; rtol=1e-3)
    end
    
    @testset "default_nknots function" begin
        @test MultistateModels.default_nknots(0) == 0
        @test MultistateModels.default_nknots(1) == 2  # min 2
        @test MultistateModels.default_nknots(10) == 2
        @test MultistateModels.default_nknots(32) == 2  # 32^(1/5) ≈ 2.0
        @test MultistateModels.default_nknots(100) == 2  # 100^(1/5) ≈ 2.51
        @test MultistateModels.default_nknots(1000) == 3  # 1000^(1/5) ≈ 3.98
        @test MultistateModels.default_nknots(10000) == 6  # 10000^(1/5) ≈ 6.31
    end

    # =========================================================================
    # Time transformation support: verify parity with non-transformed
    # =========================================================================
    
    @testset "Time Transform for Splines - parity verification" begin
        # time_transform should give identical results to standard evaluation
        h_plain = Hazard(@formula(0 ~ x), "sp", 1, 2; 
                         degree=3, knots=[0.3, 0.5, 0.7],
                         boundaryknots=[0.0, 1.0],
                         natural_spline=true,
                         time_transform=false)
        h_tt = Hazard(@formula(0 ~ x), "sp", 1, 2; 
                      degree=3, knots=[0.3, 0.5, 0.7],
                      boundaryknots=[0.0, 1.0],
                      natural_spline=true,
                      time_transform=true)
        
        model_plain = multistatemodel(h_plain; data=two_state_dat)
        model_tt = multistatemodel(h_tt; data=two_state_dat)
        
        # Set identical parameters
        Random.seed!(44444)
        npar = model_plain.hazards[1].npar_total
        test_pars = randn(npar) * 0.4
        set_parameters!(model_plain, 1, test_pars)
        set_parameters!(model_tt, 1, test_pars)
        
        pars_plain = get_parameters(model_plain, 1, scale=:log)
        pars_tt = get_parameters(model_tt, 1, scale=:log)
        
        haz_plain = model_plain.hazards[1]
        haz_tt = model_tt.hazards[1]
        
        # Test with multiple covariate values
        for x_val in [-0.3, 0.0, 0.5]
            covars = (x = x_val,)
            linpred = MultistateModels._linear_predictor(pars_tt, covars, haz_tt)
            
            # Point hazard parity
            for t in [0.1, 0.3, 0.5, 0.7, 0.9]
                h_plain_val = MultistateModels.eval_hazard(haz_plain, t, pars_plain, covars)
                h_tt_val = MultistateModels._time_transform_hazard(haz_tt, pars_tt, t, linpred)
                
                @test isapprox(h_plain_val, h_tt_val; rtol=1e-8)
            end
            
            # Cumulative hazard parity
            for (lb, ub) in [(0.1, 0.5), (0.2, 0.8), (0.0, 1.0)]
                H_plain = MultistateModels.eval_cumhaz(haz_plain, lb, ub, pars_plain, covars)
                H_tt = MultistateModels._time_transform_cumhaz(haz_tt, pars_tt, lb, ub, linpred)
                
                @test isapprox(H_plain, H_tt; rtol=1e-6)
            end
        end
    end

    # =========================================================================
    # Coefficient rectification (rectify_coefs!)
    # =========================================================================
    
    @testset "rectify_coefs! for Splines" begin
        # Create a model with monotone splines (where rectification matters most)
        h12_mono = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                          degree=3, knots=[0.3, 0.5, 0.7],
                          monotone=1)  # Increasing
        h21_mono = Hazard(@formula(0 ~ 1 + x), "sp", 2, 1; 
                          degree=3, knots=[0.3, 0.5, 0.7],
                          monotone=-1)  # Decreasing
        
        model = multistatemodel(h12_mono, h21_mono; data=large_dat)
        
        # Set reasonable parameters
        for (h, haz) in enumerate(model.hazards)
            npar = haz.npar_total
            new_pars = fill(-1.0, npar)
            set_parameters!(model, h, new_pars)
        end
        
        # Get the flat parameter vector
        ests_before = get_parameters_flat(model)
        
        # Apply rectification
        ests_after = copy(ests_before)
        MultistateModels.rectify_coefs!(ests_after, model)
        
        # Rectification should produce valid parameters
        @test all(isfinite.(ests_after))
        # Rectified parameters should be the same or modified (not NaN/Inf)
        # For monotone splines, coefficients may be adjusted
        @test length(ests_after) == length(ests_before)
        
        # Update model with rectified parameters using unflatten_natural
        pars_nested = MultistateModels.unflatten_natural(ests_after, model)
        for (hazname, hazidx) in model.hazkeys
            hazard_pars = pars_nested[hazname]
            # Extract full parameter vector (baseline + covariates)
            pars_vec = MultistateModels.extract_params_vector(hazard_pars)
            set_parameters!(model, hazidx, pars_vec)
        end
        
        # Verify cumulative hazard matches numerical integration after rectification
        # Note: rtol=1e-3 because spline cumhaz uses a different internal integration
        # method than QuadGK, introducing small numerical differences.
        # Only test hazard 1 which has no covariates; hazard 2 requires covariates.
        haz = model.hazards[1]
        pars = get_parameters(model, 1, scale=:log)
        covars = NamedTuple()
        lb, ub = 0.1, 0.8
        
        H_impl = MultistateModels.eval_cumhaz(haz, lb, ub, pars, covars)
        H_quad, _ = quadgk(t -> MultistateModels.eval_hazard(haz, t, pars, covars), lb, ub; rtol=1e-10)
        
        @test isapprox(H_impl, H_quad; rtol=1e-3)
    end
    
    @testset "rectify_coefs! round-trip consistency" begin
        # Test that rectify_coefs! produces consistent results on second application
        h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                     degree=3, knots=[0.3, 0.5, 0.7],
                     monotone=1)
        
        model = multistatemodel(h12; data=large_dat)
        
        Random.seed!(99999)
        set_parameters!(model, 1, rand(Normal(0, 1), model.hazards[1].npar_total))
        
        ests = get_parameters_flat(model)
        
        # First rectification
        ests_rect1 = copy(ests)
        MultistateModels.rectify_coefs!(ests_rect1, model)
        
        # Second rectification should be idempotent (or very close)
        ests_rect2 = copy(ests_rect1)
        MultistateModels.rectify_coefs!(ests_rect2, model)
        
        # Results should be very close after second pass
        @test maximum(abs.(ests_rect1 .- ests_rect2)) < 1e-10
    end

    # =========================================================================
    # Spline coefficient transformations
    # =========================================================================
    
    @testset "_spline_ests2coefs and _spline_coefs2ests round-trip" begin
        using BSplineKit
        
        # Create a test basis
        knots = [0.0, 0.3, 0.5, 0.7, 1.0]
        B = BSplineBasis(BSplineOrder(4), knots)  # cubic
        B_natural = RecombinedBSplineBasis(B, Natural())
        
        # Test for monotone = 0 (no constraint)
        Random.seed!(111)
        ests_orig = rand(Normal(0, 1), length(B_natural))
        coefs = MultistateModels._spline_ests2coefs(ests_orig, B_natural, 0)
        ests_back = MultistateModels._spline_coefs2ests(coefs, B_natural, 0)
        @test maximum(abs.(ests_orig .- ests_back)) < 1e-12
        
        # Test for monotone = 1 (increasing)
        ests_orig = rand(Normal(0, 1), length(B_natural))
        coefs = MultistateModels._spline_ests2coefs(ests_orig, B_natural, 1)
        ests_back = MultistateModels._spline_coefs2ests(coefs, B_natural, 1)
        @test maximum(abs.(ests_orig .- ests_back)) < 1e-12
        
        # Test for monotone = -1 (decreasing)
        ests_orig = rand(Normal(0, 1), length(B_natural))
        coefs = MultistateModels._spline_ests2coefs(ests_orig, B_natural, -1)
        ests_back = MultistateModels._spline_coefs2ests(coefs, B_natural, -1)
        @test maximum(abs.(ests_orig .- ests_back)) < 1e-12
    end

    # =========================================================================
    # Edge cases
    # =========================================================================
    
    @testset "Spline edge cases" begin
        # Linear spline (degree=1) - verify cumhaz matches QuadGK integration
        h_linear = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                          degree=1, knots=[0.3, 0.5, 0.7])
        
        model_linear = multistatemodel(h_linear; data=large_dat)
        haz_linear = model_linear.hazards[1]
        pars = get_parameters(model_linear, 1, scale=:log)
        covars = NamedTuple()
        lb, ub = 0.2, 0.8
        
        # Note: rtol=1e-3 because spline cumhaz uses internal integration that differs from QuadGK
        H_impl = MultistateModels.eval_cumhaz(haz_linear, lb, ub, pars, covars)
        H_quad, _ = quadgk(t -> MultistateModels.eval_hazard(haz_linear, t, pars, covars), lb, ub; rtol=1e-10)
        @test isapprox(H_impl, H_quad; rtol=1e-3)
        
        # Spline with constant extrapolation - verify cumhaz at boundaries
        h_const = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                        degree=3, knots=[0.3, 0.5, 0.7],
                        extrapolation="constant")
        
        model_const = multistatemodel(h_const; data=large_dat)
        haz_const = model_const.hazards[1]
        pars_const = get_parameters(model_const, 1, scale=:log)
        
        # Verify cumhaz beyond knot boundaries matches QuadGK
        # Note: rtol=1e-3 because spline cumhaz uses internal integration that differs from QuadGK
        H_impl_const = MultistateModels.eval_cumhaz(haz_const, 0.0, 0.5, pars_const, covars)
        H_quad_const, _ = quadgk(t -> MultistateModels.eval_hazard(haz_const, t, pars_const, covars), 0.0, 0.5; rtol=1e-10)
        @test isapprox(H_impl_const, H_quad_const; rtol=1e-3)
        
        # Spline with linear extrapolation - verify cumhaz  
        h_extrap = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                          degree=3, knots=[0.3, 0.5, 0.7],
                          extrapolation="linear")
        
        model_extrap = multistatemodel(h_extrap; data=large_dat)
        haz_extrap = model_extrap.hazards[1]
        pars_extrap = get_parameters(model_extrap, 1, scale=:log)
        
        # Note: rtol=2e-3 for extrapolation cases which have additional numerical differences
        H_impl_extrap = MultistateModels.eval_cumhaz(haz_extrap, 0.2, 0.8, pars_extrap, covars)
        H_quad_extrap, _ = quadgk(t -> MultistateModels.eval_hazard(haz_extrap, t, pars_extrap, covars), 0.2, 0.8; rtol=1e-10)
        @test isapprox(H_impl_extrap, H_quad_extrap; rtol=2e-3)
    end

    # =========================================================================
    # default_nknots helper
    # =========================================================================
    
    @testset "default_nknots" begin
        # Small sample → fewer knots
        @test default_nknots(10) <= default_nknots(100)
        @test default_nknots(100) <= default_nknots(1000)
        
        # Should return at least 1 knot
        @test default_nknots(5) >= 1
        
        # Should cap at reasonable maximum
        @test default_nknots(100000) <= 20
    end
    
    # =========================================================================
    # place_interior_knots helper
    # =========================================================================
    
    @testset "place_interior_knots" begin
        sojourns = collect(range(0.1, 1.0, length=100))
        
        # Basic functionality
        knots3 = place_interior_knots(sojourns, 3)
        @test length(knots3) == 3
        @test issorted(knots3)
        @test all(knots3 .>= minimum(sojourns))
        @test all(knots3 .<= maximum(sojourns))
        
        # With explicit bounds
        knots_bounded = place_interior_knots(sojourns, 2; lower_bound=0.0, upper_bound=2.0)
        @test length(knots_bounded) == 2
        @test all(knots_bounded .>= 0.0)
        @test all(knots_bounded .<= 2.0)
        
        # Different number of knots
        knots5 = place_interior_knots(sojourns, 5)
        @test length(knots5) == 5
        @test issorted(knots5)
    end

    # =========================================================================
    # calibrate_splines and calibrate_splines! 
    # =========================================================================
    
    @testset "Spline Knot Calibration" begin
        Random.seed!(54321)
        
        # Create test data with multiple transitions
        n_subj = 40
        calib_data = DataFrame(
            id = repeat(1:n_subj, inner=1),
            tstart = zeros(n_subj),
            tstop = rand(n_subj) .* 5 .+ 0.1,
            statefrom = ones(Int, n_subj),
            stateto = [rand() < 0.6 ? 2 : 3 for _ in 1:n_subj],
            obstype = fill(1, n_subj)
        )
        
        h12_calib = Hazard(@formula(0 ~ 1), "sp", 1, 2; degree=3)
        h13_calib = Hazard(@formula(0 ~ 1), "sp", 1, 3; degree=3)
        
        @testset "calibrate_splines - basic functionality" begin
            model = multistatemodel(h12_calib, h13_calib; data=calib_data)
            
            # Auto-select nknots
            knots = calibrate_splines(model)
            
            # Verify structure and values (accessing fields proves keys exist)
            @test knots.h12.boundary_knots[1] ≈ 0.0  # Lower boundary is always 0
            @test knots.h13.boundary_knots[1] ≈ 0.0
            @test knots.h12.boundary_knots[2] > 0.0  # Upper boundary is positive
            @test knots.h13.boundary_knots[2] > 0.0
            @test length(knots.h12.interior_knots) >= 1
            @test length(knots.h13.interior_knots) >= 1
            
            # Check interior knots are strictly within boundaries
            @test all(knots.h12.boundary_knots[1] .< knots.h12.interior_knots .< knots.h12.boundary_knots[2])
            @test all(knots.h13.boundary_knots[1] .< knots.h13.interior_knots .< knots.h13.boundary_knots[2])
            
            # Interior knots should be increasing
            if length(knots.h12.interior_knots) > 1
                @test issorted(knots.h12.interior_knots)
            end
        end
        
        @testset "calibrate_splines - explicit nknots" begin
            model = multistatemodel(h12_calib, h13_calib; data=calib_data)
            
            # Uniform nknots for all hazards
            knots = calibrate_splines(model; nknots=3)
            @test length(knots.h12.interior_knots) <= 3  # May be fewer due to ties
            
            # Per-hazard nknots
            knots2 = calibrate_splines(model; nknots=(h12=4, h13=2))
            @test length(knots2.h12.interior_knots) <= 4
            @test length(knots2.h13.interior_knots) <= 2
        end
        
        @testset "calibrate_splines - explicit quantiles" begin
            model = multistatemodel(h12_calib, h13_calib; data=calib_data)
            
            # Uniform quantiles for all hazards
            knots = calibrate_splines(model; quantiles=[0.25, 0.5, 0.75])
            @test length(knots.h12.interior_knots) <= 3  # May be fewer due to ties
        end
        
        @testset "calibrate_splines - error handling" begin
            model = multistatemodel(h12_calib, h13_calib; data=calib_data)
            
            # Error if both quantiles and nknots specified
            @test_throws ArgumentError calibrate_splines(model; quantiles=[0.5], nknots=3)
            
            # Error on non-spline model
            data_12_only = DataFrame(
                id = 1:10,
                tstart = zeros(10),
                tstop = rand(10) .* 5 .+ 0.1,
                statefrom = ones(Int, 10),
                stateto = fill(2, 10),
                obstype = fill(1, 10)
            )
            h_exp = Hazard(@formula(0 ~ 1), "exp", 1, 2)
            model_exp = multistatemodel(h_exp; data=data_12_only)
            @test_throws ArgumentError calibrate_splines(model_exp)
            
            # Error on fitted model
            fitted = fit(model; verbose=false)
            @test_throws ArgumentError calibrate_splines(fitted)
        end
        
        @testset "calibrate_splines! - in-place modification" begin
            model = multistatemodel(h12_calib, h13_calib; data=calib_data)
            
            old_knots = copy(model.hazards[1].knots)
            old_npar = model.hazards[1].npar_baseline
            
            knots_result = calibrate_splines!(model; nknots=4)
            
            new_knots = model.hazards[1].knots
            new_npar = model.hazards[1].npar_baseline
            
            # Verify model was modified (knots changed)
            @test length(new_knots) != length(old_knots) || new_knots != old_knots
            
            # Verify parameters were rebuilt with correct size
            expected_npar = sum(h.npar_total for h in model.hazards)
            @test length(model.parameters.flat) == expected_npar
            @test all(isfinite.(model.parameters.flat))  # All params should be valid
            
            # Verify result matches model state
            @test knots_result.h12.interior_knots == new_knots[2:end-1]
            
            # Verify interior knots are strictly increasing
            if length(knots_result.h12.interior_knots) > 1
                @test issorted(knots_result.h12.interior_knots)
            end
        end
        
        @testset "calibrate_splines! - parameter structure integrity" begin
            # This test verifies the fix for the bug where _rebuild_model_parameters!
            # was creating an 'unflatten' field instead of 'reconstructor'
            model = multistatemodel(h12_calib, h13_calib; data=calib_data)
            
            calibrate_splines!(model; nknots=3, verbose=false)
            
            # Verify reconstructor field exists and is the correct type
            @test haskey(model.parameters, :reconstructor)
            @test model.parameters.reconstructor isa MultistateModels.ReConstructor
            
            # Verify unflatten operations work with the reconstructor
            flat_params = model.parameters.flat
            nested = MultistateModels.unflatten(model.parameters.reconstructor, flat_params)
            @test haskey(nested, :h12)
            @test haskey(nested, :h13)
            
            # Verify flatten/unflatten round-trip preserves values
            reflattened = MultistateModels.flatten(model.parameters.reconstructor, nested)
            @test reflattened ≈ flat_params
        end
        
        @testset "calibrate_splines! - model remains functional" begin
            # Verify model can still be fitted after calibration
            model = multistatemodel(h12_calib, h13_calib; data=calib_data)
            calibrate_splines!(model; nknots=2, verbose=false)
            
            # Model should be fittable
            fitted = fit(model; verbose=false, compute_vcov=false)
            @test fitted isa MultistateModels.MultistateModelFitted
            @test isfinite(fitted.loglik.loglik)
            
            # Parameters should have been optimized
            @test !all(fitted.parameters.flat .≈ 0.0)
        end
        
        @testset "calibrate_splines! - set_parameters! works after calibration" begin
            model = multistatemodel(h12_calib, h13_calib; data=calib_data)
            calibrate_splines!(model; nknots=2, verbose=false)
            
            # set_parameters! should work with various input formats
            npar_h12 = model.hazards[1].npar_total
            npar_h13 = model.hazards[2].npar_total
            
            # Vector{Vector{Float64}} format
            new_params = [randn(npar_h12), randn(npar_h13)]
            set_parameters!(model, new_params)
            @test model.parameters.flat[1:npar_h12] ≈ new_params[1]
            
            # NamedTuple format
            nt_params = (h12 = randn(npar_h12), h13 = randn(npar_h13))
            set_parameters!(model, nt_params)
            @test model.parameters.flat[1:npar_h12] ≈ collect(nt_params.h12)
        end
    end

    # =========================================================================
    # SMOOTH COVARIATE TERMS: s(x) and te(x,y)
    # =========================================================================
    # Tests for smooth covariate effects in hazard formulas (Phase 3)
    
    @testset "Smooth Covariate Terms s(x)" begin
        # Test data with continuous covariates
        Random.seed!(42)
        n_obs = 100
        smooth_data = DataFrame(
            id = 1:n_obs,
            tstart = zeros(n_obs),
            tstop = rand(n_obs) .* 5 .+ 0.1,
            statefrom = ones(Int, n_obs),
            stateto = fill(2, n_obs),
            obstype = ones(Int, n_obs),
            age = rand(n_obs) .* 50 .+ 20,
            bmi = rand(n_obs) .* 15 .+ 20,
            trt = rand([0.0, 1.0], n_obs)
        )
        
        @testset "s(x) formula parsing" begin
            # Basic s(x) parsing
            f = @formula(0 ~ s(age, 5, 2))
            schema = StatsModels.schema(f, smooth_data)
            fs = apply_schema(f, schema)
            
            term = fs.rhs.terms[1]
            @test term isa SmoothTerm
            @test term.knots == 5
            @test term.penalty_order == 2
            @test term.label == "s(age)"
            
            # Coefficient names
            cnames = coefnames(term)
            @test length(cnames) == 5
            @test cnames[1] == "s(age)_1"
            @test cnames[end] == "s(age)_5"
        end
        
        @testset "s(x) syntax edge cases" begin
            # Verify s() constructor with various argument counts
            
            # Default penalty order (should be 2)
            f_default = @formula(0 ~ s(age, 5))
            schema = StatsModels.schema(f_default, smooth_data)
            fs = apply_schema(f_default, schema)
            term = fs.rhs.terms[1]
            @test term isa SmoothTerm
            @test term.penalty_order == 2  # Default penalty order
            
            # Different penalty orders
            f_pen1 = @formula(0 ~ s(age, 5, 1))
            fs_pen1 = apply_schema(f_pen1, StatsModels.schema(f_pen1, smooth_data))
            @test fs_pen1.rhs.terms[1].penalty_order == 1
            
            f_pen2 = @formula(0 ~ s(age, 5, 2))
            fs_pen2 = apply_schema(f_pen2, StatsModels.schema(f_pen2, smooth_data))
            @test fs_pen2.rhs.terms[1].penalty_order == 2
            
            f_pen3 = @formula(0 ~ s(age, 5, 3))
            fs_pen3 = apply_schema(f_pen3, StatsModels.schema(f_pen3, smooth_data))
            @test fs_pen3.rhs.terms[1].penalty_order == 3
            
            # s() with different knot counts - minimum is 4 for cubic splines
            # Test that k=3 (too few) throws an error
            f_k3 = @formula(0 ~ s(age, 3, 2))
            @test_throws ArgumentError apply_schema(f_k3, StatsModels.schema(f_k3, smooth_data))
            
            # k=4 is minimum valid
            f_k4 = @formula(0 ~ s(age, 4, 2))
            fs_k4 = apply_schema(f_k4, StatsModels.schema(f_k4, smooth_data))
            @test fs_k4.rhs.terms[1].knots == 4
            @test length(coefnames(fs_k4.rhs.terms[1])) == 4
            
            f_k10 = @formula(0 ~ s(age, 10, 2))
            fs_k10 = apply_schema(f_k10, StatsModels.schema(f_k10, smooth_data))
            @test fs_k10.rhs.terms[1].knots == 10
            @test length(coefnames(fs_k10.rhs.terms[1])) == 10
        end
        
        @testset "s(x) penalty matrix properties" begin
            f = @formula(0 ~ s(age, 6, 2))
            schema = StatsModels.schema(f, smooth_data)
            fs = apply_schema(f, schema)
            term = fs.rhs.terms[1]
            
            S = term.S
            
            # Penalty matrix should be symmetric
            @test issymmetric(S)
            
            # Penalty matrix should be positive semi-definite
            eigvals = eigen(Symmetric(S)).values
            @test all(eigvals .>= -1e-10)
            
            # Size should be k × k
            @test size(S) == (6, 6)
        end
        
        @testset "s(x) modelcols basis evaluation" begin
            f = @formula(0 ~ s(age, 5, 2))
            schema = StatsModels.schema(f, smooth_data)
            fs = apply_schema(f, schema)
            term = fs.rhs.terms[1]
            
            # Evaluate basis on columns
            cols = (age = smooth_data.age,)
            B = modelcols(term, cols)
            
            @test size(B) == (n_obs, 5)
            
            # B-splines should sum to approximately 1 at each point
            row_sums = sum(B, dims=2)
            @test all(x -> 0.99 < x < 1.01, row_sums)
            
            # All basis values should be non-negative
            @test all(B .>= 0)
        end
        
        @testset "s(x) column expansion" begin
            h12 = Hazard(@formula(0 ~ s(age, 5, 2)), :exp, 1, 2)
            data_copy = copy(smooth_data)
            
            expand_smooth_term_columns!(data_copy, h12)
            
            # Should have added 5 new columns
            @test Symbol("s(age)_1") in propertynames(data_copy)
            @test Symbol("s(age)_5") in propertynames(data_copy)
            
            # Values should be valid basis function values
            @test all(data_copy[!, Symbol("s(age)_1")] .>= 0)
        end
        
        @testset "s(x) model creation" begin
            h12 = Hazard(@formula(0 ~ s(age, 5, 2)), :exp, 1, 2)
            model = multistatemodel(h12; data=smooth_data)
            
            # Should have 1 intercept + 5 basis coefficients
            @test length(model.parameters.flat) == 6
            
            # Check parameter names
            parnames = get_parnames(model)[1]
            @test :h12_Intercept in parnames
            @test Symbol("h12_s(age)_1") in parnames
            @test Symbol("h12_s(age)_5") in parnames
        end
        
        @testset "s(x) smooth_info extraction" begin
            h12 = Hazard(@formula(0 ~ s(age, 5, 2)), :exp, 1, 2)
            model = multistatemodel(h12; data=smooth_data)
            
            haz = model.hazards[1]
            @test hasproperty(haz, :smooth_info)
            @test length(haz.smooth_info) == 1
            
            info = haz.smooth_info[1]
            @test info.label == "s(age)"
            @test length(info.par_indices) == 5
            @test size(info.S) == (5, 5)
        end
        
        @testset "s(x) penalty config building" begin
            h12 = Hazard(@formula(0 ~ s(age, 5, 2)), :exp, 1, 2)
            model = multistatemodel(h12; data=smooth_data)
            
            config = build_penalty_config(model, SplinePenalty())
            
            @test length(config.smooth_covariate_terms) == 1
            @test config.n_lambda == 1
            
            term = config.smooth_covariate_terms[1]
            @test term.label == "s(age)"
            @test length(term.param_indices) == 5
        end
        
        @testset "s(x) + linear term" begin
            h12 = Hazard(@formula(0 ~ s(age, 5, 2) + trt), :exp, 1, 2)
            model = multistatemodel(h12; data=smooth_data)
            
            # Should have 1 intercept + 5 basis + 1 trt
            @test length(model.parameters.flat) == 7
            
            # Penalty config should only have smooth term, not linear term
            config = build_penalty_config(model, SplinePenalty())
            @test length(config.smooth_covariate_terms) == 1
        end
        
        @testset "Multiple s(x) terms" begin
            h12 = Hazard(@formula(0 ~ s(age, 5, 2) + s(bmi, 4, 2)), :exp, 1, 2)
            model = multistatemodel(h12; data=smooth_data)
            
            # 1 intercept + 5 (age) + 4 (bmi)
            @test length(model.parameters.flat) == 10
            
            # Two smooth terms in penalty config
            config = build_penalty_config(model, SplinePenalty())
            @test length(config.smooth_covariate_terms) == 2
            @test config.n_lambda == 2
            
            labels = [t.label for t in config.smooth_covariate_terms]
            @test "s(age)" in labels
            @test "s(bmi)" in labels
        end
    end
    
    @testset "Tensor Product Smooths te(x,y)" begin
        # Test data
        Random.seed!(42)
        n_obs = 100
        tensor_data = DataFrame(
            id = 1:n_obs,
            tstart = zeros(n_obs),
            tstop = rand(n_obs) .* 5 .+ 0.1,
            statefrom = ones(Int, n_obs),
            stateto = fill(2, n_obs),
            obstype = ones(Int, n_obs),
            age = rand(n_obs) .* 50 .+ 20,
            bmi = rand(n_obs) .* 15 .+ 20
        )
        
        @testset "te(x,y) formula parsing - same k" begin
            # te(x, y, k, m) - same k for both dimensions
            f = @formula(0 ~ te(age, bmi, 4, 2))
            schema = StatsModels.schema(f, tensor_data)
            fs = apply_schema(f, schema)
            
            term = fs.rhs.terms[1]
            @test term isa TensorProductTerm
            @test term.kx == 4
            @test term.ky == 4
            @test term.penalty_order == 2
            @test term.label == "te(age,bmi)"
        end
        
        @testset "te(x,y) formula parsing - different k" begin
            # te(x, y, kx, ky, m) - different k per dimension
            f = @formula(0 ~ te(age, bmi, 5, 6, 2))
            schema = StatsModels.schema(f, tensor_data)
            fs = apply_schema(f, schema)
            
            term = fs.rhs.terms[1]
            @test term.kx == 5
            @test term.ky == 6
            @test size(term.S) == (30, 30)  # 5*6
        end
        
        @testset "te(x,y) coefficient names" begin
            f = @formula(0 ~ te(age, bmi, 4, 2))
            schema = StatsModels.schema(f, tensor_data)
            fs = apply_schema(f, schema)
            term = fs.rhs.terms[1]
            
            cnames = coefnames(term)
            @test length(cnames) == 16  # 4*4
            @test cnames[1] == "te(age,bmi)_1"
            @test cnames[end] == "te(age,bmi)_16"
        end
        
        @testset "te(x,y) penalty matrix properties" begin
            f = @formula(0 ~ te(age, bmi, 4, 2))
            schema = StatsModels.schema(f, tensor_data)
            fs = apply_schema(f, schema)
            term = fs.rhs.terms[1]
            
            S = term.S
            
            # Should be symmetric
            @test issymmetric(S)
            
            # Should be positive semi-definite
            eigvals = eigen(Symmetric(S)).values
            @test all(eigvals .>= -1e-10)
            
            # Size: (kx*ky) × (kx*ky)
            @test size(S) == (16, 16)
        end
        
        @testset "te(x,y) modelcols - Kronecker product" begin
            f = @formula(0 ~ te(age, bmi, 4, 2))
            schema = StatsModels.schema(f, tensor_data)
            fs = apply_schema(f, schema)
            term = fs.rhs.terms[1]
            
            cols = (age = tensor_data.age, bmi = tensor_data.bmi)
            B = modelcols(term, cols)
            
            @test size(B) == (n_obs, 16)
            
            # Tensor product of B-splines should sum to approximately 1
            row_sums = sum(B, dims=2)
            @test all(x -> 0.99 < x < 1.01, row_sums)
        end
        
        @testset "te(x,y) column expansion" begin
            h12 = Hazard(@formula(0 ~ te(age, bmi, 4, 2)), :exp, 1, 2)
            data_copy = copy(tensor_data)
            
            expand_smooth_term_columns!(data_copy, h12)
            
            @test Symbol("te(age,bmi)_1") in propertynames(data_copy)
            @test Symbol("te(age,bmi)_16") in propertynames(data_copy)
        end
        
        @testset "te(x,y) model creation" begin
            h12 = Hazard(@formula(0 ~ te(age, bmi, 4, 2)), :exp, 1, 2)
            model = multistatemodel(h12; data=tensor_data)
            
            # 1 intercept + 16 tensor basis
            @test length(model.parameters.flat) == 17
            
            # Check smooth_info
            haz = model.hazards[1]
            @test length(haz.smooth_info) == 1
            @test haz.smooth_info[1].label == "te(age,bmi)"
            @test size(haz.smooth_info[1].S) == (16, 16)
        end
        
        @testset "te(x,y) penalty config" begin
            h12 = Hazard(@formula(0 ~ te(age, bmi, 4, 2)), :exp, 1, 2)
            model = multistatemodel(h12; data=tensor_data)
            
            config = build_penalty_config(model, SplinePenalty())
            
            @test length(config.smooth_covariate_terms) == 1
            @test config.n_lambda == 1
            @test config.smooth_covariate_terms[1].label == "te(age,bmi)"
        end
        
        @testset "build_tensor_penalty_matrix" begin
            # Manual test of tensor penalty construction
            kx, ky = 3, 4
            Sx = randn(kx, kx)
            Sx = Sx * Sx'  # Make symmetric PD
            Sy = randn(ky, ky)
            Sy = Sy * Sy'
            
            S_te = build_tensor_penalty_matrix(Sx, Sy)
            
            # Size check
            @test size(S_te) == (kx * ky, kx * ky)
            
            # Symmetry
            @test issymmetric(S_te)
        end
    end
    
    @testset "Lambda Sharing for Smooth Covariates" begin
        Random.seed!(42)
        n_obs = 100
        share_data = DataFrame(
            id = 1:n_obs,
            tstart = zeros(n_obs),
            tstop = rand(n_obs) .* 5 .+ 0.1,
            statefrom = ones(Int, n_obs),
            stateto = fill(2, n_obs),
            obstype = ones(Int, n_obs),
            age = rand(n_obs) .* 50 .+ 20,
            bmi = rand(n_obs) .* 15 .+ 20
        )
        
        @testset "Default: separate lambda per term" begin
            h12 = Hazard(@formula(0 ~ s(age, 5, 2) + s(bmi, 4, 2)), :exp, 1, 2)
            model = multistatemodel(h12; data=share_data)
            
            # Default: share_covariate_lambda=false
            config = build_penalty_config(model, SplinePenalty())
            
            @test length(config.smooth_covariate_terms) == 2
            @test config.n_lambda == 2  # One per term
            @test isempty(config.shared_smooth_groups)
        end
        
        @testset "share_covariate_lambda=:global" begin
            h12 = Hazard(@formula(0 ~ s(age, 5, 2) + s(bmi, 4, 2)), :exp, 1, 2)
            model = multistatemodel(h12; data=share_data)
            
            config = build_penalty_config(model, SplinePenalty(share_covariate_lambda=:global))
            
            @test length(config.smooth_covariate_terms) == 2
            @test config.n_lambda == 1  # One shared lambda
            @test length(config.shared_smooth_groups) == 1
            @test config.shared_smooth_groups[1] == [1, 2]
        end
        
        @testset "share_covariate_lambda=:hazard" begin
            h12 = Hazard(@formula(0 ~ s(age, 5, 2) + s(bmi, 4, 2)), :exp, 1, 2)
            model = multistatemodel(h12; data=share_data)
            
            config = build_penalty_config(model, SplinePenalty(share_covariate_lambda=:hazard))
            
            @test length(config.smooth_covariate_terms) == 2
            @test config.n_lambda == 1  # One per hazard (only h12)
            @test length(config.shared_smooth_groups) == 1
        end
    end
    
    @testset "Combined s(x) + te(x,y)" begin
        Random.seed!(42)
        n_obs = 100
        combined_data = DataFrame(
            id = 1:n_obs,
            tstart = zeros(n_obs),
            tstop = rand(n_obs) .* 5 .+ 0.1,
            statefrom = ones(Int, n_obs),
            stateto = fill(2, n_obs),
            obstype = ones(Int, n_obs),
            age = rand(n_obs) .* 50 .+ 20,
            bmi = rand(n_obs) .* 15 .+ 20,
            trt = rand([0.0, 1.0], n_obs)
        )
        
        @testset "s(x) + te(x,y) model creation" begin
            h12 = Hazard(@formula(0 ~ s(age, 5, 2) + te(age, bmi, 4, 2)), :exp, 1, 2)
            model = multistatemodel(h12; data=combined_data)
            
            # 1 intercept + 5 (s(age)) + 16 (te(age,bmi))
            @test length(model.parameters.flat) == 22
            
            # Two smooth terms
            haz = model.hazards[1]
            @test length(haz.smooth_info) == 2
            
            labels = [info.label for info in haz.smooth_info]
            @test "s(age)" in labels
            @test "te(age,bmi)" in labels
        end
        
        @testset "s(x) + te(x,y) + linear term" begin
            h12 = Hazard(@formula(0 ~ s(age, 5, 2) + te(age, bmi, 4, 2) + trt), :exp, 1, 2)
            model = multistatemodel(h12; data=combined_data)
            
            # 1 intercept + 5 + 16 + 1 trt
            @test length(model.parameters.flat) == 23
            
            config = build_penalty_config(model, SplinePenalty())
            @test length(config.smooth_covariate_terms) == 2
            @test config.n_lambda == 2
        end
        
        @testset "s(x) + te(x,y) penalty config" begin
            h12 = Hazard(@formula(0 ~ s(age, 5, 2) + te(age, bmi, 4, 2)), :exp, 1, 2)
            model = multistatemodel(h12; data=combined_data)
            
            config = build_penalty_config(model, SplinePenalty())
            
            @test length(config.smooth_covariate_terms) == 2
            @test config.n_lambda == 2
            
            labels = [t.label for t in config.smooth_covariate_terms]
            @test "s(age)" in labels
            @test "te(age,bmi)" in labels
        end
    end
    
    @testset "Smooth Covariates with Spline Baseline" begin
        Random.seed!(42)
        n_obs = 100
        sp_data = DataFrame(
            id = 1:n_obs,
            tstart = zeros(n_obs),
            tstop = rand(n_obs) .* 5 .+ 0.1,
            statefrom = ones(Int, n_obs),
            stateto = fill(2, n_obs),
            obstype = ones(Int, n_obs),
            age = rand(n_obs) .* 50 .+ 20
        )
        
        @testset "Spline baseline + s(x)" begin
            # Spline baseline hazard with smooth covariate
            h12 = Hazard(@formula(0 ~ s(age, 5, 2)), :sp, 1, 2;
                         knots=[1.0, 2.0, 3.0], boundaryknots=[0.0, 5.0])
            model = multistatemodel(h12; data=sp_data)
            
            config = build_penalty_config(model, SplinePenalty())
            
            # Should have both baseline penalty and smooth covariate penalty
            @test length(config.terms) >= 1  # Baseline spline penalty
            @test length(config.smooth_covariate_terms) == 1  # s(age)
            
            # Total lambda count
            @test config.n_lambda >= 2
        end
    end
end
