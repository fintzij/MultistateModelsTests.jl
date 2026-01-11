# =============================================================================
# Analytical Log-Likelihood Verification Tests
# =============================================================================
#
# This test file verifies that log-likelihood functions return mathematically
# correct values by comparing against hand-calculated analytical formulas.
#
# Test scenarios cover each hazard family and data type used in longtests:
#
# EXACT DATA (obstype=1) - directly observed transitions
# -------------------------------------------------------
# Single subject, single transition: l = log(h(t)) - H(t)
# where h(t) = hazard, H(t) = cumulative hazard from 0 to t
#
# PANEL DATA (obstype=2) - interval-censored Markov
# --------------------------------------------------
# For Markov models, P(t) = exp(Qt) where Q is the generator matrix.
# For a 2-state absorbing model: P12(t) = 1 - exp(-lambda*t)
# l = log(P_{s0,s1}(t1 - t0))
#
# Analytical Hazard Formulas (MultistateModels parameterization):
# ---------------------------------------------------------------
# Exponential: h(t) = lambda, H(t) = lambda*t
# Weibull:     h(t) = kappa*lambda*t^{kappa-1}, H(t) = lambda*t^kappa  (shape kappa, scale lambda)
# Gompertz:    h(t) = b*exp(a*t), H(t) = (b/a)*(exp(a*t) - 1)  (shape a, rate b)
# Spline:      h(t) = exp(spline(t)), H(t) = integral of exp(spline(u))du
#
# Covariate Effects (PH):
# ----------------------
# h(t|x) = h0(t) * exp(beta'x)
# H(t|x) = H0(t) * exp(beta'x)
#
# =============================================================================

using Test
using MultistateModels
using DataFrames
using LinearAlgebra

import MultistateModels: 
    SamplePath, ExactData, MPanelData, loglik_exact, loglik_markov,
    set_parameters!, get_parameters_flat, build_tpm_mapping, extract_paths,
    unflatten_parameters

# Tolerance for analytical comparisons (accounting for numerical precision)
const ANALYTICAL_RTOL = 1e-10
const NUMERICAL_RTOL = 1e-6  # For matrix exponential / numerical comparisons

@testset "Analytical Log-Likelihood Verification" begin
    
    # =========================================================================
    # PART 1: EXACT DATA - Single Subject, Single Transition
    # =========================================================================
    
    @testset "Exact Data: Single Subject, Single Transition" begin
        
        # =====================================================================
        # 1.1 Exponential Hazard
        # =====================================================================
        @testset "Exponential: l = log(lambda) - lambda*t" begin
            # Setup: Single transition 1->2 at time t=3.0
            rate = 0.5  # rate parameter
            t = 3.0     # transition time
            
            h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
            
            dat = DataFrame(
                id = [1],
                tstart = [0.0],
                tstop = [t],
                statefrom = [1],
                stateto = [2],
                obstype = [1]  # exact observation
            )
            
            model = multistatemodel(h12; data=dat)
            set_parameters!(model, (h12 = [rate],))
            
            # Analytical likelihood: f(t) = lambda*exp(-lambda*t)
            # log-likelihood: log(lambda) - lambda*t
            ll_analytical = log(rate) - rate * t
            
            # Package computation
            paths = extract_paths(model)
            exact_data = ExactData(model, paths)
            ll_package = loglik_exact(model.parameters.flat, exact_data; neg=false)
            
            @test ll_package ≈ ll_analytical rtol=ANALYTICAL_RTOL
        end
        
        @testset "Exponential with covariate (PH)" begin
            # Setup: Single transition with binary covariate x=1
            rate = 0.3   # baseline rate
            beta = 0.7   # covariate effect
            x = 1.0      # covariate value
            t = 4.0      # transition time
            
            h12 = Hazard(@formula(0 ~ 1 + x), "exp", 1, 2)
            
            dat = DataFrame(
                id = [1],
                tstart = [0.0],
                tstop = [t],
                statefrom = [1],
                stateto = [2],
                obstype = [1],
                x = [x]
            )
            
            model = multistatemodel(h12; data=dat)
            set_parameters!(model, (h12 = [rate, beta],))
            
            # Effective rate: rate_eff = rate*exp(beta*x)
            rate_eff = rate * exp(beta * x)
            ll_analytical = log(rate_eff) - rate_eff * t
            
            paths = extract_paths(model)
            exact_data = ExactData(model, paths)
            ll_package = loglik_exact(model.parameters.flat, exact_data; neg=false)
            
            @test ll_package ≈ ll_analytical rtol=ANALYTICAL_RTOL
        end
        
        # =====================================================================
        # 1.2 Weibull Hazard
        # =====================================================================
        @testset "Weibull: l = log(kappa*lambda*t^{kappa-1}) - lambda*t^kappa" begin
            # Setup: Single transition 1->2 at time t=2.5
            shape = 1.5   # shape parameter (kappa)
            scale = 0.2   # scale parameter (lambda)
            t = 2.5       # transition time
            
            h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
            
            dat = DataFrame(
                id = [1],
                tstart = [0.0],
                tstop = [t],
                statefrom = [1],
                stateto = [2],
                obstype = [1]
            )
            
            model = multistatemodel(h12; data=dat)
            set_parameters!(model, (h12 = [shape, scale],))
            
            # Weibull hazard: h(t) = kappa*lambda*t^{kappa-1}
            # Cumulative hazard: H(t) = lambda*t^kappa
            h_t = shape * scale * t^(shape - 1)
            H_t = scale * t^shape
            ll_analytical = log(h_t) - H_t
            
            paths = extract_paths(model)
            exact_data = ExactData(model, paths)
            ll_package = loglik_exact(model.parameters.flat, exact_data; neg=false)
            
            @test ll_package ≈ ll_analytical rtol=ANALYTICAL_RTOL
        end
        
        @testset "Weibull with covariate (PH): scale adjustment" begin
            # PH effect: h(t|x) = kappa*lambda*exp(beta*x)*t^{kappa-1}
            #            H(t|x) = lambda*exp(beta*x)*t^kappa
            shape = 1.3   # shape (kappa)
            scale = 0.15  # scale (lambda)
            beta = 0.5    # covariate effect
            x = 1.0       # covariate value
            t = 3.0       # transition time
            
            h12 = Hazard(@formula(0 ~ 1 + x), "wei", 1, 2)
            
            dat = DataFrame(
                id = [1],
                tstart = [0.0],
                tstop = [t],
                statefrom = [1],
                stateto = [2],
                obstype = [1],
                x = [x]
            )
            
            model = multistatemodel(h12; data=dat)
            set_parameters!(model, (h12 = [shape, scale, beta],))
            
            # Effective scale: scale_eff = scale*exp(beta*x)
            scale_eff = scale * exp(beta * x)
            h_t = shape * scale_eff * t^(shape - 1)
            H_t = scale_eff * t^shape
            ll_analytical = log(h_t) - H_t
            
            paths = extract_paths(model)
            exact_data = ExactData(model, paths)
            ll_package = loglik_exact(model.parameters.flat, exact_data; neg=false)
            
            @test ll_package ≈ ll_analytical rtol=ANALYTICAL_RTOL
        end
        
        # =====================================================================
        # 1.3 Gompertz Hazard (flexsurv parameterization)
        # =====================================================================
        @testset "Gompertz: l = log(b*exp(a*t)) - (b/a)*(exp(a*t) - 1)" begin
            # Setup: Single transition 1->2 at time t=2.0
            # Gompertz: h(t) = b*exp(a*t), H(t) = (b/a)*(exp(a*t) - 1)
            a = 0.1   # shape (can be negative for decreasing hazard)
            b = 0.3   # rate
            t = 2.0   # transition time
            
            h12 = Hazard(@formula(0 ~ 1), "gom", 1, 2)
            
            dat = DataFrame(
                id = [1],
                tstart = [0.0],
                tstop = [t],
                statefrom = [1],
                stateto = [2],
                obstype = [1]
            )
            
            model = multistatemodel(h12; data=dat)
            set_parameters!(model, (h12 = [a, b],))
            
            # Gompertz hazard and cumulative hazard
            h_t = b * exp(a * t)
            H_t = (b / a) * (exp(a * t) - 1)
            ll_analytical = log(h_t) - H_t
            
            paths = extract_paths(model)
            exact_data = ExactData(model, paths)
            ll_package = loglik_exact(model.parameters.flat, exact_data; neg=false)
            
            @test ll_package ≈ ll_analytical rtol=ANALYTICAL_RTOL
        end
        
        @testset "Gompertz with negative shape (decreasing hazard)" begin
            # Decreasing hazard: a < 0
            a = -0.15  # negative shape
            b = 0.5    # rate
            t = 3.0    # transition time
            
            h12 = Hazard(@formula(0 ~ 1), "gom", 1, 2)
            
            dat = DataFrame(
                id = [1],
                tstart = [0.0],
                tstop = [t],
                statefrom = [1],
                stateto = [2],
                obstype = [1]
            )
            
            model = multistatemodel(h12; data=dat)
            set_parameters!(model, (h12 = [a, b],))
            
            h_t = b * exp(a * t)
            H_t = (b / a) * (exp(a * t) - 1)
            ll_analytical = log(h_t) - H_t
            
            paths = extract_paths(model)
            exact_data = ExactData(model, paths)
            ll_package = loglik_exact(model.parameters.flat, exact_data; neg=false)
            
            @test ll_package ≈ ll_analytical rtol=ANALYTICAL_RTOL
        end
        
        # =====================================================================
        # 1.4 Competing Risks (Multiple Exit Hazards)
        # =====================================================================
        @testset "Competing risks: l = log(h_exit) - H_12 - H_13" begin
            # State 1 can exit to state 2 or state 3
            # Subject transitions 1->2 at time t
            rate_12 = 0.3
            rate_13 = 0.2
            t = 2.0
            
            h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
            h13 = Hazard(@formula(0 ~ 1), "exp", 1, 3)
            
            dat = DataFrame(
                id = [1],
                tstart = [0.0],
                tstop = [t],
                statefrom = [1],
                stateto = [2],  # transition to state 2
                obstype = [1]
            )
            
            model = multistatemodel(h12, h13; data=dat)
            set_parameters!(model, (h12 = [rate_12], h13 = [rate_13]))
            
            # Likelihood: density of 1->2 * survival from 1->3
            # = h_12(t) * exp(-H_12(t)) * exp(-H_13(t))
            # = rate_12 * exp(-rate_12*t) * exp(-rate_13*t)
            # log-lik = log(rate_12) - rate_12*t - rate_13*t
            ll_analytical = log(rate_12) - rate_12 * t - rate_13 * t
            
            paths = extract_paths(model)
            exact_data = ExactData(model, paths)
            ll_package = loglik_exact(model.parameters.flat, exact_data; neg=false)
            
            @test ll_package ≈ ll_analytical rtol=ANALYTICAL_RTOL
        end
        
        # =====================================================================
        # 1.5 Right-Censored Observation (No Transition)
        # =====================================================================
        @testset "Right-censored: l = -H(t) (survival only)" begin
            # Subject observed in state 1 at time t, no transition
            rate = 0.4
            t = 5.0
            
            h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
            
            dat = DataFrame(
                id = [1],
                tstart = [0.0],
                tstop = [t],
                statefrom = [1],
                stateto = [1],  # no transition
                obstype = [1]
            )
            
            model = multistatemodel(h12; data=dat)
            set_parameters!(model, (h12 = [rate],))
            
            # No transition: just survival S(t) = exp(-rate*t)
            # log-lik = -rate*t
            ll_analytical = -rate * t
            
            paths = extract_paths(model)
            exact_data = ExactData(model, paths)
            ll_package = loglik_exact(model.parameters.flat, exact_data; neg=false)
            
            @test ll_package ≈ ll_analytical rtol=ANALYTICAL_RTOL
        end
        
        # =====================================================================
        # 1.6 Spline Hazard (constant hazard via equal coefficients)
        # =====================================================================
        @testset "Spline constant hazard (equal coefficients)" begin
            # B-spline hazards model the HAZARD directly (not log-hazard)
            # Due to partition of unity: Σᵢ Bᵢ(t) = 1 for all t
            # So if all coefficients βᵢ = c, then h(t) = c × Σᵢ Bᵢ(t) = c
            # This gives a constant hazard, equivalent to exponential with rate=c
            #
            # For exact transition data:
            # ℓ = log(h(t)) - H(t) = log(c) - c×t
            
            t = 1.0
            hazard_rate = 0.5  # Constant hazard value (coefficients all = 0.5)
            
            h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; degree=0, knots=nothing)
            
            dat = DataFrame(
                id = [1],
                tstart = [0.0],
                tstop = [t],
                statefrom = [1],
                stateto = [2],
                obstype = [1]
            )
            
            model = multistatemodel(h12; data=dat)
            
            # Get actual number of spline coefficients (auto-placed knots)
            n_coefs = length(model.parameters.flat)
            
            # Set all coefficients equal → constant hazard = hazard_rate
            set_parameters!(model, (h12 = fill(hazard_rate, n_coefs),))
            
            # Analytical formula for constant hazard:
            # h(t) = hazard_rate (constant)
            # H(t) = hazard_rate × t
            # ℓ = log(hazard_rate) - hazard_rate × t
            ll_analytical = log(hazard_rate) - hazard_rate * t
            
            paths = extract_paths(model)
            exact_data = ExactData(model, paths)
            ll_package = loglik_exact(model.parameters.flat, exact_data; neg=false)
            
            # Should match exponential with same rate
            @test ll_package ≈ ll_analytical rtol=NUMERICAL_RTOL
        end
        
        @testset "Spline degree 0 piecewise constant: 2 intervals" begin
            # Degree 0 B-splines are indicator functions on disjoint intervals.
            # With boundary [0, 2] and one interior knot at 1.0, we have 2 basis functions:
            # B₁(t) = I(0 ≤ t < 1), B₂(t) = I(1 ≤ t ≤ 2)
            #
            # h(t) = β₁⋅B₁(t) + β₂⋅B₂(t) = piecewise constant hazard
            #      = β₁ for t ∈ [0, 1), β₂ for t ∈ [1, 2]
            #
            # For transition at t = 1.5 (in second interval):
            # H(1.5) = β₁⋅1 + β₂⋅0.5  (integrate over [0,1] and [1,1.5])
            # h(1.5) = β₂
            # ℓ = log(β₂) - β₁ - 0.5⋅β₂
            
            t = 1.5
            β₁ = 0.3  # hazard in [0, 1)
            β₂ = 0.6  # hazard in [1, 2]
            
            h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                         degree=0, 
                         knots=[1.0],  # one interior knot
                         boundaryknots=[0.0, 2.0])
            
            dat = DataFrame(
                id = [1],
                tstart = [0.0],
                tstop = [t],
                statefrom = [1],
                stateto = [2],
                obstype = [1]
            )
            
            model = multistatemodel(h12; data=dat)
            set_parameters!(model, (h12 = [β₁, β₂],))
            
            # Analytical log-likelihood:
            # h(1.5) = β₂ (in second interval)
            # H(1.5) = β₁×1 + β₂×0.5 = β₁ + 0.5β₂
            h_t = β₂
            H_t = β₁ * 1.0 + β₂ * 0.5
            ll_analytical = log(h_t) - H_t
            
            paths = extract_paths(model)
            exact_data = ExactData(model, paths)
            ll_package = loglik_exact(model.parameters.flat, exact_data; neg=false)
            
            @test ll_package ≈ ll_analytical rtol=NUMERICAL_RTOL
        end
        
        @testset "Spline degree 0 piecewise constant: 3 intervals" begin
            # Three intervals with knots at 1.0 and 2.0, boundary [0, 3]
            # h(t) = β₁ for t ∈ [0,1), β₂ for t ∈ [1,2), β₃ for t ∈ [2,3]
            #
            # For transition at t = 2.5:
            # H(2.5) = β₁⋅1 + β₂⋅1 + β₃⋅0.5
            # h(2.5) = β₃
            
            t = 2.5
            β₁ = 0.2
            β₂ = 0.5
            β₃ = 0.8
            
            h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2;
                         degree=0,
                         knots=[1.0, 2.0],
                         boundaryknots=[0.0, 3.0])
            
            dat = DataFrame(
                id = [1],
                tstart = [0.0],
                tstop = [t],
                statefrom = [1],
                stateto = [2],
                obstype = [1]
            )
            
            model = multistatemodel(h12; data=dat)
            set_parameters!(model, (h12 = [β₁, β₂, β₃],))
            
            # Analytical: ℓ = log(β₃) - (β₁ + β₂ + 0.5β₃)
            h_t = β₃
            H_t = β₁ * 1.0 + β₂ * 1.0 + β₃ * 0.5
            ll_analytical = log(h_t) - H_t
            
            paths = extract_paths(model)
            exact_data = ExactData(model, paths)
            ll_package = loglik_exact(model.parameters.flat, exact_data; neg=false)
            
            @test ll_package ≈ ll_analytical rtol=NUMERICAL_RTOL
        end
        
        @testset "Spline degree 0: censored in first interval" begin
            # Transition at t = 0.4 (within first interval [0, 1))
            # h(0.4) = β₁, H(0.4) = β₁⋅0.4
            
            t = 0.4
            β₁ = 0.5
            β₂ = 1.0
            
            h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2;
                         degree=0,
                         knots=[1.0],
                         boundaryknots=[0.0, 2.0])
            
            dat = DataFrame(
                id = [1],
                tstart = [0.0],
                tstop = [t],
                statefrom = [1],
                stateto = [2],
                obstype = [1]
            )
            
            model = multistatemodel(h12; data=dat)
            set_parameters!(model, (h12 = [β₁, β₂],))
            
            h_t = β₁
            H_t = β₁ * t
            ll_analytical = log(h_t) - H_t
            
            paths = extract_paths(model)
            exact_data = ExactData(model, paths)
            ll_package = loglik_exact(model.parameters.flat, exact_data; neg=false)
            
            @test ll_package ≈ ll_analytical rtol=NUMERICAL_RTOL
        end
        
        @testset "Spline degree 0: survival only (right-censored)" begin
            # No transition, just survival
            # Subject observed in state 1 at t = 1.5 (spans both intervals)
            # ℓ = -H(1.5) = -(β₁⋅1 + β₂⋅0.5)
            
            t = 1.5
            β₁ = 0.3
            β₂ = 0.7
            
            h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2;
                         degree=0,
                         knots=[1.0],
                         boundaryknots=[0.0, 2.0])
            
            dat = DataFrame(
                id = [1],
                tstart = [0.0],
                tstop = [t],
                statefrom = [1],
                stateto = [1],  # no transition
                obstype = [1]
            )
            
            model = multistatemodel(h12; data=dat)
            set_parameters!(model, (h12 = [β₁, β₂],))
            
            H_t = β₁ * 1.0 + β₂ * 0.5
            ll_analytical = -H_t
            
            paths = extract_paths(model)
            exact_data = ExactData(model, paths)
            ll_package = loglik_exact(model.parameters.flat, exact_data; neg=false)
            
            @test ll_package ≈ ll_analytical rtol=NUMERICAL_RTOL
        end
        
        @testset "Spline degree 1: linear with known integral" begin
            # Degree 1 B-splines create a piecewise linear hazard.
            # With boundary knots [0, 2], we get a "hat function" basis.
            # 
            # For a single interior knot at t=1:
            # - B₁(t): peaks at 0, linear decrease to 1
            # - B₂(t): peaks at 1, linear from 0 and to 2
            # - B₃(t): peaks at 2, linear increase from 1
            #
            # Equal coefficients β still produce constant hazard (partition of unity).
            # Here we verify this property with degree=1 basis.
            
            t = 1.5
            β = 0.4  # constant hazard value
            
            h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2;
                         degree=1,
                         knots=[1.0],
                         boundaryknots=[0.0, 2.0])
            
            dat = DataFrame(
                id = [1],
                tstart = [0.0],
                tstop = [t],
                statefrom = [1],
                stateto = [2],
                obstype = [1]
            )
            
            model = multistatemodel(h12; data=dat)
            n_coefs = length(model.parameters.flat)
            set_parameters!(model, (h12 = fill(β, n_coefs),))
            
            # Partition of unity: equal coefficients → h(t) = β (constant)
            # ℓ = log(β) - β⋅t
            ll_analytical = log(β) - β * t
            
            paths = extract_paths(model)
            exact_data = ExactData(model, paths)
            ll_package = loglik_exact(model.parameters.flat, exact_data; neg=false)
            
            @test ll_package ≈ ll_analytical rtol=NUMERICAL_RTOL
        end
        
        @testset "Spline degree 3 (cubic): constant hazard" begin
            # Cubic B-splines also satisfy partition of unity.
            # Equal coefficients → constant hazard regardless of degree.
            
            t = 2.5
            β = 0.25
            
            h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2;
                         degree=3,
                         knots=[1.0, 2.0],
                         boundaryknots=[0.0, 4.0])
            
            dat = DataFrame(
                id = [1],
                tstart = [0.0],
                tstop = [t],
                statefrom = [1],
                stateto = [2],
                obstype = [1]
            )
            
            model = multistatemodel(h12; data=dat)
            n_coefs = length(model.parameters.flat)
            set_parameters!(model, (h12 = fill(β, n_coefs),))
            
            ll_analytical = log(β) - β * t
            
            paths = extract_paths(model)
            exact_data = ExactData(model, paths)
            ll_package = loglik_exact(model.parameters.flat, exact_data; neg=false)
            
            @test ll_package ≈ ll_analytical rtol=NUMERICAL_RTOL
        end
        
        @testset "Spline with PH covariate: degree 0 piecewise constant" begin
            # Piecewise constant hazard with proportional hazards covariate effect
            # h(t|x) = h₀(t)⋅exp(β⋅x) where h₀ is piecewise constant
            #
            # For x=1, β_cov=0.5: exp(0.5) ≈ 1.6487
            # h(t|x=1) = h₀(t)⋅exp(0.5)
            
            t = 1.5
            β₁ = 0.3   # baseline hazard in [0, 1)
            β₂ = 0.6   # baseline hazard in [1, 2]
            β_cov = 0.5  # covariate coefficient
            x = 1.0
            
            h12 = Hazard(@formula(0 ~ 1 + x), "sp", 1, 2;
                         degree=0,
                         knots=[1.0],
                         boundaryknots=[0.0, 2.0])
            
            dat = DataFrame(
                id = [1],
                tstart = [0.0],
                tstop = [t],
                statefrom = [1],
                stateto = [2],
                obstype = [1],
                x = [x]
            )
            
            model = multistatemodel(h12; data=dat)
            set_parameters!(model, (h12 = [β₁, β₂, β_cov],))
            
            # PH: h(t|x) = h₀(t)⋅exp(β_cov⋅x)
            # H(t|x) = H₀(t)⋅exp(β_cov⋅x)
            hr = exp(β_cov * x)
            
            # Baseline: h₀(1.5) = β₂, H₀(1.5) = β₁⋅1 + β₂⋅0.5
            h0_t = β₂
            H0_t = β₁ * 1.0 + β₂ * 0.5
            
            h_t = h0_t * hr
            H_t = H0_t * hr
            ll_analytical = log(h_t) - H_t
            
            paths = extract_paths(model)
            exact_data = ExactData(model, paths)
            ll_package = loglik_exact(model.parameters.flat, exact_data; neg=false)
            
            @test ll_package ≈ ll_analytical rtol=NUMERICAL_RTOL
        end
        
        @testset "Spline competing risks: degree 0 piecewise constant" begin
            # State 1 can exit to state 2 or state 3
            # Both hazards are piecewise constant with degree 0 splines
            #
            # h₁₂(t) = β₁ for t ∈ [0,1), β₂ for t ∈ [1,2]
            # h₁₃(t) = γ₁ for t ∈ [0,1), γ₂ for t ∈ [1,2]
            #
            # Transition to state 2 at t=1.5:
            # ℓ = log(h₁₂(1.5)) - H₁₂(1.5) - H₁₃(1.5)
            
            t = 1.5
            β₁, β₂ = 0.3, 0.5  # 1→2 hazard
            γ₁, γ₂ = 0.2, 0.4  # 1→3 hazard
            
            h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2;
                         degree=0, knots=[1.0], boundaryknots=[0.0, 2.0])
            h13 = Hazard(@formula(0 ~ 1), "sp", 1, 3;
                         degree=0, knots=[1.0], boundaryknots=[0.0, 2.0])
            
            dat = DataFrame(
                id = [1],
                tstart = [0.0],
                tstop = [t],
                statefrom = [1],
                stateto = [2],  # transition to state 2
                obstype = [1]
            )
            
            model = multistatemodel(h12, h13; data=dat)
            set_parameters!(model, (h12 = [β₁, β₂], h13 = [γ₁, γ₂]))
            
            # ℓ = log(h₁₂(1.5)) - H₁₂(1.5) - H₁₃(1.5)
            h_12_t = β₂
            H_12_t = β₁ * 1.0 + β₂ * 0.5
            H_13_t = γ₁ * 1.0 + γ₂ * 0.5
            
            ll_analytical = log(h_12_t) - H_12_t - H_13_t
            
            paths = extract_paths(model)
            exact_data = ExactData(model, paths)
            ll_package = loglik_exact(model.parameters.flat, exact_data; neg=false)
            
            @test ll_package ≈ ll_analytical rtol=NUMERICAL_RTOL
        end
        
        # =====================================================================
        # 1.7 AFT (Accelerated Failure Time) Covariate Effects
        # =====================================================================
        # AFT: h(t|x) = h₀(t⋅exp(-βx))⋅exp(-βx), H(t|x) = H₀(t⋅exp(-βx))
        # Time is "accelerated" or "decelerated" depending on sign of βx
        
        @testset "Weibull AFT: time acceleration" begin
            # Weibull with AFT parameterization
            # h₀(t) = κλt^{κ-1}, H₀(t) = λt^κ
            # AFT: h(t|x) = κλ(t⋅e^{-βx})^{κ-1}⋅e^{-βx} = κλt^{κ-1}⋅e^{-βx⋅κ}
            # AFT: H(t|x) = H₀(t⋅e^{-βx}) = λ(t⋅e^{-βx})^κ = λt^κ⋅e^{-βxκ}
            
            shape = 1.5   # shape (kappa)
            scale = 0.2   # scale (lambda)
            beta = 0.6    # AFT covariate effect (positive = deceleration)
            x = 1.0       # covariate value
            t = 2.0       # transition time
            
            h12 = Hazard(@formula(0 ~ 1 + x), "wei", 1, 2; linpred_effect=:aft)
            
            dat = DataFrame(
                id = [1],
                tstart = [0.0],
                tstop = [t],
                statefrom = [1],
                stateto = [2],
                obstype = [1],
                x = [x]
            )
            
            model = multistatemodel(h12; data=dat)
            set_parameters!(model, (h12 = [shape, scale, beta],))
            
            # AFT formula: time_scale = exp(-β*x)
            time_scale = exp(-beta * x)
            t_eff = t * time_scale  # Effective time
            
            # h(t|x) = h₀(t_eff) * time_scale
            h_t = shape * scale * t_eff^(shape - 1) * time_scale
            # H(t|x) = H₀(t_eff)
            H_t = scale * t_eff^shape
            ll_analytical = log(h_t) - H_t
            
            paths = extract_paths(model)
            exact_data = ExactData(model, paths)
            ll_package = loglik_exact(model.parameters.flat, exact_data; neg=false)
            
            @test ll_package ≈ ll_analytical rtol=ANALYTICAL_RTOL
        end
        
        @testset "Gompertz AFT: time acceleration" begin
            # Gompertz with AFT parameterization
            # h₀(t) = b⋅exp(a⋅t), H₀(t) = (b/a)(exp(at) - 1)
            # AFT: time_scale = exp(-βx), t_eff = t⋅time_scale
            
            a = 0.1       # shape
            b = 0.3       # rate  
            beta = 0.5    # AFT covariate effect
            x = 1.0       # covariate value
            t = 2.5       # transition time
            
            h12 = Hazard(@formula(0 ~ 1 + x), "gom", 1, 2; linpred_effect=:aft)
            
            dat = DataFrame(
                id = [1],
                tstart = [0.0],
                tstop = [t],
                statefrom = [1],
                stateto = [2],
                obstype = [1],
                x = [x]
            )
            
            model = multistatemodel(h12; data=dat)
            set_parameters!(model, (h12 = [a, b, beta],))
            
            # AFT: time_scale = exp(-β*x)
            time_scale = exp(-beta * x)
            t_eff = t * time_scale
            
            # h(t|x) = h₀(t_eff) * time_scale = b⋅exp(a⋅t_eff)⋅time_scale
            h_t = b * exp(a * t_eff) * time_scale
            # H(t|x) = H₀(t_eff) = (b/a)(exp(a⋅t_eff) - 1)
            H_t = (b / a) * (exp(a * t_eff) - 1)
            ll_analytical = log(h_t) - H_t
            
            paths = extract_paths(model)
            exact_data = ExactData(model, paths)
            ll_package = loglik_exact(model.parameters.flat, exact_data; neg=false)
            
            @test ll_package ≈ ll_analytical rtol=ANALYTICAL_RTOL
        end
        
        @testset "Exponential AFT: reduces to PH (for exponential)" begin
            # For exponential: AFT and PH are equivalent (constant hazard)
            # h₀(t) = λ, h(t|x) = λ⋅exp(-βx)
            # Note: For exponential, AFT gives h(t|x) = λ⋅time_scale = λ⋅exp(-βx)
            # This is the same as PH with negative coefficient
            
            rate = 0.4
            beta = 0.3    # AFT effect
            x = 1.0
            t = 3.0
            
            h12 = Hazard(@formula(0 ~ 1 + x), "exp", 1, 2; linpred_effect=:aft)
            
            dat = DataFrame(
                id = [1],
                tstart = [0.0],
                tstop = [t],
                statefrom = [1],
                stateto = [2],
                obstype = [1],
                x = [x]
            )
            
            model = multistatemodel(h12; data=dat)
            set_parameters!(model, (h12 = [rate, beta],))
            
            # For exponential AFT: effective rate = rate * exp(-β*x)
            time_scale = exp(-beta * x)
            rate_eff = rate * time_scale
            
            h_t = rate_eff
            H_t = rate_eff * t
            ll_analytical = log(h_t) - H_t
            
            paths = extract_paths(model)
            exact_data = ExactData(model, paths)
            ll_package = loglik_exact(model.parameters.flat, exact_data; neg=false)
            
            @test ll_package ≈ ll_analytical rtol=ANALYTICAL_RTOL
        end
        
        # NOTE: Phase-type (pt) hazards are NOT tested here because:
        # 1. Phase-type models operate on an expanded state space internally
        # 2. The user-facing likelihood is a convolution over hidden phases
        # 3. There's no simple closed-form analytical expression for comparison
        # 4. Phase-type correctness is validated in integration tests
        #    (longtest_phasetype_exact.jl, longtest_phasetype_panel.jl)
        
        # NOTE: Time-varying covariates (TVC) are NOT tested here because:
        # 1. TVC requires piecewise integration over covariate change points
        # 2. No single analytical formula applies - depends on change times
        # 3. TVC correctness is validated in integration tests
        #    (longtest_mcem_tvc.jl, longtest_aft_suite.jl)
        
    end
    
    # =========================================================================
    # PART 2: PANEL DATA (Interval-Censored Markov)
    # =========================================================================
    
    @testset "Panel Data: Markov Transition Probabilities" begin
        
        # =====================================================================
        # 2.1 Two-State Absorbing Model
        # =====================================================================
        @testset "2-state absorbing: P12(t) = 1 - exp(-lambda*t)" begin
            # Single observation: state 1 at t=0, state 2 at t=2
            rate = 0.5
            t = 2.0
            
            h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
            
            dat = DataFrame(
                id = [1],
                tstart = [0.0],
                tstop = [t],
                statefrom = [1],
                stateto = [2],
                obstype = [2]  # panel data
            )
            
            model = multistatemodel(h12; data=dat)
            set_parameters!(model, (h12 = [rate],))
            
            # Analytical transition probability: P12(t) = 1 - exp(-rate*t)
            P_12 = 1 - exp(-rate * t)
            ll_analytical = log(P_12)
            
            books = build_tpm_mapping(model.data)
            mpd = MPanelData(model, books)
            ll_package = loglik_markov(model.parameters.flat, mpd; neg=false)
            
            @test ll_package ≈ ll_analytical rtol=ANALYTICAL_RTOL
        end
        
        @testset "2-state absorbing: P11(t) = exp(-lambda*t) (no transition)" begin
            # Subject stays in state 1
            rate = 0.3
            t = 3.0
            
            h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
            
            dat = DataFrame(
                id = [1],
                tstart = [0.0],
                tstop = [t],
                statefrom = [1],
                stateto = [1],  # stays in state 1
                obstype = [2]
            )
            
            model = multistatemodel(h12; data=dat)
            set_parameters!(model, (h12 = [rate],))
            
            # Survival probability: P11(t) = exp(-rate*t)
            P_11 = exp(-rate * t)
            ll_analytical = log(P_11)
            
            books = build_tpm_mapping(model.data)
            mpd = MPanelData(model, books)
            ll_package = loglik_markov(model.parameters.flat, mpd; neg=false)
            
            @test ll_package ≈ ll_analytical rtol=ANALYTICAL_RTOL
        end
        
        # =====================================================================
        # 2.2 Three-State Progressive Model
        # =====================================================================
        @testset "3-state progressive: P13(t) via matrix exponential" begin
            # Model: 1 -> 2 -> 3 (progressive)
            # Q = [-rate12  rate12    0  ]
            #     [  0    -rate23  rate23]
            #     [  0       0      0    ]
            
            rate_12 = 0.3
            rate_23 = 0.4
            t = 4.0
            
            h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
            h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
            
            dat = DataFrame(
                id = [1],
                tstart = [0.0],
                tstop = [t],
                statefrom = [1],
                stateto = [3],  # observed in absorbing state 3
                obstype = [2]
            )
            
            model = multistatemodel(h12, h23; data=dat)
            set_parameters!(model, (h12 = [rate_12], h23 = [rate_23]))
            
            # Analytical P13(t) for progressive 3-state:
            # P13 = 1 - P11 - P12
            # where P11 = exp(-rate12*t)
            # and P12 = (rate12/(rate23-rate12))*(exp(-rate12*t) - exp(-rate23*t)) for rate12 != rate23
            
            P_11 = exp(-rate_12 * t)
            if abs(rate_12 - rate_23) > 1e-10
                P_12 = (rate_12 / (rate_23 - rate_12)) * (exp(-rate_12 * t) - exp(-rate_23 * t))
            else
                # rate12 = rate23 case: P12 = rate12*t*exp(-rate12*t)
                P_12 = rate_12 * t * exp(-rate_12 * t)
            end
            P_13 = 1 - P_11 - P_12
            ll_analytical = log(P_13)
            
            books = build_tpm_mapping(model.data)
            mpd = MPanelData(model, books)
            ll_package = loglik_markov(model.parameters.flat, mpd; neg=false)
            
            @test ll_package ≈ ll_analytical rtol=NUMERICAL_RTOL
        end
        
        @testset "3-state progressive: P12(t) (intermediate state)" begin
            # Subject observed in state 2 (not yet absorbed)
            rate_12 = 0.25
            rate_23 = 0.35
            t = 2.5
            
            h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
            h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
            
            dat = DataFrame(
                id = [1],
                tstart = [0.0],
                tstop = [t],
                statefrom = [1],
                stateto = [2],  # observed in intermediate state 2
                obstype = [2]
            )
            
            model = multistatemodel(h12, h23; data=dat)
            set_parameters!(model, (h12 = [rate_12], h23 = [rate_23]))
            
            # P12(t) = (rate12/(rate23-rate12))*(exp(-rate12*t) - exp(-rate23*t))
            if abs(rate_12 - rate_23) > 1e-10
                P_12 = (rate_12 / (rate_23 - rate_12)) * (exp(-rate_12 * t) - exp(-rate_23 * t))
            else
                P_12 = rate_12 * t * exp(-rate_12 * t)
            end
            ll_analytical = log(P_12)
            
            books = build_tpm_mapping(model.data)
            mpd = MPanelData(model, books)
            ll_package = loglik_markov(model.parameters.flat, mpd; neg=false)
            
            @test ll_package ≈ ll_analytical rtol=NUMERICAL_RTOL
        end
        
        # =====================================================================
        # 2.3 Multiple Intervals (Product of Transition Probabilities)
        # =====================================================================
        @testset "Multiple intervals: l = sum log(P_{si,si+1}(dt_i))" begin
            # Two observation intervals:
            # [0,2]: state 1 -> state 1 (survival)
            # [2,5]: state 1 -> state 2 (transition)
            rate = 0.4
            
            h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
            
            dat = DataFrame(
                id = [1, 1],
                tstart = [0.0, 2.0],
                tstop = [2.0, 5.0],
                statefrom = [1, 1],
                stateto = [1, 2],
                obstype = [2, 2]
            )
            
            model = multistatemodel(h12; data=dat)
            set_parameters!(model, (h12 = [rate],))
            
            # Interval 1: P11(2) = exp(-rate*2)
            # Interval 2: P12(3) = 1 - exp(-rate*3)
            P_11_interval1 = exp(-rate * 2)
            P_12_interval2 = 1 - exp(-rate * 3)
            ll_analytical = log(P_11_interval1) + log(P_12_interval2)
            
            books = build_tpm_mapping(model.data)
            mpd = MPanelData(model, books)
            ll_package = loglik_markov(model.parameters.flat, mpd; neg=false)
            
            @test ll_package ≈ ll_analytical rtol=ANALYTICAL_RTOL
        end
        
        # =====================================================================
        # 2.4 Panel Data with Covariate (PH effect on rates)
        # =====================================================================
        @testset "Panel with covariate: effective rate lambda*exp(beta*x)" begin
            rate = 0.2
            beta = 0.6
            x = 1.0
            t = 3.0
            
            h12 = Hazard(@formula(0 ~ 1 + x), "exp", 1, 2)
            
            dat = DataFrame(
                id = [1],
                tstart = [0.0],
                tstop = [t],
                statefrom = [1],
                stateto = [2],
                obstype = [2],
                x = [x]
            )
            
            model = multistatemodel(h12; data=dat)
            set_parameters!(model, (h12 = [rate, beta],))
            
            # Effective rate: rate_eff = rate*exp(beta*x)
            rate_eff = rate * exp(beta * x)
            P_12 = 1 - exp(-rate_eff * t)
            ll_analytical = log(P_12)
            
            books = build_tpm_mapping(model.data)
            mpd = MPanelData(model, books)
            ll_package = loglik_markov(model.parameters.flat, mpd; neg=false)
            
            @test ll_package ≈ ll_analytical rtol=ANALYTICAL_RTOL
        end
        
        # =====================================================================
        # 2.5 Reversible 2-State Model (Birth-Death Process)
        # =====================================================================
        @testset "Reversible 2-state: analytical matrix exponential" begin
            # Q = [-rate12   rate12]
            #     [ rate21  -rate21]
            # Eigenvalues: 0, -(rate12 + rate21)
            # P12(t) = (rate12/(rate12+rate21))*(1 - exp(-(rate12+rate21)*t))
            
            rate_12 = 0.3
            rate_21 = 0.2
            t = 2.0
            
            h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
            h21 = Hazard(@formula(0 ~ 1), "exp", 2, 1)
            
            dat = DataFrame(
                id = [1],
                tstart = [0.0],
                tstop = [t],
                statefrom = [1],
                stateto = [2],
                obstype = [2]
            )
            
            model = multistatemodel(h12, h21; data=dat)
            set_parameters!(model, (h12 = [rate_12], h21 = [rate_21]))
            
            # Analytical P12(t) for reversible 2-state
            total_rate = rate_12 + rate_21
            pi_2 = rate_12 / total_rate  # stationary probability of state 2
            P_12 = pi_2 * (1 - exp(-total_rate * t))
            ll_analytical = log(P_12)
            
            books = build_tpm_mapping(model.data)
            mpd = MPanelData(model, books)
            ll_package = loglik_markov(model.parameters.flat, mpd; neg=false)
            
            @test ll_package ≈ ll_analytical rtol=NUMERICAL_RTOL
        end
        
        # =====================================================================
        # 2.6 Panel Data with Spline Hazards
        # =====================================================================
        # For Markov panel data, P(t) = exp(Qt) where Q is the rate matrix.
        # For a 2-state absorbing model with constant hazard λ:
        # P12(t) = 1 - exp(-λ*t), P11(t) = exp(-λ*t)
        #
        # Spline hazards with equal coefficients produce constant hazard (partition of unity).
        
        @testset "Panel spline: constant hazard via equal coefficients" begin
            # Degree 0 spline with equal coefficients = constant hazard
            # This is equivalent to exponential model
            rate = 0.4
            t = 2.5
            
            h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2;
                         degree=0,
                         knots=[1.0, 2.0],
                         boundaryknots=[0.0, 3.0])
            
            dat = DataFrame(
                id = [1],
                tstart = [0.0],
                tstop = [t],
                statefrom = [1],
                stateto = [2],
                obstype = [2]  # panel data
            )
            
            model = multistatemodel(h12; data=dat)
            n_coefs = length(model.parameters.flat)
            set_parameters!(model, (h12 = fill(rate, n_coefs),))
            
            # With equal coefficients, h(t) = rate (constant), equivalent to exponential
            P_12 = 1 - exp(-rate * t)
            ll_analytical = log(P_12)
            
            books = build_tpm_mapping(model.data)
            mpd = MPanelData(model, books)
            ll_package = loglik_markov(model.parameters.flat, mpd; neg=false)
            
            @test ll_package ≈ ll_analytical rtol=NUMERICAL_RTOL
        end
        
        @testset "Panel spline: piecewise constant - Markov approximation" begin
            # IMPORTANT: For panel data, the package uses Markov approximation.
            # The hazard rate is evaluated at t=tstart and treated as CONSTANT
            # for the entire interval. This is NOT the same as integrating
            # the piecewise hazard over the interval.
            #
            # For interval [0, 1.5] with h(0) = β₁:
            # P12(1.5) = 1 - exp(-β₁ * 1.5)  (using rate at t=0)
            #
            # This differs from exact data where cumulative hazard is integrated.
            
            t = 1.5
            β₁ = 0.3  # hazard in [0, 1)
            β₂ = 0.6  # hazard in [1, 2]
            
            h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2;
                         degree=0,
                         knots=[1.0],
                         boundaryknots=[0.0, 2.0])
            
            dat = DataFrame(
                id = [1],
                tstart = [0.0],
                tstop = [t],
                statefrom = [1],
                stateto = [2],
                obstype = [2]
            )
            
            model = multistatemodel(h12; data=dat)
            set_parameters!(model, (h12 = [β₁, β₂],))
            
            # Markov approximation: rate = h(tstart) = h(0) = β₁
            # P12(dt) = 1 - exp(-rate * dt) = 1 - exp(-β₁ * 1.5)
            rate = β₁  # hazard at tstart=0
            dt = t
            P_12 = 1 - exp(-rate * dt)
            ll_analytical = log(P_12)
            
            books = build_tpm_mapping(model.data)
            mpd = MPanelData(model, books)
            ll_package = loglik_markov(model.parameters.flat, mpd; neg=false)
            
            @test ll_package ≈ ll_analytical rtol=NUMERICAL_RTOL
        end
        
        @testset "Panel spline: survival - Markov approximation" begin
            # Subject stays in state 1
            # With Markov approximation: P11(dt) = exp(-h(tstart) * dt)
            
            t = 1.5
            β₁ = 0.4
            β₂ = 0.8
            
            h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2;
                         degree=0,
                         knots=[1.0],
                         boundaryknots=[0.0, 2.0])
            
            dat = DataFrame(
                id = [1],
                tstart = [0.0],
                tstop = [t],
                statefrom = [1],
                stateto = [1],  # no transition
                obstype = [2]
            )
            
            model = multistatemodel(h12; data=dat)
            set_parameters!(model, (h12 = [β₁, β₂],))
            
            # Markov approximation: rate = h(0) = β₁
            rate = β₁
            dt = t
            P_11 = exp(-rate * dt)
            ll_analytical = log(P_11)
            
            books = build_tpm_mapping(model.data)
            mpd = MPanelData(model, books)
            ll_package = loglik_markov(model.parameters.flat, mpd; neg=false)
            
            @test ll_package ≈ ll_analytical rtol=NUMERICAL_RTOL
        end
        
        @testset "Panel spline: degree 1 with constant hazard" begin
            # Degree 1 spline with equal coefficients = constant hazard
            rate = 0.35
            t = 3.0
            
            h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2;
                         degree=1,
                         knots=[1.0, 2.0],
                         boundaryknots=[0.0, 4.0])
            
            dat = DataFrame(
                id = [1],
                tstart = [0.0],
                tstop = [t],
                statefrom = [1],
                stateto = [2],
                obstype = [2]
            )
            
            model = multistatemodel(h12; data=dat)
            n_coefs = length(model.parameters.flat)
            set_parameters!(model, (h12 = fill(rate, n_coefs),))
            
            P_12 = 1 - exp(-rate * t)
            ll_analytical = log(P_12)
            
            books = build_tpm_mapping(model.data)
            mpd = MPanelData(model, books)
            ll_package = loglik_markov(model.parameters.flat, mpd; neg=false)
            
            @test ll_package ≈ ll_analytical rtol=NUMERICAL_RTOL
        end
        
        @testset "Panel spline: multiple intervals - h(0) for all intervals" begin
            # Two observation intervals with piecewise constant spline
            # [0, 1.5]: state 1 → state 1 (survival)
            # [1.5, 2.5]: state 1 → state 2 (transition)
            #
            # IMPORTANT: For Markov panel data, the hazard is evaluated at t=0
            # for ALL intervals because Markov models assume time-invariant rates.
            # This means non-constant splines effectively use only h(0) for panel data.
            
            β₁ = 0.3  # hazard in [0, 1)
            β₂ = 0.5  # hazard in [1, 2)
            β₃ = 0.4  # hazard in [2, 3]
            
            h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2;
                         degree=0,
                         knots=[1.0, 2.0],
                         boundaryknots=[0.0, 3.0])
            
            dat = DataFrame(
                id = [1, 1],
                tstart = [0.0, 1.5],
                tstop = [1.5, 2.5],
                statefrom = [1, 1],
                stateto = [1, 2],
                obstype = [2, 2]
            )
            
            model = multistatemodel(h12; data=dat)
            set_parameters!(model, (h12 = [β₁, β₂, β₃],))
            
            # Markov assumption: rate = h(0) = β₁ for ALL intervals
            rate = β₁  # h(0) is used for all panel intervals
            
            # Interval 1 [0, 1.5]: P11 = exp(-rate * 1.5)
            P_11_interval1 = exp(-rate * 1.5)
            
            # Interval 2 [1.5, 2.5]: P12 = 1 - exp(-rate * 1.0)
            P_12_interval2 = 1 - exp(-rate * 1.0)
            
            ll_analytical = log(P_11_interval1) + log(P_12_interval2)
            
            books = build_tpm_mapping(model.data)
            mpd = MPanelData(model, books)
            ll_package = loglik_markov(model.parameters.flat, mpd; neg=false)
            
            @test ll_package ≈ ll_analytical rtol=NUMERICAL_RTOL
        end
    end
    
    # =========================================================================
    # PART 3: MULTI-SUBJECT VERIFICATION
    # =========================================================================
    
    @testset "Multi-Subject: Sum of Individual Likelihoods" begin
        
        @testset "Two subjects with same parameters" begin
            rate = 0.5
            t1, t2 = 2.0, 3.0
            
            h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
            
            dat = DataFrame(
                id = [1, 2],
                tstart = [0.0, 0.0],
                tstop = [t1, t2],
                statefrom = [1, 1],
                stateto = [2, 2],
                obstype = [1, 1]  # exact data
            )
            
            model = multistatemodel(h12; data=dat)
            set_parameters!(model, (h12 = [rate],))
            
            # Each subject's likelihood
            ll_subj1 = log(rate) - rate * t1
            ll_subj2 = log(rate) - rate * t2
            ll_analytical = ll_subj1 + ll_subj2
            
            paths = extract_paths(model)
            exact_data = ExactData(model, paths)
            ll_package = loglik_exact(model.parameters.flat, exact_data; neg=false)
            
            @test ll_package ≈ ll_analytical rtol=ANALYTICAL_RTOL
        end
        
        @testset "Two subjects with different covariates" begin
            rate = 0.3
            beta = 0.5
            t = 2.0
            x1, x2 = 0.0, 1.0  # different covariate values
            
            h12 = Hazard(@formula(0 ~ 1 + x), "exp", 1, 2)
            
            dat = DataFrame(
                id = [1, 2],
                tstart = [0.0, 0.0],
                tstop = [t, t],
                statefrom = [1, 1],
                stateto = [2, 2],
                obstype = [1, 1],
                x = [x1, x2]
            )
            
            model = multistatemodel(h12; data=dat)
            set_parameters!(model, (h12 = [rate, beta],))
            
            rate_eff1 = rate * exp(beta * x1)
            rate_eff2 = rate * exp(beta * x2)
            ll_subj1 = log(rate_eff1) - rate_eff1 * t
            ll_subj2 = log(rate_eff2) - rate_eff2 * t
            ll_analytical = ll_subj1 + ll_subj2
            
            paths = extract_paths(model)
            exact_data = ExactData(model, paths)
            ll_package = loglik_exact(model.parameters.flat, exact_data; neg=false)
            
            @test ll_package ≈ ll_analytical rtol=ANALYTICAL_RTOL
        end
        
        @testset "Two subjects panel data" begin
            rate = 0.4
            t1, t2 = 2.0, 3.0
            
            h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
            
            dat = DataFrame(
                id = [1, 2],
                tstart = [0.0, 0.0],
                tstop = [t1, t2],
                statefrom = [1, 1],
                stateto = [2, 1],  # subject 1: transition, subject 2: no transition
                obstype = [2, 2]
            )
            
            model = multistatemodel(h12; data=dat)
            set_parameters!(model, (h12 = [rate],))
            
            # Subject 1: P12(t1) = 1 - exp(-rate*t1)
            # Subject 2: P11(t2) = exp(-rate*t2)
            P_12_subj1 = 1 - exp(-rate * t1)
            P_11_subj2 = exp(-rate * t2)
            ll_analytical = log(P_12_subj1) + log(P_11_subj2)
            
            books = build_tpm_mapping(model.data)
            mpd = MPanelData(model, books)
            ll_package = loglik_markov(model.parameters.flat, mpd; neg=false)
            
            @test ll_package ≈ ll_analytical rtol=ANALYTICAL_RTOL
        end
    end
    
    # =========================================================================
    # PART 4: EDGE CASES AND BOUNDARY CONDITIONS
    # =========================================================================
    
    @testset "Edge Cases" begin
        
        @testset "Very short interval (t close to 0)" begin
            rate = 0.5
            t = 1e-6  # very small time
            
            h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
            
            dat = DataFrame(
                id = [1],
                tstart = [0.0],
                tstop = [t],
                statefrom = [1],
                stateto = [2],
                obstype = [1]
            )
            
            model = multistatemodel(h12; data=dat)
            set_parameters!(model, (h12 = [rate],))
            
            ll_analytical = log(rate) - rate * t
            
            paths = extract_paths(model)
            exact_data = ExactData(model, paths)
            ll_package = loglik_exact(model.parameters.flat, exact_data; neg=false)
            
            @test ll_package ≈ ll_analytical rtol=1e-5  # Slightly relaxed for small t
        end
        
        @testset "Large time interval" begin
            rate = 0.1
            t = 50.0  # large time
            
            h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
            
            dat = DataFrame(
                id = [1],
                tstart = [0.0],
                tstop = [t],
                statefrom = [1],
                stateto = [2],
                obstype = [1]
            )
            
            model = multistatemodel(h12; data=dat)
            set_parameters!(model, (h12 = [rate],))
            
            ll_analytical = log(rate) - rate * t
            
            paths = extract_paths(model)
            exact_data = ExactData(model, paths)
            ll_package = loglik_exact(model.parameters.flat, exact_data; neg=false)
            
            @test ll_package ≈ ll_analytical rtol=ANALYTICAL_RTOL
        end
        
        @testset "Weibull shape = 1 (reduces to exponential)" begin
            shape = 1.0   # shape = 1 -> Weibull becomes exponential
            scale = 0.4   # scale = rate for shape=1
            t = 3.0
            
            h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
            
            dat = DataFrame(
                id = [1],
                tstart = [0.0],
                tstop = [t],
                statefrom = [1],
                stateto = [2],
                obstype = [1]
            )
            
            model = multistatemodel(h12; data=dat)
            set_parameters!(model, (h12 = [shape, scale],))
            
            # When shape=1: h(t) = scale, H(t) = scale*t (exponential)
            ll_analytical = log(scale) - scale * t
            
            paths = extract_paths(model)
            exact_data = ExactData(model, paths)
            ll_package = loglik_exact(model.parameters.flat, exact_data; neg=false)
            
            @test ll_package ≈ ll_analytical rtol=ANALYTICAL_RTOL
        end
        
        @testset "Gompertz shape near 0 (approximately exponential)" begin
            # As a -> 0: h(t) -> b (constant), H(t) -> b*t
            # Use small a to test near-exponential behavior
            a = 1e-8   # very small shape
            b = 0.3    # rate
            t = 2.0
            
            h12 = Hazard(@formula(0 ~ 1), "gom", 1, 2)
            
            dat = DataFrame(
                id = [1],
                tstart = [0.0],
                tstop = [t],
                statefrom = [1],
                stateto = [2],
                obstype = [1]
            )
            
            model = multistatemodel(h12; data=dat)
            set_parameters!(model, (h12 = [a, b],))
            
            # Limit formula: exponential with rate b
            ll_exponential = log(b) - b * t
            
            paths = extract_paths(model)
            exact_data = ExactData(model, paths)
            ll_package = loglik_exact(model.parameters.flat, exact_data; neg=false)
            
            # Should be close to exponential
            @test ll_package ≈ ll_exponential rtol=1e-4
        end
        
        @testset "Panel: near-certain transition (high rate, long time)" begin
            rate = 2.0
            t = 10.0  # P12 should be very close to 1
            
            h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
            
            dat = DataFrame(
                id = [1],
                tstart = [0.0],
                tstop = [t],
                statefrom = [1],
                stateto = [2],
                obstype = [2]
            )
            
            model = multistatemodel(h12; data=dat)
            set_parameters!(model, (h12 = [rate],))
            
            P_12 = 1 - exp(-rate * t)
            ll_analytical = log(P_12)
            
            books = build_tpm_mapping(model.data)
            mpd = MPanelData(model, books)
            ll_package = loglik_markov(model.parameters.flat, mpd; neg=false)
            
            # Relaxed tolerance for very small probabilities (near 1-P close to 0)
            # Matrix exponential numerical precision limits at very small values
            @test ll_package ≈ ll_analytical rtol=1e-6
        end
    end
end

println("\nAll analytical log-likelihood verification tests completed")
