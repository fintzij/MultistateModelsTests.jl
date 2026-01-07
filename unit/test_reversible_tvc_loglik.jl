# =============================================================================
# Unit Tests for Reversible Semi-Markov Models with TVC
# =============================================================================
#
# Tests targeting reversible models with semi-Markov hazards and TVC.
# NOTE: Basic analytical likelihood validation is in test_loglik.jl
# This file focuses on reversible model-specific behaviors.

using Test
using MultistateModels
using DataFrames
using Random
using Statistics

import MultistateModels: 
    SamplePath, make_subjdat, set_parameters!, SMPanelData, loglik!

@testset "Reversible Semi-Markov with TVC" begin
    
    @testset "Sojourn time resets in reversible model" begin
        # Path: 1 (0-3) → 2 (3-7) → 1 (7-12) → 2 (12-15)
        # Sojourn should reset each time we enter a new state
        
        h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
        h21 = Hazard(@formula(0 ~ 1), "wei", 2, 1)
        
        dat = DataFrame(
            id = fill(1, 4),
            tstart = [0.0, 3.0, 7.0, 12.0],
            tstop = [3.0, 7.0, 12.0, 15.0],
            statefrom = [1, 2, 1, 2],
            stateto = [2, 1, 2, 2],
            obstype = fill(1, 4)
        )
        
        model = multistatemodel(h12, h21; data=dat)
        set_parameters!(model, (h12 = [2.0, 0.3], h21 = [1.5, 0.25]))
        
        paths = MultistateModels.extract_paths(model)
        @test length(paths) == 1
        
        path = paths[1]
        @test path.times ≈ [0.0, 3.0, 7.0, 12.0, 15.0]
        @test path.states == [1, 2, 1, 2, 2]
        
        # Compute likelihood
        subjectdata = view(model.data, model.data.id .== 1, :)
        subjdat_path = make_subjdat(path, subjectdata)
        
        # Verify sojourn times are correct - should reset each transition
        @test subjdat_path.sojourn ≈ [0.0, 0.0, 0.0, 0.0]
    end
    
    @testset "Manual vs package path likelihood - reversible Weibull" begin
        # Create a simple reversible model and manually compute likelihood
        
        h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
        h21 = Hazard(@formula(0 ~ 1), "wei", 2, 1)
        
        dat = DataFrame(
            id = [1, 1],
            tstart = [0.0, 5.0],
            tstop = [5.0, 10.0],
            statefrom = [1, 2],
            stateto = [2, 2],
            obstype = [1, 1]
        )
        
        model = multistatemodel(h12, h21; data=dat)
        
        # Parameters: shape, scale (natural scale)
        shape_12, scale_12 = 1.5, 0.2
        shape_21, scale_21 = 1.2, 0.3
        set_parameters!(model, (
            h12 = [shape_12, scale_12],
            h21 = [shape_21, scale_21]
        ))
        
        # Path: state 1 for [0, 5], then state 2 for [5, 10]
        # Transition from 1→2 at t=5
        
        # Manual calculation for Weibull hazard:
        # h(t) = scale * shape * t^(shape-1)
        # H(t) = scale * t^shape  (cumulative hazard)
        # 
        # Interval [0, 5] in state 1:
        #   - Sojourn time = 5
        #   - h12 density at t=5: scale_12 * shape_12 * 5^(shape_12-1)
        #   - h12 cumulative: scale_12 * 5^shape_12
        #   - h21 not active (we're in state 1)
        # 
        # Interval [5, 10] in state 2:
        #   - SOJOURN RESETS! sojourn = 0 to 5
        #   - Duration in state 2: 5 time units
        #   - h21 survival: exp(-scale_21 * 5^shape_21)
        #   - No transition (right-censored)
        
        # Part 1: [0,5] in state 1, transition to 2
        h_val = scale_12 * shape_12 * 5^(shape_12-1)
        H_val = scale_12 * 5^shape_12
        ll_part1 = log(h_val) - H_val
        
        # Part 2: [5,10] in state 2, NO transition (sojourn = 5)
        ll_part2 = -scale_21 * 5^shape_21  # Just survival, no density
        
        ll_manual = ll_part1 + ll_part2
        
        # Package computation
        paths = MultistateModels.extract_paths(model)
        exact_data = MultistateModels.ExactData(model, paths)
        pars = model.parameters.flat
        ll_package = MultistateModels.loglik_exact(pars, exact_data; neg=false)
        
        @test ll_package ≈ ll_manual rtol=1e-6
    end
    
    @testset "Reversible model with TVC and multiple sojourns" begin
        # Path: 1→2→1→2 with covariate change
        # Tests that TVC is handled correctly with sojourn resets
        
        h12 = Hazard(@formula(0 ~ 1 + x), "wei", 1, 2)
        h21 = Hazard(@formula(0 ~ 1 + x), "wei", 2, 1)
        
        # Parameters
        shape_12, scale_12, beta_12 = 1.3, 0.2, 0.5
        shape_21, scale_21, beta_21 = 1.1, 0.15, -0.3
        
        # Covariate x = 1 throughout (no TVC change, simplifies manual calc)
        dat = DataFrame(
            id = fill(1, 4),
            tstart = [0.0, 2.0, 6.0, 9.0],
            tstop = [2.0, 6.0, 9.0, 12.0],
            statefrom = [1, 2, 1, 2],
            stateto = [2, 1, 2, 2],
            obstype = fill(1, 4),
            x = fill(1.0, 4)
        )
        
        model = multistatemodel(h12, h21; data=dat)
        set_parameters!(model, (
            h12 = [shape_12, scale_12, beta_12],
            h21 = [shape_21, scale_21, beta_21]
        ))
        
        # Manual calculation:
        # Linear predictor for h12: log(scale_12) + beta_12 * x = log(scale_12) + beta_12
        # Effective scale_12: exp(log(scale_12) + beta_12) = scale_12 * exp(beta_12)
        eff_scale_12 = scale_12 * exp(beta_12)
        eff_scale_21 = scale_21 * exp(beta_21)
        
        # Interval 1: [0, 2], state 1, transition 1→2, sojourn = 2
        t1 = 2.0
        h1 = eff_scale_12 * shape_12 * t1^(shape_12-1)
        H1 = eff_scale_12 * t1^shape_12
        ll_1 = log(h1) - H1
        
        # Interval 2: [2, 6], state 2, transition 2→1, sojourn = 4
        t2 = 4.0  # sojourn time
        h2 = eff_scale_21 * shape_21 * t2^(shape_21-1)
        H2 = eff_scale_21 * t2^shape_21
        ll_2 = log(h2) - H2
        
        # Interval 3: [6, 9], state 1, transition 1→2, sojourn = 3
        t3 = 3.0
        h3 = eff_scale_12 * shape_12 * t3^(shape_12-1)
        H3 = eff_scale_12 * t3^shape_12
        ll_3 = log(h3) - H3
        
        # Interval 4: [9, 12], state 2, no transition, sojourn = 3
        t4 = 3.0
        H4 = eff_scale_21 * t4^shape_21
        ll_4 = -H4  # survival only
        
        ll_manual = ll_1 + ll_2 + ll_3 + ll_4
        
        # Package computation
        paths = MultistateModels.extract_paths(model)
        exact_data = MultistateModels.ExactData(model, paths)
        pars = model.parameters.flat
        ll_package = MultistateModels.loglik_exact(pars, exact_data; neg=false)
        
        @test ll_package ≈ ll_manual rtol=1e-6
    end
    
end

@testset "AFT + TVC Machine-Precision Validation" begin
    # This testset verifies AFT (accelerated failure time) models with TVC
    # compute cumulative hazard correctly using effective time τ(t) = ∫ exp(-βx(s)) ds
    
    @testset "Weibull AFT with TVC and sojourn reset - comprehensive" begin
        # This is the most complex case: AFT + TVC + multiple sojourns
        # Tests that effective time τ resets when entering a new state
        # 
        # Path: 1 (t=0-2, x=0.5) → 2 (t=2-4, x=1.0) → 2 (t=4-6, x=0.0)
        #       transition at t=2, right-censored at t=6
        #
        # For AFT in a semi-Markov model:
        # - Each sojourn starts with τ = 0
        # - τ accumulates as τ(s) = ∫₀ˢ exp(-β·x(u)) du where s is sojourn time
        # - Cumulative hazard H(s) = H₀(τ(s))
        # - Instantaneous hazard h(s|x) = h₀(τ) · exp(-β·x(s))
        
        shape_12 = 1.5
        scale_12 = 0.8
        beta_12 = 0.5
        
        shape_21 = 1.2  # for completeness, though no 2→1 transition
        scale_21 = 0.3
        beta_21 = -0.2
        
        h12 = Hazard(@formula(0 ~ x), "wei", 1, 2; linpred_effect=:aft)
        h21 = Hazard(@formula(0 ~ x), "wei", 2, 1; linpred_effect=:aft)
        
        dat = DataFrame(
            id = [1, 1, 1],
            tstart = [0.0, 2.0, 4.0],
            tstop = [2.0, 4.0, 6.0],
            statefrom = [1, 2, 2],
            stateto = [2, 2, 2],  # transition at t=2, then right-censored
            obstype = [1, 1, 1],
            x = [0.5, 1.0, 0.0]
        )
        
        model = multistatemodel(h12, h21; data=dat)
        set_parameters!(model, (
            h12 = [shape_12, scale_12, beta_12],
            h21 = [shape_21, scale_21, beta_21]
        ))
        
        # === Manual calculation ===
        
        # Sojourn 1: State 1, interval [0, 2), x=0.5, transition to state 2
        # Effective time: τ₁ = 2 * exp(-0.5 * 0.5) = 2 * exp(-0.25)
        sojourn_1_duration = 2.0
        x_sojourn_1 = 0.5
        tau_sojourn_1 = sojourn_1_duration * exp(-beta_12 * x_sojourn_1)
        
        # Weibull AFT cumulative hazard: H₀(τ) = scale * τ^shape
        H_12_sojourn_1 = scale_12 * tau_sojourn_1^shape_12
        
        # Instantaneous hazard at transition: h₀(τ) * exp(-β*x)
        h_12_at_transition = shape_12 * scale_12 * tau_sojourn_1^(shape_12 - 1) * exp(-beta_12 * x_sojourn_1)
        
        # Log-likelihood contribution from sojourn 1: log(h) - H
        ll_sojourn_1 = log(h_12_at_transition) - H_12_sojourn_1
        
        # Sojourn 2: State 2, interval [2, 6), NO transition (right-censored)
        # IMPORTANT: τ RESETS to 0 when we enter state 2!
        # Two intervals within this sojourn: [2,4) with x=1.0, [4,6) with x=0.0
        
        # Effective time for h21 (2→1 hazard):
        delta_tau_21_int1 = 2.0 * exp(-beta_21 * 1.0)  # interval [2,4), duration=2, x=1.0
        delta_tau_21_int2 = 2.0 * exp(-beta_21 * 0.0)  # interval [4,6), duration=2, x=0.0
        tau_sojourn_2 = delta_tau_21_int1 + delta_tau_21_int2
        
        # Cumulative hazard (survival term only - no transition)
        H_21_sojourn_2 = scale_21 * tau_sojourn_2^shape_21
        
        # Log-likelihood contribution from sojourn 2: just -H (survival)
        ll_sojourn_2 = -H_21_sojourn_2
        
        ll_manual = ll_sojourn_1 + ll_sojourn_2
        
        # === Package calculation ===
        paths = MultistateModels.extract_paths(model)
        exact_data = MultistateModels.ExactData(model, paths)
        ll_package = MultistateModels.loglik_exact(model.parameters.flat, exact_data; neg=false)
        
        # Verify intermediate values for debugging
        @test tau_sojourn_1 ≈ 2.0 * exp(-0.25) rtol=1e-10
        @test tau_sojourn_2 ≈ 2.0 * exp(0.2) + 2.0 * 1.0 rtol=1e-10
        
        @test ll_package ≈ ll_manual rtol=1e-6
    end
    
    @testset "Weibull AFT with TVC - likelihood" begin
        # Test parameters (natural scale)
        shape = 1.5
        scale = 0.8
        beta = 0.5
        
        # TVC: x changes at t=1 and t=2
        # x = 0.0 for [0,1), x = 1.0 for [1,2), x = 0.5 for [2,∞)
        h12 = Hazard(@formula(0 ~ x), "wei", 1, 2; linpred_effect=:aft)
        
        dat = DataFrame(
            id = [1, 1, 1],
            tstart = [0.0, 1.0, 2.0],
            tstop = [1.0, 2.0, 3.0],
            statefrom = [1, 1, 1],
            stateto = [1, 1, 2],  # transition at t=3
            obstype = [1, 1, 1],
            x = [0.0, 1.0, 0.5]
        )
        
        model = multistatemodel(h12; data=dat)
        # Parameters stored as: [shape, scale, beta] (natural scale)
        set_parameters!(model, (h12 = [shape, scale, beta],))
        
        # Manual calculation of effective time τ
        # τ = 1.0 * exp(-0.5 * 0.0) + 1.0 * exp(-0.5 * 1.0) + 1.0 * exp(-0.5 * 0.5)
        tau_manual = 1.0 * exp(-beta * 0.0) + 1.0 * exp(-beta * 1.0) + 1.0 * exp(-beta * 0.5)
        
        # Weibull AFT: H(t) = H₀(τ) = scale * τ^shape
        H_manual = scale * tau_manual^shape
        
        # For obstype=1 at transition (state 1→2), need instantaneous hazard:
        # h(t) = h₀(τ) * exp(-β * x(t)) = shape * scale * τ^(shape-1) * exp(-β * x_final)
        x_final = 0.5
        h_manual = shape * scale * tau_manual^(shape - 1) * exp(-beta * x_final)
        
        # Log-likelihood = log(h) - H
        ll_manual = log(h_manual) - H_manual
        
        # Package computation
        paths = MultistateModels.extract_paths(model)
        exact_data = MultistateModels.ExactData(model, paths)
        ll_package = MultistateModels.loglik_exact(model.parameters.flat, exact_data; neg=false)
        
        @test ll_package ≈ ll_manual rtol=1e-6
        @test tau_manual ≈ 2.385331442784038 rtol=1e-10  # regression test
    end
    
    @testset "Gompertz AFT with TVC - likelihood" begin
        # Test parameters (natural scale)
        shape = 0.3
        rate = 0.2
        beta = 0.5
        
        h12 = Hazard(@formula(0 ~ x), "gom", 1, 2; linpred_effect=:aft)
        
        dat = DataFrame(
            id = [1, 1, 1],
            tstart = [0.0, 1.0, 2.0],
            tstop = [1.0, 2.0, 3.0],
            statefrom = [1, 1, 1],
            stateto = [1, 1, 2],
            obstype = [1, 1, 1],
            x = [0.0, 1.0, 0.5]
        )
        
        model = multistatemodel(h12; data=dat)
        # Gompertz: [shape, rate, beta] - shape on identity scale, rate on natural scale
        set_parameters!(model, (h12 = [shape, rate, beta],))
        
        # Manual effective time
        tau_manual = 1.0 * exp(-beta * 0.0) + 1.0 * exp(-beta * 1.0) + 1.0 * exp(-beta * 0.5)
        
        # Gompertz AFT: H(τ) = (rate/shape) * (exp(shape * τ) - 1)
        H_manual = (rate / shape) * (exp(shape * tau_manual) - 1.0)
        
        # Instantaneous hazard: h(τ) * exp(-β * x(t)) = rate * exp(shape * τ) * exp(-β * x_final)
        x_final = 0.5
        h_manual = rate * exp(shape * tau_manual) * exp(-beta * x_final)
        
        ll_manual = log(h_manual) - H_manual
        
        paths = MultistateModels.extract_paths(model)
        exact_data = MultistateModels.ExactData(model, paths)
        ll_package = MultistateModels.loglik_exact(model.parameters.flat, exact_data; neg=false)
        
        @test ll_package ≈ ll_manual rtol=1e-6
        @test H_manual ≈ 0.6969416014725967 rtol=1e-10  # regression test
    end
    
    @testset "Gompertz AFT with TVC and sojourn reset" begin
        # Test Gompertz AFT with multiple sojourns to verify effective time resets
        # Path: 1 (t=0-3, x=0.3) → 2 (t=3-5, x=0.8) → 1 (t=5-8, x=0.1)
        
        shape_12 = 0.2  # Gompertz shape (a)
        rate_12 = 0.15  # Gompertz rate (b)
        beta_12 = 0.4
        
        shape_21 = 0.3
        rate_21 = 0.2
        beta_21 = -0.3
        
        h12 = Hazard(@formula(0 ~ x), "gom", 1, 2; linpred_effect=:aft)
        h21 = Hazard(@formula(0 ~ x), "gom", 2, 1; linpred_effect=:aft)
        
        dat = DataFrame(
            id = [1, 1, 1],
            tstart = [0.0, 3.0, 5.0],
            tstop = [3.0, 5.0, 8.0],
            statefrom = [1, 2, 1],
            stateto = [2, 1, 1],  # 1→2 at t=3, 2→1 at t=5, then censored
            obstype = [1, 1, 1],
            x = [0.3, 0.8, 0.1]
        )
        
        model = multistatemodel(h12, h21; data=dat)
        set_parameters!(model, (
            h12 = [shape_12, rate_12, beta_12],
            h21 = [shape_21, rate_21, beta_21]
        ))
        
        # === Manual calculation ===
        
        # Sojourn 1: State 1, interval [0, 3), x=0.3, transition 1→2
        tau_1 = 3.0 * exp(-beta_12 * 0.3)
        
        # Gompertz: H₀(τ) = (rate/shape) * (exp(shape*τ) - 1)
        H_12 = (rate_12 / shape_12) * (exp(shape_12 * tau_1) - 1.0)
        
        # h₀(τ) * exp(-β*x) = rate * exp(shape*τ) * exp(-β*x)
        h_12 = rate_12 * exp(shape_12 * tau_1) * exp(-beta_12 * 0.3)
        
        ll_1 = log(h_12) - H_12
        
        # Sojourn 2: State 2, interval [3, 5), x=0.8, transition 2→1
        # τ RESETS to 0
        tau_2 = 2.0 * exp(-beta_21 * 0.8)
        
        H_21 = (rate_21 / shape_21) * (exp(shape_21 * tau_2) - 1.0)
        h_21 = rate_21 * exp(shape_21 * tau_2) * exp(-beta_21 * 0.8)
        
        ll_2 = log(h_21) - H_21
        
        # Sojourn 3: State 1, interval [5, 8), x=0.1, NO transition (censored)
        # τ RESETS to 0 again
        tau_3 = 3.0 * exp(-beta_12 * 0.1)
        
        H_12_final = (rate_12 / shape_12) * (exp(shape_12 * tau_3) - 1.0)
        
        ll_3 = -H_12_final  # survival only
        
        ll_manual = ll_1 + ll_2 + ll_3
        
        # === Package calculation ===
        paths = MultistateModels.extract_paths(model)
        exact_data = MultistateModels.ExactData(model, paths)
        ll_package = MultistateModels.loglik_exact(model.parameters.flat, exact_data; neg=false)
        
        @test ll_package ≈ ll_manual rtol=1e-6
    end
    
    @testset "Exponential AFT with TVC - equivalence to PH" begin
        # For exponential, AFT with β is equivalent to PH with -β
        rate = 0.5
        beta = 0.3
        
        h12_aft = Hazard(@formula(0 ~ x), "exp", 1, 2; linpred_effect=:aft)
        h12_ph = Hazard(@formula(0 ~ x), "exp", 1, 2; linpred_effect=:ph)
        
        dat = DataFrame(
            id = [1, 1],
            tstart = [0.0, 1.5],
            tstop = [1.5, 3.0],
            statefrom = [1, 1],
            stateto = [1, 2],
            obstype = [1, 1],
            x = [0.0, 1.0]
        )
        
        model_aft = multistatemodel(h12_aft; data=dat)
        model_ph = multistatemodel(h12_ph; data=dat)
        
        set_parameters!(model_aft, (h12 = [rate, beta],))
        set_parameters!(model_ph, (h12 = [rate, -beta],))  # negated beta for PH
        
        paths_aft = MultistateModels.extract_paths(model_aft)
        paths_ph = MultistateModels.extract_paths(model_ph)
        exact_data_aft = MultistateModels.ExactData(model_aft, paths_aft)
        exact_data_ph = MultistateModels.ExactData(model_ph, paths_ph)
        
        ll_aft = MultistateModels.loglik_exact(model_aft.parameters.flat, exact_data_aft; neg=false)
        ll_ph = MultistateModels.loglik_exact(model_ph.parameters.flat, exact_data_ph; neg=false)
        
        @test ll_aft ≈ ll_ph rtol=1e-10
    end
end
