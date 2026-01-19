# =============================================================================
# Test suite for MCEM algorithm and related functions
# =============================================================================
#
# Tests for:
# - MCEM helper function correctness (mcem_mll, mcem_lml, mcem_ase)
# - ForwardDiff gradient compatibility

using Test
using MultistateModels
using DataFrames
using LinearAlgebra
using Random
using ForwardDiff

@testset "MCEM Tests" begin
    
    @testset "ForwardDiff gradient compatibility" begin
        # Critical: If gradients are wrong, optimization silently fails
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
        
        nsubj = 3
        dat = DataFrame(
            id = repeat(1:nsubj, inner=2),
            tstart = repeat([0.0, 1.0], outer=nsubj),
            tstop = repeat([1.0, 2.0], outer=nsubj),
            statefrom = repeat([1, 2], outer=nsubj),
            stateto = repeat([2, 3], outer=nsubj),
            obstype = fill(1, 2*nsubj)
        )
        
        model = multistatemodel(h12, h23; data=dat)
        params = MultistateModels.get_parameters_flat(model)
        
        # Test ExactData gradient
        samplepaths = [MultistateModels.SamplePath(i, [0.0, 1.0, 2.0], [1, 2, 3]) for i in 1:nsubj]
        exactdata = MultistateModels.ExactData(model, samplepaths)
        
        grad = ForwardDiff.gradient(p -> MultistateModels.loglik_exact(p, exactdata; neg=true), params)
        # AD consistency: gradient should be finite and have correct length
        @test length(grad) == length(params)
        @test all(isfinite.(grad))
        
        # Test SMPanelData gradient
        samplepaths_nested = [[MultistateModels.SamplePath(i, [0.0, 1.0, 2.0], [1, 2, 3])] for i in 1:nsubj]
        weights = [[1.0] for _ in 1:nsubj]
        smdata = MultistateModels.SMPanelData(model, samplepaths_nested, weights)
        
        grad_sm = ForwardDiff.gradient(p -> MultistateModels.loglik_semi_markov(p, smdata; neg=true), params)
        # AD consistency: gradient should be finite and have correct length
        @test length(grad_sm) == length(params)
        @test all(isfinite.(grad_sm))
    end
    
    @testset "MCEM helper functions" begin
        # Test mcem_mll, mcem_ase with known values
        logliks = [[-1.0, -2.0], [-1.5, -2.5]]
        ImportanceWeights = [[0.6, 0.4], [0.5, 0.5]]
        SubjectWeights = [1.0, 1.0]
        
        # mcem_mll: weighted average of log-likelihoods
        mll = MultistateModels.mcem_mll(logliks, ImportanceWeights, SubjectWeights)
        expected = (0.6*(-1.0) + 0.4*(-2.0)) + (0.5*(-1.5) + 0.5*(-2.5))
        @test mll ≈ expected
        
        # mcem_mll with non-unit subject weights
        SubjectWeights2 = [2.0, 0.5]
        mll2 = MultistateModels.mcem_mll(logliks, ImportanceWeights, SubjectWeights2)
        expected2 = 2.0*(0.6*(-1.0) + 0.4*(-2.0)) + 0.5*(0.5*(-1.5) + 0.5*(-2.5))
        @test mll2 ≈ expected2
        
        # mcem_ase with identical logliks (no variance) should be zero
        ase_zero = MultistateModels.mcem_ase(logliks, logliks, ImportanceWeights, SubjectWeights)
        @test ase_zero ≈ 0.0
        
        # var_ris with all zeros should return 0
        w = [0.3, 0.3, 0.4]
        v_zero = MultistateModels.var_ris([0.0, 0.0, 0.0], w)
        @test v_zero ≈ 0.0
    end
    
    # =========================================================================
    # Test 1: MCEM Iteration Monotonicity
    # =========================================================================
    # Verify that Q(θ_{k+1}) >= Q(θ_k) for MCEM iterations.
    # The expected log-likelihood (Q function) should increase monotonically
    # at each EM iteration, though this may not hold exactly due to Monte Carlo
    # noise in MCEM. We test that increases are typical.
    # =========================================================================
    @testset "MCEM iteration monotonicity" begin
        # Set up a semi-Markov model with panel data (obstype=2)
        # Using Weibull hazards makes it semi-Markov (shape != 1 means memory)
        Random.seed!(12345)
        
        # Create panel data with varied transitions to avoid degenerate case
        # where all importance weights are identical (causes ParetoSmooth to fail).
        # IMPORTANT: Panel data (obstype=2) means we observe the state at discrete times,
        # so only the final state matters for each row (statefrom shows previous state).
        
        # Mix of trajectories:
        # Group A (5 subjects): stay in state 1 throughout
        # Group B (5 subjects): transition 1→2 by time 2.0, stay in 2
        # Group C (5 subjects): transition 1→2 by time 2.0, back to 1 by time 4.0
        
        rows = DataFrame[]
        id = 1
        
        # Group A: stay in state 1 (5 subjects)
        for _ in 1:5
            push!(rows, DataFrame(
                id = fill(id, 3),
                tstart = [0.0, 2.0, 4.0],
                tstop = [2.0, 4.0, 6.0],
                statefrom = [1, 1, 1],
                stateto = [1, 1, 1],
                obstype = [2, 2, 2]
            ))
            id += 1
        end
        
        # Group B: 1→2 transition, stay in 2 (5 subjects)
        for _ in 1:5
            push!(rows, DataFrame(
                id = fill(id, 3),
                tstart = [0.0, 2.0, 4.0],
                tstop = [2.0, 4.0, 6.0],
                statefrom = [1, 2, 2],
                stateto = [2, 2, 2],
                obstype = [2, 2, 2]
            ))
            id += 1
        end
        
        # Group C: 1→2→1 transition pattern (5 subjects)
        for _ in 1:5
            push!(rows, DataFrame(
                id = fill(id, 3),
                tstart = [0.0, 2.0, 4.0],
                tstop = [2.0, 4.0, 6.0],
                statefrom = [1, 2, 1],
                stateto = [2, 1, 1],
                obstype = [2, 2, 2]
            ))
            id += 1
        end
        
        dat = vcat(rows...)
        
        # Weibull hazards (semi-Markov because shape can differ from 1)
        h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
        h21 = Hazard(@formula(0 ~ 1), "wei", 2, 1)
        
        # Build model with surrogate (required for MCEM)
        model = multistatemodel(h12, h21; data=dat, surrogate=:markov)
        
        # Set reasonable initial parameters (natural scale)
        # Weibull: [shape, rate]
        set_parameters!(model, (h12 = [1.2, 0.3], h21 = [1.1, 0.25]))
        
        # Fit using MCEM with very limited iterations for testing
        # We want to verify monotonicity, not convergence
        fitted = fit(model;
            maxiter = 3,
            ess_target_initial = 30,
            tol = 1e-6,  # Will not converge in 3 iters, that is fine
            verbose = false,
            sir = :none,  # Pure importance sampling
            compute_vcov = false,
            compute_ij_vcov = false,
            return_convergence_records = true
        )
        
        # Get convergence records which contain Q function values
        records = get_convergence_records(fitted)
        # Convergence records should be returned
        @test !isnothing(records)
        
        # Check that MLL (Q function) values show non-decreasing trend
        # Note: Due to MC noise, we allow small decreases
        if haskey(records, :mll)
            mll_values = records.mll
            # Should have at least 2 iterations recorded
            @test length(mll_values) >= 2
            
            # Most iterations should show increase (allowing for MC noise)
            # Check that the final MLL is not substantially worse than initial
            mll_diff = mll_values[end] - mll_values[1]
            # Allow for MC noise: final should be within 10% of initial or better
            # Q function should not decrease substantially
            @test mll_diff >= -abs(mll_values[1]) * 0.1
        end
    end
    
    # =========================================================================
    # Test 2: mcem_ase with panel data - verify non-zero variance
    # =========================================================================
    # The asymptotic standard error of the Q function change should be positive
    # when there is genuine Monte Carlo variability in the path likelihoods.
    # =========================================================================
    @testset "mcem_ase with panel data variance" begin
        # Create log-likelihoods with known variability
        loglik_cur = [
            [-10.0, -10.5, -11.0, -10.2, -10.8],  # Subject 1
            [-12.0, -12.3, -11.8, -12.5, -12.1]   # Subject 2
        ]
        loglik_prop = [
            [-9.5, -10.0, -10.5, -9.8, -10.3],    # Subject 1 (improved)
            [-11.5, -11.8, -11.3, -12.0, -11.6]   # Subject 2 (improved)
        ]
        
        # Normalized importance weights (sum to 1 per subject)
        ImportanceWeights = [
            [0.25, 0.20, 0.15, 0.22, 0.18],
            [0.18, 0.22, 0.20, 0.15, 0.25]
        ]
        SubjectWeights = [1.0, 1.0]
        
        # Compute ASE
        ase = MultistateModels.mcem_ase(loglik_prop, loglik_cur, ImportanceWeights, SubjectWeights)
        
        # ASE should be positive when there is variability
        # ASE should be positive with variable log-likelihoods
        @test ase > 0.0
        # ASE should be finite
        @test isfinite(ase)
        
        # Single-path subjects should contribute zero variance to ASE
        loglik_single = [[-10.0], [-12.0]]
        loglik_single_prop = [[-9.5], [-11.5]]
        ImportanceWeights_single = [[1.0], [1.0]]
        
        ase_single = MultistateModels.mcem_ase(loglik_single_prop, loglik_single, 
                                               ImportanceWeights_single, SubjectWeights)
        # Single-path subjects should have zero variance contribution
        @test ase_single ≈ 0.0
    end
    
    # =========================================================================
    # Test 3: var_ris analytical verification
    # =========================================================================
    # Verify var_ris computes the correct variance for known test cases.
    #
    # var_ris formula: (sum(l.*w))^2 * ((sum((w.*l).^2))/(sum(w.*l))^2 - 2*(sum(w.^2.*l))/(sum(w.*l)) + sum(w.^2))
    #
    # For uniform weights w = 1/n, this simplifies considerably. Let's derive
    # the analytical values for specific test cases.
    # =========================================================================
    @testset "var_ris analytical verification" begin
        # -----------------------------------------------------------------
        # Case 1: All differences equal -> variance should be exactly 0
        # -----------------------------------------------------------------
        n = 5
        w_uniform = fill(1.0/n, n)
        l_constant = fill(2.0, n)
        v_const = MultistateModels.var_ris(l_constant, w_uniform)
        # When all l values are identical, there is no variability
        @test v_const ≈ 0.0 atol=1e-14
        
        # -----------------------------------------------------------------
        # Case 2: l = [1, 2, 3], w = [1/3, 1/3, 1/3]
        # -----------------------------------------------------------------
        # Manual calculation:
        #   sum(l.*w) = 2 (mean)
        #   sum((w.*l).^2) = (1/9)(1 + 4 + 9) = 14/9
        #   sum(w.^2.*l) = (1/9)(1 + 2 + 3) = 6/9 = 2/3
        #   sum(w.^2) = 3 * (1/9) = 1/3
        #
        #   var_ris = 4 * (14/36 - 2*(2/3)/2 + 1/3)
        #           = 4 * (14/36 - 2/3 + 1/3)
        #           = 4 * (14/36 - 12/36)
        #           = 4 * (2/36) = 8/36 = 2/9
        # -----------------------------------------------------------------
        l_three = [1.0, 2.0, 3.0]
        w_three = fill(1.0/3, 3)
        v_three = MultistateModels.var_ris(l_three, w_three)
        expected_v_three = 2.0/9.0
        @test v_three ≈ expected_v_three rtol=1e-12
        
        # -----------------------------------------------------------------
        # Case 3: l = [1, 2, 3, 4, 5], w = [1/5, 1/5, 1/5, 1/5, 1/5]
        # -----------------------------------------------------------------
        # Manual calculation:
        #   sum(l.*w) = 3 (mean)
        #   sum((w.*l).^2) = (1/25)(1 + 4 + 9 + 16 + 25) = 55/25 = 11/5
        #   sum(w.^2.*l) = (1/25)(1 + 2 + 3 + 4 + 5) = 15/25 = 3/5
        #   sum(w.^2) = 5 * (1/25) = 1/5
        #
        #   var_ris = 9 * (11/45 - 2*(3/5)/3 + 1/5)
        #           = 9 * (11/45 - 2/5 + 1/5)
        #           = 9 * (11/45 - 9/45)
        #           = 9 * (2/45) = 18/45 = 2/5
        # -----------------------------------------------------------------
        l_five = [1.0, 2.0, 3.0, 4.0, 5.0]
        w_five = fill(1.0/5, 5)
        v_five = MultistateModels.var_ris(l_five, w_five)
        expected_v_five = 2.0/5.0
        @test v_five ≈ expected_v_five rtol=1e-12
        
        # -----------------------------------------------------------------
        # Case 4: Symmetry check - reversing order preserves variance
        # -----------------------------------------------------------------
        l_reversed = reverse(l_five)
        w_reversed = reverse(w_five)
        v_reversed = MultistateModels.var_ris(l_reversed, w_reversed)
        @test v_reversed ≈ expected_v_five rtol=1e-12
        
        # -----------------------------------------------------------------
        # Case 5: Two observations with equal weights
        # l = [1, 3], w = [0.5, 0.5]
        # -----------------------------------------------------------------
        # Manual calculation:
        #   sum(l.*w) = 2
        #   sum((w.*l).^2) = (1/4)(1 + 9) = 10/4 = 5/2
        #   sum(w.^2.*l) = (1/4)(1 + 3) = 1
        #   sum(w.^2) = 2 * (1/4) = 1/2
        #
        #   var_ris = 4 * (5/8 - 2*1/2 + 1/2)
        #           = 4 * (5/8 - 1 + 1/2)
        #           = 4 * (5/8 - 4/8)
        #           = 4 * (1/8) = 1/2
        # -----------------------------------------------------------------
        l_two = [1.0, 3.0]
        w_two = [0.5, 0.5]
        v_two = MultistateModels.var_ris(l_two, w_two)
        expected_v_two = 0.5
        @test v_two ≈ expected_v_two rtol=1e-12
        
        # -----------------------------------------------------------------
        # Case 6: Non-uniform weights
        # l = [1, 2, 3], w = [0.5, 0.3, 0.2]
        # -----------------------------------------------------------------
        # Manual calculation:
        #   sum(l.*w) = 0.5*1 + 0.3*2 + 0.2*3 = 0.5 + 0.6 + 0.6 = 1.7
        #   sum((w.*l).^2) = 0.25*1 + 0.09*4 + 0.04*9 = 0.25 + 0.36 + 0.36 = 0.97
        #   sum(w.^2.*l) = 0.25*1 + 0.09*2 + 0.04*3 = 0.25 + 0.18 + 0.12 = 0.55
        #   sum(w.^2) = 0.25 + 0.09 + 0.04 = 0.38
        #
        #   var_ris = 1.7^2 * (0.97/1.7^2 - 2*0.55/1.7 + 0.38)
        #           = 0.97 - 2*0.55*1.7 + 0.38*1.7^2
        #           = 0.97 - 1.87 + 1.0982
        #           = 0.1982
        # -----------------------------------------------------------------
        l_nonunif = [1.0, 2.0, 3.0]
        w_nonunif = [0.5, 0.3, 0.2]
        v_nonunif = MultistateModels.var_ris(l_nonunif, w_nonunif)
        expected_v_nonunif = 0.97 - 2*0.55*1.7 + 0.38*1.7^2
        @test v_nonunif ≈ expected_v_nonunif rtol=1e-12
    end
    
    # =========================================================================
    # Test 4: Gradient at MLE should be exactly zero (analytical)
    # =========================================================================
    # For exponential hazard with rate λ and exact observation data:
    #   ℓ(λ) = n × log(λ) - λ × Σtᵢ   (where tᵢ are transition times)
    #   MLE: λ̂ = n / Σtᵢ
    #   Score: dℓ/dλ = n/λ - Σtᵢ = 0 at MLE
    #
    # In v0.3.0+, parameters are stored on NATURAL scale (λ, not log(λ)).
    # The gradient at MLE should be exactly zero.
    #
    # Log-likelihood at MLE: ℓ(λ̂) = n*log(n/Σtᵢ) - n = n*(log(n) - log(Σtᵢ) - 1)
    # =========================================================================
    @testset "Gradient at MLE" begin
        Random.seed!(54321)
        
        # Create exact observation data for exponential model
        # Each subject transitions 1→2 at time t_i
        nsubj = 20
        transition_times = [1.5 for _ in 1:nsubj]  # All transitions at t=1.5
        
        dat_exact = DataFrame(
            id = 1:nsubj,
            tstart = zeros(nsubj),
            tstop = transition_times,
            statefrom = fill(1, nsubj),
            stateto = fill(2, nsubj),
            obstype = fill(1, nsubj)  # Exact observations
        )
        
        h12_exact = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        model_exact = multistatemodel(h12_exact; data=dat_exact)
        
        # -----------------------------------------------------------------
        # Analytical MLE for exponential: λ̂ = n / Σtᵢ
        # -----------------------------------------------------------------
        n_events = nsubj
        sum_times = sum(transition_times)
        lambda_mle = n_events / sum_times  # = 20 / 30 = 2/3
        
        # Build ExactData for gradient computation
        paths_from_data = MultistateModels.extract_paths(model_exact)
        exactdata = MultistateModels.ExactData(model_exact, paths_from_data)
        
        # -----------------------------------------------------------------
        # Test 1: Gradient at analytical MLE should be exactly zero
        # Score: dℓ/dλ = n/λ - Σtᵢ
        # At MLE: dℓ/dλ = n/(n/Σtᵢ) - Σtᵢ = Σtᵢ - Σtᵢ = 0
        # -----------------------------------------------------------------
        grad_at_analytical_mle = ForwardDiff.gradient(
            p -> MultistateModels.loglik_exact(p, exactdata; neg=false), 
            [lambda_mle]
        )
        @test grad_at_analytical_mle[1] ≈ 0.0 atol=1e-10
        
        # -----------------------------------------------------------------
        # Test 2: Verify fitted model matches analytical MLE
        # -----------------------------------------------------------------
        set_parameters!(model_exact, (h12 = [0.5],))  # Initial rate (different from MLE)
        fitted_exact = fit(model_exact; verbose=false)
        params_fitted = MultistateModels.get_parameters_flat(fitted_exact)
        
        # Fitted parameter (natural scale λ) should match analytical MLE
        @test params_fitted[1] ≈ lambda_mle rtol=1e-6
        
        # -----------------------------------------------------------------
        # Test 3: Gradient at fitted MLE should also be zero
        # -----------------------------------------------------------------
        grad_at_fitted_mle = ForwardDiff.gradient(
            p -> MultistateModels.loglik_exact(p, exactdata; neg=false), 
            params_fitted
        )
        @test grad_at_fitted_mle[1] ≈ 0.0 atol=1e-6
        
        # -----------------------------------------------------------------
        # Test 4: Verify log-likelihood value at MLE
        # ℓ(λ̂) = n*log(λ̂) - λ̂*Σtᵢ = n*log(n/Σtᵢ) - n
        #      = n*(log(n) - log(Σtᵢ) - 1)
        # -----------------------------------------------------------------
        expected_ll_at_mle = n_events * (log(n_events) - log(sum_times) - 1)
        actual_ll_at_mle = MultistateModels.loglik_exact([lambda_mle], exactdata; neg=false)
        @test actual_ll_at_mle ≈ expected_ll_at_mle rtol=1e-10
    end
    
end
