using Test
using MultistateModels
using DataFrames
using LinearAlgebra
using Random

# =============================================================================
# MATHEMATICAL BACKGROUND: Emission Matrix Expansion for Phase-Type Models
# =============================================================================
#
# In a Hidden Markov Model (HMM), the emission matrix encodes:
#   e[i, s] = P(Y_i | X_i = s)
#
# where Y_i is the observation and X_i is the latent state.
#
# The forward algorithm computes α_t(s) = P(Y_{1:t}, X_t = s) via:
#   α_t(s) = P(Y_t | X_t=s) × Σ_r P(X_t=s | X_{t-1}=r) × α_{t-1}(r)
#
# For phase-type expansion:
# - User specifies e[i, k] = P(Y_i | State k) for observed states k = 1,...,K
# - We expand to e_exp[i, p] = P(Y_i | Phase p) for phases p = 1,...,P
#
# KEY INSIGHT: If phase p belongs to observed state k, then:
#   P(Y_i | Phase p) = P(Y_i | State k)
#
# because the observation "I'm in state k" is equally likely regardless of
# which internal phase of state k the process is in. The phases are latent
# and have no observable signature.
#
# CONSEQUENCE: The expanded emission matrix does NOT preserve row sums.
# If State 1 has 2 phases and e[i, 1] = 0.7, e[i, 2] = 0.3, then:
#   e_exp[i, :] = [0.7, 0.7, 0.3]  (sum = 1.7, not 1.0)
#
# This is mathematically correct because e is P(Y|X), not a distribution over X.
# The forward algorithm marginalizes correctly over phases.
# =============================================================================

@testset "Phase-type Emission Matrix Expansion" begin

    @testset "Matrix dimensions and values" begin
        # Setup: 2-state model where state 1 has 2 phases, state 2 has 1 phase
        h12 = Hazard(@formula(0 ~ 1), "pt", 1, 2)
        
        dat = DataFrame(
            id = [1, 1, 2],
            tstart = [0.0, 1.0, 0.0],
            tstop = [1.0, 2.0, 1.0],
            statefrom = [1, 2, 1],
            stateto = [2, 2, 1],
            obstype = [1, 2, 2]
        )
        
        # User-supplied emission matrix (3 obs x 2 states)
        EmissionMatrix = [
            1.0 0.0;  # Obs 1: P(Y|state1)=1, P(Y|state2)=0
            0.0 1.0;  # Obs 2: P(Y|state1)=0, P(Y|state2)=1
            0.5 0.5   # Obs 3: soft evidence
        ]
        
        model = multistatemodel(h12; data=dat, EmissionMatrix=EmissionMatrix, n_phases=Dict(1=>2))
        
        # Expanded: 4 rows, 3 states (phase 1, phase 2, state 2)
        @test size(model.emat) == (4, 3)
        
        # Row 1 & 2: from Obs 1, P(Y|state1)=1 -> P(Y|phase1)=P(Y|phase2)=1
        @test model.emat[1, 1] == 1.0
        @test model.emat[1, 2] == 1.0
        @test model.emat[1, 3] == 0.0
        
        @test model.emat[2, 1] == 1.0
        @test model.emat[2, 2] == 1.0
        @test model.emat[2, 3] == 0.0
        
        # Row 3: from Obs 2, P(Y|state2)=1 -> P(Y|phase3)=1
        @test model.emat[3, 1] == 0.0
        @test model.emat[3, 2] == 0.0
        @test model.emat[3, 3] == 1.0
        
        # Row 4: from Obs 3, P(Y|state1)=0.5, P(Y|state2)=0.5
        @test model.emat[4, 1] == 0.5
        @test model.emat[4, 2] == 0.5
        @test model.emat[4, 3] == 0.5
    end

    @testset "Emission probability expansion preserves semantics" begin
        # Test that P(Y|State k) is correctly replicated to all phases of state k
        h12 = Hazard(@formula(0 ~ 1), "pt", 1, 2)
        
        dat = DataFrame(
            id = [1],
            tstart = [0.0],
            tstop = [1.0],
            statefrom = [1],
            stateto = [1],
            obstype = [2]
        )
        
        # P(obs | state 1) = 0.7, P(obs | state 2) = 0.3
        EmissionMatrix = [0.7 0.3]
        
        model = multistatemodel(h12; data=dat, EmissionMatrix=EmissionMatrix, n_phases=Dict(1=>2))
        
        # State 1 -> phases 1,2; State 2 -> phase 3
        @test model.emat[1, 1] ≈ 0.7
        @test model.emat[1, 2] ≈ 0.7
        @test model.emat[1, 3] ≈ 0.3
        
        # Row sum is 1.7, NOT 1.0 - this is correct!
        # The emission matrix is P(Y|X), not a distribution over X
        @test sum(model.emat[1, :]) ≈ 1.7
    end

    @testset "Likelihood computation with emission matrix" begin
        # Verify likelihood is computed correctly using internal API
        
        h12 = Hazard(@formula(0 ~ 1), "pt", 1, 2)
        
        dat = DataFrame(
            id = [1],
            tstart = [0.0],
            tstop = [1.0],
            statefrom = [1],
            stateto = [1],
            obstype = [2]
        )
        
        # Soft evidence: 80% state 1, 20% state 2
        EmissionMatrix_soft = [0.8 0.2]
        model_soft = multistatemodel(h12; data=dat, EmissionMatrix=EmissionMatrix_soft, n_phases=Dict(1=>2))
        
        # Hard evidence: 100% state 1
        EmissionMatrix_hard = [1.0 0.0]
        model_hard = multistatemodel(h12; data=dat, EmissionMatrix=EmissionMatrix_hard, n_phases=Dict(1=>2))
        
        # Initialize parameters
        initialize_parameters!(model_soft)
        initialize_parameters!(model_hard)
        
        # Compute likelihoods using internal API
        books_soft = MultistateModels.build_tpm_mapping(model_soft.data)
        mpd_soft = MultistateModels.MPanelData(model_soft, books_soft)
        pars_soft = MultistateModels.get_parameters_flat(model_soft)
        ll_soft = MultistateModels.loglik_markov(pars_soft, mpd_soft; neg=false)
        
        books_hard = MultistateModels.build_tpm_mapping(model_hard.data)
        mpd_hard = MultistateModels.MPanelData(model_hard, books_hard)
        pars_hard = MultistateModels.get_parameters_flat(model_hard)
        ll_hard = MultistateModels.loglik_markov(pars_hard, mpd_hard; neg=false)
        
        @test isfinite(ll_soft)
        @test isfinite(ll_hard)
        
        # With soft evidence e=[0.8, 0.2] vs hard evidence e=[1, 0],
        # the likelihood is: L = e1*P(state1) + e2*P(state2)
        # 
        # For this model: P(state1) ≈ 0.61, P(state2) ≈ 0.39
        # Hard: L = 1.0*0.61 + 0*0.39 = 0.61 → log(L) ≈ -0.50
        # Soft: L = 0.8*0.61 + 0.2*0.39 = 0.57 → log(L) ≈ -0.57
        #
        # Soft evidence gives LOWER likelihood because it down-weights 
        # the more probable state (state 1). This is mathematically correct.
        @test ll_soft < ll_hard
    end

    @testset "Analytical verification: simple exponential model" begin
        # =================================================================
        # Verify likelihood computation analytically for a simple case
        # =================================================================
        # 
        # Model: State 1 -> State 2 with rate λ
        # P(in state 1 at time t) = exp(-λt)
        # P(in state 2 at time t) = 1 - exp(-λt)
        #
        # Observation: At time T, with soft evidence:
        #   e = [e1, e2] where e1 = P(Y | state 1), e2 = P(Y | state 2)
        #
        # Likelihood (marginalized over states):
        #   L = e1 * P(state 1 at T) + e2 * P(state 2 at T)
        #     = e1 * exp(-λT) + e2 * (1 - exp(-λT))
        # =================================================================
        
        # Use phase-type with 1 phase (reduces to exponential)
        h12 = Hazard(@formula(0 ~ 1), "pt", 1, 2)
        
        dat = DataFrame(
            id = [1],
            tstart = [0.0],
            tstop = [1.0],
            statefrom = [1],
            stateto = [1],  # observed state doesn't matter with soft evidence
            obstype = [2]
        )
        
        # Soft evidence
        e1, e2 = 0.8, 0.2
        EmissionMatrix = [e1 e2]
        
        # Use 1 phase so it's a simple exponential
        model = multistatemodel(h12; data=dat, EmissionMatrix=EmissionMatrix, n_phases=Dict(1=>1))
        
        # Set known rate parameter
        λ = 0.5
        T = 1.0
        
        # Set parameter (log scale for exponential hazard)
        pars = get_parameters(model)
        pars[:h12][1] = log(λ)
        
        # Compute likelihood
        books = MultistateModels.build_tpm_mapping(model.data)
        mpd = MultistateModels.MPanelData(model, books)
        pars_flat = MultistateModels.get_parameters_flat(model)
        pars_flat[1] = log(λ)
        ll_computed = MultistateModels.loglik_markov(pars_flat, mpd; neg=false)
        
        # Analytical likelihood
        p1 = exp(-λ * T)
        p2 = 1 - exp(-λ * T)
        L_analytical = e1 * p1 + e2 * p2
        ll_analytical = log(L_analytical)
        
        @test isfinite(ll_computed)
        @test ll_computed ≈ ll_analytical atol=1e-6
    end

    @testset "Parameter recovery with misclassification" begin
        # =================================================================
        # Test that MLE can recover true parameters with known error rates
        # =================================================================
        # 
        # Data generating process:
        # - True rate λ = 0.5
        # - Observe state at time T with misclassification:
        #   P(observe 1 | true state 1) = 0.9
        #   P(observe 2 | true state 2) = 0.9
        #
        # Emission matrix encodes the inverse (P(Y | X)):
        #   If observed state is 1: e = [0.9, 0.1]
        #   If observed state is 2: e = [0.1, 0.9]
        # =================================================================
        
        Random.seed!(42)
        
        true_λ = 0.5
        T_obs = 2.0
        n_subjects = 100
        
        # P(state 1 at T) under true model
        p_state1 = exp(-true_λ * T_obs)
        
        # Simulate true states
        true_states = [rand() < p_state1 ? 1 : 2 for _ in 1:n_subjects]
        
        # Add misclassification
        observed_states = [
            true_states[i] == 1 ? (rand() < 0.9 ? 1 : 2) : (rand() < 0.9 ? 2 : 1)
            for i in 1:n_subjects
        ]
        
        # Build data
        dat = DataFrame(
            id = 1:n_subjects,
            tstart = fill(0.0, n_subjects),
            tstop = fill(T_obs, n_subjects),
            statefrom = fill(1, n_subjects),
            stateto = observed_states,
            obstype = fill(2, n_subjects)
        )
        
        # Build emission matrix
        emat_rows = Matrix{Float64}(undef, n_subjects, 2)
        for i in 1:n_subjects
            if observed_states[i] == 1
                emat_rows[i, :] = [0.9, 0.1]
            else
                emat_rows[i, :] = [0.1, 0.9]
            end
        end
        
        h12 = Hazard(@formula(0 ~ 1), "pt", 1, 2)
        model = multistatemodel(h12; data=dat, EmissionMatrix=emat_rows, n_phases=Dict(1=>1))
        
        initialize_parameters!(model)
        
        # Evaluate likelihood at grid of λ values
        λ_grid = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        lls = Float64[]
        
        books = MultistateModels.build_tpm_mapping(model.data)
        mpd = MultistateModels.MPanelData(model, books)
        
        for λ in λ_grid
            pars_flat = MultistateModels.get_parameters_flat(model)
            pars_flat[1] = log(λ)
            ll = MultistateModels.loglik_markov(pars_flat, mpd; neg=false)
            push!(lls, ll)
        end
        
        # Find MLE on grid
        best_idx = argmax(lls)
        λ_mle = λ_grid[best_idx]
        
        # MLE should be close to true value (within grid resolution)
        @test abs(λ_mle - true_λ) <= 0.2
        
        # Print for diagnostics
        println("\nParameter recovery test:")
        println("  True λ = $true_λ")
        println("  Grid MLE λ = $λ_mle")
        println("  Log-likelihoods: ", round.(lls, digits=2))
    end

    @testset "Soft evidence vs hard evidence comparison" begin
        # =================================================================
        # Compare likelihoods with identical data but different evidence
        # =================================================================
        
        h12 = Hazard(@formula(0 ~ 1), "pt", 1, 2)
        
        # Two subjects observed in state 1 at time 1
        dat = DataFrame(
            id = [1, 2],
            tstart = [0.0, 0.0],
            tstop = [1.0, 1.0],
            statefrom = [1, 1],
            stateto = [1, 1],
            obstype = [2, 2]
        )
        
        # Model 1: Hard evidence (certainty about state)
        emat_hard = [1.0 0.0; 1.0 0.0]
        model_hard = multistatemodel(h12; data=dat, EmissionMatrix=emat_hard, n_phases=Dict(1=>1))
        
        # Model 2: Soft evidence (some probability of being in state 2)
        emat_soft = [0.8 0.2; 0.8 0.2]
        model_soft = multistatemodel(h12; data=dat, EmissionMatrix=emat_soft, n_phases=Dict(1=>1))
        
        initialize_parameters!(model_hard)
        initialize_parameters!(model_soft)
        
        # At rate λ = 0.5, P(state 1 at t=1) ≈ 0.61
        λ = 0.5
        
        books_hard = MultistateModels.build_tpm_mapping(model_hard.data)
        mpd_hard = MultistateModels.MPanelData(model_hard, books_hard)
        pars_hard = MultistateModels.get_parameters_flat(model_hard)
        pars_hard[1] = log(λ)
        ll_hard = MultistateModels.loglik_markov(pars_hard, mpd_hard; neg=false)
        
        books_soft = MultistateModels.build_tpm_mapping(model_soft.data)
        mpd_soft = MultistateModels.MPanelData(model_soft, books_soft)
        pars_soft = MultistateModels.get_parameters_flat(model_soft)
        pars_soft[1] = log(λ)
        ll_soft = MultistateModels.loglik_markov(pars_soft, mpd_soft; neg=false)
        
        # With soft evidence e=[0.8, 0.2] vs hard evidence e=[1, 0], the likelihood is:
        #   L = e1*P(state1) + e2*P(state2)
        #
        # For λ=0.5, t=1: P(state1) = exp(-0.5) ≈ 0.607, P(state2) ≈ 0.393
        # Hard: L = 1*0.607 = 0.607 per subject
        # Soft: L = 0.8*0.607 + 0.2*0.393 = 0.564 per subject
        #
        # Soft evidence gives LOWER likelihood because it down-weights 
        # the more probable state (state 1). This is mathematically correct.
        @test ll_soft < ll_hard
        
        # Verify analytically
        p1 = exp(-λ)  # P(state 1 at t=1)
        p2 = 1 - p1   # P(state 2 at t=1)
        
        L_hard = p1^2  # Both subjects definitely in state 1
        L_soft = (0.8*p1 + 0.2*p2)^2  # Soft evidence for both
        
        @test log(L_hard) ≈ ll_hard atol=1e-6
        @test log(L_soft) ≈ ll_soft atol=1e-6
    end
end
