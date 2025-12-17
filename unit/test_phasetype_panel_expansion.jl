using Test
using MultistateModels
using DataFrames
using LinearAlgebra

@testset "Phase-Type Panel Data Expansion" begin
    
    # Create simple test case: 1 -> 2 (panel observation)
    # State 1 has 2 phases, State 2 has 2 phases.
    # Observation: Start in state 1 at t=0, observed in state 2 at t=2.
    # This means at t=2, the subject could be in phase 3 or phase 4 (phases of state 2).
    
    data = DataFrame(
        id = [1],
        tstart = [0.0],
        tstop = [2.0],
        statefrom = [1],
        stateto = [2],
        obstype = [2]  # panel observation
    )

    h12 = Hazard(:pt, 1, 2)
    h23 = Hazard(:pt, 2, 3)
    model = multistatemodel(h12, h23; data = data, n_phases = Dict(1 => 2, 2 => 2), coxian_structure = :sctp)

    # 1. Check Data Expansion
    # Expectation: stateto should be 0 (censored), obstype should be 4 (censoring pattern for state 2)
    @test model.data.stateto[1] == 0
    @test model.data.obstype[1] == 4
    
    # Check Emission Matrix
    # Expectation: emat row 1 should allow phases 3 and 4 (indices of state 2 phases)
    # Phases: 1,2 (State 1); 3,4 (State 2); 5 (State 3)
    @test model.emat[1, 1] == 0.0
    @test model.emat[1, 2] == 0.0
    @test model.emat[1, 3] == 1.0
    @test model.emat[1, 4] == 1.0
    @test model.emat[1, 5] == 0.0

    # 2. Check Likelihood Calculation
    # Set parameters to log(0.5) so all rates are 0.5
    log_rate = log(0.5)
    flat_params = fill(log_rate, length(model.parameters.flat))
    
    # Build MPanelData
    books = MultistateModels.build_tpm_mapping(model.data)
    mpd = MultistateModels.MPanelData(model, books)
    
    # Compute loglik
    ll_markov = MultistateModels.loglik_markov(flat_params, mpd; neg=false)
    
    # Manual Verification
    # Q matrix for rate = 0.5
    # State 1 (Phases 1,2) -> State 2 (Phases 3,4) -> State 3 (Phase 5)
    # SCTP structure:
    # Phase 1 -> Phase 2 (0.5), Phase 1 -> Phase 3 (0.5)
    # Phase 2 -> Phase 3 (0.5)
    # Phase 3 -> Phase 4 (0.5), Phase 3 -> Phase 5 (0.5)
    # Phase 4 -> Phase 5 (0.5)
    
    n_phases = 5
    Q = zeros(n_phases, n_phases)
    rate = 0.5
    
    Q[1,2] = rate; Q[1,3] = rate; Q[1,1] = -1.0
    Q[2,3] = rate; Q[2,2] = -0.5
    Q[3,4] = rate; Q[3,5] = rate; Q[3,3] = -1.0
    Q[4,5] = rate; Q[4,4] = -0.5
    
    P = exp(Q * 2.0)
    
    # Probability of being in State 2 (Phases 3 or 4) at t=2 given start in State 1 (Phase 1)
    prob_state2 = P[1,3] + P[1,4]
    expected_ll = log(prob_state2)
    
    @test isapprox(ll_markov, expected_ll, atol=1e-6)
    
    # Verify the specific value (exp(-1) case)
    @test isapprox(prob_state2, exp(-1.0), atol=1e-6)
    @test isapprox(ll_markov, -1.0, atol=1e-6)

end
