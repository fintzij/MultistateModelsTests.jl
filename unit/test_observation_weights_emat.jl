# =============================================================================
# Test suite for ObservationWeights and EmissionMatrix/CensoringPatterns
# =============================================================================
#
# Tests for:
# - ObservationWeights propagation through model construction
# - ObservationWeights application in likelihood computation
# - CensoringPatterns construction of emission matrices
# - EmissionMatrix direct specification
# - Emission matrix usage in forward algorithm for censored observations
#
# NOTE: ObservationWeights are only applied to exact (obstype=1) and panel 
#       (obstype=2) data in loglik_markov. They are NOT currently applied in 
#       loglik_exact.
#
# =============================================================================

using Test
using MultistateModels
using DataFrames
using LinearAlgebra
using Random
using Statistics

# Helper function to set parameters using nested vectors
function set_test_params!(model, values)
    nested = [[v] for v in values]
    set_parameters!(model, nested)
end

@testset "ObservationWeights and EmissionMatrix Validation" begin
    
    # =========================================================================
    # ObservationWeights Tests
    # =========================================================================
    
    @testset "ObservationWeights model construction" begin
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
        
        nsubj = 3
        dat = DataFrame(
            id = repeat(1:nsubj, inner=2),
            tstart = repeat([0.0, 1.0], outer=nsubj),
            tstop = repeat([1.0, 2.0], outer=nsubj),
            statefrom = repeat([1, 2], outer=nsubj),
            stateto = repeat([2, 3], outer=nsubj),
            obstype = fill(2, 2*nsubj)
        )
        
        # Test: ObservationWeights stored correctly
        obs_weights = [1.0, 2.0, 1.0, 0.5, 1.0, 1.5]  # one per observation
        model = multistatemodel(h12, h23; data=dat, ObservationWeights=obs_weights)
        @test model.ObservationWeights == obs_weights
        
        # Test: Wrong length should error
        @test_throws ArgumentError multistatemodel(h12, h23; data=dat, ObservationWeights=[1.0, 2.0])
        
        # Test: Non-positive weights should error
        @test_throws ArgumentError multistatemodel(h12, h23; data=dat, ObservationWeights=[1.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        @test_throws ArgumentError multistatemodel(h12, h23; data=dat, ObservationWeights=[1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
        
        # Test: SubjectWeights and ObservationWeights are mutually exclusive
        @test_throws ArgumentError multistatemodel(h12, h23; data=dat, 
            SubjectWeights=[1.0, 1.0, 1.0], 
            ObservationWeights=obs_weights)
    end
    
    @testset "ObservationWeights in panel data likelihood" begin
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
        
        nsubj = 2
        dat = DataFrame(
            id = repeat(1:nsubj, inner=2),
            tstart = repeat([0.0, 1.0], outer=nsubj),
            tstop = repeat([1.0, 2.0], outer=nsubj),
            statefrom = repeat([1, 2], outer=nsubj),
            stateto = repeat([2, 3], outer=nsubj),
            obstype = fill(2, 2*nsubj)  # panel data
        )
        
        # Model with uniform weights
        model_uniform = multistatemodel(h12, h23; data=dat)
        set_test_params!(model_uniform, [0.5, 0.3])
        
        # Model with observation weights (double first obs of each subject)
        obs_weights = [2.0, 1.0, 2.0, 1.0]
        model_weighted = multistatemodel(h12, h23; data=dat, ObservationWeights=obs_weights)
        set_test_params!(model_weighted, [0.5, 0.3])
        
        books_uniform = MultistateModels.build_tpm_mapping(model_uniform.data)
        mpd_uniform = MultistateModels.MPanelData(model_uniform, books_uniform)
        ll_uniform = MultistateModels.loglik_markov(
            MultistateModels.get_parameters_flat(model_uniform), 
            mpd_uniform; neg=false
        )
        
        books_weighted = MultistateModels.build_tpm_mapping(model_weighted.data)
        mpd_weighted = MultistateModels.MPanelData(model_weighted, books_weighted)
        ll_weighted = MultistateModels.loglik_markov(
            MultistateModels.get_parameters_flat(model_weighted), 
            mpd_weighted; neg=false
        )
        
        # Should differ (total weight 6 vs 4)
        @test !isapprox(ll_weighted, ll_uniform, rtol=0.01)
        @test isfinite(ll_weighted)
        @test isfinite(ll_uniform)
    end
    
    @testset "ObservationWeights in exact data likelihood (loglik_markov)" begin
        # Test observation weights in loglik_markov for exact data
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
        
        nsubj = 2
        dat = DataFrame(
            id = repeat(1:nsubj, inner=2),
            tstart = repeat([0.0, 1.0], outer=nsubj),
            tstop = repeat([1.0, 2.0], outer=nsubj),
            statefrom = repeat([1, 2], outer=nsubj),
            stateto = repeat([2, 3], outer=nsubj),
            obstype = fill(1, 2*nsubj)  # exact data
        )
        
        # Model with uniform weights
        model_uniform = multistatemodel(h12, h23; data=dat)
        set_test_params!(model_uniform, [0.5, 0.3])
        
        # Model with observation weights
        obs_weights = [2.0, 1.0, 2.0, 1.0]
        model_weighted = multistatemodel(h12, h23; data=dat, ObservationWeights=obs_weights)
        set_test_params!(model_weighted, [0.5, 0.3])
        
        books_uniform = MultistateModels.build_tpm_mapping(model_uniform.data)
        mpd_uniform = MultistateModels.MPanelData(model_uniform, books_uniform)
        ll_uniform = MultistateModels.loglik_markov(
            MultistateModels.get_parameters_flat(model_uniform), 
            mpd_uniform; neg=false
        )
        
        books_weighted = MultistateModels.build_tpm_mapping(model_weighted.data)
        mpd_weighted = MultistateModels.MPanelData(model_weighted, books_weighted)
        ll_weighted = MultistateModels.loglik_markov(
            MultistateModels.get_parameters_flat(model_weighted), 
            mpd_weighted; neg=false
        )
        
        # Should differ (total weight 6 vs 4)
        @test !isapprox(ll_weighted, ll_uniform, rtol=0.01)
        @test isfinite(ll_weighted)
        @test isfinite(ll_uniform)
    end
    
    # =========================================================================
    # CensoringPatterns Tests
    # =========================================================================
    
    @testset "CensoringPatterns model construction" begin
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
        
        # Data with censored observations (obstype > 2)
        # obstype = 3 means use CensoringPatterns row 1
        # obstype = 4 means use CensoringPatterns row 2
        # When obstype > 2, stateto must be 0
        dat = DataFrame(
            id = [1, 1, 2, 2],
            tstart = [0.0, 1.0, 0.0, 1.0],
            tstop = [1.0, 2.0, 1.0, 2.0],
            statefrom = [1, 1, 1, 1],
            stateto = [2, 0, 2, 0],   # stateto=0 for censored observations
            obstype = [2, 3, 2, 4]    # panel, censored, panel, censored
        )
        
        # CensoringPatterns: [pattern_id, state1_prob, state2_prob, state3_prob]
        # Pattern 1 (obstype=3): could be state 2 or 3
        # Pattern 2 (obstype=4): could be state 1 or 2
        CensoringPatterns = [
            3.0 0.0 1.0 1.0;  # pattern 1: not state 1, could be 2 or 3
            4.0 1.0 1.0 0.0   # pattern 2: not state 3, could be 1 or 2
        ]
        
        model = multistatemodel(h12, h23; data=dat, CensoringPatterns=CensoringPatterns)
        
        # Check emat was constructed
        @test size(model.emat) == (4, 3)  # 4 observations, 3 states
        
        # Row 1 (obstype=2, panel): only state 2 allowed (stateto=2)
        @test model.emat[1, :] == [0.0, 1.0, 0.0]
        
        # Row 2 (obstype=3, pattern 1): states 2 and 3 allowed
        @test model.emat[2, :] == [0.0, 1.0, 1.0]
        
        # Row 3 (obstype=2, panel): only state 2 allowed
        @test model.emat[3, :] == [0.0, 1.0, 0.0]
        
        # Row 4 (obstype=4, pattern 2): states 1 and 2 allowed
        @test model.emat[4, :] == [1.0, 1.0, 0.0]
    end
    
    @testset "EmissionMatrix direct specification" begin
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
        
        dat = DataFrame(
            id = [1, 1],
            tstart = [0.0, 1.0],
            tstop = [1.0, 2.0],
            statefrom = [1, 2],
            stateto = [2, 3],
            obstype = [2, 2]
        )
        
        # Custom emission matrix with soft evidence
        EmissionMatrix = [
            0.0 0.9 0.1;   # obs 1: 90% sure state 2, 10% state 3
            0.1 0.1 0.8    # obs 2: mostly state 3
        ]
        
        model = multistatemodel(h12, h23; data=dat, EmissionMatrix=EmissionMatrix)
        
        @test model.emat == EmissionMatrix
    end
    
    @testset "EmissionMatrix validation" begin
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
        
        dat = DataFrame(
            id = [1, 1],
            tstart = [0.0, 1.0],
            tstop = [1.0, 2.0],
            statefrom = [1, 2],
            stateto = [2, 3],
            obstype = [2, 2]
        )
        
        # Wrong dimensions should error
        @test_throws ArgumentError multistatemodel(h12, h23; data=dat, 
            EmissionMatrix=[1.0 0.0 0.0])  # 1 row instead of 2
        
        # Values outside [0,1] should error
        @test_throws ArgumentError multistatemodel(h12, h23; data=dat, 
            EmissionMatrix=[0.0 1.5 0.0; 0.0 0.0 1.0])
        
        # All zeros in a row should error
        @test_throws ArgumentError multistatemodel(h12, h23; data=dat, 
            EmissionMatrix=[0.0 0.0 0.0; 0.0 0.0 1.0])
    end
    
    @testset "Emission matrix in forward algorithm (censored likelihood)" begin
        # Test that emission matrix is correctly used in forward algorithm
        # for censored observations
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
        
        # Single subject with censored observation
        # stateto=0 required for censored (obstype > 2)
        dat = DataFrame(
            id = [1, 1],
            tstart = [0.0, 1.0],
            tstop = [1.0, 2.0],
            statefrom = [1, 1],  # start in state 1
            stateto = [2, 0],    # known -> unknown
            obstype = [2, 3]     # panel, then censored
        )
        
        # Pattern 1 (obstype=3): could be state 2 or 3
        CensoringPatterns = [3.0 0.0 1.0 1.0]
        
        model = multistatemodel(h12, h23; data=dat, CensoringPatterns=CensoringPatterns)
        set_test_params!(model, [0.5, 0.3])
        
        books = MultistateModels.build_tpm_mapping(model.data)
        mpd = MultistateModels.MPanelData(model, books)
        ll = MultistateModels.loglik_markov(
            MultistateModels.get_parameters_flat(model), 
            mpd; neg=false
        )
        
        # Should be finite and computable
        @test isfinite(ll)
        
        # Compare with a model where state is known exactly (state 2)
        dat_exact = DataFrame(
            id = [1, 1],
            tstart = [0.0, 1.0],
            tstop = [1.0, 2.0],
            statefrom = [1, 1],
            stateto = [2, 2],    # known to be state 2
            obstype = [2, 2]
        )
        
        model_exact = multistatemodel(h12, h23; data=dat_exact)
        set_test_params!(model_exact, [0.5, 0.3])
        
        books_exact = MultistateModels.build_tpm_mapping(model_exact.data)
        mpd_exact = MultistateModels.MPanelData(model_exact, books_exact)
        ll_exact = MultistateModels.loglik_markov(
            MultistateModels.get_parameters_flat(model_exact), 
            mpd_exact; neg=false
        )
        
        # Censored likelihood should differ from exact likelihood
        @test ll != ll_exact
        @test isfinite(ll_exact)
    end
    
    @testset "CensoringPatterns forward algorithm computes correct sum" begin
        # When a state is censored, the likelihood should sum over allowed states
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
        
        # Single observation, censored to allow states 2 or 3
        dat_censored = DataFrame(
            id = [1],
            tstart = [0.0],
            tstop = [1.0],
            statefrom = [1],
            stateto = [0],   # censored
            obstype = [3]    # use pattern 1
        )
        
        CensoringPatterns = [3.0 0.0 1.0 1.0]  # states 2 or 3 allowed
        
        model_censored = multistatemodel(h12, h23; data=dat_censored, CensoringPatterns=CensoringPatterns)
        set_test_params!(model_censored, [0.5, 0.3])
        
        # Compute censored likelihood
        books = MultistateModels.build_tpm_mapping(model_censored.data)
        mpd = MultistateModels.MPanelData(model_censored, books)
        ll_censored = MultistateModels.loglik_markov(
            MultistateModels.get_parameters_flat(model_censored), 
            mpd; neg=false
        )
        
        # Now compute explicitly: P(end in 2 or 3 | start in 1)
        # This is P(1->2) + P(1->3)
        dat_to2 = DataFrame(
            id = [1],
            tstart = [0.0],
            tstop = [1.0],
            statefrom = [1],
            stateto = [2],
            obstype = [2]
        )
        dat_to3 = DataFrame(
            id = [1],
            tstart = [0.0],
            tstop = [1.0],
            statefrom = [1],
            stateto = [3],
            obstype = [2]
        )
        
        model_to2 = multistatemodel(h12, h23; data=dat_to2)
        model_to3 = multistatemodel(h12, h23; data=dat_to3)
        set_test_params!(model_to2, [0.5, 0.3])
        set_test_params!(model_to3, [0.5, 0.3])
        
        books_to2 = MultistateModels.build_tpm_mapping(model_to2.data)
        mpd_to2 = MultistateModels.MPanelData(model_to2, books_to2)
        ll_to2 = MultistateModels.loglik_markov(
            MultistateModels.get_parameters_flat(model_to2), 
            mpd_to2; neg=false
        )
        
        books_to3 = MultistateModels.build_tpm_mapping(model_to3.data)
        mpd_to3 = MultistateModels.MPanelData(model_to3, books_to3)
        ll_to3 = MultistateModels.loglik_markov(
            MultistateModels.get_parameters_flat(model_to3), 
            mpd_to3; neg=false
        )
        
        # Censored likelihood should be log(P(->2) + P(->3))
        expected_ll = log(exp(ll_to2) + exp(ll_to3))
        @test ll_censored ≈ expected_ll rtol=1e-10
    end
    
end

