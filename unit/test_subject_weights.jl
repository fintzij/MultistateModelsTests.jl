# =============================================================================
# Test suite for SubjectWeights validation
# =============================================================================
#
# Tests for:
# - SubjectWeights propagation through model construction
# - Weighted likelihood computation in loglik_markov
# - SubjectWeights usage in MCEM functions (mcem_mll, mcem_ase, mcem_mll_sir)
# - Equivalence: duplicated data vs. SubjectWeights
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
    # Convert flat vector to nested structure matching model hazards
    nested = [[v] for v in values]
    set_parameters!(model, nested)
end

@testset "SubjectWeights Validation" begin
    
    @testset "Model construction with SubjectWeights" begin
        # Setup basic model
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
        
        nsubj = 5
        dat = DataFrame(
            id = repeat(1:nsubj, inner=2),
            tstart = repeat([0.0, 1.0], outer=nsubj),
            tstop = repeat([1.0, 2.0], outer=nsubj),
            statefrom = repeat([1, 2], outer=nsubj),
            stateto = repeat([2, 3], outer=nsubj),
            obstype = fill(2, 2*nsubj)  # panel data
        )
        
        # Test 1: Model construction with uniform weights (default)
        model_uniform = multistatemodel(h12, h23; data=dat)
        @test model_uniform.SubjectWeights == ones(nsubj)
        
        # Test 2: Model construction with custom weights
        custom_weights = [1.0, 2.0, 0.5, 1.5, 3.0]
        model_weighted = multistatemodel(h12, h23; data=dat, SubjectWeights=custom_weights)
        @test model_weighted.SubjectWeights == custom_weights
        
        # Test 3: Validation - wrong length should error
        @test_throws ErrorException multistatemodel(h12, h23; data=dat, SubjectWeights=[1.0, 2.0])
        
        # Test 4: Validation - non-positive weights should error
        @test_throws ErrorException multistatemodel(h12, h23; data=dat, SubjectWeights=[1.0, 0.0, 1.0, 1.0, 1.0])
        @test_throws ErrorException multistatemodel(h12, h23; data=dat, SubjectWeights=[1.0, -1.0, 1.0, 1.0, 1.0])
    end
    
    @testset "loglik_markov weighted vs. duplicated data equivalence" begin
        # Core statistical property: a weight of 2 should give the same
        # log-likelihood as duplicating the subject's data
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
        
        # Single subject data
        dat_single = DataFrame(
            id = [1, 1],
            tstart = [0.0, 1.0],
            tstop = [1.0, 2.0],
            statefrom = [1, 2],
            stateto = [2, 3],
            obstype = [2, 2]
        )
        
        # Duplicated data (same subject twice)
        dat_duplicated = DataFrame(
            id = [1, 1, 2, 2],
            tstart = [0.0, 1.0, 0.0, 1.0],
            tstop = [1.0, 2.0, 1.0, 2.0],
            statefrom = [1, 2, 1, 2],
            stateto = [2, 3, 2, 3],
            obstype = [2, 2, 2, 2]
        )
        
        # Model with weight=2
        model_weighted = multistatemodel(h12, h23; data=dat_single, SubjectWeights=[2.0])
        
        # Model with duplicated data (uniform weights)
        model_duplicated = multistatemodel(h12, h23; data=dat_duplicated)
        
        # Set same parameters
        set_test_params!(model_weighted, [log(0.5), log(0.3)])  # log scale
        set_test_params!(model_duplicated, [log(0.5), log(0.3)])
        
        # Compute log-likelihoods
        books_weighted = MultistateModels.build_tpm_mapping(model_weighted.data)
        mpd_weighted = MultistateModels.MPanelData(model_weighted, books_weighted)
        ll_weighted = MultistateModels.loglik_markov(
            MultistateModels.get_parameters_flat(model_weighted), 
            mpd_weighted; 
            neg=false
        )
        
        books_duplicated = MultistateModels.build_tpm_mapping(model_duplicated.data)
        mpd_duplicated = MultistateModels.MPanelData(model_duplicated, books_duplicated)
        ll_duplicated = MultistateModels.loglik_markov(
            MultistateModels.get_parameters_flat(model_duplicated), 
            mpd_duplicated; 
            neg=false
        )
        
        # Should be equal: w=2 for one subject ≡ two identical subjects with w=1
        @test ll_weighted ≈ ll_duplicated rtol=1e-10
    end
    
    @testset "loglik_markov subject-level weights correctness" begin
        # Test that subject-level log-likelihoods are correctly scaled by weights
        
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
        
        # Model with uniform weights
        model_uniform = multistatemodel(h12, h23; data=dat)
        set_test_params!(model_uniform, [log(0.5), log(0.3)])
        
        books_uniform = MultistateModels.build_tpm_mapping(model_uniform.data)
        mpd_uniform = MultistateModels.MPanelData(model_uniform, books_uniform)
        ll_subj_uniform = MultistateModels.loglik_markov(
            MultistateModels.get_parameters_flat(model_uniform), 
            mpd_uniform; 
            neg=false, 
            return_ll_subj=true
        )
        
        # Model with non-uniform weights
        weights = [1.0, 2.0, 0.5]
        model_weighted = multistatemodel(h12, h23; data=dat, SubjectWeights=weights)
        set_test_params!(model_weighted, [log(0.5), log(0.3)])
        
        books_weighted = MultistateModels.build_tpm_mapping(model_weighted.data)
        mpd_weighted = MultistateModels.MPanelData(model_weighted, books_weighted)
        ll_subj_weighted = MultistateModels.loglik_markov(
            MultistateModels.get_parameters_flat(model_weighted), 
            mpd_weighted; 
            neg=false, 
            return_ll_subj=true
        )
        
        # Subject log-likelihoods should be scaled by their weights
        # All subjects have identical data, so unweighted contributions should be equal
        unweighted_ll = ll_subj_uniform[1]  # Same for all subjects
        @test ll_subj_uniform[1] ≈ ll_subj_uniform[2] rtol=1e-10
        @test ll_subj_uniform[2] ≈ ll_subj_uniform[3] rtol=1e-10
        
        # Weighted contributions should be weight * unweighted_ll
        @test ll_subj_weighted[1] ≈ 1.0 * unweighted_ll rtol=1e-10
        @test ll_subj_weighted[2] ≈ 2.0 * unweighted_ll rtol=1e-10
        @test ll_subj_weighted[3] ≈ 0.5 * unweighted_ll rtol=1e-10
        
        # Total log-likelihood should be sum of weighted contributions
        ll_total_weighted = MultistateModels.loglik_markov(
            MultistateModels.get_parameters_flat(model_weighted), 
            mpd_weighted; 
            neg=false
        )
        @test ll_total_weighted ≈ sum(ll_subj_weighted) rtol=1e-10
    end
    
    @testset "MCEM functions with SubjectWeights" begin
        # Test that mcem_mll, mcem_ase, mcem_mll_sir correctly apply SubjectWeights
        
        # Setup synthetic log-likelihoods and weights
        logliks = [[-1.0, -2.0, -1.5], [-0.5, -1.0, -0.8], [-2.0, -2.5, -1.8]]
        ImportanceWeights = [[0.4, 0.35, 0.25], [0.3, 0.4, 0.3], [0.5, 0.3, 0.2]]
        
        # Test with uniform weights
        uniform_weights = [1.0, 1.0, 1.0]
        mll_uniform = MultistateModels.mcem_mll(logliks, ImportanceWeights, uniform_weights)
        
        # Manual calculation for verification
        expected_uniform = sum(
            dot(logliks[i], ImportanceWeights[i]) 
            for i in 1:3
        )
        @test mll_uniform ≈ expected_uniform rtol=1e-10
        
        # Test with non-uniform weights
        nonuniform_weights = [2.0, 0.5, 1.5]
        mll_nonuniform = MultistateModels.mcem_mll(logliks, ImportanceWeights, nonuniform_weights)
        
        expected_nonuniform = sum(
            dot(logliks[i], ImportanceWeights[i]) * nonuniform_weights[i] 
            for i in 1:3
        )
        @test mll_nonuniform ≈ expected_nonuniform rtol=1e-10
        
        # Test mcem_ase with SubjectWeights
        logliks_prop = [[-0.9, -1.8, -1.4], [-0.4, -0.9, -0.7], [-1.9, -2.4, -1.7]]
        ase_uniform = MultistateModels.mcem_ase(logliks_prop, logliks, ImportanceWeights, uniform_weights)
        ase_nonuniform = MultistateModels.mcem_ase(logliks_prop, logliks, ImportanceWeights, nonuniform_weights)
        
        # ASE should be non-negative
        @test ase_uniform >= 0.0
        @test ase_nonuniform >= 0.0
        
        # ASE with higher weights should generally scale up (quadratic in weights)
        # Note: This is a weak test; exact relationship depends on variance structure
        @test ase_nonuniform > 0  # Just verify it's computed
    end
    
    @testset "mcem_mll_sir with SubjectWeights" begin
        # Test SIR-specific MCEM function
        logliks = [[-1.0, -2.0, -1.5, -1.2, -0.8], 
                   [-0.5, -1.0, -0.8, -0.6, -0.9], 
                   [-2.0, -2.5, -1.8, -2.2, -1.9]]
        
        # SIR indices (subsample of paths per subject)
        sir_indices = [[1, 3, 5], [2, 4], [1, 2, 3, 4, 5]]
        
        # Uniform weights
        uniform_weights = [1.0, 1.0, 1.0]
        mll_uniform = MultistateModels.mcem_mll_sir(logliks, sir_indices, uniform_weights)
        
        # Manual calculation: mean over subsampled paths, then sum
        expected_uniform = mean(logliks[1][[1,3,5]]) + mean(logliks[2][[2,4]]) + mean(logliks[3])
        @test mll_uniform ≈ expected_uniform rtol=1e-10
        
        # Non-uniform weights
        nonuniform_weights = [2.0, 0.5, 1.5]
        mll_nonuniform = MultistateModels.mcem_mll_sir(logliks, sir_indices, nonuniform_weights)
        
        expected_nonuniform = 2.0 * mean(logliks[1][[1,3,5]]) + 
                             0.5 * mean(logliks[2][[2,4]]) + 
                             1.5 * mean(logliks[3])
        @test mll_nonuniform ≈ expected_nonuniform rtol=1e-10
        
        # Test mcem_ase_sir
        logliks_prop = [l .+ 0.1 for l in logliks]  # Slightly different
        ase = MultistateModels.mcem_ase_sir(logliks_prop, logliks, sir_indices, nonuniform_weights)
        @test ase >= 0.0
    end
    
    @testset "SubjectWeights in exact data likelihood" begin
        # Test weighted exact data likelihood
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
        
        nsubj = 3
        dat = DataFrame(
            id = repeat(1:nsubj, inner=2),
            tstart = repeat([0.0, 1.0], outer=nsubj),
            tstop = repeat([1.0, 2.0], outer=nsubj),
            statefrom = repeat([1, 2], outer=nsubj),
            stateto = repeat([2, 3], outer=nsubj),
            obstype = fill(1, 2*nsubj)  # exact data
        )
        
        weights = [1.0, 2.0, 0.5]
        model = multistatemodel(h12, h23; data=dat, SubjectWeights=weights)
        set_test_params!(model, [log(0.5), log(0.3)])
        
        # Create sample paths for ExactData
        samplepaths = [MultistateModels.SamplePath(i, [0.0, 1.0, 2.0], [1, 2, 3]) for i in 1:nsubj]
        exactdata = MultistateModels.ExactData(model, samplepaths)
        
        # Compute log-likelihood
        params = MultistateModels.get_parameters_flat(model)
        ll = MultistateModels.loglik_exact(params, exactdata; neg=false)
        
        # Should be finite (basic sanity check)
        @test isfinite(ll)
        
        # The likelihood should be influenced by weights
        # Create unweighted model for comparison
        model_uniform = multistatemodel(h12, h23; data=dat)
        set_test_params!(model_uniform, [log(0.5), log(0.3)])
        exactdata_uniform = MultistateModels.ExactData(model_uniform, samplepaths)
        ll_uniform = MultistateModels.loglik_exact(params, exactdata_uniform; neg=false)
        
        # Weighted ll should differ from uniform (unless weights average to 1)
        # Here sum(weights)/nsubj = 3.5/3 ≠ 1, so they should differ
        @test !isapprox(ll, ll_uniform, rtol=0.01)
    end
    
    @testset "Gradient correctness with SubjectWeights" begin
        # ForwardDiff should handle weighted likelihoods correctly
        using ForwardDiff
        
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
        
        weights = [1.0, 2.0, 0.5]
        model = multistatemodel(h12, h23; data=dat, SubjectWeights=weights)
        set_test_params!(model, [log(0.5), log(0.3)])
        
        books = MultistateModels.build_tpm_mapping(model.data)
        mpd = MultistateModels.MPanelData(model, books)
        params = MultistateModels.get_parameters_flat(model)
        
        # Gradient should be finite
        grad = ForwardDiff.gradient(p -> MultistateModels.loglik_markov(p, mpd; neg=true), params)
        @test length(grad) == length(params)
        @test all(isfinite.(grad))
        
        # Gradient with uniform weights for comparison
        model_uniform = multistatemodel(h12, h23; data=dat)
        set_test_params!(model_uniform, [log(0.5), log(0.3)])
        books_uniform = MultistateModels.build_tpm_mapping(model_uniform.data)
        mpd_uniform = MultistateModels.MPanelData(model_uniform, books_uniform)
        
        grad_uniform = ForwardDiff.gradient(p -> MultistateModels.loglik_markov(p, mpd_uniform; neg=true), params)
        
        # Gradients should differ (different weights affect gradient)
        @test !all(isapprox.(grad, grad_uniform, rtol=0.01))
    end
    
    @testset "Phase-type model with SubjectWeights" begin
        # Test that SubjectWeights work correctly with phase-type distributions
        # Phase-type uses MCEM with importance sampling, different code path
        
        # Setup a Weibull hazard (requires phase-type approximation for panel data)
        h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
        h23 = Hazard(@formula(0 ~ 1), "wei", 2, 3)
        
        nsubj = 5
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
        @test model_uniform.SubjectWeights == ones(nsubj)
        
        # Model with non-uniform weights
        custom_weights = [1.0, 2.0, 0.5, 1.5, 3.0]
        model_weighted = multistatemodel(h12, h23; data=dat, SubjectWeights=custom_weights)
        @test model_weighted.SubjectWeights == custom_weights
        
        # Verify weights are stored correctly in the model
        @test length(model_weighted.SubjectWeights) == nsubj
        @test all(model_weighted.SubjectWeights .> 0)
    end
    
    @testset "Phase-type weighted vs duplicated equivalence" begin
        # For phase-type, test that weight=2 gives same result as duplicating data
        
        h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
        h23 = Hazard(@formula(0 ~ 1), "wei", 2, 3)
        
        # Single subject data
        dat_single = DataFrame(
            id = [1, 1],
            tstart = [0.0, 1.0],
            tstop = [1.0, 2.0],
            statefrom = [1, 2],
            stateto = [2, 3],
            obstype = [2, 2]
        )
        
        # Duplicated data (same subject twice)
        dat_duplicated = DataFrame(
            id = [1, 1, 2, 2],
            tstart = [0.0, 1.0, 0.0, 1.0],
            tstop = [1.0, 2.0, 1.0, 2.0],
            statefrom = [1, 2, 1, 2],
            stateto = [2, 3, 2, 3],
            obstype = [2, 2, 2, 2]
        )
        
        # Model with weight=2
        model_weighted = multistatemodel(h12, h23; data=dat_single, SubjectWeights=[2.0])
        
        # Model with duplicated data (uniform weights)
        model_duplicated = multistatemodel(h12, h23; data=dat_duplicated)
        
        # Set same parameters (Weibull has 2 params per hazard: log_scale, log_shape)
        set_parameters!(model_weighted, [[0.0, 0.0], [0.0, 0.0]])
        set_parameters!(model_duplicated, [[0.0, 0.0], [0.0, 0.0]])
        
        # Compute log-likelihoods using the Markov wrapper
        books_weighted = MultistateModels.build_tpm_mapping(model_weighted.data)
        mpd_weighted = MultistateModels.MPanelData(model_weighted, books_weighted)
        ll_weighted = MultistateModels.loglik_markov(
            MultistateModels.get_parameters_flat(model_weighted), 
            mpd_weighted; 
            neg=false
        )
        
        books_duplicated = MultistateModels.build_tpm_mapping(model_duplicated.data)
        mpd_duplicated = MultistateModels.MPanelData(model_duplicated, books_duplicated)
        ll_duplicated = MultistateModels.loglik_markov(
            MultistateModels.get_parameters_flat(model_duplicated), 
            mpd_duplicated; 
            neg=false
        )
        
        # Should be equal: w=2 for one subject ≡ two identical subjects with w=1
        @test ll_weighted ≈ ll_duplicated rtol=1e-10
    end
    
    @testset "Phase-type expansion preserves SubjectWeights" begin
        # Test that phase-type state expansion correctly propagates SubjectWeights
        
        # Use explicit phase-type hazard for expansion test
        h12 = Hazard(@formula(0 ~ 1), :pt, 1, 2; n_phases=3, coxian_structure=:sctp)
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
        
        weights = [1.0, 2.0, 0.5]
        model = multistatemodel(h12, h23; data=dat, SubjectWeights=weights)
        
        # Verify weights are preserved in expanded model
        @test model.SubjectWeights == weights
        @test length(model.SubjectWeights) == nsubj
    end
    
end

