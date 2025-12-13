# =============================================================================
# Unit tests for MLL consistency across SIR, LHS, and IS-weighted full pool
# =============================================================================
#
# Tests that marginal log-likelihood estimates from:
# - :sir (multinomial resampling)
# - :lhs (Latin Hypercube Sampling resampling)
# - IS-weighted full pool (gold standard)
# 
# agree within sampling error at both subject and sample levels.
#
# Also compares variance of resampled estimates to full pool (zero resampling variance)
# and includes runtime benchmarks.

using Test
using MultistateModels
using DataFrames
using Random
using Statistics
using LinearAlgebra

@testset "MLL Consistency Tests" begin
    
    # =========================================================================
    # Setup: Create a test model with known parameters
    # =========================================================================
    
    Random.seed!(20241213)
    
    # Create panel data for a 3-state progressive model: 1 → 2 → 3
    n_subjects = 50
    rows = []
    
    for i in 1:n_subjects
        # Each subject has observations at t=0, 1, 2, 3
        state_at_0 = 1
        state_at_1 = rand() < 0.35 ? 2 : 1
        state_at_2 = state_at_1 == 2 ? (rand() < 0.4 ? 3 : 2) : (rand() < 0.25 ? 2 : 1)
        state_at_3 = state_at_2 == 3 ? 3 : (state_at_2 == 2 ? (rand() < 0.5 ? 3 : 2) : (rand() < 0.2 ? 2 : 1))
        
        push!(rows, (id=i, tstart=0.0, tstop=1.0, statefrom=state_at_0, stateto=state_at_1, obstype=2))
        push!(rows, (id=i, tstart=1.0, tstop=2.0, statefrom=state_at_1, stateto=state_at_2, obstype=2))
        push!(rows, (id=i, tstart=2.0, tstop=3.0, statefrom=state_at_2, stateto=state_at_3, obstype=2))
    end
    
    data = DataFrame(rows)
    
    # Create Weibull hazards
    h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
    h23 = Hazard(@formula(0 ~ 1), "wei", 2, 3)
    
    # Create model
    model = multistatemodel(
        h12, h23;
        data = data,
        CensoringPatterns = nothing
    )
    
    # Set up surrogate
    set_surrogate!(model)
    
    nsubj = length(unique(data.id))
    
    # =========================================================================
    # Fit model with large ESS and return paths for testing
    # =========================================================================
    
    # Fit with large ESS to get a good pool of paths
    Random.seed!(12345)
    fitted = fit(model;
        ess_target_initial = 500,
        max_ess = 1000,
        maxiter = 3,  # Just a few iterations to get paths
        verbose = false,
        return_proposed_paths = true,
        compute_vcov = false,
        compute_ij_vcov = false
    )
    
    # Extract paths and weights from fitted model
    samplepaths = fitted.ProposedPaths.paths
    ImportanceWeights = fitted.ProposedPaths.weights
    
    # Compute target log-likelihoods for all paths at the fitted parameters
    loglik_target = [Float64[] for _ in 1:nsubj]
    fitted_params = fitted.parameters.flat  # Get flat parameter vector
    for i in 1:nsubj
        loglik_target[i] = zeros(length(samplepaths[i]))
        for (j, path) in enumerate(samplepaths[i])
            loglik_target[i][j] = MultistateModels.loglik(fitted_params, path, fitted.hazards, fitted)
        end
    end
    
    # =========================================================================
    # Test: Subject-level and sample-level MLL consistency
    # =========================================================================
    
    @testset "Subject-level and sample-level MLL consistency" begin
        ess_target = 200  # Subsample size for SIR/LHS
        
        # =====================================================================
        # Compute MLL using three methods with timing
        # =====================================================================
        
        # 1. Full IS-weighted estimate (gold standard)
        time_is = @elapsed begin
            mll_is_total = MultistateModels.mcem_mll(loglik_target, ImportanceWeights, model.SubjectWeights)
            mll_is_subj = [dot(loglik_target[i], ImportanceWeights[i]) for i in 1:nsubj]
        end
        
        # 2. SIR resampled estimate
        Random.seed!(12345)
        time_sir = @elapsed begin
            sir_indices = [MultistateModels.get_sir_subsample_indices(ImportanceWeights[i], ess_target, :sir) for i in 1:nsubj]
            mll_sir_total = MultistateModels.mcem_mll_sir(loglik_target, sir_indices, model.SubjectWeights)
            mll_sir_subj = [mean(loglik_target[i][sir_indices[i]]) for i in 1:nsubj]
        end
        
        # 3. LHS resampled estimate  
        Random.seed!(12345)
        time_lhs = @elapsed begin
            lhs_indices = [MultistateModels.get_sir_subsample_indices(ImportanceWeights[i], ess_target, :lhs) for i in 1:nsubj]
            mll_lhs_total = MultistateModels.mcem_mll_sir(loglik_target, lhs_indices, model.SubjectWeights)
            mll_lhs_subj = [mean(loglik_target[i][lhs_indices[i]]) for i in 1:nsubj]
        end
        
        # =====================================================================
        # Tests: Sample-level (summed over subjects)
        # =====================================================================
        
        @testset "Sample-level MLL agreement" begin
            # All three methods should agree within sampling error
            # With large pool and ESS, expect agreement within ~10% of MLL magnitude
            
            mll_range = abs(mll_is_total) * 0.15  # Allow 15% relative tolerance
            mll_tol = max(mll_range, 5.0)  # At least 5 log-likelihood units
            
            @test abs(mll_sir_total - mll_is_total) < mll_tol
            @test abs(mll_lhs_total - mll_is_total) < mll_tol
            @test abs(mll_sir_total - mll_lhs_total) < mll_tol
            
            # Print diagnostics
            println("\n  Sample-level MLL estimates:")
            println("    IS-weighted (full pool): $(round(mll_is_total, digits=4))")
            println("    SIR (ESS=$ess_target):    $(round(mll_sir_total, digits=4))")
            println("    LHS (ESS=$ess_target):    $(round(mll_lhs_total, digits=4))")
            println("    Max absolute difference: $(round(max(abs(mll_sir_total - mll_is_total), abs(mll_lhs_total - mll_is_total)), digits=4))")
        end
        
        # =====================================================================
        # Tests: Subject-level agreement
        # =====================================================================
        
        @testset "Subject-level MLL agreement" begin
            # Check correlation between methods
            cor_sir_is = cor(mll_sir_subj, mll_is_subj)
            cor_lhs_is = cor(mll_lhs_subj, mll_is_subj)
            cor_sir_lhs = cor(mll_sir_subj, mll_lhs_subj)
            
            # High correlation expected (> 0.90)
            @test cor_sir_is > 0.85
            @test cor_lhs_is > 0.85
            @test cor_sir_lhs > 0.85
            
            # Mean absolute error per subject
            mae_sir_is = mean(abs.(mll_sir_subj .- mll_is_subj))
            mae_lhs_is = mean(abs.(mll_lhs_subj .- mll_is_subj))
            mae_sir_lhs = mean(abs.(mll_sir_subj .- mll_lhs_subj))
            
            # MAE should be small relative to typical subject MLL magnitude
            mean_subj_mll = mean(abs.(mll_is_subj))
            relative_mae_sir = mae_sir_is / mean_subj_mll
            relative_mae_lhs = mae_lhs_is / mean_subj_mll
            
            @test relative_mae_sir < 0.25  # Less than 25% relative MAE
            @test relative_mae_lhs < 0.25
            
            println("\n  Subject-level MLL agreement:")
            println("    Correlation (SIR vs IS):  $(round(cor_sir_is, digits=4))")
            println("    Correlation (LHS vs IS):  $(round(cor_lhs_is, digits=4))")
            println("    Correlation (SIR vs LHS): $(round(cor_sir_lhs, digits=4))")
            println("    Relative MAE (SIR vs IS): $(round(100*relative_mae_sir, digits=2))%")
            println("    Relative MAE (LHS vs IS): $(round(100*relative_mae_lhs, digits=2))%")
        end
        
        # =====================================================================
        # Tests: Runtime comparison
        # =====================================================================
        
        @testset "Runtime benchmarks" begin
            # Warm up all functions to exclude compilation time
            _ = MultistateModels.mcem_mll(loglik_target, ImportanceWeights, model.SubjectWeights)
            _ = MultistateModels.get_sir_subsample_indices(ImportanceWeights[1], ess_target, :sir)
            _ = MultistateModels.get_sir_subsample_indices(ImportanceWeights[1], ess_target, :lhs)
            sir_idx_warmup = [MultistateModels.get_sir_subsample_indices(ImportanceWeights[i], ess_target, :sir) for i in 1:nsubj]
            _ = MultistateModels.mcem_mll_sir(loglik_target, sir_idx_warmup, model.SubjectWeights)
            
            # Time MLL computation only (excluding resampling)
            n_timing_reps = 50
            
            time_is_mll = 0.0
            for _ in 1:n_timing_reps
                time_is_mll += @elapsed MultistateModels.mcem_mll(loglik_target, ImportanceWeights, model.SubjectWeights)
            end
            
            # Pre-compute indices for fair MLL-only comparison
            Random.seed!(999)
            sir_idx_precomputed = [MultistateModels.get_sir_subsample_indices(ImportanceWeights[i], ess_target, :sir) for i in 1:nsubj]
            lhs_idx_precomputed = [MultistateModels.get_sir_subsample_indices(ImportanceWeights[i], ess_target, :lhs) for i in 1:nsubj]
            
            time_sir_mll = 0.0
            for _ in 1:n_timing_reps
                time_sir_mll += @elapsed MultistateModels.mcem_mll_sir(loglik_target, sir_idx_precomputed, model.SubjectWeights)
            end
            
            time_lhs_mll = 0.0
            for _ in 1:n_timing_reps
                time_lhs_mll += @elapsed MultistateModels.mcem_mll_sir(loglik_target, lhs_idx_precomputed, model.SubjectWeights)
            end
            
            # Time resampling step separately
            time_sir_resample = 0.0
            for _ in 1:n_timing_reps
                Random.seed!(999)
                time_sir_resample += @elapsed [MultistateModels.get_sir_subsample_indices(ImportanceWeights[i], ess_target, :sir) for i in 1:nsubj]
            end
            
            time_lhs_resample = 0.0
            for _ in 1:n_timing_reps
                Random.seed!(999)
                time_lhs_resample += @elapsed [MultistateModels.get_sir_subsample_indices(ImportanceWeights[i], ess_target, :lhs) for i in 1:nsubj]
            end
            
            avg_time_is_mll = time_is_mll / n_timing_reps * 1000  # ms
            avg_time_sir_mll = time_sir_mll / n_timing_reps * 1000
            avg_time_lhs_mll = time_lhs_mll / n_timing_reps * 1000
            avg_time_sir_resample = time_sir_resample / n_timing_reps * 1000
            avg_time_lhs_resample = time_lhs_resample / n_timing_reps * 1000
            
            # Pool size info
            pool_sizes = [length(ImportanceWeights[i]) for i in 1:nsubj]
            avg_pool_size = mean(pool_sizes)
            
            println("\n  Runtime benchmarks ($(n_timing_reps) reps, warmed up):")
            println("    Pool size: $(round(Int, avg_pool_size)) paths/subject avg, ESS target: $(ess_target)")
            println("")
            println("    MLL computation only (excluding resampling):")
            println("      IS-weighted (full pool, $(round(Int, avg_pool_size)) paths): $(round(avg_time_is_mll, digits=4)) ms")
            println("      SIR resampled ($(ess_target) paths):             $(round(avg_time_sir_mll, digits=4)) ms")
            println("      LHS resampled ($(ess_target) paths):             $(round(avg_time_lhs_mll, digits=4)) ms")
            println("      Speedup (IS/SIR): $(round(avg_time_is_mll / avg_time_sir_mll, digits=2))x")
            println("      Speedup (IS/LHS): $(round(avg_time_is_mll / avg_time_lhs_mll, digits=2))x")
            println("")
            println("    Resampling overhead:")
            println("      SIR resampling: $(round(avg_time_sir_resample, digits=4)) ms")
            println("      LHS resampling: $(round(avg_time_lhs_resample, digits=4)) ms")
            println("")
            println("    Total (MLL + resampling):")
            println("      IS-weighted:    $(round(avg_time_is_mll, digits=4)) ms")
            println("      SIR:            $(round(avg_time_sir_mll + avg_time_sir_resample, digits=4)) ms")
            println("      LHS:            $(round(avg_time_lhs_mll + avg_time_lhs_resample, digits=4)) ms")
            
            # MLL on fewer paths should be faster
            @test avg_time_sir_mll < avg_time_is_mll * 1.5  # Allow some overhead tolerance
            @test avg_time_lhs_mll < avg_time_is_mll * 1.5
            
            # =================================================================
            # Log-likelihood computation benchmark (the expensive part in M-step)
            # This is where SIR/LHS provide real speedup during optimization
            # =================================================================
            
            # Create SIR-subsampled data structure (as used in actual M-step)
            sir_data = MultistateModels.create_sir_subsampled_data(samplepaths, sir_idx_precomputed)
            lhs_data = MultistateModels.create_sir_subsampled_data(samplepaths, lhs_idx_precomputed)
            
            # Warm up loglik computation
            loglik_full_warmup = [zeros(length(samplepaths[i])) for i in 1:nsubj]
            loglik_sir_warmup = [zeros(length(sir_data.paths[i])) for i in 1:nsubj]
            loglik_lhs_warmup = [zeros(length(lhs_data.paths[i])) for i in 1:nsubj]
            
            full_panel = MultistateModels.SMPanelData(model, samplepaths, ImportanceWeights)
            sir_panel = MultistateModels.SMPanelData(model, sir_data.paths, sir_data.weights)
            lhs_panel = MultistateModels.SMPanelData(model, lhs_data.paths, lhs_data.weights)
            
            MultistateModels.loglik!(fitted_params, loglik_full_warmup, full_panel)
            MultistateModels.loglik!(fitted_params, loglik_sir_warmup, sir_panel)
            MultistateModels.loglik!(fitted_params, loglik_lhs_warmup, lhs_panel)
            
            # Time log-likelihood computation (the actual bottleneck in optimization)
            n_loglik_reps = 10
            
            loglik_full = [zeros(length(samplepaths[i])) for i in 1:nsubj]
            time_loglik_full = 0.0
            for _ in 1:n_loglik_reps
                time_loglik_full += @elapsed MultistateModels.loglik!(fitted_params, loglik_full, full_panel)
            end
            
            loglik_sir = [zeros(length(sir_data.paths[i])) for i in 1:nsubj]
            time_loglik_sir = 0.0
            for _ in 1:n_loglik_reps
                time_loglik_sir += @elapsed MultistateModels.loglik!(fitted_params, loglik_sir, sir_panel)
            end
            
            loglik_lhs = [zeros(length(lhs_data.paths[i])) for i in 1:nsubj]
            time_loglik_lhs = 0.0
            for _ in 1:n_loglik_reps
                time_loglik_lhs += @elapsed MultistateModels.loglik!(fitted_params, loglik_lhs, lhs_panel)
            end
            
            avg_time_loglik_full = time_loglik_full / n_loglik_reps * 1000  # ms
            avg_time_loglik_sir = time_loglik_sir / n_loglik_reps * 1000
            avg_time_loglik_lhs = time_loglik_lhs / n_loglik_reps * 1000
            
            total_paths_full = sum(length.(samplepaths))
            total_paths_sir = sum(length.(sir_data.paths))
            total_paths_lhs = sum(length.(lhs_data.paths))
            
            println("")
            println("    Log-likelihood computation (M-step bottleneck, $(n_loglik_reps) reps):")
            println("      Full pool ($(total_paths_full) paths total): $(round(avg_time_loglik_full, digits=2)) ms")
            println("      SIR subset ($(total_paths_sir) paths total):  $(round(avg_time_loglik_sir, digits=2)) ms")
            println("      LHS subset ($(total_paths_lhs) paths total):  $(round(avg_time_loglik_lhs, digits=2)) ms")
            println("      Speedup (Full/SIR): $(round(avg_time_loglik_full / avg_time_loglik_sir, digits=2))x")
            println("      Speedup (Full/LHS): $(round(avg_time_loglik_full / avg_time_loglik_lhs, digits=2))x")
            println("      Expected speedup (paths ratio): $(round(total_paths_full / total_paths_sir, digits=2))x")
            
            # SIR/LHS should be significantly faster for loglik computation
            @test avg_time_loglik_sir < avg_time_loglik_full
            @test avg_time_loglik_lhs < avg_time_loglik_full
        end
    end
    
    # =========================================================================
    # Test: Variance comparison - SIR, LHS, and Full Pool
    # =========================================================================
    
    @testset "Variance comparison: SIR vs LHS vs Full Pool" begin
        ess_target = 200
        n_reps = 50
        
        # Gold standard (full pool IS-weighted) - zero resampling variance
        mll_is_total = MultistateModels.mcem_mll(loglik_target, ImportanceWeights, model.SubjectWeights)
        
        sir_totals = zeros(n_reps)
        lhs_totals = zeros(n_reps)
        
        time_sir_var = @elapsed for rep in 1:n_reps
            Random.seed!(rep * 1000)
            sir_idx = [MultistateModels.get_sir_subsample_indices(ImportanceWeights[i], ess_target, :sir) for i in 1:nsubj]
            sir_totals[rep] = MultistateModels.mcem_mll_sir(loglik_target, sir_idx, model.SubjectWeights)
        end
        
        time_lhs_var = @elapsed for rep in 1:n_reps
            Random.seed!(rep * 1000)
            lhs_idx = [MultistateModels.get_sir_subsample_indices(ImportanceWeights[i], ess_target, :lhs) for i in 1:nsubj]
            lhs_totals[rep] = MultistateModels.mcem_mll_sir(loglik_target, lhs_idx, model.SubjectWeights)
        end
        
        var_sir = var(sir_totals)
        var_lhs = var(lhs_totals)
        # Full pool has zero resampling variance (deterministic given weights)
        var_full_pool = 0.0
        
        # Both should have reasonably low variance
        @test var_sir < (abs(mll_is_total) * 0.10)^2  # SD < 10% of MLL
        @test var_lhs < (abs(mll_is_total) * 0.10)^2
        
        # Check that means are close to IS estimate
        @test abs(mean(sir_totals) - mll_is_total) < 3 * sqrt(var_sir)
        @test abs(mean(lhs_totals) - mll_is_total) < 3 * sqrt(var_lhs)
        
        println("\n  Variance comparison ($(n_reps) replications, ESS=$(ess_target)):")
        println("    Full Pool IS variance: $(round(var_full_pool, digits=6)) (deterministic)")
        println("    SIR variance:          $(round(var_sir, digits=4)), SD: $(round(sqrt(var_sir), digits=4))")
        println("    LHS variance:          $(round(var_lhs, digits=4)), SD: $(round(sqrt(var_lhs), digits=4))")
        println("    Variance ratio (SIR/LHS): $(round(var_sir/var_lhs, digits=2))")
        println("")
        println("    Full Pool IS mean:     $(round(mll_is_total, digits=4)) (exact)")
        println("    SIR mean:              $(round(mean(sir_totals), digits=4))")
        println("    LHS mean:              $(round(mean(lhs_totals), digits=4))")
        println("")
        println("    Variance computation runtimes:")
        println("      SIR ($n_reps reps):  $(round(time_sir_var, digits=3)) s")
        println("      LHS ($n_reps reps):  $(round(time_lhs_var, digits=3)) s")
    end
    
    # =========================================================================
    # Test: Unbiasedness check
    # =========================================================================
    
    @testset "Unbiasedness of resampling estimators" begin
        ess_target = 100
        n_bootstrap = 100
        
        # Gold standard
        mll_is = MultistateModels.mcem_mll(loglik_target, ImportanceWeights, model.SubjectWeights)
        
        # Bootstrap
        sir_estimates = zeros(n_bootstrap)
        lhs_estimates = zeros(n_bootstrap)
        
        for b in 1:n_bootstrap
            Random.seed!(b)
            sir_idx = [MultistateModels.get_sir_subsample_indices(ImportanceWeights[i], ess_target, :sir) for i in 1:nsubj]
            sir_estimates[b] = MultistateModels.mcem_mll_sir(loglik_target, sir_idx, model.SubjectWeights)
            
            lhs_idx = [MultistateModels.get_sir_subsample_indices(ImportanceWeights[i], ess_target, :lhs) for i in 1:nsubj]
            lhs_estimates[b] = MultistateModels.mcem_mll_sir(loglik_target, lhs_idx, model.SubjectWeights)
        end
        
        # Test unbiasedness: mean of bootstrap estimates should be close to IS estimate
        sir_bias = mean(sir_estimates) - mll_is
        lhs_bias = mean(lhs_estimates) - mll_is
        
        # Bias should be small relative to standard deviation
        # Allow 3 standard deviations
        @test abs(sir_bias) < 3 * std(sir_estimates)
        @test abs(lhs_bias) < 3 * std(lhs_estimates)
        
        println("\n  Unbiasedness check ($(n_bootstrap) bootstrap samples):")
        println("    IS estimate:   $(round(mll_is, digits=4))")
        println("    SIR mean:      $(round(mean(sir_estimates), digits=4)), bias: $(round(sir_bias, digits=4))")
        println("    LHS mean:      $(round(mean(lhs_estimates), digits=4)), bias: $(round(lhs_bias, digits=4))")
        println("    SIR SD:        $(round(std(sir_estimates), digits=4))")
        println("    LHS SD:        $(round(std(lhs_estimates), digits=4))")
    end
    
    # =========================================================================
    # Test: Edge cases
    # =========================================================================
    
    @testset "Edge cases" begin
        # Test with uniform weights (all paths equally likely)
        n_paths = 100
        uniform_weights = fill(1.0/n_paths, n_paths)
        logliks_uniform = randn(n_paths) .- 5.0  # Random log-likelihoods
        
        # SIR and LHS should both reduce to uniform sampling
        Random.seed!(111)
        sir_idx = MultistateModels.get_sir_subsample_indices(uniform_weights, 50, :sir)
        lhs_idx = MultistateModels.get_sir_subsample_indices(uniform_weights, 50, :lhs)
        
        @test length(sir_idx) == 50
        @test length(lhs_idx) == 50
        @test all(1 .<= sir_idx .<= n_paths)
        @test all(1 .<= lhs_idx .<= n_paths)
        
        # Test with highly skewed weights (one dominant path)
        skewed_weights = zeros(n_paths)
        skewed_weights[1] = 0.99
        skewed_weights[2:end] .= 0.01 / (n_paths - 1)
        
        Random.seed!(222)
        sir_idx_skewed = MultistateModels.get_sir_subsample_indices(skewed_weights, 50, :sir)
        lhs_idx_skewed = MultistateModels.get_sir_subsample_indices(skewed_weights, 50, :lhs)
        
        # Most indices should be 1
        @test count(==(1), sir_idx_skewed) > 30  # At least 60% should be index 1
        @test count(==(1), lhs_idx_skewed) > 30
        
        println("\n  Edge case tests passed")
    end
end

println("\n✓ All MLL consistency tests completed")
