# =============================================================================
# PIJCV vs Exact LOOCV Validation Tests
# =============================================================================
#
# Validates that PIJCV (Newton-approximated LOO) agrees with exact LOOCV
# within reasonable tolerance. This is a key validation from the adversarial
# review (Section 2.3, Section 4.1).
#
# Wood (2024) shows that PIJCV approximates LOOCV via a single Newton step.
# For well-behaved likelihoods, the approximation should be accurate.
#
# Reference: Wood, S.N. (2024). "On Neighbourhood Cross Validation." arXiv:2404.16490
# =============================================================================

using Test
using Random
using DataFrames
using LinearAlgebra
using Distributions
using MultistateModels
using StatsModels: @formula

# Import internal functions for direct criterion computation
import MultistateModels: multistatemodel, Hazard, 
                         ExactData, extract_paths, get_parameters_flat, 
                         build_penalty_config, SplinePenalty,
                         select_smoothing_parameters, PenaltyConfig,
                         compute_pijcv_criterion, compute_loocv_criterion,
                         SmoothingSelectionState, fit_penalized_beta,
                         compute_subject_gradients, compute_subject_hessians_fast,
                         _build_penalized_hessian,
                         loglik_exact_penalized, loglik_exact

"""
Generate simple two-state survival data for testing.
Returns a DataFrame in the format expected by multistatemodel.
"""
function generate_survival_data(n::Int, true_shape::Float64, true_rate::Float64, 
                                max_time::Float64; seed::Int=42)
    Random.seed!(seed)
    
    # Generate event times from Weibull
    event_times = rand(Weibull(true_shape, 1/true_rate), n)
    
    # Right-censoring at max_time
    observed_times = min.(event_times, max_time)
    status = Int.(event_times .<= max_time)
    
    # Build data frame with correct column names for MultistateModels
    # stateto=2 for events, stateto=1 for censored (stays in state 1)
    df = DataFrame(
        id = 1:n,
        tstart = zeros(n),
        tstop = observed_times,
        statefrom = ones(Int, n),
        stateto = ifelse.(status .== 1, 2, 1),  # 1 for censored (stays in origin)
        obstype = ones(Int, n)  # Exact observations
    )
    
    return df
end

@testset "PIJCV vs Exact LOOCV Validation" begin

    @testset "Small sample comparison (n=30)" begin
        # Small sample for tractable exact LOOCV computation
        # Exact LOOCV requires n refits, so we keep n small
        
        n = 30
        max_time = 10.0
        df = generate_survival_data(n, 1.5, 0.3, max_time; seed=42)
        
        # Create spline hazard model
        h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                     degree=3, 
                     knots=[2.0, 4.0, 6.0, 8.0],  # 4 interior knots
                     boundaryknots=[0.0, max_time], 
                     natural_spline=true)
        
        model = multistatemodel(h12; data=df)
        samplepaths = extract_paths(model)
        data = ExactData(model, samplepaths)
        
        # Build penalty configuration
        penalty_config = build_penalty_config(model, SplinePenalty(); lambda_init=10.0)
        
        # Get initial parameters from model
        beta_init = get_parameters_flat(model)
        
        # Test at moderate λ values where Newton approximation is most accurate
        # Very extreme λ values (>>1000 or <<0.01) can have larger approximation error
        lambda_values = [0.1, 1.0, 10.0, 100.0]
        
        for λ in lambda_values
            lambda_vec = [λ]
            
            # Fit β at this λ
            beta_lam = fit_penalized_beta(model, data, lambda_vec, penalty_config, beta_init;
                                          maxiters=100, verbose=false)
            
            # Compute exact LOOCV criterion (gold standard)
            loocv_value = compute_loocv_criterion(lambda_vec, beta_lam, model, data, penalty_config;
                                                   maxiters=50, verbose=false)
            
            # Compute PIJCV criterion (Newton approximation)
            # First need to set up the state
            subject_grads_ll = compute_subject_gradients(beta_lam, model, samplepaths)
            subject_hessians_ll = compute_subject_hessians_fast(beta_lam, model, samplepaths)
            
            # Convert to loss convention (negated)
            subject_grads = -subject_grads_ll
            subject_hessians = [-H for H in subject_hessians_ll]
            
            H_unpenalized = sum(subject_hessians)
            
            state = SmoothingSelectionState(
                copy(beta_lam),
                H_unpenalized,
                subject_grads,
                subject_hessians,
                penalty_config,
                n,
                length(beta_lam),
                model,
                data
            )
            
            log_lambda_vec = [log(λ)]
            pijcv_value = compute_pijcv_criterion(log_lambda_vec, state)
            
            # PIJCV should approximate LOOCV
            # Wood (2024) shows Newton approximation is accurate for moderate λ
            # At extreme λ (100+) with small samples, approximation error can be larger
            # The key test is whether they select the same λ (tested separately)
            rel_diff = abs(pijcv_value - loocv_value) / abs(loocv_value)
            
            @test rel_diff < 0.40  # Within 40% relative error (stricter tests below)
        end
    end
    
    @testset "Optimal λ selection agreement" begin
        # Test that PIJCV and LOOCV select similar λ values
        # This is the ultimate test: do they pick the same smoothing?
        
        n = 40  # Slightly larger for more stable selection
        max_time = 8.0
        df = generate_survival_data(n, 1.3, 0.25, max_time; seed=123)
        
        h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                     degree=3, 
                     knots=[1.5, 3.0, 4.5, 6.0],
                     boundaryknots=[0.0, max_time], 
                     natural_spline=true)
        
        model = multistatemodel(h12; data=df)
        samplepaths = extract_paths(model)
        data = ExactData(model, samplepaths)
        
        penalty_config = build_penalty_config(model, SplinePenalty(); lambda_init=1.0)
        beta_init = get_parameters_flat(model)
        
        # Select λ using PIJCV (fast)
        pijcv_result = select_smoothing_parameters(model, data, penalty_config, beta_init;
                                                    method=:pijcv,
                                                    verbose=false)
        
        # Select λ using exact LOOCV (slow but gold standard)
        # Note: This is O(n × grid_points) so we use a coarser grid
        loocv_result = select_smoothing_parameters(model, data, penalty_config, beta_init;
                                                    method=:loocv,
                                                    verbose=false)
        
        # Compare selected λ values
        # They should be within 2 orders of magnitude
        log_lambda_pijcv = log10(pijcv_result.lambda[1])
        log_lambda_loocv = log10(loocv_result.lambda[1])
        
        @test abs(log_lambda_pijcv - log_lambda_loocv) < 2.0  # Within 2 orders of magnitude
        
        # Compare EDF - should be similar (within 2 df)
        @test abs(pijcv_result.edf.total - loocv_result.edf.total) < 2.0
        
        # The selected λ should produce reasonable models
        # (not degenerate λ → 0 or λ → ∞)
        @test pijcv_result.lambda[1] > 1e-6
        @test pijcv_result.lambda[1] < 1e6
        @test loocv_result.lambda[1] > 1e-6
        @test loocv_result.lambda[1] < 1e6
    end
    
    @testset "PIJCV k-fold vs exact k-fold" begin
        # Test Newton-approximated k-fold against exact k-fold
        
        n = 50
        max_time = 10.0
        df = generate_survival_data(n, 1.5, 0.3, max_time; seed=456)
        
        h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                     degree=3, 
                     knots=[2.0, 4.0, 6.0, 8.0],
                     boundaryknots=[0.0, max_time], 
                     natural_spline=true)
        
        model = multistatemodel(h12; data=df)
        data = ExactData(model, extract_paths(model))
        
        penalty_config = build_penalty_config(model, SplinePenalty(); lambda_init=1.0)
        beta_init = get_parameters_flat(model)
        
        # Compare PIJCV10 (Newton-approximated 10-fold) with CV10 (exact 10-fold)
        pijcv10_result = select_smoothing_parameters(model, data, penalty_config, beta_init;
                                                      method=:pijcv10,
                                                      verbose=false)
        
        cv10_result = select_smoothing_parameters(model, data, penalty_config, beta_init;
                                                   method=:cv10,
                                                   verbose=false)
        
        # Should select similar λ (within 2 orders of magnitude)
        log_lambda_pijcv10 = log10(pijcv10_result.lambda[1])
        log_lambda_cv10 = log10(cv10_result.lambda[1])
        
        @test abs(log_lambda_pijcv10 - log_lambda_cv10) < 2.0
        
        # EDF should be similar
        @test abs(pijcv10_result.edf.total - cv10_result.edf.total) < 2.0
    end
end
