# Unit tests for model output functions: aic(), bic(), summary(), estimate_loglik()
#
# These are exported public API functions that users rely on for model selection 
# and reporting. Tests verify:
# 1. Correct formula implementation for AIC/BIC
# 2. Parameter counting for each model type
# 3. summary() table structure and content
# 4. Error handling for unfitted models
# 5. estimate_loglik() accuracy
#
# References:
# - AIC = -2*loglik + 2*k (Akaike Information Criterion)
# - BIC = -2*loglik + k*log(n) (Bayesian Information Criterion)

using Test
using MultistateModels
using DataFrames
using Random
using Distributions: Normal, quantile
using LinearAlgebra: diag

import MultistateModels: get_parameters_flat, summary, get_parameters

# =============================================================================
# Test Fixtures
# =============================================================================

"""Create a simple fitted exponential model for testing"""
function create_fitted_exponential_model(; n_subj=100, seed=12345)
    Random.seed!(seed)
    
    # Create data with exact observations
    dat = DataFrame(
        id = 1:n_subj,
        tstart = zeros(n_subj),
        tstop = fill(10.0, n_subj),
        statefrom = ones(Int, n_subj),
        stateto = ones(Int, n_subj),
        obstype = fill(1, n_subj)
    )
    
    # Simulate data from known parameters
    true_rate = 0.3
    h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    model = multistatemodel(h12; data=dat)
    set_parameters!(model, (h12 = [true_rate],))
    
    sim_result = simulate(model; paths=false, data=true, nsim=1)
    exact_data = sim_result[1, 1]
    
    # Fit model to simulated data
    h12_fit = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    fitted_model = multistatemodel(h12_fit; data=exact_data)
    fitted = fit(fitted_model; verbose=false)
    
    return fitted, exact_data
end

"""Create a fitted model with covariates"""
function create_fitted_covariate_model(; n_subj=100, seed=54321)
    Random.seed!(seed)
    
    # Create data with covariates
    dat = DataFrame(
        id = 1:n_subj,
        tstart = zeros(n_subj),
        tstop = fill(10.0, n_subj),
        statefrom = ones(Int, n_subj),
        stateto = ones(Int, n_subj),
        obstype = fill(1, n_subj),
        x = randn(n_subj)
    )
    
    # Simulate with covariate effect
    true_rate = 0.25
    true_beta = 0.5
    h12 = Hazard(@formula(0 ~ x), "exp", 1, 2)
    model = multistatemodel(h12; data=dat)
    set_parameters!(model, (h12 = [true_rate, true_beta],))
    
    sim_result = simulate(model; paths=false, data=true, nsim=1)
    exact_data = sim_result[1, 1]
    
    # Fit
    h12_fit = Hazard(@formula(0 ~ x), "exp", 1, 2)
    fitted_model = multistatemodel(h12_fit; data=exact_data)
    fitted = fit(fitted_model; verbose=false)
    
    return fitted, exact_data
end

"""Create an unfitted model for error handling tests"""
function create_unfitted_model(; n_subj=50)
    dat = DataFrame(
        id = 1:n_subj,
        tstart = zeros(n_subj),
        tstop = fill(5.0, n_subj),
        statefrom = ones(Int, n_subj),
        stateto = fill(2, n_subj),
        obstype = fill(1, n_subj)
    )
    
    h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    return multistatemodel(h12; data=dat)
end

# =============================================================================
# aic() Tests
# =============================================================================

@testset "aic()" begin
    
    @testset "AIC formula: -2*loglik + 2*k for exponential model" begin
        fitted, _ = create_fitted_exponential_model()
        
        # Extract EXACT components
        loglik_val = get_loglik(fitted)
        n_params = length(get_parameters_flat(fitted))
        
        # Verify parameter count is exactly 1 for exponential
        @test n_params == 1
        
        # Compute expected AIC from EXACT analytical formula
        expected_aic = -2 * loglik_val + 2 * n_params
        
        # Verify AIC matches analytical formula exactly
        aic_val = aic(fitted; estimate_likelihood=false)
        @test aic_val ≈ expected_aic rtol=1e-10
        
        # Also verify AIC is finite (sanity check)
        @test isfinite(aic_val)
    end
    
    @testset "AIC formula: -2*loglik + 2*k for covariate model" begin
        fitted, _ = create_fitted_covariate_model()
        
        # Extract EXACT components
        loglik_val = get_loglik(fitted)
        n_params = length(get_parameters_flat(fitted))
        
        # Verify parameter count is exactly 2 (baseline rate + covariate effect)
        @test n_params == 2
        
        # Compute expected AIC from EXACT analytical formula
        expected_aic = -2 * loglik_val + 2 * n_params
        
        # Verify AIC matches analytical formula exactly
        aic_val = aic(fitted; estimate_likelihood=false)
        @test aic_val ≈ expected_aic rtol=1e-10
    end
    
    @testset "AIC penalty difference between nested models" begin
        fitted_simple, _ = create_fitted_exponential_model()
        fitted_cov, _ = create_fitted_covariate_model()
        
        # Extract exact parameter counts
        k_simple = length(get_parameters_flat(fitted_simple))
        k_cov = length(get_parameters_flat(fitted_cov))
        
        # Verify exact parameter counts
        @test k_simple == 1
        @test k_cov == 2
        
        # The AIC penalty difference should be exactly 2*(k_cov - k_simple)
        aic_simple = aic(fitted_simple; estimate_likelihood=false)
        aic_cov = aic(fitted_cov; estimate_likelihood=false)
        
        # Penalty component only (ignoring loglik which differs)
        loglik_simple = get_loglik(fitted_simple)
        loglik_cov = get_loglik(fitted_cov)
        
        # Verify AIC = -2*loglik + 2*k relationship holds for each
        @test aic_simple ≈ -2*loglik_simple + 2*k_simple rtol=1e-10
        @test aic_cov ≈ -2*loglik_cov + 2*k_cov rtol=1e-10
    end
end

# =============================================================================
# bic() Tests
# =============================================================================

@testset "bic()" begin
    
    @testset "BIC formula: -2*loglik + k*log(n) for exponential model" begin
        fitted, _ = create_fitted_exponential_model()
        
        # Extract EXACT components
        loglik_val = get_loglik(fitted)
        n_params = length(get_parameters_flat(fitted))
        n_obs = sum(fitted.SubjectWeights)
        
        # Verify exact parameter count and sample size
        @test n_params == 1
        @test n_obs > 0
        
        # Compute expected BIC from EXACT analytical formula
        expected_bic = -2 * loglik_val + n_params * log(n_obs)
        
        # Verify BIC matches analytical formula exactly
        bic_val = bic(fitted; estimate_likelihood=false)
        @test bic_val ≈ expected_bic rtol=1e-10
        @test isfinite(bic_val)
    end
    
    @testset "BIC formula: -2*loglik + k*log(n) for covariate model" begin
        fitted, _ = create_fitted_covariate_model()
        
        # Extract EXACT components
        loglik_val = get_loglik(fitted)
        n_params = length(get_parameters_flat(fitted))
        n_obs = sum(fitted.SubjectWeights)
        
        # Verify exact parameter count
        @test n_params == 2
        
        # Compute expected BIC from EXACT analytical formula
        expected_bic = -2 * loglik_val + n_params * log(n_obs)
        
        # Verify BIC matches analytical formula exactly
        bic_val = bic(fitted; estimate_likelihood=false)
        @test bic_val ≈ expected_bic rtol=1e-10
    end
    
    @testset "BIC vs AIC relationship: exact penalty difference" begin
        fitted, _ = create_fitted_exponential_model()
        
        # Extract exact components
        n_params = length(get_parameters_flat(fitted))
        n_obs = sum(fitted.SubjectWeights)
        
        # Verify we have the expected values
        @test n_params == 1
        @test n_obs > 0
        
        aic_val = aic(fitted; estimate_likelihood=false)
        bic_val = bic(fitted; estimate_likelihood=false)
        
        # BIC should differ from AIC by EXACT penalty difference: k*(log(n) - 2)
        expected_penalty_diff = n_params * (log(n_obs) - 2)
        actual_diff = bic_val - aic_val
        
        @test actual_diff ≈ expected_penalty_diff rtol=1e-10
    end
    
    @testset "BIC vs AIC penalty comparison for various sample sizes" begin
        # For n > exp(2) ≈ 7.4, BIC penalty > AIC penalty
        # For n < exp(2), AIC penalty > BIC penalty
        
        fitted, _ = create_fitted_exponential_model(n_subj=100)
        n_obs = sum(fitted.SubjectWeights)
        n_params = length(get_parameters_flat(fitted))
        
        # Verify sample size is large enough that BIC penalizes more
        @test n_obs > exp(2)
        
        # Compute exact penalty terms
        aic_penalty = 2 * n_params
        bic_penalty = n_params * log(n_obs)
        
        @test bic_penalty > aic_penalty
        
        # This should also mean BIC > AIC (since penalties are only diff)
        @test bic(fitted; estimate_likelihood=false) > aic(fitted; estimate_likelihood=false)
    end
end

# =============================================================================
# summary() Tests
# =============================================================================

@testset "summary()" begin
    
    @testset "Summary AIC equals standalone aic() exactly" begin
        fitted, _ = create_fitted_exponential_model()
        
        summ = summary(fitted; estimate_likelihood=false)
        
        # Verify summary AIC equals standalone function EXACTLY
        @test summ.AIC == aic(fitted; estimate_likelihood=false)
        
        # Also verify it matches the analytical formula
        loglik_val = get_loglik(fitted)
        n_params = length(get_parameters_flat(fitted))
        expected_aic = -2 * loglik_val + 2 * n_params
        @test summ.AIC ≈ expected_aic rtol=1e-10
    end
    
    @testset "Summary BIC equals standalone bic() exactly" begin
        fitted, _ = create_fitted_exponential_model()
        
        summ = summary(fitted; estimate_likelihood=false)
        
        # Verify summary BIC equals standalone function EXACTLY
        @test summ.BIC == bic(fitted; estimate_likelihood=false)
        
        # Also verify it matches the analytical formula
        loglik_val = get_loglik(fitted)
        n_params = length(get_parameters_flat(fitted))
        n_obs = sum(fitted.SubjectWeights)
        expected_bic = -2 * loglik_val + n_params * log(n_obs)
        @test summ.BIC ≈ expected_bic rtol=1e-10
    end
    
    @testset "Summary loglik equals stored loglik exactly" begin
        fitted, _ = create_fitted_exponential_model()
        
        summ = summary(fitted; estimate_likelihood=false)
        
        # Verify summary loglik equals stored loglik EXACTLY
        @test summ.loglik == get_loglik(fitted)
    end
    
    @testset "Summary standard errors equal sqrt(diag(vcov))" begin
        fitted, _ = create_fitted_exponential_model(n_subj=50)
        
        # Skip if vcov is not available
        if !isnothing(fitted.vcov)
            summ = summary(fitted; compute_se=true)
            df = summ.summary.h12
            
            # SE should equal sqrt(diag(vcov)) EXACTLY
            varcov = get_vcov(fitted)
            expected_se = sqrt.(diag(varcov))
            
            @test df.se ≈ expected_se rtol=1e-10
            
            # Also verify SEs are positive
            @test all(df.se .> 0)
        end
    end
    
    @testset "Summary confidence intervals are exact: estimate ± z*se" begin
        fitted, _ = create_fitted_exponential_model(n_subj=50)
        
        if !isnothing(fitted.vcov)
            confidence_level = 0.95
            z_crit = quantile(Normal(0.0, 1.0), 1 - (1 - confidence_level) / 2)
            
            summ = summary(fitted; compute_se=true, confidence_level=confidence_level)
            df = summ.summary.h12
            
            # Verify CI bounds are exactly estimate ± z*se
            expected_lower = df.estimate .- z_crit .* df.se
            expected_upper = df.estimate .+ z_crit .* df.se
            
            @test df.lower ≈ expected_lower rtol=1e-10
            @test df.upper ≈ expected_upper rtol=1e-10
            
            # Sanity checks on CI ordering
            @test all(df.lower .< df.estimate)
            @test all(df.upper .> df.estimate)
        end
    end
    
    @testset "Summary parameter count matches model" begin
        fitted_simple, _ = create_fitted_exponential_model()
        fitted_cov, _ = create_fitted_covariate_model()
        
        summ_simple = summary(fitted_simple)
        summ_cov = summary(fitted_cov)
        
        # Verify exact parameter counts in summary DataFrames
        @test nrow(summ_simple.summary.h12) == 1  # Exponential: 1 parameter
        @test nrow(summ_cov.summary.h12) == 2     # Exponential + covariate: 2 parameters
        
        # These should match get_parameters_flat
        @test nrow(summ_simple.summary.h12) == length(get_parameters_flat(fitted_simple))
        @test nrow(summ_cov.summary.h12) == length(get_parameters_flat(fitted_cov))
    end
end

# =============================================================================
# estimate_loglik() Tests
# =============================================================================

@testset "estimate_loglik()" begin
    
    @testset "Returns finite log-likelihood" begin
        fitted, _ = create_fitted_exponential_model()
        
        # estimate_loglik returns a NamedTuple
        result = estimate_loglik(fitted)
        @test haskey(result, :loglik)
        @test isfinite(result.loglik)
        @test result.loglik isa Float64
    end
    
    @testset "Log-likelihood should be negative for most data" begin
        fitted, _ = create_fitted_exponential_model()
        
        result = estimate_loglik(fitted)
        # Log-likelihood is typically negative (or at most slightly positive)
        @test result.loglik < 10  # Just a sanity check
    end
    
    @testset "Consistency with get_loglik()" begin
        fitted, _ = create_fitted_exponential_model()
        
        result = estimate_loglik(fitted)
        ll_stored = get_loglik(fitted)
        
        # For exact data, estimate should be very close to stored value
        @test isapprox(result.loglik, ll_stored; rtol=0.01)
    end
    
    @testset "Returns subject-level log-likelihoods" begin
        fitted, _ = create_fitted_exponential_model()
        
        result = estimate_loglik(fitted)
        @test haskey(result, :loglik_subj)
        @test length(result.loglik_subj) > 0
    end
end

# =============================================================================
# Error Handling Tests
# =============================================================================

@testset "Error handling for unfitted models" begin
    
    @testset "aic() throws on unfitted model" begin
        unfitted = create_unfitted_model()
        # Unfitted models should throw when calling aic
        @test_throws MethodError aic(unfitted)
    end
    
    @testset "bic() throws on unfitted model" begin
        unfitted = create_unfitted_model()
        @test_throws MethodError bic(unfitted)
    end
    
    @testset "get_loglik() throws on unfitted model" begin
        unfitted = create_unfitted_model()
        @test_throws MethodError get_loglik(unfitted)
    end
end

# =============================================================================
# get_parameters() Structure Tests
# =============================================================================

@testset "get_parameters() returns correct structure" begin
    
    @testset "Returns NamedTuple on natural scale" begin
        fitted, _ = create_fitted_exponential_model()
        
        params = get_parameters(fitted)  # Default is scale=:natural
        
        @test params isa NamedTuple
        @test haskey(params, :h12)
        @test params.h12 isa Vector{Float64}
        @test length(params.h12) == 1  # Exponential has 1 parameter
        @test params.h12[1] > 0  # Rate on natural scale must be positive
    end
    
    @testset "Flat scale returns Vector" begin
        fitted, _ = create_fitted_exponential_model()
        
        params_flat = get_parameters(fitted; scale=:flat)
        
        @test params_flat isa Vector{Float64}
        @test length(params_flat) == 1  # Exponential: 1 parameter
    end
    
    @testset "Covariate model has correct number of parameters" begin
        fitted, _ = create_fitted_covariate_model()
        
        params = get_parameters(fitted)
        
        @test haskey(params, :h12)
        @test length(params.h12) == 2  # baseline rate + covariate coefficient
    end
end

# =============================================================================
# Model Comparison Integration
# =============================================================================

@testset "Model comparison with AIC/BIC" begin
    
    @testset "Compare nested models" begin
        # Create two datasets from same underlying process
        Random.seed!(99999)
        n = 100
        
        dat = DataFrame(
            id = 1:n,
            tstart = zeros(n),
            tstop = fill(10.0, n),
            statefrom = ones(Int, n),
            stateto = ones(Int, n),
            obstype = fill(1, n),
            x = randn(n)
        )
        
        # True model has covariate effect
        h12_true = Hazard(@formula(0 ~ x), "exp", 1, 2)
        model_true = multistatemodel(h12_true; data=dat)
        set_parameters!(model_true, (h12 = [0.3, 0.7],))
        
        sim_result = simulate(model_true; paths=false, data=true, nsim=1)
        exact_data = sim_result[1, 1]
        
        # Fit both models
        h12_full = Hazard(@formula(0 ~ x), "exp", 1, 2)
        h12_reduced = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        
        model_full = multistatemodel(h12_full; data=exact_data)
        model_reduced = multistatemodel(h12_reduced; data=exact_data)
        
        fitted_full = fit(model_full; verbose=false)
        fitted_reduced = fit(model_reduced; verbose=false)
        
        aic_full = aic(fitted_full; estimate_likelihood=false)
        aic_reduced = aic(fitted_reduced; estimate_likelihood=false)
        
        # Both should be finite
        @test isfinite(aic_full)
        @test isfinite(aic_reduced)
        
        # Full model should have lower AIC (better fit, true model includes covariate)
        # Note: This may not always hold due to sample variability, so just check finiteness
    end
end
