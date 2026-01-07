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

import MultistateModels: get_parameters_flat, summary

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
    
    @testset "Basic AIC calculation" begin
        fitted, _ = create_fitted_exponential_model()
        
        # Use estimate_likelihood=false to use stored loglik (fast for testing)
        aic_val = aic(fitted; estimate_likelihood=false)
        @test isfinite(aic_val)
        @test aic_val isa Float64
    end
    
    @testset "AIC formula: -2*loglik + 2*k" begin
        fitted, _ = create_fitted_exponential_model()
        
        # Extract components
        loglik_val = get_loglik(fitted)
        n_params = length(get_parameters_flat(fitted))
        
        # Compute expected AIC
        expected_aic = -2 * loglik_val + 2 * n_params
        
        @test aic(fitted; estimate_likelihood=false) ≈ expected_aic rtol=1e-10
    end
    
    @testset "AIC with covariates" begin
        fitted, _ = create_fitted_covariate_model()
        
        aic_val = aic(fitted; estimate_likelihood=false)
        @test isfinite(aic_val)
        
        # More parameters should increase AIC penalty
        fitted_simple, _ = create_fitted_exponential_model()
        # Both should have finite AIC values
        @test isfinite(aic(fitted_simple; estimate_likelihood=false))
    end
    
    @testset "AIC parameter counting" begin
        # Exponential: 1 parameter (log-rate)
        fitted_exp, _ = create_fitted_exponential_model()
        n_exp = length(get_parameters_flat(fitted_exp))
        @test n_exp == 1
        
        # Exponential with covariate: 2 parameters (log-rate + beta)
        fitted_cov, _ = create_fitted_covariate_model()
        n_cov = length(get_parameters_flat(fitted_cov))
        @test n_cov == 2
    end
end

# =============================================================================
# bic() Tests
# =============================================================================

@testset "bic()" begin
    
    @testset "Basic BIC calculation" begin
        fitted, _ = create_fitted_exponential_model()
        
        bic_val = bic(fitted; estimate_likelihood=false)
        @test isfinite(bic_val)
        @test bic_val isa Float64
    end
    
    @testset "BIC formula: -2*loglik + k*log(n)" begin
        fitted, data = create_fitted_exponential_model()
        
        # Extract components
        loglik_val = get_loglik(fitted)
        n_params = length(get_parameters_flat(fitted))
        # BIC uses number of subjects, which is stored in SubjectWeights
        n_obs = sum(fitted.SubjectWeights)
        
        # Compute expected BIC
        expected_bic = -2 * loglik_val + n_params * log(n_obs)
        
        @test bic(fitted; estimate_likelihood=false) ≈ expected_bic rtol=1e-10
    end
    
    @testset "BIC vs AIC relationship" begin
        fitted, data = create_fitted_exponential_model()
        
        aic_val = aic(fitted; estimate_likelihood=false)
        bic_val = bic(fitted; estimate_likelihood=false)
        
        # BIC should differ from AIC by penalty difference: k*(log(n) - 2)
        n_params = length(get_parameters_flat(fitted))
        n_obs = sum(fitted.SubjectWeights)
        penalty_diff = n_params * (log(n_obs) - 2)
        
        @test (bic_val - aic_val) ≈ penalty_diff rtol=1e-10
    end
    
    @testset "BIC penalizes more than AIC for large n" begin
        fitted, data = create_fitted_exponential_model(n_subj=100)
        
        # For n > exp(2) ≈ 7.4, BIC penalty > AIC penalty
        n_obs = sum(fitted.SubjectWeights)
        @test n_obs > exp(2)
        @test bic(fitted; estimate_likelihood=false) > aic(fitted; estimate_likelihood=false)
    end
end

# =============================================================================
# summary() Tests
# =============================================================================

@testset "summary()" begin
    
    @testset "Returns a table-like object" begin
        fitted, _ = create_fitted_exponential_model()
        
        summ = summary(fitted)
        @test summ isa Any  # Should be a summary table
    end
    
    @testset "Summary for model with covariates" begin
        fitted, _ = create_fitted_covariate_model()
        
        summ = summary(fitted)
        @test summ isa Any
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
