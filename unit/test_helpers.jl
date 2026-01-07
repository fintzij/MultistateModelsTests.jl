# =============================================================================
# Helper Utility Tests
# =============================================================================
#
# Tests that verify critical algorithmic correctness:
# 1. ForwardDiff compatibility (gradients/Hessians work correctly)
# 2. Batched vs sequential likelihood parity (optimization bugs)
using .TestFixtures
using ForwardDiff

# --- ForwardDiff compatibility -------------------------------------------------
# Critical: If gradients/Hessians are wrong, optimization silently fails
@testset "ForwardDiff compatibility" begin
    using MultistateModels: ExactData, loglik_exact
    
    @testset "gradient computation" begin
        h12 = Hazard(@formula(0 ~ 1 + age), "wei", 1, 2)
        dat = DataFrame(
            id = [1, 2],
            tstart = [0.0, 0.0],
            tstop = [5.0, 7.0],
            statefrom = [1, 1],
            stateto = [2, 2],
            obstype = [1, 1],
            age = [30.0, 50.0]
        )
        model = multistatemodel(h12; data = dat)
        set_parameters!(model, (h12 = [0.1, 1.0, 0.01],))
        paths = MultistateModels.extract_paths(model)
        exact_data = ExactData(model, paths)
        pars = model.parameters.flat
        
        grad = ForwardDiff.gradient(p -> loglik_exact(p, exact_data; neg=false), pars)
        @test length(grad) == length(pars)
        @test all(isfinite.(grad))
    end
    
    @testset "Hessian computation" begin
        h12 = Hazard(@formula(0 ~ 1 + trt), "exp", 1, 2)
        dat = DataFrame(
            id = [1, 2, 3],
            tstart = [0.0, 0.0, 0.0],
            tstop = [4.0, 6.0, 3.0],
            statefrom = [1, 1, 1],
            stateto = [2, 2, 2],
            obstype = [1, 1, 1],
            trt = [0.0, 1.0, 0.0]
        )
        model = multistatemodel(h12; data = dat)
        set_parameters!(model, (h12 = [0.2, 0.3],))
        paths = MultistateModels.extract_paths(model)
        exact_data = ExactData(model, paths)
        pars = model.parameters.flat
        
        hess = ForwardDiff.hessian(p -> loglik_exact(p, exact_data; neg=false), pars)
        @test size(hess) == (length(pars), length(pars))
        @test all(isfinite.(hess))
        # Hessian of log-likelihood should be negative semi-definite at/near MLE
        # (i.e., -hess should be PSD). We check symmetry and eigenvalue structure.
        @test issymmetric(hess)
        # All eigenvalues should be <= 0 for concave log-likelihood
        eigs = eigvals(Symmetric(hess))
        @test all(eigs .<= sqrt(eps()))
    end
end

# --- Batched vs sequential parity ----------------------------------------------
# Critical: Batched optimization must give same answer as sequential
@testset "batched_vs_sequential_parity" begin
    using MultistateModels: SMPanelData, loglik_semi_markov!, loglik_semi_markov_batched!
    
    # Illness-death model tests batched path likelihood
    dat = DataFrame(
        id = [1, 1, 2, 2, 3],
        tstart = [0.0, 3.0, 0.0, 2.0, 0.0],
        tstop = [3.0, 7.0, 2.0, 5.0, 6.0],
        statefrom = [1, 2, 1, 1, 1],
        stateto = [2, 3, 1, 3, 3],
        obstype = [1, 1, 1, 1, 1],
        age = [40.0, 40.0, 50.0, 50.0, 60.0]
    )
    h12 = Hazard(@formula(0 ~ 1 + age), "wei", 1, 2)
    h13 = Hazard(@formula(0 ~ 1), "exp", 1, 3)
    h23 = Hazard(@formula(0 ~ 1 + age), "gom", 2, 3)
    model = multistatemodel(h12, h13, h23; data = dat)
    set_parameters!(model, (
        h12 = [0.1, 1.2, 0.01],
        h13 = [0.05],
        h23 = [0.15, 0.02, 0.01]
    ))
    
    base_paths = MultistateModels.extract_paths(model)
    n_subjects = length(base_paths)
    n_paths = 3
    nested_paths = [[deepcopy(base_paths[i]) for _ in 1:n_paths] for i in 1:n_subjects]
    weights = [ones(n_paths) for _ in 1:n_subjects]
    smpanel = SMPanelData(model, nested_paths, weights)
    pars = model.parameters.flat
    
    logliks_seq = [zeros(n_paths) for _ in 1:n_subjects]
    logliks_bat = [zeros(n_paths) for _ in 1:n_subjects]
    loglik_semi_markov!(pars, logliks_seq, smpanel)
    loglik_semi_markov_batched!(pars, logliks_bat, smpanel)
    
    for i in 1:n_subjects, j in 1:n_paths
        @test isapprox(logliks_seq[i][j], logliks_bat[i][j], rtol=1e-12)
    end
end

# --- Phase 1 Parameter Handling Tests -----------------------------------------
# Tests for NamedTuple parameter structure with named fields

@testset "build_hazard_params - NamedTuple structure" begin
    using MultistateModels: build_hazard_params
    
    @testset "Weibull baseline only" begin
        params = build_hazard_params(
            [log(1.5), log(0.2)],
            [:h12_shape, :h12_scale],
            2,
            2  # npar_total = npar_baseline
        )
        
        # Verify structure and values directly
        @test params isa NamedTuple
        @test params.baseline isa NamedTuple
        @test params.baseline.h12_shape ≈ log(1.5)
        @test params.baseline.h12_scale ≈ log(0.2)
        @test !hasproperty(params, :covariates)  # No covariates
    end
    
    @testset "Weibull with covariates" begin
        params = build_hazard_params(
            [log(1.5), log(0.2), 0.3, 0.1],
            [:h12_shape, :h12_scale, :h12_age, :h12_sex],
            2,
            4  # npar_total
        )
        
        # Verify all values directly (accessing them proves keys exist)
        @test params.baseline.h12_shape ≈ log(1.5)
        @test params.baseline.h12_scale ≈ log(0.2)
        @test params.covariates.h12_age ≈ 0.3
        @test params.covariates.h12_sex ≈ 0.1
    end
    
    @testset "Exponential baseline only" begin
        params = build_hazard_params(
            [log(0.5)],
            [:h13_intercept],
            1,
            1  # npar_total
        )
        
        @test params.baseline.h13_intercept ≈ log(0.5)
        @test !hasproperty(params, :covariates)  # No covariates
    end
    
    @testset "Error on mismatched lengths" begin
        @test_throws AssertionError build_hazard_params(
            [1.0, 2.0],
            [:h12_shape],  # Wrong length!
            2,
            2
        )
    end
    
    @testset "Error on invalid npar_baseline" begin
        @test_throws AssertionError build_hazard_params(
            [1.0, 2.0],
            [:h12_shape, :h12_scale],
            3,  # More baseline params than total!
            2
        )
    end
end

@testset "Parameter extraction helpers" begin
    using MultistateModels: extract_baseline_values, extract_covariate_values, 
                            extract_params_vector, extract_natural_vector
    
    # v0.3.0+: Parameters on natural scale
    params_with_covars = (
        baseline = (h12_shape = 1.5, h12_scale = 0.2),
        covariates = (h12_age = 0.3, h12_sex = 0.1)
    )
    
    params_no_covars = (
        baseline = (h13_intercept = 0.8,),
    )
    
    @testset "extract_baseline_values" begin
        baseline_vals = extract_baseline_values(params_with_covars)
        @test baseline_vals ≈ [1.5, 0.2]
        @test baseline_vals isa Vector{Float64}
        
        baseline_vals_single = extract_baseline_values(params_no_covars)
        @test baseline_vals_single ≈ [0.8]
    end
    
    @testset "extract_covariate_values" begin
        covar_vals = extract_covariate_values(params_with_covars)
        @test covar_vals ≈ [0.3, 0.1]
        @test covar_vals isa Vector{Float64}
        
        covar_vals_empty = extract_covariate_values(params_no_covars)
        @test isempty(covar_vals_empty)
        @test covar_vals_empty isa Vector{Float64}
    end
    
    @testset "extract_params_vector" begin
        all_params = extract_params_vector(params_with_covars)
        @test all_params ≈ [1.5, 0.2, 0.3, 0.1]
        
        baseline_only = extract_params_vector(params_no_covars)
        @test baseline_only ≈ [0.8]
    end
    
    @testset "extract_natural_vector" begin
        # v0.3.0+: All params on natural scale, no transformation needed
        # params_with_covars is for a Weibull-style hazard (shape, scale)
        natural_vals = extract_natural_vector(params_with_covars, :wei)
        @test natural_vals ≈ [1.5, 0.2, 0.3, 0.1]  # all unchanged (already natural scale)
        
        # params_no_covars is for an exponential-style hazard (intercept only)
        natural_baseline = extract_natural_vector(params_no_covars, :exp)
        @test natural_baseline ≈ [0.8]  # unchanged
        
        # Test Gompertz: shape unconstrained (can be negative), rate positive
        params_gom = (
            baseline = (h12_shape = -0.5, h12_rate = 0.2),
            covariates = (h12_age = 0.1,)
        )
        natural_gom = extract_natural_vector(params_gom, :gom)
        @test natural_gom ≈ [-0.5, 0.2, 0.1]  # all unchanged
    end
end

@testset "ParameterHandling.jl with nested NamedTuples" begin
    using ParameterHandling
    using MultistateModels: build_hazard_params
    
    @testset "Flatten and unflatten with named fields" begin
        # Build parameter structure with named NamedTuples
        params = (
            h12 = build_hazard_params(
                [log(1.5), log(0.2), 0.3, 0.1],
                [:h12_shape, :h12_scale, :h12_age, :h12_sex],
                2,
                4  # npar_total
            ),
            h23 = build_hazard_params(
                [log(0.8)],
                [:h23_intercept],
                1,
                1  # npar_total
            )
        )
        
        # Flatten and unflatten
        flat, unflatten = ParameterHandling.flatten(params)
        reconstructed = unflatten(flat)
        
        # Verify structure preserved
        @test reconstructed.h12.baseline.h12_shape ≈ log(1.5)
        @test reconstructed.h12.baseline.h12_scale ≈ log(0.2)
        @test reconstructed.h12.covariates.h12_age ≈ 0.3
        @test reconstructed.h12.covariates.h12_sex ≈ 0.1
        @test reconstructed.h23.baseline.h23_intercept ≈ log(0.8)
        
        # Test modification (as in optimization)
        modified_flat = flat .+ 0.1
        modified = unflatten(modified_flat)
        @test modified.h12.baseline.h12_shape ≈ log(1.5) + 0.1
        @test modified.h12.baseline.h12_scale ≈ log(0.2) + 0.1
        @test modified.h12.covariates.h12_age ≈ 0.3 + 0.1
        @test modified.h23.baseline.h23_intercept ≈ log(0.8) + 0.1
    end
    
    @testset "Named access works correctly" begin
        params = (
            h12 = (
                baseline = (h12_shape = log(2.0), h12_scale = log(1.0)),
                covariates = (h12_trt = 0.5,)
            ),
        )
        
        # Test that we can access by name
        @test params.h12.baseline.h12_shape == log(2.0)
        @test params.h12.baseline.h12_scale == log(1.0)
        @test params.h12.covariates.h12_trt == 0.5
        
        # Flatten/unflatten preserves named access
        flat, unflatten = ParameterHandling.flatten(params)
        restored = unflatten(flat)
        @test restored.h12.baseline.h12_shape == log(2.0)
        @test restored.h12.covariates.h12_trt == 0.5
    end
end
# --- Parameter Transformation Tests --------------------------------------------
# v0.3.0+: All parameters stored on natural scale, transform functions are identity

@testset "parameter_transformations" begin
    using MultistateModels: transform_baseline_to_natural, transform_baseline_to_estimation
    
    @testset "any family works (identity transform)" begin
        # v0.3.0+: Parameters stored on natural scale, transformations are identity
        # The functions don't throw for unknown families anymore
        baseline = (h12_Intercept = 0.5,)
        
        # All families now return identity (parameters already on natural scale)
        # These should NOT throw - they're all identity transforms
        nat_exp = transform_baseline_to_natural(baseline, :exp, Float64)
        @test nat_exp.h12_Intercept ≈ 0.5
        
        nat_wei = transform_baseline_to_natural(baseline, :wei, Float64)
        @test nat_wei.h12_Intercept ≈ 0.5
        
        nat_unknown = transform_baseline_to_natural(baseline, :unknown, Float64)
        @test nat_unknown.h12_Intercept ≈ 0.5
    end
    
    @testset "known families work correctly (identity)" begin
        # v0.3.0+: All transforms are identity - params already on natural scale
        baseline = (h12_Intercept = 0.5,)
        
        # Exponential: identity
        nat_exp = transform_baseline_to_natural(baseline, :exp, Float64)
        @test nat_exp.h12_Intercept ≈ 0.5
        
        # Weibull: identity (params already natural scale)
        baseline_wei = (h12_shape = 1.2, h12_scale = 0.3)
        nat_wei = transform_baseline_to_natural(baseline_wei, :wei, Float64)
        @test nat_wei.h12_shape ≈ 1.2  # identity
        @test nat_wei.h12_scale ≈ 0.3  # identity
        
        # Gompertz: identity (shape can be negative, rate positive)
        baseline_gom = (h12_shape = -0.5, h12_rate = 0.2)
        nat_gom = transform_baseline_to_natural(baseline_gom, :gom, Float64)
        @test nat_gom.h12_shape ≈ -0.5  # identity
        @test nat_gom.h12_rate ≈ 0.2  # identity
        
        # Spline: identity
        baseline_sp = (h12_b1 = 0.1, h12_b2 = 0.2)
        nat_sp = transform_baseline_to_natural(baseline_sp, :sp, Float64)
        @test nat_sp.h12_b1 ≈ 0.1  # identity
        @test nat_sp.h12_b2 ≈ 0.2  # identity
    end
    
    @testset "round-trip transformations" begin
        # v0.3.0+: Both transforms are identity, so round-trip is always exact
        
        # Exponential
        baseline_exp = (h12_Intercept = 0.5,)
        nat = transform_baseline_to_natural(baseline_exp, :exp, Float64)
        back = transform_baseline_to_estimation(nat, :exp)
        @test back.h12_Intercept ≈ baseline_exp.h12_Intercept rtol=1e-10
        
        # Weibull (shape and scale both positive)
        baseline_wei = (h12_shape = 1.5, h12_scale = 0.3)
        nat = transform_baseline_to_natural(baseline_wei, :wei, Float64)
        back = transform_baseline_to_estimation(nat, :wei)
        @test back.h12_shape ≈ baseline_wei.h12_shape rtol=1e-10
        @test back.h12_scale ≈ baseline_wei.h12_scale rtol=1e-10
        
        # Gompertz (shape can be negative, rate positive)
        baseline_gom = (h12_shape = -0.3, h12_rate = 0.2)
        nat = transform_baseline_to_natural(baseline_gom, :gom, Float64)
        back = transform_baseline_to_estimation(nat, :gom)
        @test back.h12_shape ≈ baseline_gom.h12_shape rtol=1e-10
        @test back.h12_rate ≈ baseline_gom.h12_rate rtol=1e-10
    end
end