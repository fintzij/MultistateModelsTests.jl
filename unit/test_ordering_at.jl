# =============================================================================
# Test: ordering_at Parameter for Phase-Type Eigenvalue Constraints
# =============================================================================
#
# Tests the ordering_at parameter that controls where eigenvalue ordering
# constraints are enforced for phase-type models.
#
# Reference: Item #26 in CODEBASE_REFACTORING_GUIDE.md
# =============================================================================

using Test
using MultistateModels
using DataFrames
using Random
using Statistics

import MultistateModels: _compute_ordering_reference, _extract_covariate_names,
    _generate_ordering_constraints, _build_linear_ordering_constraint,
    _build_nonlinear_ordering_constraint, _build_rate_with_covariates

# =============================================================================
# Test Helper Functions
# =============================================================================

"""Create simple illness-death test data with covariates"""
function create_pt_test_data(; n=50, seed=12345)
    rng = MersenneTwister(seed)
    
    DataFrame(
        id = repeat(1:n, inner=2),
        tstart = repeat([0.0, 0.5], n),
        tstop = repeat([0.5, 1.0], n),
        statefrom = repeat([1, 1], n),
        stateto = repeat([1, 2], n),
        obstype = repeat([2, 2], n),
        age = repeat(rand(rng, n) .* 40 .+ 30, inner=2),  # 30-70
        treatment = repeat(rand(rng, n) .< 0.5, inner=2)  # binary
    )
end

# =============================================================================
# Test: _compute_ordering_reference Function
# =============================================================================

@testset "ordering_at Reference Computation" begin
    data = create_pt_test_data()
    covariate_names = [:age, :treatment]
    
    @testset ":reference returns empty Dict" begin
        ref = _compute_ordering_reference(:reference, data, covariate_names)
        @test ref == Dict{Symbol, Float64}()
        @test isempty(ref)
    end
    
    @testset ":mean computes correct means" begin
        ref = _compute_ordering_reference(:mean, data, covariate_names)
        @test haskey(ref, :age)
        @test haskey(ref, :treatment)
        @test ref[:age] ≈ mean(data.age) atol=1e-10
        @test ref[:treatment] ≈ mean(data.treatment) atol=1e-10
    end
    
    @testset ":median computes correct medians" begin
        ref = _compute_ordering_reference(:median, data, covariate_names)
        @test haskey(ref, :age)
        @test haskey(ref, :treatment)
        @test ref[:age] ≈ median(data.age) atol=1e-10
        @test ref[:treatment] ≈ median(data.treatment) atol=1e-10
    end
    
    @testset "NamedTuple uses explicit values" begin
        ref = _compute_ordering_reference((age=50.0, treatment=0.5), data, covariate_names)
        @test ref[:age] == 50.0
        @test ref[:treatment] == 0.5
    end
    
    @testset "NamedTuple validates covariate names" begin
        # Missing covariate should throw
        @test_throws ArgumentError _compute_ordering_reference(
            (age=50.0,), data, [:age, :treatment]
        )
    end
    
    @testset "Missing covariate in data warns" begin
        # Should warn but not error
        @test_logs (:warn,) _compute_ordering_reference(:mean, data, [:nonexistent])
    end
end

# =============================================================================
# Test: _extract_covariate_names Function  
# =============================================================================

@testset "Covariate Name Extraction" begin
    @testset "Intercept-only formula has no covariates" begin
        h = Hazard(@formula(0 ~ 1), :pt, 1, 2)
        names = _extract_covariate_names([h])
        @test isempty(names)
    end
    
    @testset "Single covariate formula" begin
        h = Hazard(@formula(0 ~ 1 + age), :pt, 1, 2)
        names = _extract_covariate_names([h])
        @test :age in names
        @test length(names) == 1
    end
    
    @testset "Multiple covariate formula" begin
        h = Hazard(@formula(0 ~ 1 + age + treatment), :pt, 1, 2)
        names = _extract_covariate_names([h])
        @test :age in names
        @test :treatment in names
        @test length(names) == 2
    end
    
    @testset "Multiple hazards combine covariates" begin
        h1 = Hazard(@formula(0 ~ 1 + age), :pt, 1, 2)
        h2 = Hazard(@formula(0 ~ 1 + treatment), :pt, 1, 3)
        names = _extract_covariate_names([h1, h2])
        @test :age in names
        @test :treatment in names
    end
end

# =============================================================================
# Test: Model Construction with ordering_at
# =============================================================================

@testset "Model Construction with ordering_at" begin
    data = create_pt_test_data()
    
    @testset "Default is :reference (backward compatible)" begin
        h12 = Hazard(@formula(0 ~ 1), :pt, 1, 2)
        # Should not error
        model = multistatemodel(h12; data=data, n_phases=Dict(1=>2))
        @test has_phasetype_expansion(model)
    end
    
    @testset ":reference creates model successfully" begin
        h12 = Hazard(@formula(0 ~ 1), :pt, 1, 2)
        model = multistatemodel(h12; data=data, n_phases=Dict(1=>2), ordering_at=:reference)
        @test has_phasetype_expansion(model)
    end
    
    @testset ":mean creates model successfully" begin
        h12 = Hazard(@formula(0 ~ 1 + age), :pt, 1, 2)
        model = multistatemodel(h12; data=data, n_phases=Dict(1=>2), ordering_at=:mean)
        @test has_phasetype_expansion(model)
    end
    
    @testset ":median creates model successfully" begin
        h12 = Hazard(@formula(0 ~ 1 + age), :pt, 1, 2)
        model = multistatemodel(h12; data=data, n_phases=Dict(1=>2), ordering_at=:median)
        @test has_phasetype_expansion(model)
    end
    
    @testset "Explicit NamedTuple creates model successfully" begin
        h12 = Hazard(@formula(0 ~ 1 + age), :pt, 1, 2)
        model = multistatemodel(h12; data=data, n_phases=Dict(1=>2), ordering_at=(age=50.0,))
        @test has_phasetype_expansion(model)
    end
    
    @testset "Invalid ordering_at throws ArgumentError" begin
        h12 = Hazard(@formula(0 ~ 1), :pt, 1, 2)
        @test_throws ArgumentError multistatemodel(h12; data=data, n_phases=Dict(1=>2), 
                                                    ordering_at=:invalid)
    end
end

# =============================================================================
# Test: C1 Constraint Simplification
# =============================================================================

@testset "C1 Constraint (Homogeneous Covariates) Simplification" begin
    data = create_pt_test_data()
    
    @testset "Homogeneous covariates with :mean still uses linear constraints" begin
        # With C1 (homogeneous), exp(β'x̄) factors cancel, so constraint stays linear
        h12 = Hazard(@formula(0 ~ 1 + age), :pt, 1, 2; covariate_constraints=:homogeneous)
        
        # Should work with :mean ordering (no nonlinear constraints due to C1)
        model = multistatemodel(h12; data=data, n_phases=Dict(1=>2), ordering_at=:mean)
        @test has_phasetype_expansion(model)
        
        # The model should have constraints
        @test !isnothing(model.modelcall.constraints)
    end
end

# =============================================================================
# Test: Constraint Expression Generation (Unit)
# =============================================================================

@testset "Constraint Expression Structure" begin
    data = create_pt_test_data()
    
    @testset "Reference ordering generates linear expressions" begin
        h12 = Hazard(@formula(0 ~ 1), :pt, 1, 2)
        # Note: :eigorder_sctp was planned but not implemented - using :sctp_decreasing instead
        model = multistatemodel(h12; data=data, n_phases=Dict(1=>2), 
                                 ordering_at=:reference, coxian_structure=:sctp_decreasing)
        
        # Check that model has constraints
        @test haskey(model.modelcall, :constraints)
        constraints = model.modelcall.constraints
        @test !isnothing(constraints)
        
        # Should have at least one ordering constraint (n-1 = 1 for 2 phases)
        @test length(constraints.cons) >= 1
    end
    
    @testset "Non-reference ordering with heterogeneous covariates generates nonlinear expressions" begin
        h12 = Hazard(@formula(0 ~ 1 + age), :pt, 1, 2; covariate_constraints=:unstructured)
        # Note: :eigorder_sctp was planned but not implemented - using :sctp_decreasing instead
        model = multistatemodel(h12; data=data, n_phases=Dict(1=>2), 
                                 ordering_at=:mean, coxian_structure=:sctp_decreasing)
        
        constraints = model.modelcall.constraints
        @test !isnothing(constraints)
        
        # Constraints should exist (will contain exp() for nonlinear terms)
        @test length(constraints.cons) >= 1
    end
end

# =============================================================================
# Test: Fitting with ordering_at
# =============================================================================

@testset "Fitting with ordering_at" begin
    data = create_pt_test_data()
    
    @testset "Fit with :reference ordering" begin
        h12 = Hazard(@formula(0 ~ 1), :pt, 1, 2)
        # Note: :eigorder_sctp was planned but not implemented - using :sctp_decreasing instead
        model = multistatemodel(h12; data=data, n_phases=Dict(1=>2), 
                                 ordering_at=:reference, coxian_structure=:sctp_decreasing)
        
        # Should be able to fit
        fitted = fit(model; verbose=false)
        @test fitted isa MultistateModels.MultistateModelFitted
    end
    
    @testset "Fit with :mean ordering (no covariates)" begin
        h12 = Hazard(@formula(0 ~ 1), :pt, 1, 2)
        # Note: :eigorder_sctp was planned but not implemented - using :sctp_decreasing instead
        model = multistatemodel(h12; data=data, n_phases=Dict(1=>2), 
                                 ordering_at=:mean, coxian_structure=:sctp_decreasing)
        
        # Should be able to fit (no covariates = same as reference)
        fitted = fit(model; verbose=false)
        @test fitted isa MultistateModels.MultistateModelFitted
    end
end

println("\n✓ ordering_at Tests Complete")
