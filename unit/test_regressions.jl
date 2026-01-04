# =============================================================================
# Regression Tests for MultistateModels.jl
# =============================================================================
#
# This file contains regression tests for bugs that have been fixed.
# Each test should document:
#   1. The original bug (issue number if applicable)
#   2. What was broken
#   3. The minimal reproducer
#   4. Expected behavior after fix
#
# Add new regression tests as bugs are fixed to prevent regressions.
# =============================================================================

using Test
using MultistateModels
using DataFrames
using StatsModels

# =============================================================================
# Template for adding new regression tests
# =============================================================================
#
# @testset "Issue #XXX: Brief description" begin
#     # Description: [What the bug was]
#     # Fixed in: [Commit or PR]
#     # Root cause: [Why it happened]
#     
#     # Minimal reproducer that used to fail:
#     # ... setup code ...
#     
#     # Assertion that now passes:
#     @test <condition>
# end
# =============================================================================

@testset "Regression Tests" begin
    
    # =========================================================================
    # Regression: compute_hazard() API parameter structure mismatch (Fixed Jan 3, 2026)
    # =========================================================================
    @testset "Regression: compute_hazard() API works with public interface" begin
        # Description: compute_hazard() and compute_cumulative_hazard() were passing
        #   a plain Vector to internal eval_hazard()/eval_cumhaz(), which expected
        #   a NamedTuple with .baseline field.
        # Fixed in: api.jl - changed get_parameters() to get_hazard_params()
        # Root cause: get_parameters(model, hazind, scale=:log) returns Vector,
        #   but eval_hazard() expects NamedTuple from get_hazard_params()
        
        n = 10
        dat = DataFrame(
            id = 1:n,
            tstart = zeros(n),
            tstop = fill(5.0, n),
            statefrom = ones(Int, n),
            stateto = fill(2, n),
            obstype = fill(1, n)
        )
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        model = multistatemodel(h12; data=dat)
        
        λ = 0.3
        set_parameters!(model, (h12 = [log(λ)],))
        
        # This used to throw: FieldError: type Array has no field 'baseline'
        hazards = compute_hazard([1.0, 2.0], model, :h12)
        @test all(isapprox.(hazards, λ; rtol=1e-6))
        
        # This also used to fail
        cumhaz = compute_cumulative_hazard([0.0], [5.0], model, :h12)
        @test isapprox(cumhaz[1], λ * 5.0; rtol=1e-6)
    end
    
    # Placeholder test - replace with actual regression tests as bugs are fixed
    @testset "Template: basic model construction works" begin
        # This is a placeholder to ensure the regression test file runs
        # Replace with actual regression tests as bugs are identified and fixed
        
        n = 10
        dat = DataFrame(
            id = 1:n,
            tstart = zeros(n),
            tstop = fill(5.0, n),
            statefrom = ones(Int, n),
            stateto = fill(2, n),
            obstype = fill(1, n)
        )
        
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        model = multistatemodel(h12; data=dat)
        
        @test model isa MultistateModels.MultistateModel
        @test length(model.hazards) == 1
    end
    
end
