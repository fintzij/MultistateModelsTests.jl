# =============================================================================
# Test MCEM RNG Reproducibility
# =============================================================================
# Validates that MCEM produces identical results when run with the same seed.
# Part of Phase 5 validation for the MCEM surrogate-agnostic refactor.
# =============================================================================

using MultistateModels
using DataFrames
using Random
using Test

println("=" ^ 60)
println("MCEM RNG Reproducibility Tests")
println("=" ^ 60)

# Create panel data for semi-Markov model
function make_test_data()
    rows = DataFrame[]
    id = 1
    
    # Group A: stay in state 1
    for _ in 1:5
        push!(rows, DataFrame(
            id = fill(id, 2),
            tstart = [0.0, 2.0],
            tstop = [2.0, 4.0],
            statefrom = [1, 1],
            stateto = [1, 1],
            obstype = [2, 2]
        ))
        id += 1
    end
    
    # Group B: 1→2 transition
    for _ in 1:5
        push!(rows, DataFrame(
            id = fill(id, 2),
            tstart = [0.0, 2.0],
            tstop = [2.0, 4.0],
            statefrom = [1, 2],
            stateto = [2, 2],
            obstype = [2, 2]
        ))
        id += 1
    end
    
    return vcat(rows...)
end

# Test 1: Markov Surrogate Reproducibility
println("\n[1/2] Testing Markov surrogate reproducibility...")

dat = make_test_data()
h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)

model1 = multistatemodel(h12; data=dat, surrogate=:markov)
model2 = multistatemodel(h12; data=dat, surrogate=:markov)
set_parameters!(model1, (h12 = [1.2, 0.3],))
set_parameters!(model2, (h12 = [1.2, 0.3],))

Random.seed!(98765)
fitted1 = fit(model1;
    maxiter = 3,
    ess_target_initial = 30,
    tol = 1e-6,
    verbose = false,
    sir = :none,
    vcov_type = :none
)
params1 = get_parameters(fitted1)

Random.seed!(98765)
fitted2 = fit(model2;
    maxiter = 3,
    ess_target_initial = 30,
    tol = 1e-6,
    verbose = false,
    sir = :none,
    vcov_type = :none
)
params2 = get_parameters(fitted2)

println("  params1.h12 = ", params1.h12)
println("  params2.h12 = ", params2.h12)

@testset "Markov surrogate reproducibility" begin
    @test params1.h12 ≈ params2.h12 rtol=1e-12
end
println("  ✓ Markov surrogate test PASSED")

# Test 2: PhaseType Surrogate Reproducibility
println("\n[2/2] Testing PhaseType surrogate reproducibility...")

dat = make_test_data()
h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)

model1 = multistatemodel(h12; data=dat, surrogate=:phasetype)
model2 = multistatemodel(h12; data=dat, surrogate=:phasetype)
set_parameters!(model1, (h12 = [1.2, 0.3],))
set_parameters!(model2, (h12 = [1.2, 0.3],))

Random.seed!(87654)
fitted1 = fit(model1;
    maxiter = 3,
    ess_target_initial = 30,
    tol = 1e-6,
    verbose = false,
    sir = :none,
    vcov_type = :none
)
params1 = get_parameters(fitted1)

Random.seed!(87654)
fitted2 = fit(model2;
    maxiter = 3,
    ess_target_initial = 30,
    tol = 1e-6,
    verbose = false,
    sir = :none,
    vcov_type = :none
)
params2 = get_parameters(fitted2)

println("  params1.h12 = ", params1.h12)
println("  params2.h12 = ", params2.h12)

@testset "PhaseType surrogate reproducibility" begin
    @test params1.h12 ≈ params2.h12 rtol=1e-12
end
println("  ✓ PhaseType surrogate test PASSED")

println("\n" * "=" ^ 60)
println("All MCEM reproducibility tests PASSED")
println("=" ^ 60)
