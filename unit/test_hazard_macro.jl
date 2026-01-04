# Unit tests for @hazard macro
#
# The @hazard macro provides a declarative front-end for creating Hazard objects.
# Tests verify:
# 1. All family aliases work correctly (exp/exponential, wei/weibull, etc.)
# 2. Transition specification syntax is parsed correctly
# 3. Formula specification works
# 4. Keyword arguments are forwarded
# 5. Equivalence to explicit Hazard() constructor
# 6. Error handling for invalid inputs

using Test
using MultistateModels
using DataFrames
using StatsModels

import MultistateModels: _hazard_macro_entry, _ANALYTIC_HAZARD_FAMILY_ALIASES, HazardFunction

# =============================================================================
# Family Alias Tests
# =============================================================================

@testset "Family Aliases" begin
    
    @testset "Exponential aliases" begin
        h1 = @hazard begin
            family = :exp
            statefrom = 1
            stateto = 2
        end
        h2 = @hazard begin
            family = :exponential
            statefrom = 1
            stateto = 2
        end
        
        @test h1 isa HazardFunction
        @test h2 isa HazardFunction
    end
    
    @testset "Weibull aliases" begin
        h1 = @hazard begin
            family = :wei
            statefrom = 1
            stateto = 2
        end
        h2 = @hazard begin
            family = :weibull
            statefrom = 1
            stateto = 2
        end
        
        @test h1 isa HazardFunction
        @test h2 isa HazardFunction
    end
    
    @testset "Gompertz aliases" begin
        h1 = @hazard begin
            family = :gom
            statefrom = 1
            stateto = 2
        end
        h2 = @hazard begin
            family = :gompertz
            statefrom = 1
            stateto = 2
        end
        
        @test h1 isa HazardFunction
        @test h2 isa HazardFunction
    end
    
    @testset "Spline aliases" begin
        h1 = @hazard begin
            family = :sp
            statefrom = 1
            stateto = 2
        end
        h2 = @hazard begin
            family = :spline
            statefrom = 1
            stateto = 2
        end
        
        @test h1 isa HazardFunction
        @test h2 isa HazardFunction
    end
    
    @testset "Phase-type aliases" begin
        h1 = @hazard begin
            family = :pt
            statefrom = 1
            stateto = 2
        end
        h2 = @hazard begin
            family = :phasetype
            statefrom = 1
            stateto = 2
        end
        
        @test h1 isa HazardFunction
        @test h2 isa HazardFunction
    end
    
    @testset "Alias map is correct" begin
        @test haskey(_ANALYTIC_HAZARD_FAMILY_ALIASES, :exp)
        @test haskey(_ANALYTIC_HAZARD_FAMILY_ALIASES, :wei)
        @test haskey(_ANALYTIC_HAZARD_FAMILY_ALIASES, :gom)
        @test haskey(_ANALYTIC_HAZARD_FAMILY_ALIASES, :sp)
        @test haskey(_ANALYTIC_HAZARD_FAMILY_ALIASES, :pt)
    end
end

# =============================================================================
# Transition Syntax Tests
# =============================================================================

@testset "Transition Syntax" begin
    
    @testset "statefrom/stateto specification" begin
        h = @hazard begin
            family = :exp
            statefrom = 1
            stateto = 2
        end
        @test h isa HazardFunction
    end
    
    @testset "from/to aliases" begin
        h = @hazard begin
            family = :exp
            from = 1
            to = 2
        end
        @test h isa HazardFunction
    end
    
    @testset "transition pair" begin
        h = @hazard begin
            family = :exp
            transition = 1 => 2
        end
        @test h isa HazardFunction
    end
    
    @testset "Multiple transitions" begin
        h12 = @hazard begin
            family = :exp
            statefrom = 1
            stateto = 2
        end
        h13 = @hazard begin
            family = :exp
            statefrom = 1
            stateto = 3
        end
        h23 = @hazard begin
            family = :exp
            statefrom = 2
            stateto = 3
        end
        
        @test h12 isa HazardFunction
        @test h13 isa HazardFunction
        @test h23 isa HazardFunction
    end
end

# =============================================================================
# Formula Specification Tests
# =============================================================================

@testset "Formula Specification" begin
    
    @testset "Intercept-only formula (default)" begin
        h = @hazard begin
            family = :exp
            statefrom = 1
            stateto = 2
        end
        @test h isa HazardFunction
    end
    
    @testset "Explicit intercept-only formula" begin
        h = @hazard begin
            family = :exp
            statefrom = 1
            stateto = 2
            formula = @formula(0 ~ 1)
        end
        @test h isa HazardFunction
    end
    
    @testset "Single covariate formula" begin
        h = @hazard begin
            family = :exp
            statefrom = 1
            stateto = 2
            formula = @formula(0 ~ x)
        end
        @test h isa HazardFunction
    end
    
    @testset "Multiple covariate formula" begin
        h = @hazard begin
            family = :exp
            statefrom = 1
            stateto = 2
            formula = @formula(0 ~ x + y)
        end
        @test h isa HazardFunction
    end
end

# =============================================================================
# Keyword Forwarding Tests
# =============================================================================

@testset "Keyword Forwarding" begin
    
    @testset "Spline with knots" begin
        h = @hazard begin
            family = :sp
            statefrom = 1
            stateto = 2
            knots = [1.0, 2.0, 3.0]
        end
        @test h isa HazardFunction
    end
    
    @testset "Phase-type with n_phases" begin
        h = @hazard begin
            family = :pt
            statefrom = 1
            stateto = 2
            n_phases = 3
        end
        @test h isa HazardFunction
    end
    
    @testset "Effect type specification" begin
        h = @hazard begin
            family = :wei
            statefrom = 1
            stateto = 2
            formula = @formula(0 ~ x)
            linpred_effect = :aft
        end
        @test h isa HazardFunction
    end
end

# =============================================================================
# Equivalence to Hazard() Constructor Tests
# =============================================================================

@testset "Equivalence to Hazard()" begin
    
    @testset "Exponential equivalence" begin
        h_macro = @hazard begin
            family = :exp
            statefrom = 1
            stateto = 2
        end
        h_explicit = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        
        @test typeof(h_macro) == typeof(h_explicit)
    end
    
    @testset "Weibull equivalence" begin
        h_macro = @hazard begin
            family = :wei
            statefrom = 1
            stateto = 2
        end
        h_explicit = Hazard(@formula(0 ~ 1), "wei", 1, 2)
        
        @test typeof(h_macro) == typeof(h_explicit)
    end
    
    @testset "Both can be used in model construction" begin
        n = 50
        dat = DataFrame(
            id = 1:n,
            tstart = zeros(n),
            tstop = fill(5.0, n),
            statefrom = ones(Int, n),
            stateto = fill(2, n),
            obstype = fill(1, n)
        )
        
        h_macro = @hazard begin
            family = :exp
            statefrom = 1
            stateto = 2
        end
        h_explicit = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        
        model_macro = multistatemodel(h_macro; data=dat)
        model_explicit = multistatemodel(h_explicit; data=dat)
        
        @test model_macro isa MultistateModels.MultistateProcess
        @test model_explicit isa MultistateModels.MultistateProcess
    end
end

# =============================================================================
# Error Handling Tests
# =============================================================================

@testset "Error Handling" begin
    
    @testset "Missing family throws error" begin
        @test_throws ArgumentError @hazard begin
            statefrom = 1
            stateto = 2
        end
    end
    
    @testset "Invalid family throws error" begin
        @test_throws ArgumentError @hazard begin
            family = :invalid_family
            statefrom = 1
            stateto = 2
        end
    end
    
    @testset "Missing transition info throws error" begin
        @test_throws ArgumentError @hazard begin
            family = :exp
        end
    end
end

# =============================================================================
# Integration Tests: Multi-transition Models
# =============================================================================

@testset "Multi-transition Models" begin
    
    @testset "Three-state illness-death model" begin
        n = 100
        dat = DataFrame(
            id = 1:n,
            tstart = zeros(n),
            tstop = fill(10.0, n),
            statefrom = ones(Int, n),
            stateto = ones(Int, n),
            obstype = fill(1, n)
        )
        
        h12 = @hazard begin
            family = :exp
            transition = 1 => 2
        end
        h13 = @hazard begin
            family = :exp
            transition = 1 => 3
        end
        h23 = @hazard begin
            family = :exp
            transition = 2 => 3
        end
        
        model = multistatemodel(h12, h13, h23; data=dat)
        @test model isa MultistateModels.MultistateProcess
    end
    
    @testset "Mixed hazard families" begin
        n = 100
        dat = DataFrame(
            id = 1:n,
            tstart = zeros(n),
            tstop = fill(10.0, n),
            statefrom = ones(Int, n),
            stateto = ones(Int, n),
            obstype = fill(1, n)
        )
        
        h12 = @hazard begin
            family = :exp
            transition = 1 => 2
        end
        h13 = @hazard begin
            family = :wei
            transition = 1 => 3
        end
        
        model = multistatemodel(h12, h13; data=dat)
        @test model isa MultistateModels.MultistateProcess
    end
end

# =============================================================================
# Macro Hygiene Tests
# =============================================================================

@testset "Macro Hygiene" begin
    
    @testset "No variable capture issues" begin
        exp = 42  # Shadow the "exp" alias with a local variable
        
        # This should still work because macro uses symbol lookup
        h = @hazard begin
            family = :exponential
            statefrom = 1
            stateto = 2
        end
        @test h isa HazardFunction
        
        # Verify local variable wasn't affected
        @test exp == 42
    end
end
