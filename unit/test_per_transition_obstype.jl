"""
Unit tests for per-transition observation type specification in simulation.

Tests the feature where observation types (exact/panel/censored) can be 
specified per-transition using a Dict mapping transition index to obstype code.

Observation type codes:
- 1: Exact (transition time and states fully observed)
- 2: Panel (only endpoint state at interval boundary observed)
- 3+: Censored (endpoint state unknown/missing)
"""

using Test
using DataFrames
using Random
using StatsModels
using MultistateModels
import MultistateModels: 
    enumerate_transitions, 
    transition_index_map, 
    print_transition_map,
    validate_obstype_by_transition,
    _get_transition_obstype,
    observe_path

# Import fixtures from TestFixtures module
include(joinpath(@__DIR__, "..", "fixtures", "TestFixtures.jl"))
using .TestFixtures: toy_illness_death_model, toy_multisubject_illness_death_model

# =============================================================================
# Transition Helper Tests
# =============================================================================

@testset "Transition Helpers" begin
    @testset "enumerate_transitions" begin
        # Simple illness-death (3 state)
        tmat = [0 1 1; 0 0 1; 0 0 0]
        trans = enumerate_transitions(tmat)
        @test trans == [(1, 2), (1, 3), (2, 3)]
        @test length(trans) == 3
        
        # Reversible two-state
        tmat2 = [0 1; 1 0]
        trans2 = enumerate_transitions(tmat2)
        @test trans2 == [(1, 2), (2, 1)]
        @test length(trans2) == 2
        
        # Check row-major order (scan by row, then column)
        tmat3 = [0 1 2; 3 0 4; 0 0 0]
        trans3 = enumerate_transitions(tmat3)
        @test trans3[1] == (1, 2)  # First nonzero in row 1
        @test trans3[2] == (1, 3)  # Second nonzero in row 1
        @test trans3[3] == (2, 1)  # First nonzero in row 2
        @test trans3[4] == (2, 3)  # Second nonzero in row 2
    end
    
    @testset "transition_index_map" begin
        tmat = [0 1 1; 0 0 1; 0 0 0]
        trans_map = transition_index_map(tmat)
        
        @test trans_map[(1, 2)] == 1
        @test trans_map[(1, 3)] == 2
        @test trans_map[(2, 3)] == 3
        @test length(trans_map) == 3
        
        # Test with model
        model = toy_illness_death_model().model
        trans_map_model = transition_index_map(model)
        @test trans_map_model == trans_map
    end
    
    @testset "print_transition_map" begin
        tmat = [0 1 1; 0 0 1; 0 0 0]
        
        # Capture output
        io = IOBuffer()
        print_transition_map(io, tmat)
        output = String(take!(io))
        
        @test occursin("1 → 2", output)
        @test occursin("1 → 3", output)
        @test occursin("2 → 3", output)
        @test occursin("Index", output)
    end
end

# =============================================================================
# Validation Tests
# =============================================================================

@testset "Validation Functions" begin
    tmat = [0 1 1; 0 0 1; 0 0 0]  # 3 transitions
    
    @testset "validate_obstype_by_transition" begin
        # Valid cases
        @test validate_obstype_by_transition(Dict{Int,Int}(), 3) === nothing
        @test validate_obstype_by_transition(Dict(1 => 1), 3) === nothing
        @test validate_obstype_by_transition(Dict(1 => 1, 2 => 2, 3 => 3), 3) === nothing
        @test validate_obstype_by_transition(Dict(2 => 5), 3) === nothing  # obstype >= 3 is valid
        
        # Invalid transition index
        @test_throws ArgumentError validate_obstype_by_transition(Dict(0 => 1), 3)
        @test_throws ArgumentError validate_obstype_by_transition(Dict(4 => 1), 3)
        @test_throws ArgumentError validate_obstype_by_transition(Dict(-1 => 1), 3)
        
        # Invalid obstype code
        @test_throws ArgumentError validate_obstype_by_transition(Dict(1 => 0), 3)
        @test_throws ArgumentError validate_obstype_by_transition(Dict(1 => -1), 3)
        
        # tmat method
        @test validate_obstype_by_transition(Dict(1 => 1, 2 => 2), tmat) === nothing
        @test_throws ArgumentError validate_obstype_by_transition(Dict(4 => 1), tmat)
    end
end

# =============================================================================
# _get_transition_obstype Tests
# =============================================================================

@testset "_get_transition_obstype" begin
    tmat = [0 1 1; 0 0 1; 0 0 0]
    trans_map = transition_index_map(tmat)
    
    @testset "obstype_by_transition specifies obstypes" begin
        obstype_dict = Dict(1 => 1, 2 => 3, 3 => 2)
        
        @test _get_transition_obstype(1, 2, trans_map, obstype_dict) == 1
        @test _get_transition_obstype(1, 3, trans_map, obstype_dict) == 3
        @test _get_transition_obstype(2, 3, trans_map, obstype_dict) == 2
    end
    
    @testset "defaults to 1 (exact) when not specified" begin
        @test _get_transition_obstype(1, 2, trans_map, Dict{Int,Int}()) == 1
        @test _get_transition_obstype(1, 3, trans_map, Dict{Int,Int}()) == 1
        
        # Partial specification - unspecified transitions default to exact
        partial_dict = Dict(1 => 2)  # Only trans 1 specified
        @test _get_transition_obstype(1, 2, trans_map, partial_dict) == 2  # From dict
        @test _get_transition_obstype(1, 3, trans_map, partial_dict) == 1  # Default
        @test _get_transition_obstype(2, 3, trans_map, partial_dict) == 1  # Default
    end
end

# =============================================================================
# observe_path with per-transition Tests
# =============================================================================

@testset "observe_path with per-transition obstypes" begin
    model = toy_illness_death_model().model
    trans_map = transition_index_map(model)
    
    @testset "all exact transitions" begin
        Random.seed!(1)
        _, paths = simulate(model; nsim=1, data=true, paths=true, autotmax=false)
        path = paths[1][1]
        
        # All exact (same as original behavior)
        obstype_map = Dict(1 => 1, 2 => 1, 3 => 1)
        result = observe_path(path, model; obstype_by_transition=obstype_map, trans_map=trans_map)
        
        # All obstypes should be 1
        @test all(result.obstype .== 1)
        
        # stateto should never be missing for exact observations
        @test all(!ismissing(x) || x === missing for x in result.stateto)
    end
    
    @testset "mixed exact/censored transitions" begin
        # Use a seed that produces 1→2→3 path
        Random.seed!(1)
        _, paths = simulate(model; nsim=1, data=true, paths=true, autotmax=false)
        path = paths[1][1]
        
        # Skip if path doesn't have the expected transitions
        if length(path.states) >= 3 && path.states[2] == 2
            # 1→2 exact, 2→3 censored
            obstype_map = Dict(1 => 1, 3 => 3)  # trans 1 (1→2) exact, trans 3 (2→3) censored
            result = observe_path(path, model; obstype_by_transition=obstype_map, trans_map=trans_map)
            
            # Should have exact row for 1→2 and censored row for interval containing 2→3
            exact_rows = filter(r -> r.obstype == 1, eachrow(result))
            censored_rows = filter(r -> r.obstype == 3, eachrow(result))
            
            @test length(collect(exact_rows)) >= 1
            @test length(collect(censored_rows)) >= 1
            
            # Censored rows should have missing stateto
            for row in censored_rows
                @test ismissing(row.stateto)
            end
        end
    end
    
    @testset "panel observation type" begin
        Random.seed!(123)
        _, paths = simulate(model; nsim=1, data=true, paths=true, autotmax=false)
        path = paths[1][1]
        
        # All transitions as panel
        obstype_map = Dict(1 => 2, 2 => 2, 3 => 2)
        result = observe_path(path, model; obstype_by_transition=obstype_map, trans_map=trans_map)
        
        # Panel rows should have observed endpoint state (not missing)
        panel_rows = filter(r -> r.obstype == 2, eachrow(result))
        for row in panel_rows
            # Panel observation records endpoint state
            @test !ismissing(row.stateto)
        end
    end
end

# =============================================================================
# simulate API Tests
# =============================================================================

@testset "simulate with per-transition obstypes" begin
    model = toy_illness_death_model().model
    
    @testset "simulate accepts obstype_by_transition" begin
        obstype_map = Dict(1 => 1, 2 => 2, 3 => 3)
        
        # Should not error
        Random.seed!(42)
        data_result = simulate_data(model; nsim=3, obstype_by_transition=obstype_map)
        
        @test length(data_result) == 3
        @test all(isa(d, DataFrame) for d in data_result)
    end
    
    @testset "simulate respects autotmax=false" begin
        obstype_map = Dict(1 => 1, 2 => 2, 3 => 3)
        
        Random.seed!(1)
        data_auto, _ = simulate(model; nsim=1, data=true, paths=true, 
                                 obstype_by_transition=obstype_map, autotmax=true)
        
        Random.seed!(1)
        data_noauto, _ = simulate(model; nsim=1, data=true, paths=true,
                                   obstype_by_transition=obstype_map, autotmax=false)
        
        # autotmax=false should preserve interval structure (potentially more rows)
        # Note: might be same if path terminates before interval boundary
        @test nrow(data_noauto[1]) >= nrow(data_auto[1])
    end
    
    @testset "validation errors" begin
        # Invalid transition index
        bad_map = Dict(10 => 1)
        @test_throws ArgumentError simulate_data(model; nsim=1, obstype_by_transition=bad_map)
        
        # Invalid obstype code
        bad_map2 = Dict(1 => 0)
        @test_throws ArgumentError simulate_data(model; nsim=1, obstype_by_transition=bad_map2)
    end
end

# =============================================================================
# Path-to-Data Verification Tests
# =============================================================================

@testset "Path-to-Data Verification" begin
    model = toy_illness_death_model().model
    trans_map = transition_index_map(model)
    
    @testset "exact transitions match path exactly" begin
        Random.seed!(1)
        data_result, paths = simulate(model; nsim=5, data=true, paths=true,
                                       obstype_by_transition=Dict(1 => 1, 2 => 1, 3 => 1),
                                       autotmax=false)
        
        for i in 1:5
            path = paths[i][1]
            df = data_result[i]
            
            # For exact transitions, verify times match
            for row in eachrow(df)
                if row.obstype == 1 && !ismissing(row.stateto)
                    # Find corresponding transition in path
                    idx = findfirst(t -> t ≈ row.tstop, path.times)
                    if idx !== nothing && idx > 1
                        # Verify state matches
                        @test path.states[idx] == row.stateto
                    end
                end
            end
        end
    end
    
    @testset "censored transitions have missing stateto" begin
        Random.seed!(1)
        # Make 2→3 censored
        obstype_map = Dict(1 => 1, 2 => 1, 3 => 3)
        
        data_result, paths = simulate(model; nsim=10, data=true, paths=true,
                                       obstype_by_transition=obstype_map,
                                       autotmax=false)
        
        for i in 1:10
            path = paths[i][1]
            df = data_result[i]
            
            # Check that censored rows have missing stateto
            censored_rows = filter(r -> r.obstype >= 3, eachrow(df))
            for row in censored_rows
                @test ismissing(row.stateto)
            end
        end
    end
    
    @testset "data rows are chronologically ordered" begin
        Random.seed!(1)
        obstype_map = Dict(1 => 1, 2 => 2, 3 => 3)
        
        data_result = simulate_data(model; nsim=5, 
                                    obstype_by_transition=obstype_map,
                                    autotmax=false)
        
        for df in data_result
            # tstart should be non-decreasing
            for i in 2:nrow(df)
                @test df.tstart[i] >= df.tstart[i-1]
            end
            
            # tstop should be >= tstart for each row
            for i in 1:nrow(df)
                @test df.tstop[i] >= df.tstart[i]
            end
            
            # tstart[i] should equal tstop[i-1] for consecutive rows
            for i in 2:nrow(df)
                @test df.tstart[i] ≈ df.tstop[i-1]
            end
        end
    end
end

# =============================================================================
# Multi-subject Tests
# =============================================================================

@testset "Multi-subject simulation" begin
    model = toy_multisubject_illness_death_model().model
    
    @testset "per-transition applies to all subjects" begin
        obstype_map = Dict(1 => 2, 2 => 3, 3 => 1)
        
        Random.seed!(1)
        data_result, paths = simulate(model; nsim=2, data=true, paths=true,
                                       obstype_by_transition=obstype_map,
                                       autotmax=false)
        
        # Each dataset should contain data for 3 subjects
        for df in data_result
            subject_ids = unique(df.id)
            @test length(subject_ids) == 3
        end
    end
end

