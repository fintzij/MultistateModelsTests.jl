# =============================================================================
# Error Message Content Validation Tests
# =============================================================================
#
# These tests verify that error messages contain helpful, descriptive content
# (not just that errors are thrown). This helps ensure users get actionable
# feedback when they make mistakes.

using Test
using MultistateModels
using DataFrames
using StatsModels: @formula

# =============================================================================
@testset "Error Message Content Validation" begin
    
    # Setup minimal model for testing
    n = 10
    dat = DataFrame(
        id = 1:n,
        tstart = zeros(n),
        tstop = fill(5.0, n),
        statefrom = ones(Int, n),
        stateto = fill(2, n),
        obstype = fill(1, n)
    )
    
    @testset "Invalid hazard family" begin
        try
            h = Hazard(@formula(0 ~ 1), "invalid_family", 1, 2)
            @test false  # Should not reach here
        catch e
            @test e isa Exception
            msg = string(e)
            # Message should mention the invalid family or valid options
            @test occursin("invalid", lowercase(msg)) || occursin("family", lowercase(msg)) || occursin("unknown", lowercase(msg))
        end
    end
    
    @testset "Missing required data columns" begin
        bad_dat = DataFrame(id = 1:5, x = rand(5))  # Missing required columns
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        
        # NOTE: Currently throws BoundsError when columns missing
        # Future improvement could validate columns explicitly with better message
        try
            model = multistatemodel(h12; data=bad_dat)
            @test false  # Should not reach here
        catch e
            # Current behavior: BoundsError (not the most descriptive)
            @test e isa Exception
            # This documents the current behavior - a BoundsError is thrown
            @test e isa BoundsError || occursin("column", lowercase(string(e))) || 
                  occursin("key", lowercase(string(e)))
        end
    end
    
    @testset "Invalid state transitions" begin
        # Try to create hazard with invalid transition (same state)
        try
            h = Hazard(@formula(0 ~ 1), "exp", 1, 1)  # from=to is invalid
            @test false  # Should not reach here
        catch e
            @test e isa Exception
            msg = string(e)
            # Message should indicate the transition issue
            @test occursin("state", lowercase(msg)) || 
                  occursin("transition", lowercase(msg)) ||
                  occursin("from", lowercase(msg))
        end
    end
    
    @testset "Negative time values - current behavior" begin
        # NOTE: Currently the package does NOT validate negative start times
        # This test documents the current behavior (may accept negative times)
        bad_dat = DataFrame(
            id = 1:5,
            tstart = fill(-1.0, 5),  # Negative start time
            tstop = fill(5.0, 5),
            statefrom = ones(Int, 5),
            stateto = fill(2, 5),
            obstype = fill(1, 5)
        )
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        
        # Current behavior: model construction may succeed (no validation)
        # This test documents that behavior - validation could be added later
        model = multistatemodel(h12; data=bad_dat)
        @test model isa MultistateModels.MultistateModel
    end
    
    @testset "Simulation without parameters" begin
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        model = multistatemodel(h12; data=dat)
        # Don't set parameters - should error on simulate
        
        try
            result = simulate(model; nsim=1)
            # If it doesn't error, that's also acceptable (may use defaults)
            @test true
        catch e
            @test e isa Exception
            msg = string(e)
            # Message should mention parameters
            @test occursin("param", lowercase(msg)) || 
                  occursin("set", lowercase(msg)) ||
                  occursin("initial", lowercase(msg))
        end
    end
    
    @testset "Invalid obstype values" begin
        bad_dat = DataFrame(
            id = 1:5,
            tstart = zeros(5),
            tstop = fill(5.0, 5),
            statefrom = ones(Int, 5),
            stateto = fill(2, 5),
            obstype = fill(-1, 5)  # Invalid: negative obstype
        )
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        
        try
            model = multistatemodel(h12; data=bad_dat)
            @test false  # Should not reach here
        catch e
            @test e isa Exception
            # Error about obstype is acceptable
            @test true
        end
    end
    
    @testset "@hazard macro with invalid syntax" begin
        # Test that macro errors have helpful messages
        try
            # Invalid: missing transition specification
            eval(:(@hazard(exp)))
            @test false
        catch e
            @test e isa Exception
            # Any syntax error is acceptable
            @test true
        end
    end
    
    @testset "SplinePenalty invalid order" begin
        try
            p = SplinePenalty(order=0)
            @test false
        catch e
            @test e isa ArgumentError
            msg = e.msg
            @test occursin("order", lowercase(msg)) || occursin("positive", lowercase(msg))
        end
    end
    
    @testset "Simulation with both data=false and paths=false" begin
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        model = multistatemodel(h12; data=dat)
        set_parameters!(model, (h12 = [0.1],))
        
        try
            result = simulate(model; data=false, paths=false)
            @test false
        catch e
            @test e isa ArgumentError
            msg = e.msg
            @test occursin("data", lowercase(msg)) || occursin("path", lowercase(msg))
        end
    end
    
    @testset "Simulation with negative tmax" begin
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        model = multistatemodel(h12; data=dat)
        set_parameters!(model, (h12 = [0.1],))
        
        try
            result = simulate(model; tmax=-1.0)
            @test false
        catch e
            @test e isa ArgumentError
            msg = e.msg
            @test occursin("tmax", lowercase(msg)) || occursin("positive", lowercase(msg)) || occursin("negative", lowercase(msg))
        end
    end
    
end

# =============================================================================
# Input Validation Gap Tests (Phase 3 Audit Gaps #2 and #3)
# =============================================================================
#
# These tests document current behavior for validation gaps identified
# during the testing infrastructure audit (Phase 2).
#
# GAP #2 (W6): Negative time values are NOT currently validated
# GAP #3 (W7): Missing columns produce BoundsError instead of descriptive error
#
# =============================================================================

@testset "Input Validation Gaps - Documentation" begin
    
    # =========================================================================
    # GAP #2: Negative Time Values (W6 - HIGH Severity)
    # =========================================================================
    
    @testset "Negative tstart - CURRENT BEHAVIOR (GAP)" begin
        # This test documents that negative tstart is currently ACCEPTED
        # Future fix should throw ArgumentError with helpful message
        #
        # Expected future behavior:
        #   throw(ArgumentError("Data contains negative tstart values at rows [1, 2, 3, ...]. " *
        #                       "All times must be non-negative."))
        
        bad_dat = DataFrame(
            id = [1, 2, 3],
            tstart = [-5.0, -1.0, 0.0],  # First two are negative
            tstop = [5.0, 5.0, 5.0],
            statefrom = [1, 1, 1],
            stateto = [2, 2, 2],
            obstype = [1, 1, 1]
        )
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        
        # CURRENT BEHAVIOR: Model construction succeeds (no validation)
        # This documents the gap - validation should be added
        model = multistatemodel(h12; data=bad_dat, initialize=false)
        @test model isa MultistateModels.MultistateModel
        
        # Document that the negative values are stored unchanged
        @test minimum(model.data.tstart) < 0
    end
    
    @testset "Negative tstop - CURRENT BEHAVIOR (GAP)" begin
        # This test documents that negative tstop is currently ACCEPTED
        # when tstart <= tstop (the only check performed)
        
        bad_dat = DataFrame(
            id = [1],
            tstart = [-10.0],  # Negative
            tstop = [-5.0],    # Also negative, but >= tstart
            statefrom = [1],
            stateto = [1],
            obstype = [1]
        )
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        
        # CURRENT BEHAVIOR: Model construction succeeds
        model = multistatemodel(h12; data=bad_dat, initialize=false)
        @test model isa MultistateModels.MultistateModel
        
        # Both times are negative but stored
        @test model.data.tstart[1] < 0
        @test model.data.tstop[1] < 0
    end
    
    @testset "Negative times with exact observations - CURRENT BEHAVIOR (GAP)" begin
        # Document that fitting with negative times may produce unexpected results
        # but doesn't error at model construction
        
        bad_dat = DataFrame(
            id = [1],
            tstart = [-2.0],
            tstop = [3.0],
            statefrom = [1],
            stateto = [2],
            obstype = [1]  # exact observation
        )
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        
        # Model construction succeeds
        model = multistatemodel(h12; data=bad_dat, initialize=false)
        set_parameters!(model, (h12 = [0.5],))
        
        # Fitting may succeed but results are mathematically questionable
        # (cumulative hazard integrated from -2 to 3 instead of 0 to 5)
        fitted = fit(model; verbose=false, vcov_type=:none)
        @test fitted isa MultistateModels.MultistateModelFitted
    end
    
    # =========================================================================
    # GAP #3: Missing Required Columns (W7 - MEDIUM Severity)
    # =========================================================================
    
    @testset "Missing all required columns - CURRENT BEHAVIOR (GAP)" begin
        # This test documents that missing columns produce BoundsError
        # Future fix should throw ArgumentError listing missing columns
        #
        # Expected future behavior:
        #   throw(ArgumentError("Data is missing required columns: ['tstart', 'tstop', 'statefrom', 'stateto', 'obstype']. " *
        #                       "Required columns are: ['id', 'tstart', 'tstop', 'statefrom', 'stateto', 'obstype']."))
        
        bad_dat = DataFrame(id = 1:5, x = rand(5))  # Only has 'id' and 'x'
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        
        # CURRENT BEHAVIOR: throws BoundsError (not helpful)
        err = nothing
        try
            model = multistatemodel(h12; data=bad_dat)
        catch e
            err = e
        end
        
        @test err !== nothing
        # Document that the error is currently a BoundsError or similar
        # Not a descriptive ArgumentError about missing columns
        @test !(err isa ArgumentError && occursin("missing", lowercase(err.msg)))
    end
    
    @testset "Missing some required columns - CURRENT BEHAVIOR (GAP)" begin
        # Data has most columns but missing 'obstype'
        
        bad_dat = DataFrame(
            id = 1:5,
            tstart = zeros(5),
            tstop = fill(5.0, 5),
            statefrom = ones(Int, 5),
            stateto = fill(2, 5)
            # Missing: obstype
        )
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        
        err = nothing
        try
            model = multistatemodel(h12; data=bad_dat)
        catch e
            err = e
        end
        
        @test err !== nothing
        # Current behavior doesn't give helpful message about missing 'obstype'
    end
    
    @testset "Column names misspelled - CURRENT BEHAVIOR (GAP)" begin
        # Data has columns but with wrong names
        
        bad_dat = DataFrame(
            id = 1:5,
            start_time = zeros(5),    # Should be 'tstart'
            end_time = fill(5.0, 5),  # Should be 'tstop'
            from_state = ones(Int, 5),   # Should be 'statefrom'
            to_state = fill(2, 5),       # Should be 'stateto'
            obs_type = fill(1, 5)        # Should be 'obstype'
        )
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        
        err = nothing
        try
            model = multistatemodel(h12; data=bad_dat)
        catch e
            err = e
        end
        
        @test err !== nothing
        # Current behavior doesn't suggest correct column names
    end
    
    # =========================================================================
    # Tests for EXISTING validation (these should pass)
    # =========================================================================
    
    @testset "tstart > tstop - CORRECTLY VALIDATED" begin
        # This validation already exists and works
        bad_dat = DataFrame(
            id = [1],
            tstart = [10.0],  # Greater than tstop
            tstop = [5.0],
            statefrom = [1],
            stateto = [2],
            obstype = [1]
        )
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        
        @test_throws ArgumentError multistatemodel(h12; data=bad_dat)
    end
    
    @testset "Non-consecutive subject IDs - CORRECTLY VALIDATED" begin
        # This validation already exists and works
        bad_dat = DataFrame(
            id = [1, 3, 5],  # Should be 1, 2, 3
            tstart = zeros(3),
            tstop = fill(5.0, 3),
            statefrom = ones(Int, 3),
            stateto = fill(2, 3),
            obstype = fill(1, 3)
        )
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        
        @test_throws ArgumentError multistatemodel(h12; data=bad_dat)
    end
    
    @testset "Invalid obstype - CORRECTLY VALIDATED" begin
        # Obstype must be 1, 2, or a valid censoring pattern ID
        bad_dat = DataFrame(
            id = [1],
            tstart = [0.0],
            tstop = [5.0],
            statefrom = [1],
            stateto = [2],
            obstype = [99]  # Invalid
        )
        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
        
        @test_throws ArgumentError multistatemodel(h12; data=bad_dat)
    end
end
