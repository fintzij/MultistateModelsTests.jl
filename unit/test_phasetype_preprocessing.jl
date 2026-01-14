# =============================================================================
# Phase-Type Data Preprocessing Tests
# =============================================================================
#
# Rigorous unit tests for phase-type data preprocessing, specifically testing
# the bugs that were found and fixed in src/phasetype/expansion_hazards.jl:
#
# Bug 1: _build_phase_censoring_patterns was creating rows for ALL observed states
#        instead of only multi-phase states. Single-phase states have no phase 
#        uncertainty and should not have censoring pattern rows.
#
# Bug 2: expand_data_for_phasetype_fitting was assigning censoring obstypes to
#        sojourn rows for single-phase states. Single-phase states should use
#        obstype=2 (panel) since there's no phase uncertainty.
#
# Bug 3: _merge_censoring_patterns_with_shift was shifting user codes instead of
#        auto-generated phase codes, and was not producing consecutive codes.
#
# Test scenarios use a RECURRENT illness-death model:
#   States: 1 (healthy) <-> 2 (ill) -> 3 (dead), also 1 -> 3
#   Transitions: 1->2 (panel), 2->1 (panel), 1->3 (exact), 2->3 (exact)
#
# =============================================================================

using Test
using MultistateModels
using DataFrames
using LinearAlgebra

# Helper function to compare DataFrames on key columns
function assert_expanded_data_equals(actual::DataFrame, expected::DataFrame; cols=[:id, :tstart, :tstop, :statefrom, :stateto, :obstype])
    @test nrow(actual) == nrow(expected)
    @test ncol(actual) >= length(cols)
    for col in cols
        @test actual[!, col] == expected[!, col]
    end
end

@testset "Phase-Type Preprocessing" begin

    # =========================================================================
    # Scenario 1: h12 PT(2), h21 exp, h13 exp, h23 exp
    # Only state 1 has multiple phases
    # =========================================================================
    @testset "Scenario 1: Only h12 is phase-type" begin
        # Recurrent illness-death data: 1 -> 2 (panel) -> 1 (panel) -> 3 (exact)
        data = DataFrame(
            id = [1, 1, 1],
            tstart = [0.0, 1.0, 2.0],
            tstop = [1.0, 2.0, 3.0],
            statefrom = [1, 2, 2],
            stateto = [2, 1, 3],
            obstype = [2, 2, 1]  # panel, panel, exact death
        )

        h12 = Hazard(@formula(0 ~ 1), "pt", 1, 2; n_phases=2)
        h21 = Hazard(@formula(0 ~ 1), "exp", 2, 1)
        h13 = Hazard(@formula(0 ~ 1), "exp", 1, 3)
        h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)

        model = multistatemodel(h12, h21, h13, h23; data=data, n_phases=Dict(1=>2))
        
        pt_exp = model.phasetype_expansion
        
        @testset "State-to-phases mapping" begin
            # State 1 -> phases 1,2 (PT with 2 phases)
            # State 2 -> phase 3 (single phase, exp)
            # State 3 -> phase 4 (absorbing, single phase)
            @test pt_exp.mappings.n_observed == 3
            @test pt_exp.mappings.n_expanded == 4
            @test pt_exp.mappings.state_to_phases[1] == 1:2
            @test pt_exp.mappings.state_to_phases[2] == 3:3
            @test pt_exp.mappings.state_to_phases[3] == 4:4
            @test pt_exp.mappings.n_phases_per_state == [2, 1, 1]
        end
        
        @testset "CensoringPatterns matrix - exact equality" begin
            # Only 1 row because only state 1 has multiple phases
            # Row 1 (obstype=3): phases 1,2 allowed (state 1 sojourn)
            expected_CP = [3.0 1.0 1.0 0.0 0.0]  # code + 4 phases
            @test model.CensoringPatterns == expected_CP
        end
        
        @testset "Expanded data - exact equality" begin
            # Original: 3 rows
            # Row 3 (exact obs 2->3) splits into 2 rows: sojourn + instantaneous
            # Expected: 4 rows total
            expected_data = DataFrame(
                id = [1, 1, 1, 1],
                tstart = [0.0, 1.0, 2.0, 3.0],
                tstop = [1.0, 2.0, 3.0, 3.0],
                statefrom = [1, 3, 3, 0],
                stateto = [3, 0, 3, 4],
                obstype = [2, 3, 2, 1]
            )
            assert_expanded_data_equals(model.data, expected_data)
        end
        
        @testset "Emission matrix - exact equality" begin
            # Row 1: panel obs to state 2 (single phase) -> phase 3 allowed
            # Row 2: panel obs to state 1 (multi-phase) -> obstype=3 -> phases 1,2
            # Row 3: sojourn in state 2 (single phase) -> phase 3 allowed  
            # Row 4: exact death -> phase 4 allowed
            expected_emat = [
                0.0 0.0 1.0 0.0;  # row 1: panel to state 2 -> phase 3
                1.0 1.0 0.0 0.0;  # row 2: panel to state 1 -> phases 1,2 (obstype=3)
                0.0 0.0 1.0 0.0;  # row 3: sojourn state 2 -> phase 3
                0.0 0.0 0.0 1.0   # row 4: exact to state 3 -> phase 4
            ]
            @test model.emat == expected_emat
        end
    end

    # =========================================================================
    # Scenario 2: h12 PT(2), h21 PT(2), h13 exp, h23 exp  
    # Both states 1 and 2 have multiple phases
    # =========================================================================
    @testset "Scenario 2: h12 and h21 both phase-type" begin
        data = DataFrame(
            id = [1, 1, 1],
            tstart = [0.0, 1.0, 2.0],
            tstop = [1.0, 2.0, 3.0],
            statefrom = [1, 2, 2],
            stateto = [2, 1, 3],
            obstype = [2, 2, 1]
        )

        h12 = Hazard(@formula(0 ~ 1), "pt", 1, 2; n_phases=2)
        h21 = Hazard(@formula(0 ~ 1), "pt", 2, 1; n_phases=2)
        h13 = Hazard(@formula(0 ~ 1), "exp", 1, 3)
        h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)

        model = multistatemodel(h12, h21, h13, h23; data=data, n_phases=Dict(1=>2, 2=>2))
        
        pt_exp = model.phasetype_expansion
        
        @testset "State-to-phases mapping" begin
            @test pt_exp.mappings.n_observed == 3
            @test pt_exp.mappings.n_expanded == 5
            @test pt_exp.mappings.state_to_phases[1] == 1:2
            @test pt_exp.mappings.state_to_phases[2] == 3:4
            @test pt_exp.mappings.state_to_phases[3] == 5:5
            @test pt_exp.mappings.n_phases_per_state == [2, 2, 1]
        end
        
        @testset "CensoringPatterns matrix - exact equality" begin
            # 2 rows: states 1 and 2 both have multiple phases
            # Row 1 (obstype=3): phases 1,2 allowed (state 1 sojourn)
            # Row 2 (obstype=4): phases 3,4 allowed (state 2 sojourn)
            expected_CP = [
                3.0 1.0 1.0 0.0 0.0 0.0;  # state 1: phases 1,2
                4.0 0.0 0.0 1.0 1.0 0.0   # state 2: phases 3,4
            ]
            @test model.CensoringPatterns == expected_CP
        end
        
        @testset "Expanded data - exact equality" begin
            expected_data = DataFrame(
                id = [1, 1, 1, 1],
                tstart = [0.0, 1.0, 2.0, 3.0],
                tstop = [1.0, 2.0, 3.0, 3.0],
                statefrom = [1, 3, 3, 0],
                stateto = [0, 0, 0, 5],    # censored for phase uncertainty
                obstype = [4, 3, 4, 1]     # 4=state2 phases, 3=state1 phases
            )
            assert_expanded_data_equals(model.data, expected_data)
        end
        
        @testset "Emission matrix - exact equality" begin
            expected_emat = [
                0.0 0.0 1.0 1.0 0.0;  # row 1: panel to state 2 -> phases 3,4
                1.0 1.0 0.0 0.0 0.0;  # row 2: panel to state 1 -> phases 1,2
                0.0 0.0 1.0 1.0 0.0;  # row 3: sojourn state 2 -> phases 3,4
                0.0 0.0 0.0 0.0 1.0   # row 4: exact to state 3 -> phase 5
            ]
            @test model.emat == expected_emat
        end
    end

    # =========================================================================
    # Scenario 3: User-supplied CensoringPatterns with phase expansion
    # Critical test for the merge logic bug fix
    # =========================================================================
    @testset "Scenario 3: User-supplied CensoringPatterns" begin
        # First obs is state-censored: could be state 1 or 2
        data = DataFrame(
            id = [1, 1, 1],
            tstart = [0.0, 1.0, 2.0],
            tstop = [1.0, 2.0, 3.0],
            statefrom = [1, 0, 2],
            stateto = [0, 2, 3],
            obstype = [3, 2, 1]  # state-censored, panel, exact death
        )
        
        # User censoring pattern: obstype=3 means could be state 1 or 2 (not 3)
        CensoringPatterns = [3.0 1.0 1.0 0.0]

        h12 = Hazard(@formula(0 ~ 1), "pt", 1, 2; n_phases=2)
        h21 = Hazard(@formula(0 ~ 1), "pt", 2, 1; n_phases=2)
        h13 = Hazard(@formula(0 ~ 1), "exp", 1, 3)
        h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)

        model = multistatemodel(h12, h21, h13, h23; data=data, n_phases=Dict(1=>2, 2=>2), 
                                CensoringPatterns=CensoringPatterns)
        
        @testset "Merged CensoringPatterns - exact equality" begin
            # Should have 3 rows with consecutive codes [3, 4, 5]:
            # - Code 3: User pattern (expanded to phases)
            # - Code 4: Auto state 1 phase uncertainty
            # - Code 5: Auto state 2 phase uncertainty
            expected_CP = [
                3.0 1.0 1.0 1.0 1.0 0.0;  # user: states 1,2 -> phases 1,2,3,4
                4.0 1.0 1.0 0.0 0.0 0.0;  # auto: state 1 -> phases 1,2
                5.0 0.0 0.0 1.0 1.0 0.0   # auto: state 2 -> phases 3,4
            ]
            @test model.CensoringPatterns == expected_CP
        end
        
        @testset "Expanded data - exact equality" begin
            # Row 1: state-censored (obstype=3) stays obstype=3
            # Row 2: panel to state 2 -> becomes obstype=5 (state 2 phases)
            # Row 3: sojourn state 2 + exact death -> splits into 2 rows
            expected_data = DataFrame(
                id = [1, 1, 1, 1],
                tstart = [0.0, 1.0, 2.0, 3.0],
                tstop = [1.0, 2.0, 3.0, 3.0],
                statefrom = [1, 0, 3, 0],
                stateto = [0, 0, 0, 5],
                obstype = [3, 5, 5, 1]
            )
            assert_expanded_data_equals(model.data, expected_data)
        end
        
        @testset "Emission matrix - exact equality" begin
            expected_emat = [
                1.0 1.0 1.0 1.0 0.0;  # row 1: user obstype=3 -> phases 1,2,3,4
                0.0 0.0 1.0 1.0 0.0;  # row 2: panel to state 2 -> phases 3,4 (obstype=5)
                0.0 0.0 1.0 1.0 0.0;  # row 3: sojourn state 2 -> phases 3,4 (obstype=5)
                0.0 0.0 0.0 0.0 1.0   # row 4: exact to state 3 -> phase 5
            ]
            @test model.emat == expected_emat
        end
    end

    # =========================================================================
    # Scenario 4: All hazards phase-type with varying phases
    # =========================================================================
    @testset "Scenario 4: All phase-type with different phase counts" begin
        data = DataFrame(
            id = [1, 1],
            tstart = [0.0, 1.0],
            tstop = [1.0, 2.0],
            statefrom = [1, 2],
            stateto = [2, 3],
            obstype = [1, 1]  # both exact
        )

        h12 = Hazard(@formula(0 ~ 1), "pt", 1, 2; n_phases=3)
        h21 = Hazard(@formula(0 ~ 1), "pt", 2, 1; n_phases=2)
        h13 = Hazard(@formula(0 ~ 1), "pt", 1, 3; n_phases=3)
        h23 = Hazard(@formula(0 ~ 1), "pt", 2, 3; n_phases=2)

        model = multistatemodel(h12, h21, h13, h23; data=data, 
                                n_phases=Dict(1=>3, 2=>2))
        
        pt_exp = model.phasetype_expansion
        
        @testset "State-to-phases mapping" begin
            # State 1: 3 phases, State 2: 2 phases, State 3: 1 phase (absorbing)
            @test pt_exp.mappings.n_expanded == 6
            @test pt_exp.mappings.state_to_phases[1] == 1:3
            @test pt_exp.mappings.state_to_phases[2] == 4:5
            @test pt_exp.mappings.state_to_phases[3] == 6:6
            @test pt_exp.mappings.n_phases_per_state == [3, 2, 1]
        end
        
        @testset "CensoringPatterns - exact equality" begin
            # 2 rows: states 1 and 2 have multiple phases, state 3 is single-phase
            expected_CP = [
                3.0 1.0 1.0 1.0 0.0 0.0 0.0;  # state 1: phases 1,2,3
                4.0 0.0 0.0 0.0 1.0 1.0 0.0   # state 2: phases 4,5
            ]
            @test model.CensoringPatterns == expected_CP
        end
        
        @testset "Expanded data - exact equality" begin
            # 2 exact observations -> 4 rows (sojourn + instantaneous for each)
            expected_data = DataFrame(
                id = [1, 1, 1, 1],
                tstart = [0.0, 1.0, 1.0, 2.0],
                tstop = [1.0, 1.0, 2.0, 2.0],
                statefrom = [1, 0, 4, 0],
                stateto = [0, 4, 0, 6],
                obstype = [3, 1, 4, 1]
            )
            assert_expanded_data_equals(model.data, expected_data)
        end
        
        @testset "Emission matrix - exact equality" begin
            expected_emat = [
                1.0 1.0 1.0 0.0 0.0 0.0;  # sojourn state 1 -> phases 1,2,3
                0.0 0.0 0.0 1.0 0.0 0.0;  # exact to state 2 -> first phase (4)
                0.0 0.0 0.0 1.0 1.0 0.0;  # sojourn state 2 -> phases 4,5
                0.0 0.0 0.0 0.0 0.0 1.0   # exact to state 3 -> phase 6
            ]
            @test model.emat == expected_emat
        end
    end

    # =========================================================================
    # Edge case: First state has 1 phase, second has multiple
    # =========================================================================
    @testset "Edge case: First state single-phase" begin
        data = DataFrame(
            id = [1, 1],
            tstart = [0.0, 1.0],
            tstop = [1.0, 2.0],
            statefrom = [1, 2],
            stateto = [2, 3],
            obstype = [2, 1]
        )

        h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)  # single phase
        h23 = Hazard(@formula(0 ~ 1), "pt", 2, 3; n_phases=2)  # multi-phase

        model = multistatemodel(h12, h23; data=data, n_phases=Dict(2=>2))
        
        pt_exp = model.phasetype_expansion
        
        @testset "Mappings" begin
            @test pt_exp.mappings.state_to_phases[1] == 1:1  # single phase
            @test pt_exp.mappings.state_to_phases[2] == 2:3  # 2 phases
            @test pt_exp.mappings.state_to_phases[3] == 4:4  # absorbing
        end
        
        @testset "CensoringPatterns - exact equality" begin
            # Only 1 row for state 2 (multi-phase)
            # State 1 is single-phase, no uncertainty
            # Note: obstype = 3 (first consecutive code)
            expected_CP = [3.0 0.0 1.0 1.0 0.0]  # state 2: phases 2,3
            @test model.CensoringPatterns == expected_CP
        end
        
        @testset "Expanded data - exact equality" begin
            # Row 1: panel to state 2 -> obstype=3 (multi-phase)
            # Row 2: exact 2->3 splits into sojourn + instantaneous
            expected_data = DataFrame(
                id = [1, 1, 1],
                tstart = [0.0, 1.0, 2.0],
                tstop = [1.0, 2.0, 2.0],
                statefrom = [1, 2, 0],
                stateto = [0, 0, 4],
                obstype = [3, 3, 1]
            )
            assert_expanded_data_equals(model.data, expected_data)
        end
        
        @testset "Emission matrix - exact equality" begin
            expected_emat = [
                0.0 1.0 1.0 0.0;  # panel to state 2 -> phases 2,3
                0.0 1.0 1.0 0.0;  # sojourn state 2 -> phases 2,3
                0.0 0.0 0.0 1.0   # exact to state 3 -> phase 4
            ]
            @test model.emat == expected_emat
        end
    end

    # =========================================================================
    # Edge case: All states single-phase (no phase-type expansion needed)
    # =========================================================================
    @testset "Edge case: All single-phase (n_phases=1)" begin
        data = DataFrame(
            id = [1, 1],
            tstart = [0.0, 1.0],
            tstop = [1.0, 2.0],
            statefrom = [1, 2],
            stateto = [2, 3],
            obstype = [1, 1]
        )

        h12 = Hazard(@formula(0 ~ 1), "pt", 1, 2; n_phases=1)  # degenerates to exp
        h23 = Hazard(@formula(0 ~ 1), "pt", 2, 3; n_phases=1)

        model = multistatemodel(h12, h23; data=data, n_phases=Dict(1=>1, 2=>1))
        
        pt_exp = model.phasetype_expansion
        
        @testset "Mappings" begin
            @test pt_exp.mappings.n_expanded == 3  # 1 phase per state
            @test pt_exp.mappings.state_to_phases[1] == 1:1
            @test pt_exp.mappings.state_to_phases[2] == 2:2
            @test pt_exp.mappings.state_to_phases[3] == 3:3
        end
        
        @testset "No CensoringPatterns needed" begin
            # All states single-phase -> no phase uncertainty -> empty patterns
            @test size(model.CensoringPatterns, 1) == 0
        end
        
        @testset "Expanded data - exact equality" begin
            # 2 exact observations -> 4 rows (sojourn + instantaneous for each)
            # With all single-phase states, sojourn rows should be obstype=2
            expected_data = DataFrame(
                id = [1, 1, 1, 1],
                tstart = [0.0, 1.0, 1.0, 2.0],
                tstop = [1.0, 1.0, 2.0, 2.0],
                statefrom = [1, 0, 2, 0],
                stateto = [1, 2, 2, 3],  # single phase, so stateto = phase number
                obstype = [2, 1, 2, 1]   # sojourn=2, exact=1
            )
            assert_expanded_data_equals(model.data, expected_data)
        end
        
        @testset "Emission matrix - exact equality" begin
            expected_emat = [
                1.0 0.0 0.0;  # sojourn state 1 -> phase 1
                0.0 1.0 0.0;  # exact to state 2 -> phase 2
                0.0 1.0 0.0;  # sojourn state 2 -> phase 2
                0.0 0.0 1.0   # exact to state 3 -> phase 3
            ]
            @test model.emat == expected_emat
        end
    end

    # =========================================================================
    # Test emission matrix construction
    # =========================================================================
    @testset "Emission matrix correctness" begin
        data = DataFrame(
            id = [1, 1, 1],
            tstart = [0.0, 1.0, 2.0],
            tstop = [1.0, 2.0, 3.0],
            statefrom = [1, 2, 2],
            stateto = [2, 1, 3],
            obstype = [2, 2, 1]
        )

        h12 = Hazard(@formula(0 ~ 1), "pt", 1, 2; n_phases=2)
        h21 = Hazard(@formula(0 ~ 1), "pt", 2, 1; n_phases=2)
        h13 = Hazard(@formula(0 ~ 1), "exp", 1, 3)
        h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)

        model = multistatemodel(h12, h21, h13, h23; data=data, n_phases=Dict(1=>2, 2=>2))
        
        # This is a comprehensive test that the entire emission matrix is correct
        @testset "Full emission matrix verification" begin
            @test size(model.emat) == (nrow(model.data), 5)
            
            expected_emat = [
                0.0 0.0 1.0 1.0 0.0;  # row 1: panel to state 2 -> phases 3,4
                1.0 1.0 0.0 0.0 0.0;  # row 2: panel to state 1 -> phases 1,2
                0.0 0.0 1.0 1.0 0.0;  # row 3: sojourn state 2 -> phases 3,4
                0.0 0.0 0.0 0.0 1.0   # row 4: exact to state 3 -> phase 5
            ]
            @test model.emat == expected_emat
        end
    end

    # =========================================================================
    # Additional edge cases for robustness
    # =========================================================================
    @testset "Multiple subjects" begin
        data = DataFrame(
            id = [1, 1, 2, 2],
            tstart = [0.0, 1.0, 0.0, 2.0],
            tstop = [1.0, 2.0, 2.0, 3.0],
            statefrom = [1, 2, 1, 2],
            stateto = [2, 3, 2, 3],
            obstype = [2, 1, 1, 1]
        )

        h12 = Hazard(@formula(0 ~ 1), "pt", 1, 2; n_phases=2)
        h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)

        model = multistatemodel(h12, h23; data=data, n_phases=Dict(1=>2))
        
        @testset "CensoringPatterns - exact equality" begin
            expected_CP = [3.0 1.0 1.0 0.0 0.0]  # state 1: phases 1,2
            @test model.CensoringPatterns == expected_CP
        end
        
        @testset "Expanded data structure" begin
            # Subject 1: panel(1->2) + exact(2->3) = 1 + 2 = 3 rows
            # Subject 2: exact(1->2) + exact(2->3) = 2 + 2 = 4 rows
            # Total: 7 rows
            @test nrow(model.data) == 7
            @test model.data.id == [1, 1, 1, 2, 2, 2, 2]
        end
    end
    
    @testset "Panel data only (no exact)" begin
        data = DataFrame(
            id = [1, 1, 1],
            tstart = [0.0, 1.0, 2.0],
            tstop = [1.0, 2.0, 3.0],
            statefrom = [1, 2, 1],
            stateto = [2, 1, 2],
            obstype = [2, 2, 2]
        )

        h12 = Hazard(@formula(0 ~ 1), "pt", 1, 2; n_phases=2)
        h21 = Hazard(@formula(0 ~ 1), "exp", 2, 1)

        model = multistatemodel(h12, h21; data=data, n_phases=Dict(1=>2))
        
        @testset "CensoringPatterns - exact equality" begin
            expected_CP = [3.0 1.0 1.0 0.0]
            @test model.CensoringPatterns == expected_CP
        end
        
        @testset "Expanded data - exact equality" begin
            # All panel observations, no splitting
            # statefrom: state 1 -> phase 1, state 2 -> phase 3
            # stateto: state 2 -> phase 3 (single), state 1 -> 0 (censored for phase uncertainty)
            expected_data = DataFrame(
                id = [1, 1, 1],
                tstart = [0.0, 1.0, 2.0],
                tstop = [1.0, 2.0, 3.0],
                statefrom = [1, 3, 1],      # phase 1, phase 3, phase 1
                stateto = [3, 0, 3],        # phase 3 (single), censored (multi-phase), phase 3
                obstype = [2, 3, 2]         # panel, phase uncertainty (obstype=3), panel
            )
            assert_expanded_data_equals(model.data, expected_data)
        end
        
        @testset "Emission matrix - exact equality" begin
            expected_emat = [
                0.0 0.0 1.0;  # panel to state 2 -> phase 3
                1.0 1.0 0.0;  # panel to state 1 -> phases 1,2 (obstype=3)
                0.0 0.0 1.0   # panel to state 2 -> phase 3
            ]
            @test model.emat == expected_emat
        end
    end

end  # Phase-Type Preprocessing testset
