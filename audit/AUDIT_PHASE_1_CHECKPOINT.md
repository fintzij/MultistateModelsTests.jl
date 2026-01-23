# Phase 1 Checkpoint: Test Inventory and Classification

**Date**: 2026-01-22
**Subagent**: A (Inventory)
**Branch**: penalized_splines

## Phase Summary

Inventoried 72 test files: 46 unit (21343 lines), 2 integration (827 lines), 24 longtest (19821 lines). Total: 41991 lines, ~2164 assertions in 1096 testsets.

Test quality is generally high with strong analytical verification in critical modules. ~100 assertions (5%) are smoke tests. Six tests explicitly skipped with documented reasons.

## Key Findings

- Largest file: test_phasetype.jl (2380 lines, 310 assertions)
- Most analytical: test_loglik_analytical.jl (40 correctness tests)
- Weakest: test_error_paths.jl (3 smoke tests only)
- Stub file: longtest_phasetype.jl (15 lines, 0 tests)

## Flagged Issues

### Weak Test Files (>30% smoke/directional)
1. test_error_paths.jl - 100% weak
2. test_modelgeneration.jl - 100% weak
3. test_weight_validation.jl - 100% weak

### Skipped Tests
- test_phasetype_roundtrip.jl: 3 skips
- test_pijcv_reference.jl: 2 skips
- longtest_parametric_suite.jl: 2 skips

### Loose Tolerances
- test_initialization.jl: rtol=0.5
- longtest_mcem.jl: PARAM_TOL_REL=0.35
- longtest_mcem_tvc.jl: atol=0.5

## Codebase Weaknesses

### W1 (MEDIUM): Missing validation for negative times
Location: test_error_messages.jl L77-86 documents this gap

### W2 (MEDIUM): BoundsError instead of descriptive error
Location: test_error_messages.jl L44-57 documents this gap

### W3 (MEDIUM): Documented bug in test_phasetype.jl L2343
Comment indicates statefrom should be [1,1] not [1,2]

### W4 (MEDIUM): Stub file longtest_phasetype.jl
15 lines, 0 tests, oldest file (2025-12-12)

### W5-W6 (LOW): Sparse test files
test_regressions.jl and test_modelgeneration.jl have minimal content

## Handoff Statement

Phase 1 complete. No critical weaknesses found.

For Phase 2: Strong coverage in likelihood (test_loglik_analytical.jl),
phase-type (6174 lines), splines (1504 lines). Gap: no unit tests for
_fit_markov_panel per codebase-knowledge skill.

Checkpoint: 2026-01-22
