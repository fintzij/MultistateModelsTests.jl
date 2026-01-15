# MultistateModels.jl Test Summary

**Date**: January 15, 2026
**Branch**: penalized_splines
**Julia Version**: 1.12.2

## Unit Tests

**Result**: 2001 passed, 0 failed, 3 errored

3 errors in test_phasetype.jl - wrong call signature for build_phasetype_tpm_book

## Long Tests

PASS: longtest_spline_exact.jl (7/7)
PASS: longtest_mcem.jl (13/13)
PASS: longtest_mcem_tvc.jl (7/7)
PASS: longtest_phasetype_exact.jl
PARTIAL: longtest_sensitivity_check.jl (LLVM crash)

RUNNING: longtest_parametric_suite.jl

FAILED: longtest_aft_suite.jl (empty log)
FAILED: longtest_variance_validation.jl (crashed)
