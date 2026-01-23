#!/bin/bash
# Run longtests in parallel using GNU parallel or background jobs
# Usage: ./run_longtests_parallel.sh [N_JOBS]

cd "$(dirname "$0")/.."

N_JOBS=${1:-4}
LOG_DIR="cache/longtest_logs"
mkdir -p "$LOG_DIR"

echo "Running longtests with $N_JOBS parallel jobs..."
echo "Logs will be in: $LOG_DIR/"
echo ""

# List of longtests
TESTS=(
    "longtest_exact_markov.jl"
    "longtest_mcem.jl"
    "longtest_mcem_splines.jl"
    "longtest_mcem_tvc.jl"
    "longtest_phasetype.jl"
    "longtest_robust_markov_phasetype.jl"
    "longtest_robust_parametric.jl"
    "longtest_simulation_distribution.jl"
    "longtest_simulation_tvc.jl"
    "longtest_sir.jl"
    "longtest_variance_validation.jl"
)

# Function to run a single test
run_test() {
    local test=$1
    local logfile="$LOG_DIR/${test%.jl}.log"
    echo "Starting: $test"
    julia --project=. -e "
        ENV[\"MSM_TEST_LEVEL\"] = \"full\"
        using Test
        using MultistateModels
        using StatsModels
        using DataFrames
        using Distributions
        using Random
        using LinearAlgebra
        using Logging
        
        include(\"fixtures/TestFixtures.jl\")
        using .TestFixtures
        include(\"longtests/longtest_helpers.jl\")
        
        @testset \"$test\" begin
            include(\"longtests/$test\")
        end
    " > "$logfile" 2>&1
    local status=$?
    if [ $status -eq 0 ]; then
        echo "✓ PASSED: $test"
    else
        echo "✗ FAILED: $test (see $logfile)"
    fi
    return $status
}

export -f run_test
export LOG_DIR

# Start time
START_TIME=$(date +%s)

# Run tests in parallel using background jobs
pids=()
for test in "${TESTS[@]}"; do
    run_test "$test" &
    pids+=($!)
    
    # Limit concurrent jobs
    while [ $(jobs -r | wc -l) -ge $N_JOBS ]; do
        sleep 1
    done
done

# Wait for all jobs
echo ""
echo "Waiting for all tests to complete..."
failed=0
for pid in "${pids[@]}"; do
    wait $pid || ((failed++))
done

# End time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "========================================"
echo "SUMMARY"
echo "========================================"
echo "Total tests: ${#TESTS[@]}"
echo "Failed: $failed"
echo "Time: ${ELAPSED}s"
echo ""

# Show any failures
if [ $failed -gt 0 ]; then
    echo "Failed tests (check logs for details):"
    for test in "${TESTS[@]}"; do
        logfile="$LOG_DIR/${test%.jl}.log"
        if grep -q "Test Failed" "$logfile" 2>/dev/null || grep -q "Error" "$logfile" 2>/dev/null; then
            echo "  - $test"
        fi
    done
    exit 1
fi

echo "All tests passed!"
exit 0
