# Run all long tests
# IMPORTANT: ENV must be set BEFORE loading the module since TEST_LEVEL is a const
ENV["MSM_TEST_LEVEL"] = "full"

# Use include to force fresh evaluation with updated ENV
include(joinpath(@__DIR__, "..", "src", "MultistateModelsTests.jl"))
using .MultistateModelsTests
MultistateModelsTests.runtests()
