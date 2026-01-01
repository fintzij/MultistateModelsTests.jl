using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using MultistateModels
using MultistateModels: Hazard, multistatemodel, set_parameters!, simulate_path, CachedTransformStrategy
using DataFrames
using Random
using Statistics
using StatsBase

const COVARIATE_VALUE = 1.5
const SIM_SAMPLES = 10_000

const FAMILY_CONFIG = Dict(
    "exp" => (; rate = 0.35, beta = 0.6, horizon = 5.0),
    "wei" => (; shape = 1.35, scale = 0.4, beta = -0.35, horizon = 5.0),
    "gom" => (; shape = 0.6, rate = 0.4, beta = 0.5, horizon = 5.0),
)

function build_test_model(family::String, effect::Symbol, with_covariate::Bool)
    cfg = FAMILY_CONFIG[family]
    
    if with_covariate
        data = DataFrame(
            id = [1], tstart = [0.0], tstop = [cfg.horizon],
            statefrom = [1], stateto = [2], obstype = [1],
            x = [COVARIATE_VALUE]
        )
        formula = @formula(0 ~ x)
    else
        data = DataFrame(
            id = [1], tstart = [0.0], tstop = [cfg.horizon],
            statefrom = [1], stateto = [2], obstype = [1]
        )
        formula = @formula(0 ~ 1)
    end
    
    hazard = Hazard(formula, family, 1, 2; linpred_effect = effect)
    model = multistatemodel(hazard; data = data)
    
    if family == "exp"
        base = [log(cfg.rate)]
    elseif family == "wei"
        base = [log(cfg.shape), log(cfg.scale)]
    elseif family == "gom"
        base = [cfg.shape, log(cfg.rate)]
    end
    
    pars = with_covariate ? vcat(base, [cfg.beta]) : base
    hazname = model.hazards[1].hazname
    set_parameters!(model, NamedTuple{(hazname,)}((pars,)))
    
    return model, cfg
end

function expected_cdf(family::String, effect::Symbol, with_covariate::Bool, t::Float64)
    cfg = FAMILY_CONFIG[family]
    xval = with_covariate ? COVARIATE_VALUE : 0.0
    beta = with_covariate ? cfg.beta : 0.0
    
    cumhaz = if family == "exp"
        rate = effect == :ph ? cfg.rate * exp(beta * xval) : cfg.rate * exp(-beta * xval)
        rate * t
    elseif family == "wei"
        shape, scale = cfg.shape, cfg.scale
        mult = effect == :ph ? exp(beta * xval) : exp(-shape * beta * xval)
        scale * mult * (t^shape)
    elseif family == "gom"
        shape, rate = cfg.shape, cfg.rate
        linpred = beta * xval
        if effect == :ph
            (rate / shape) * exp(linpred) * (exp(shape * t) - 1)
        else
            time_scale = exp(-linpred)
            scaled_shape = shape * time_scale
            scaled_rate = rate * time_scale
            (scaled_rate / scaled_shape) * (exp(scaled_shape * t) - 1)
        end
    end
    
    return 1 - exp(-cumhaz)
end

function simulate_event_times(model, nsim::Int; rng = Random.default_rng())
    durations = Float64[]
    strategy = CachedTransformStrategy()
    while length(durations) < nsim
        path = simulate_path(model, 1; strategy = strategy, rng = rng)
        if path.states[end] != path.states[1]
            push!(durations, path.times[end] - path.times[1])
        end
    end
    return durations
end

function max_cdf_diff(durations::Vector{Float64}, cdf_func, horizon::Float64)
    sorted = sort(durations)
    n = length(sorted)
    ecdf_vals = (1:n) ./ n
    tcdf_vals = [cdf_func(t) for t in sorted]
    return maximum(abs.(ecdf_vals .- tcdf_vals))
end

println("Testing simulation diagnostic scenarios...")
Random.seed!(12345)

scenarios = [
    ("exp", :ph, false), ("exp", :ph, true),
    ("exp", :aft, false), ("exp", :aft, true),
    ("wei", :ph, false), ("wei", :ph, true),
    ("wei", :aft, false), ("wei", :aft, true),
    ("gom", :ph, false), ("gom", :ph, true),
    ("gom", :aft, false), ("gom", :aft, true),
]

for (family, effect, with_cov) in scenarios
    model, cfg = build_test_model(family, effect, with_cov)
    durations = simulate_event_times(model, SIM_SAMPLES)
    cdf_func = t -> expected_cdf(family, effect, with_cov, t)
    max_diff = max_cdf_diff(durations, cdf_func, cfg.horizon)
    status = max_diff < 0.02 ? "✅" : "❌"
    cov_str = with_cov ? "cov" : "nocov"
    println("$status $(uppercase(family)) $(uppercase(string(effect))) $cov_str: max_diff = $(round(max_diff, digits=4))")
end
