#!/usr/bin/env julia

using CairoMakie
using DataFrames
using Distributions
using Random
using StatsBase
using StatsModels
using BSplineKit

# Conditionally load MultistateModels - either from environment or develop path
if !@isdefined(MultistateModels)
    try
        using MultistateModels
    catch
        using Pkg
        Pkg.develop(path=normpath(joinpath(@__DIR__, "..", "..")))
        using MultistateModels
    end
end

# Import specific functions we need
using MultistateModels: Hazard, multistatemodel, set_parameters!, simulate_path, 
    eval_hazard, eval_cumhaz, survprob, truncate_distribution, 
    CachedTransformStrategy, DirectTransformStrategy

# Helper function to get hazard params (replaces internal get_hazard_params)
function get_hazard_params_local(model)
    # Get the nested parameter structure directly from model
    return model.parameters.nested
end

const OUTPUT_DIR = normpath(joinpath(@__DIR__, "assets"))
mkpath(OUTPUT_DIR)
CairoMakie.activate!(type = "png", px_per_unit = 2.0)

# Set theme to avoid deprecated 'resolution' warning
set_theme!(; figure_padding = 10)

const COVARIATE_VALUE = 1.5
const DELTA_U = sqrt(eps())
const DELTA_T = sqrt(eps())
const SIM_SAMPLES = 40_000
const DIST_GRID_POINTS = 400

const FAMILY_CONFIG = Dict(
    "exp" => (; rate = 0.35, beta = 0.6, horizon = 5.0, hazard_start = 0.0),
    "wei" => (; shape = 1.35, scale = 0.4, beta = -0.35, horizon = 5.0, hazard_start = 0.02),
    "gom" => (; shape = 0.6, rate = 0.4, beta = 0.5, horizon = 5.0, hazard_start = 0.0),
    "sp"  => (; degree = 3, knots = [1.5, 3.5], boundaryknots = [0.0, 5.0], beta = 0.5, horizon = 5.0, hazard_start = 0.0, coefs = [0.1, 0.2, 0.4, 0.4, 0.2, 0.1]),
)

# Time-varying covariate configuration: covariate changes at t_changes boundaries
# Using multiple change points to test more complex TVC scenarios
const TVC_CONFIG = Dict(
    "exp" => (; rate = 0.35, beta = 0.6, horizon = 5.0, hazard_start = 0.0, t_changes = [1.5, 3.0], x_values = [0.5, 1.5, 2.5]),
    "wei" => (; rate = 0.0, shape = 1.35, scale = 0.4, beta = -0.35, horizon = 5.0, hazard_start = 0.02, t_changes = [1.5, 3.0], x_values = [0.5, 1.5, 2.5]),
    "gom" => (; rate = 0.4, shape = 0.6, beta = 0.5, horizon = 5.0, hazard_start = 0.0, t_changes = [1.5, 3.0], x_values = [0.5, 1.5, 2.5]),
    "sp"  => (; degree = 3, knots = [1.5, 3.5], boundaryknots = [0.0, 5.0], beta = 0.5, horizon = 5.0, hazard_start = 0.0, coefs = [0.1, 0.2, 0.4, 0.4, 0.2, 0.1], t_changes = [1.5, 3.0], x_values = [0.5, 1.5, 2.5]),
)

struct Scenario
    family::String
    effect::Symbol
    covariate_mode::Symbol  # :baseline, :covariate, or :tvc (time-varying)
    label::String
    slug::String
    config::NamedTuple
end

function Scenario(family::String, effect::Symbol, cov_mode::Symbol)
    if cov_mode == :tvc
        config = TVC_CONFIG[family]
        label = string(uppercasefirst(lowercase(family)), " ", uppercase(String(effect)), " time-varying covariate")
    else
        config = FAMILY_CONFIG[family]
        label = string(uppercasefirst(lowercase(family)), " ", uppercase(String(effect)), " ", cov_mode == :covariate ? "with covariate" : "baseline-only")
    end
    slug = string(family, "_", effect, "_", cov_mode)
    return Scenario(family, effect, cov_mode, label, slug, config)
end

# All 24 scenarios: 4 families × 2 effects × 3 covariate modes
# TVC scenarios require careful piecewise integration for AFT
const SCENARIOS = vcat(
    # Non-TVC: 16 scenarios (4 families × 2 effects × 2 covariate modes)
    [Scenario(fam, eff, cov) for fam in keys(FAMILY_CONFIG) for eff in (:ph, :aft) for cov in (:baseline, :covariate)],
    # TVC: 8 scenarios (4 families × 2 effects)
    [Scenario(fam, eff, :tvc) for fam in ["exp", "wei", "gom", "sp"] for eff in (:ph, :aft)]
)

function scenario_subject_df(scenario::Scenario)
    horizon = scenario.config.horizon
    if scenario.covariate_mode == :tvc
        # Time-varying covariate: multiple intervals with different x values
        t_changes = scenario.config.t_changes
        x_values = scenario.config.x_values
        
        # Build interval boundaries: [0, t_changes..., horizon]
        tstart_grid = vcat(0.0, t_changes)
        tstop_grid = vcat(t_changes, horizon)
        n_intervals = length(tstart_grid)
        
        df = DataFrame(
            id = fill(1, n_intervals),
            tstart = tstart_grid,
            tstop = tstop_grid,
            statefrom = fill(1, n_intervals),
            stateto = fill(2, n_intervals),
            obstype = fill(1, n_intervals),
            x = x_values,
        )
    else
        df = DataFrame(
            id = [1],
            tstart = [0.0],
            tstop = [horizon],
            statefrom = [1],
            stateto = [2],
            obstype = [1],
        )
        if scenario.covariate_mode == :covariate
            df.x = [COVARIATE_VALUE]
        end
    end
    return df
end

function hazard_formula(scenario::Scenario)
    (scenario.covariate_mode == :covariate || scenario.covariate_mode == :tvc) ? @formula(0 ~ x) : @formula(0 ~ 1)
end

function scenario_parameter_vector(scenario::Scenario)
    # Post-v0.3.0: Parameters are on NATURAL scale (not log-transformed)
    # Box constraints (lb ≥ 0) handle positivity
    cfg = scenario.config
    if scenario.family == "exp"
        # Exponential: rate on natural scale
        base = [cfg.rate]
    elseif scenario.family == "wei"
        # Weibull: shape and scale on natural scale
        base = [cfg.shape, cfg.scale]
    elseif scenario.family == "gom"
        # Gompertz: shape is unconstrained (can be negative), rate on natural scale
        base = [cfg.shape, cfg.rate]
    elseif scenario.family == "sp"
        # Spline coefficients are on natural scale (non-negative)
        base = cfg.coefs
    else
        error("Unsupported family $(scenario.family)")
    end
    return (scenario.covariate_mode == :covariate || scenario.covariate_mode == :tvc) ? vcat(base, [cfg.beta]) : base
end

function build_model(scenario::Scenario)
    data = scenario_subject_df(scenario)
    
    kwargs = Dict{Symbol,Any}(
        :linpred_effect => scenario.effect,
        :time_transform => true,
    )
    
    if scenario.family == "sp"
        kwargs[:degree] = scenario.config.degree
        kwargs[:knots] = scenario.config.knots
        kwargs[:boundaryknots] = scenario.config.boundaryknots
        kwargs[:natural_spline] = false # Disable natural spline to match simple BSplineBasis in tests
        kwargs[:extrapolation] = "flat" # Disable constant extrapolation (which enforces D1=0) to match simple BSplineBasis
    end
    
    hazard = Hazard(
        hazard_formula(scenario),
        scenario.family,
        1,
        2;
        kwargs...
    )
    model = multistatemodel(hazard; data = data)
    pars = scenario_parameter_vector(scenario)
    hazname = model.hazards[1].hazname
    set_parameters!(model, NamedTuple{(hazname,)}((pars,)))
    return model, data
end

function covariate_value(scenario::Scenario)
    return scenario.covariate_mode == :covariate ? COVARIATE_VALUE : 0.0
end

# Helper functions for piecewise hazard/cumhaz computation with TVC
function exp_ph_hazard(t, rate, beta, x)
    return rate * exp(beta * x)
end

function exp_ph_cumhaz(t, rate, beta, x)
    return rate * exp(beta * x) * t
end

function wei_ph_hazard(t, shape, scale, beta, x)
    return shape * scale * (t^(shape - 1)) * exp(beta * x)
end

function wei_ph_cumhaz(t, shape, scale, beta, x)
    return scale * exp(beta * x) * (t^shape)
end

# Piecewise cumulative hazard for multiple TVC intervals
function piecewise_exp_ph_cumhaz(t, rate, beta, t_changes, x_values)
    cumhaz = 0.0
    prev_t = 0.0
    for (i, tc) in enumerate(t_changes)
        if t <= tc
            cumhaz += rate * exp(beta * x_values[i]) * (t - prev_t)
            return cumhaz
        else
            cumhaz += rate * exp(beta * x_values[i]) * (tc - prev_t)
            prev_t = tc
        end
    end
    cumhaz += rate * exp(beta * x_values[end]) * (t - prev_t)
    return cumhaz
end

function piecewise_wei_ph_cumhaz(t, shape, scale, beta, t_changes, x_values)
    cumhaz = 0.0
    prev_t = 0.0
    for (i, tc) in enumerate(t_changes)
        if t <= tc
            cumhaz += scale * exp(beta * x_values[i]) * (t^shape - prev_t^shape)
            return cumhaz
        else
            cumhaz += scale * exp(beta * x_values[i]) * (tc^shape - prev_t^shape)
            prev_t = tc
        end
    end
    cumhaz += scale * exp(beta * x_values[end]) * (t^shape - prev_t^shape)
    return cumhaz
end

# Gompertz PH TVC: h(t|x) = b * exp(a*t + β*x)
# H(t) = Σᵢ (b/a) * exp(β*xᵢ) * [exp(a*tᵢ) - exp(a*tᵢ₋₁)]
function piecewise_gom_ph_cumhaz(t, shape, rate, beta, t_changes, x_values)
    cumhaz = 0.0
    prev_t = 0.0
    a = shape
    b = rate
    
    for (i, tc) in enumerate(t_changes)
        if t <= tc
            cumhaz += (b / a) * exp(beta * x_values[i]) * (exp(a * t) - exp(a * prev_t))
            return cumhaz
        else
            cumhaz += (b / a) * exp(beta * x_values[i]) * (exp(a * tc) - exp(a * prev_t))
            prev_t = tc
        end
    end
    cumhaz += (b / a) * exp(beta * x_values[end]) * (exp(a * t) - exp(a * prev_t))
    return cumhaz
end

# === AFT with TVC ===
# For AFT with TVC, the scaled time is: τ(t) = ∫₀ᵗ exp(-β*x(s)) ds
# This is piecewise constant when x(s) is piecewise constant:
# τ(t) = Σᵢ exp(-β*xᵢ) * (min(t, tᵢ) - tᵢ₋₁)

# Compute scaled time τ(t) for AFT with piecewise constant TVC
function compute_scaled_time_aft(t, beta, t_changes, x_values)
    tau = 0.0
    prev_t = 0.0
    for (i, tc) in enumerate(t_changes)
        scale_factor = exp(-beta * x_values[i])
        if t <= tc
            tau += scale_factor * (t - prev_t)
            return tau
        else
            tau += scale_factor * (tc - prev_t)
            prev_t = tc
        end
    end
    tau += exp(-beta * x_values[end]) * (t - prev_t)
    return tau
end

# Exponential AFT TVC: h(t|x(t)) = λ * exp(-β*x(t)), H(t) = λ * τ(t)
function piecewise_exp_aft_cumhaz(t, rate, beta, t_changes, x_values)
    tau = compute_scaled_time_aft(t, beta, t_changes, x_values)
    return rate * tau
end

# Weibull AFT TVC: H(t) = σ * τ(t)^κ
# Note: This is the closed-form for AFT where time scaling affects cumulative hazard
function piecewise_wei_aft_cumhaz(t, shape, scale, beta, t_changes, x_values)
    tau = compute_scaled_time_aft(t, beta, t_changes, x_values)
    return scale * (tau^shape)
end

# Gompertz AFT TVC: H(t) = (b/a') * [exp(a'*τ(t)) - 1] where a' = a (shape unchanged under AFT)
# The AFT formulation: h(t|x) = h₀(τ(t)) * dτ/dt = h₀(τ(t)) * exp(-β*x(t))
# For Gompertz: h₀(τ) = b * exp(a*τ), so h(t|x) = b * exp(a*τ(t) - β*x(t))
# The cumulative hazard follows from integration
function piecewise_gom_aft_cumhaz(t, shape, rate, beta, t_changes, x_values)
    tau = compute_scaled_time_aft(t, beta, t_changes, x_values)
    a = shape
    b = rate
    # H(t) = (b/a) * [exp(a*τ(t)) - 1]
    return (b / a) * (exp(a * tau) - 1)
end

# Helper for spline evaluation
function make_spline_basis(cfg)
    # Reconstruct the basis used by MultistateModels
    # knots + boundaryknots -> full breakpoints
    breakpoints = vcat(cfg.boundaryknots[1], cfg.knots, cfg.boundaryknots[2])
    return BSplineBasis(BSplineOrder(cfg.degree + 1), breakpoints)
end

function eval_spline_hazard(t, coefs, basis)
    spline = Spline(basis, coefs)
    return spline(t)
end

function eval_spline_cumhaz(t, coefs, basis)
    spline = Spline(basis, coefs)
    # Integral from first knot (0.0) to t
    int_spline = integral(spline)
    return int_spline(t)
end

function expected_curves(scenario::Scenario, times_h::Vector{Float64}, times_cs::Vector{Float64})
    cfg = scenario.config
    
    if scenario.covariate_mode == :tvc
        # Time-varying covariate scenario with multiple change points
        t_changes = cfg.t_changes
        x_values = cfg.x_values
        beta = cfg.beta
        
        # Helper to find covariate value at time t
        function get_x_at_t(t)
            for (i, tc) in enumerate(t_changes)
                if t < tc
                    return x_values[i]
                end
            end
            return x_values[end]
        end
        
        if scenario.family == "exp"
            rate = cfg.rate
            
            if scenario.effect == :ph
                # PH: h(t|x) = λ * exp(β*x(t))
                haz_expected = [rate * exp(beta * get_x_at_t(t)) for t in times_h]
                cum_expected = [piecewise_exp_ph_cumhaz(t, rate, beta, t_changes, x_values) for t in times_cs]
            else
                # AFT: h(t|x(t)) = λ * exp(-β*x(t))
                haz_expected = [rate * exp(-beta * get_x_at_t(t)) for t in times_h]
                cum_expected = [piecewise_exp_aft_cumhaz(t, rate, beta, t_changes, x_values) for t in times_cs]
            end
            
        elseif scenario.family == "wei"
            shape = cfg.shape
            scale = cfg.scale
            
            if scenario.effect == :ph
                # PH: h(t|x) = κσt^(κ-1) * exp(β*x(t))
                haz_expected = [shape * scale * (t^(shape - 1)) * exp(beta * get_x_at_t(t)) for t in times_h]
                cum_expected = [piecewise_wei_ph_cumhaz(t, shape, scale, beta, t_changes, x_values) for t in times_cs]
            else
                # AFT: h(t|x) at time t uses τ(t) and dτ/dt = exp(-β*x(t))
                # h(t) = h₀(τ(t)) * exp(-β*x(t)) = κστ(t)^(κ-1) * exp(-β*x(t))
                haz_expected = map(times_h) do t
                    tau = compute_scaled_time_aft(t, beta, t_changes, x_values)
                    scale_factor = exp(-beta * get_x_at_t(t))
                    return shape * scale * (tau^(shape - 1)) * scale_factor
                end
                cum_expected = [piecewise_wei_aft_cumhaz(t, shape, scale, beta, t_changes, x_values) for t in times_cs]
            end
            
        elseif scenario.family == "gom"
            a = cfg.shape
            b = cfg.rate
            
            if scenario.effect == :ph
                # PH: h(t|x) = b * exp(a*t + β*x(t))
                haz_expected = [b * exp(a * t + beta * get_x_at_t(t)) for t in times_h]
                cum_expected = [piecewise_gom_ph_cumhaz(t, a, b, beta, t_changes, x_values) for t in times_cs]
            else
                # AFT: h(t) = h₀(τ(t)) * exp(-β*x(t)) = b * exp(a*τ(t)) * exp(-β*x(t))
                haz_expected = map(times_h) do t
                    tau = compute_scaled_time_aft(t, beta, t_changes, x_values)
                    scale_factor = exp(-beta * get_x_at_t(t))
                    return b * exp(a * tau) * scale_factor
                end
                cum_expected = [piecewise_gom_aft_cumhaz(t, a, b, beta, t_changes, x_values) for t in times_cs]
            end
            
        elseif scenario.family == "sp"
            basis = make_spline_basis(cfg)
            coefs = cfg.coefs
            
            if scenario.effect == :ph
                # PH: h(t|x) = h₀(t) * exp(β*x(t))
                haz_expected = [eval_spline_hazard(t, coefs, basis) * exp(beta * get_x_at_t(t)) for t in times_h]
                
                # Cumulative hazard: piecewise integration
                function piecewise_sp_ph_cumhaz(t)
                    cumhaz = 0.0
                    prev_t = 0.0
                    H0 = u -> eval_spline_cumhaz(u, coefs, basis)
                    
                    for (i, tc) in enumerate(t_changes)
                        if t <= tc
                            cumhaz += (H0(t) - H0(prev_t)) * exp(beta * x_values[i])
                            return cumhaz
                        else
                            cumhaz += (H0(tc) - H0(prev_t)) * exp(beta * x_values[i])
                            prev_t = tc
                        end
                    end
                    cumhaz += (H0(t) - H0(prev_t)) * exp(beta * x_values[end])
                    return cumhaz
                end
                
                cum_expected = [piecewise_sp_ph_cumhaz(t) for t in times_cs]
            else
                # AFT: h(t) = h₀(τ(t)) * exp(-β*x(t))
                haz_expected = map(times_h) do t
                    tau = compute_scaled_time_aft(t, beta, t_changes, x_values)
                    scale_factor = exp(-beta * get_x_at_t(t))
                    return eval_spline_hazard(tau, coefs, basis) * scale_factor
                end
                # H(t) = H₀(τ(t))
                cum_expected = map(times_cs) do t
                    tau = compute_scaled_time_aft(t, beta, t_changes, x_values)
                    return eval_spline_cumhaz(tau, coefs, basis)
                end
            end
        else
            error("TVC not implemented for family $(scenario.family)")
        end
    else
        # Non-TVC scenarios (original code)
        xval = covariate_value(scenario)
        beta = scenario.covariate_mode == :covariate ? cfg.beta : 0.0
        if scenario.family == "exp"
            base_rate = cfg.rate
            rate = scenario.effect == :ph ? base_rate * exp(beta * xval) : base_rate * exp(-beta * xval)
            haz_expected = fill(rate, length(times_h))
            cum_expected = rate .* times_cs
        elseif scenario.family == "wei"
            shape = cfg.shape
            scale = cfg.scale
            multiplier = scenario.effect == :ph ? exp(beta * xval) : exp(-shape * beta * xval)
            haz_expected = shape * scale .* times_h .^ (shape - 1) .* multiplier
            cum_expected = scale * multiplier .* times_cs .^ shape
        elseif scenario.family == "gom"
            # flexsurv parameterization: h(t) = rate * exp(shape * t)
            # H(t) = (rate/shape) * (exp(shape*t) - 1) for shape != 0
            shape = cfg.shape
            rate = cfg.rate
            linpred = beta * xval
            if scenario.effect == :ph
                # PH: h(t|x) = rate * exp(shape*t + linpred)
                haz_expected = rate .* exp.(shape .* times_h .+ linpred)
                cum_expected = (rate / shape) * exp(linpred) .* (exp.(shape .* times_cs) .- 1)
            else
                # AFT: h(t|x) = rate * exp(shape*t*exp(-linpred)) * exp(-linpred)
                time_scale = exp(-linpred)
                scaled_shape = shape * time_scale
                scaled_rate = rate * time_scale
                haz_expected = scaled_rate .* exp.(scaled_shape .* times_h)
                cum_expected = (scaled_rate / scaled_shape) .* (exp.(scaled_shape .* times_cs) .- 1)
            end
        elseif scenario.family == "sp"
            basis = make_spline_basis(cfg)
            coefs = cfg.coefs
            linpred = beta * xval
            
            if scenario.effect == :ph
                # PH: h(t|x) = h0(t) * exp(linpred)
                haz_expected = [eval_spline_hazard(t, coefs, basis) for t in times_h] .* exp(linpred)
                cum_expected = [eval_spline_cumhaz(t, coefs, basis) for t in times_cs] .* exp(linpred)
            else
                # AFT: h(t|x) = h0(t * exp(-linpred)) * exp(-linpred)
                scale = exp(-linpred)
                haz_expected = [eval_spline_hazard(t * scale, coefs, basis) for t in times_h] .* scale
                cum_expected = [eval_spline_cumhaz(t * scale, coefs, basis) for t in times_cs]
            end
        else
            error("Unsupported family $(scenario.family)")
        end
    end
    surv_expected = exp.(-cum_expected)
    return (; haz_expected, cum_expected, surv_expected)
end

function distribution_functions(scenario::Scenario)
    cfg = scenario.config
    
    if scenario.covariate_mode == :tvc
        # Time-varying covariate - piecewise distribution with multiple intervals
        t_changes = cfg.t_changes
        x_values = cfg.x_values
        beta = cfg.beta
        
        # Helper to find covariate value at time t
        get_x_at_t = t -> begin
            for (i, tc) in enumerate(t_changes)
                if t < tc
                    return x_values[i]
                end
            end
            return x_values[end]
        end
        
        if scenario.family == "exp"
            rate = cfg.rate
            if scenario.effect == :ph
                cumhaz = t -> piecewise_exp_ph_cumhaz(t, rate, beta, t_changes, x_values)
                hazard = t -> rate * exp(beta * get_x_at_t(t))
            else  # AFT
                cumhaz = t -> piecewise_exp_aft_cumhaz(t, rate, beta, t_changes, x_values)
                hazard = t -> rate * exp(-beta * get_x_at_t(t))
            end
            
        elseif scenario.family == "wei"
            shape = cfg.shape
            scale = cfg.scale
            if scenario.effect == :ph
                cumhaz = t -> piecewise_wei_ph_cumhaz(t, shape, scale, beta, t_changes, x_values)
                hazard = t -> shape * scale * exp(beta * get_x_at_t(t)) * (t^(shape - 1))
            else  # AFT
                cumhaz = t -> piecewise_wei_aft_cumhaz(t, shape, scale, beta, t_changes, x_values)
                hazard = t -> begin
                    tau = compute_scaled_time_aft(t, beta, t_changes, x_values)
                    scale_factor = exp(-beta * get_x_at_t(t))
                    return shape * scale * (tau^(shape - 1)) * scale_factor
                end
            end
            
        elseif scenario.family == "gom"
            a = cfg.shape
            b = cfg.rate
            if scenario.effect == :ph
                cumhaz = t -> piecewise_gom_ph_cumhaz(t, a, b, beta, t_changes, x_values)
                hazard = t -> b * exp(a * t + beta * get_x_at_t(t))
            else  # AFT
                cumhaz = t -> piecewise_gom_aft_cumhaz(t, a, b, beta, t_changes, x_values)
                hazard = t -> begin
                    tau = compute_scaled_time_aft(t, beta, t_changes, x_values)
                    scale_factor = exp(-beta * get_x_at_t(t))
                    return b * exp(a * tau) * scale_factor
                end
            end
            
        elseif scenario.family == "sp"
            basis = make_spline_basis(cfg)
            coefs = cfg.coefs
            
            if scenario.effect == :ph
                cumhaz = t -> begin
                    total_H = 0.0
                    boundaries = vcat(0.0, t_changes, Inf)
                    for i in 1:length(x_values)
                        t_start = boundaries[i]
                        t_end = boundaries[i+1]
                        if t <= t_start
                            break
                        end
                        eff_end = min(t, t_end)
                        linpred = beta * x_values[i]
                        delta_H0 = eval_spline_cumhaz(eff_end, coefs, basis) - eval_spline_cumhaz(t_start, coefs, basis)
                        total_H += delta_H0 * exp(linpred)
                        if t <= t_end
                            break
                        end
                    end
                    return total_H
                end
                hazard = t -> eval_spline_hazard(t, coefs, basis) * exp(beta * get_x_at_t(t))
            else  # AFT
                cumhaz = t -> begin
                    tau = compute_scaled_time_aft(t, beta, t_changes, x_values)
                    return eval_spline_cumhaz(tau, coefs, basis)
                end
                hazard = t -> begin
                    tau = compute_scaled_time_aft(t, beta, t_changes, x_values)
                    scale_factor = exp(-beta * get_x_at_t(t))
                    return eval_spline_hazard(tau, coefs, basis) * scale_factor
                end
            end
        else
            error("TVC not implemented for family $(scenario.family)")
        end
    else
        # Non-TVC scenarios (original code)
        xval = covariate_value(scenario)
        beta = scenario.covariate_mode == :covariate ? cfg.beta : 0.0
        if scenario.family == "exp"
            base_rate = cfg.rate
            rate = scenario.effect == :ph ? base_rate * exp(beta * xval) : base_rate * exp(-beta * xval)
            cumhaz = t -> rate * t
            hazard = _ -> rate
        elseif scenario.family == "wei"
            shape = cfg.shape
            scale = cfg.scale
            multiplier = scenario.effect == :ph ? exp(beta * xval) : exp(-shape * beta * xval)
            cumhaz = t -> scale * multiplier * (t^shape)
            hazard = t -> shape * scale * multiplier * (t^(shape - 1))
        elseif scenario.family == "gom"
            # flexsurv parameterization: h(t) = rate * exp(shape * t)
            shape = cfg.shape
            rate = cfg.rate
            linpred = beta * xval
            if scenario.effect == :ph
                # PH: h(t|x) = rate * exp(shape*t + linpred)
                cumhaz = t -> (rate / shape) * exp(linpred) * (exp(shape * t) - 1)
                hazard = t -> rate * exp(shape * t + linpred)
            else
                # AFT: h(t|x) = rate * exp(shape*t*exp(-linpred)) * exp(-linpred)
                time_scale = exp(-linpred)
                scaled_shape = shape * time_scale
                scaled_rate = rate * time_scale
                cumhaz = t -> (scaled_rate / scaled_shape) * (exp(scaled_shape * t) - 1)
                hazard = t -> scaled_rate * exp(scaled_shape * t)
            end
        elseif scenario.family == "sp"
            basis = make_spline_basis(cfg)
            coefs = cfg.coefs
            linpred = beta * xval
            
            if scenario.effect == :ph
                # PH: h(t|x) = h0(t) * exp(linpred)
                cumhaz = t -> eval_spline_cumhaz(t, coefs, basis) * exp(linpred)
                hazard = t -> eval_spline_hazard(t, coefs, basis) * exp(linpred)
            else
                # AFT: h(t|x) = h0(t * exp(-linpred)) * exp(-linpred)
                scale = exp(-linpred)
                cumhaz = t -> eval_spline_cumhaz(t * scale, coefs, basis)
                hazard = t -> eval_spline_hazard(t * scale, coefs, basis) * scale
            end
        else
            error("Unsupported family $(scenario.family)")
        end
    end
    cdf = t -> t <= 0 ? 0.0 : 1 - exp(-cumhaz(t))
    pdf = t -> t <= 0 ? 0.0 : hazard(t) * exp(-cumhaz(t))
    return cdf, pdf
end

function hazard_time_grid(scenario::Scenario)
    horizon = scenario.config.horizon
    start = get(scenario.config, :hazard_start, 0.0)
    haz_start = start == 0.0 ? 0.0 : start
    return collect(range(haz_start, horizon; length = 200)), collect(range(0.0, horizon; length = 200))
end

function collect_event_durations(model, nsamples; use_cached_strategy::Bool, rng::AbstractRNG)
    durations = Vector{Float64}(undef, nsamples)
    collected = 0
    attempts = 0
    max_attempts = nsamples * 200
    strategy = use_cached_strategy ? CachedTransformStrategy() : DirectTransformStrategy()
    while collected < nsamples
        path = simulate_path(model, 1; strategy = strategy, rng = rng)
        attempts += 1
        attempts > max_attempts && error("Exceeded maximum attempts without enough uncensored paths")
        if path.states[end] != path.states[1]
            collected += 1
            durations[collected] = path.times[end] - path.times[1]
        end
    end
    return durations
end

function plot_function_panel(scenario::Scenario, model, data)
    times_h, times_cs = hazard_time_grid(scenario)
    curves = expected_curves(scenario, times_h, times_cs)
    hazard = model.hazards[1]
    # Use natural scale parameters for eval_hazard
    all_pars = get_hazard_params_local(model)
    pars = all_pars[hazard.hazname]
    
    if scenario.covariate_mode == :tvc
        # For TVC, find the correct row for each time t
        # data has rows corresponding to intervals [tstart, tstop)
        
        function get_row_at_t(t)
            for i in 1:nrow(data)
                if t >= data.tstart[i] && t < data.tstop[i]
                    return data[i, :]
                end
            end
            return data[end, :] # For t == horizon
        end
        
        haz_calc = [eval_hazard(hazard, t, pars, get_row_at_t(t); apply_transform = false) for t in times_h]
        haz_tt = [eval_hazard(hazard, t, pars, get_row_at_t(t); apply_transform = true) for t in times_h]
        
        # Cumulative hazard: piecewise integration
        function piecewise_cumhaz(t, apply_transform)
            ch = 0.0
            current_t = 0.0
            for i in 1:nrow(data)
                row = data[i, :]
                interval_end = data.tstop[i]
                if t <= interval_end
                    ch += eval_cumhaz(hazard, current_t, t, pars, row; apply_transform = apply_transform)
                    return ch
                else
                    ch += eval_cumhaz(hazard, current_t, interval_end, pars, row; apply_transform = apply_transform)
                    current_t = interval_end
                end
            end
            return ch
        end

        cum_calc = [piecewise_cumhaz(t, false) for t in times_cs]
        cum_tt = [piecewise_cumhaz(t, true) for t in times_cs]
        
        # Survival: exp(-cumhaz)
        surv_calc = exp.(-cum_calc)
        surv_tt = exp.(-cum_tt)
    else
        # Non-TVC: use single row
        subj_row = data[1, :]
        haz_calc = [eval_hazard(hazard, t, pars, subj_row; apply_transform = false) for t in times_h]
        haz_tt = [eval_hazard(hazard, t, pars, subj_row; apply_transform = true) for t in times_h]
        cum_calc = [eval_cumhaz(hazard, 0.0, t, pars, subj_row; apply_transform = false) for t in times_cs]
        cum_tt = [eval_cumhaz(hazard, 0.0, t, pars, subj_row; apply_transform = true) for t in times_cs]
        surv_calc = [survprob(0.0, t, all_pars, subj_row, model.totalhazards[1], model.hazards; give_log = false, apply_transform = false) for t in times_cs]
        surv_tt = [survprob(0.0, t, all_pars, subj_row, model.totalhazards[1], model.hazards; give_log = false, apply_transform = true) for t in times_cs]
    end

    fig = Figure(size = (1400, 720))
    colors = Dict(:expected => :black, :calc => :dodgerblue, :tt => :darkorange)

    ax1 = Axis(fig[1, 1], title = "Hazard", xlabel = "Time", ylabel = "h(t)")
    lines!(ax1, times_h, curves.haz_expected, color = colors[:expected], linewidth = 3, label = "analytic")
    lines!(ax1, times_h, haz_calc, color = colors[:calc], linewidth = 2, label = "eval_hazard")
    lines!(ax1, times_h, haz_tt, color = colors[:tt], linewidth = 2, linestyle = :dash, label = "eval_hazard (time transform)")
    axislegend(ax1, position = :rb)

    ax2 = Axis(fig[1, 2], title = "Cumulative hazard", xlabel = "Time", ylabel = "Λ(t)")
    lines!(ax2, times_cs, curves.cum_expected, color = colors[:expected], linewidth = 3)
    lines!(ax2, times_cs, cum_calc, color = colors[:calc], linewidth = 2)
    lines!(ax2, times_cs, cum_tt, color = colors[:tt], linewidth = 2, linestyle = :dash)

    ax3 = Axis(fig[2, 1:2], title = "Survival", xlabel = "Time", ylabel = "S(t)")
    lines!(ax3, times_cs, curves.surv_expected, color = colors[:expected], linewidth = 3)
    lines!(ax3, times_cs, surv_calc, color = colors[:calc], linewidth = 2)
    lines!(ax3, times_cs, surv_tt, color = colors[:tt], linewidth = 2, linestyle = :dash)

    fname = joinpath(OUTPUT_DIR, "function_panel_$(scenario.slug).png")
    save(fname, fig)
    println("saved $(basename(fname))")
end

function plot_distribution_panel(scenario::Scenario, model)
    seed = hash(scenario.slug)
    rng_tt = Random.MersenneTwister(seed)
    rng_fb = Random.MersenneTwister(seed)
    durations_tt = collect_event_durations(model, SIM_SAMPLES; use_cached_strategy = true, rng = rng_tt)
    durations_fb = collect_event_durations(model, SIM_SAMPLES; use_cached_strategy = false, rng = rng_fb)

    ecdf_tt = ecdf(durations_tt)
    ecdf_fb = ecdf(durations_fb)
    horizon = scenario.config.horizon
    ts = collect(range(0.0, horizon; length = DIST_GRID_POINTS))
    cdf_base, pdf_base = distribution_functions(scenario)
    cdf_fn, pdf_fn = truncate_distribution(cdf_base, pdf_base; lower = 0.0, upper = horizon)
    expected = cdf_fn.(ts)
    empirical = ecdf_fb.(ts)
    diff_curve = ecdf_tt.(ts) .- ecdf_fb.(ts)
    max_abs_diff = maximum(abs.(diff_curve))
    ylim_span = max(max_abs_diff, 1e-6)
    xs = ts

    # Compute KS statistic at logarithmically-spaced sample sizes
    # KS_n = max_{i=1:n} |i/n - F(x_{(i)})| where x_{(i)} is the i-th order statistic
    # This should decrease as ~1/√n for correctly distributed samples
    sorted_durations = sort(durations_fb)
    n_samples = length(sorted_durations)
    expected_cdf_at_samples = cdf_fn.(sorted_durations)
    
    # Evaluate KS at specific sample sizes
    eval_ns = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, n_samples]
    eval_ns = filter(n -> n <= n_samples, eval_ns)
    ks_at_n = zeros(length(eval_ns))
    
    for (idx, n) in enumerate(eval_ns)
        # KS statistic for first n samples (order statistics x_{(1)}, ..., x_{(n)})
        # KS_n = max_{i=1:n} max(|i/n - F(x_{(i)})|, |(i-1)/n - F(x_{(i)})|)
        max_diff = 0.0
        for i in 1:n
            cdf_i = expected_cdf_at_samples[i]  # F(x_{(i)})
            # Two-sided KS: check both sides of the step
            diff_upper = abs(i / n - cdf_i)
            diff_lower = abs((i - 1) / n - cdf_i)
            max_diff = max(max_diff, diff_upper, diff_lower)
        end
        ks_at_n[idx] = max_diff
    end

    fig = Figure(size = (1700, 480))
    ax1 = Axis(fig[1, 1], title = "ECDF vs expected", xlabel = "Duration", ylabel = "F(t)")
    lines!(ax1, ts, expected, color = :black, linewidth = 3, label = "analytic")
    lines!(ax1, ts, empirical, color = :dodgerblue, linewidth = 2, label = "simulate_path")
    axislegend(ax1, position = :rt)

    ax2 = Axis(fig[1, 2], title = "KS statistic vs sample size", xlabel = "Sample size (n)", ylabel = "Dₙ = sup|F̂ₙ − F|", xscale = log10)
    scatterlines!(ax2, eval_ns, ks_at_n, color = :crimson, linewidth = 2, markersize = 8)

    ax3 = Axis(fig[1, 3], title = "Time-transform parity", xlabel = "Duration", ylabel = "ΔF(t)")
    lines!(ax3, ts, diff_curve, color = :seagreen, linewidth = 2)
    hlines!(ax3, [0.0], color = :black, linestyle = :dash)
    ylims!(ax3, (-1.1 * ylim_span, 1.1 * ylim_span))
    axislegend(ax3, [LineElement(color = :seagreen)], ["F_tt − F_fallback"], position = :rb)

    ax_hist = Axis(fig[2, 1:3], title = "Density vs histogram", xlabel = "Duration", ylabel = "Density")
    hist!(ax_hist, durations_fb; bins = 80, normalization = :pdf, color = (:steelblue, 0.5), strokewidth = 0)
    lines!(ax_hist, xs, pdf_fn.(xs), color = :black, linewidth = 3)

    fname = joinpath(OUTPUT_DIR, "simulation_panel_$(scenario.slug).png")
    save(fname, fig)
    println("saved $(basename(fname)) (max |ΔF| = $(round(max_abs_diff; digits = 3)))")
    
    return (max_abs_diff = max_abs_diff, ks_at_n = ks_at_n, eval_ns = eval_ns)
end

# ===== KS Convergence Analysis =====

# Dvoretzky-Kiefer-Wolfowitz bound: P(D_n > ε) ≤ 2*exp(-2*n*ε²)
# So for confidence 1-α, threshold is sqrt(log(2/α)/(2n))
function dkw_bound(n::Int, alpha::Float64=0.05)
    return sqrt(log(2/alpha) / (2n))
end

# Compute KS statistic from sorted samples and true CDF values
function compute_ks_statistic(sorted_samples::Vector{Float64}, cdf_values::Vector{Float64}, n::Int)
    max_diff = 0.0
    for i in 1:n
        cdf_i = cdf_values[i]
        diff_upper = abs(i / n - cdf_i)
        diff_lower = abs((i - 1) / n - cdf_i)
        max_diff = max(max_diff, diff_upper, diff_lower)
    end
    return max_diff
end

# Compute KS statistic at multiple sample sizes
function compute_ks_by_sample_size(sorted_durations::Vector{Float64}, cdf_fn::Function, 
                                   sample_sizes::Vector{Int})
    n_total = length(sorted_durations)
    cdf_values = cdf_fn.(sorted_durations)
    
    ks_stats = Float64[]
    valid_ns = Int[]
    
    for n in sample_sizes
        if n <= n_total
            ks = compute_ks_statistic(sorted_durations, cdf_values, n)
            push!(ks_stats, ks)
            push!(valid_ns, n)
        end
    end
    
    return valid_ns, ks_stats
end

# Compute slope of log(KS) vs log(n) - should be approximately -0.5 for correct distribution
function compute_ks_slope(ns::Vector{Int}, ks_stats::Vector{Float64})
    if length(ns) < 2
        return NaN
    end
    log_ns = log.(ns)
    log_ks = log.(ks_stats)
    # Simple linear regression
    n = length(ns)
    mean_x = sum(log_ns) / n
    mean_y = sum(log_ks) / n
    numerator = sum((log_ns .- mean_x) .* (log_ks .- mean_y))
    denominator = sum((log_ns .- mean_x).^2)
    slope = denominator > 0 ? numerator / denominator : NaN
    return slope
end

# Generate KS convergence plot
function plot_ks_convergence(scenario::Scenario, ns::Vector{Int}, ks_stats::Vector{Float64})
    fig = Figure(size = (800, 600))
    
    # Main plot: KS vs 1/√n with DKW bound
    ax1 = Axis(fig[1, 1], 
               title = "KS Convergence: $(scenario.label)",
               xlabel = "1/√n",
               ylabel = "KS statistic (Dₙ)")
    
    inv_sqrt_n = 1.0 ./ sqrt.(ns)
    dkw_bounds = [dkw_bound(n) for n in ns]
    
    # Plot DKW bound region
    band!(ax1, inv_sqrt_n, zeros(length(ns)), dkw_bounds, color = (:gray, 0.2))
    
    # Plot DKW bound line
    lines!(ax1, inv_sqrt_n, dkw_bounds, color = :gray, linestyle = :dash, linewidth = 2, label = "DKW bound (α=0.05)")
    
    # Plot observed KS statistics
    scatterlines!(ax1, inv_sqrt_n, ks_stats, color = :crimson, linewidth = 2, markersize = 10, label = "Observed KS")
    
    # Linear fit line
    slope = compute_ks_slope(ns, ks_stats)
    if !isnan(slope)
        # Fit line in log-log space, then transform to 1/sqrt(n) space
        log_ns = log.(ns)
        log_ks = log.(ks_stats)
        mean_log_n = sum(log_ns) / length(ns)
        mean_log_ks = sum(log_ks) / length(ns)
        intercept = mean_log_ks - slope * mean_log_n
        
        fit_ks = exp.(slope .* log.(ns) .+ intercept)
        lines!(ax1, inv_sqrt_n, fit_ks, color = :blue, linestyle = :dot, linewidth = 2, 
               label = "Fit (slope=$(round(slope; digits=3)))")
    end
    
    axislegend(ax1, position = :rt)
    
    # Add text annotation with slope check
    expected_slope = -0.5
    slope_pass = -0.6 < slope < -0.4
    status_text = slope_pass ? "✓ PASS" : "✗ FAIL"
    status_color = slope_pass ? :green : :red
    
    text!(ax1, 0.02, 0.98, text = "Slope: $(round(slope; digits=3)) (expected ≈ -0.5)\n$status_text",
          align = (:left, :top), space = :relative, fontsize = 14, color = status_color)
    
    fname = joinpath(OUTPUT_DIR, "ks_convergence_$(scenario.slug).png")
    save(fname, fig)
    println("saved $(basename(fname)) (slope = $(round(slope; digits=3)), $(status_text))")
    
    return (slope = slope, pass = slope_pass)
end

# ===== Pass/Fail Tracking =====

mutable struct DiagnosticResult
    scenario_slug::String
    function_panel_saved::Bool
    simulation_panel_saved::Bool
    ks_convergence_saved::Bool
    max_delta_f::Float64      # Tang vs fallback parity
    ks_slope::Float64         # Should be ≈ -0.5
    delta_f_pass::Bool        # max |ΔF| < 1e-4
    ks_slope_pass::Bool       # slope ∈ [-0.6, -0.4]
end

DiagnosticResult(slug::String) = DiagnosticResult(slug, false, false, false, NaN, NaN, false, false)

const DELTA_F_THRESHOLD = 1e-4
const KS_SLOPE_MIN = -0.6
const KS_SLOPE_MAX = -0.4

function print_summary(results::Vector{DiagnosticResult})
    println("\n" * "="^80)
    println("DIAGNOSTIC SUMMARY")
    println("="^80)
    
    n_total = length(results)
    n_function_pass = count(r -> r.function_panel_saved, results)
    n_sim_pass = count(r -> r.simulation_panel_saved, results)
    n_ks_pass = count(r -> r.ks_convergence_saved, results)
    n_delta_f_pass = count(r -> r.delta_f_pass, results)
    n_slope_pass = count(r -> r.ks_slope_pass, results)
    
    println("\nPanels Generated:")
    println("  Function panels:   $n_function_pass / $n_total")
    println("  Simulation panels: $n_sim_pass / $n_total")
    println("  KS convergence:    $n_ks_pass / $n_total")
    
    println("\nQuality Checks:")
    println("  Time-transform parity (|ΔF| < $(DELTA_F_THRESHOLD)): $n_delta_f_pass / $n_total")
    println("  KS slope ∈ [$(KS_SLOPE_MIN), $(KS_SLOPE_MAX)]: $n_slope_pass / $n_total")
    
    # List failures
    failures = filter(r -> !r.delta_f_pass || !r.ks_slope_pass, results)
    if !isempty(failures)
        println("\n⚠️  FAILURES:")
        for r in failures
            reasons = String[]
            if !r.delta_f_pass
                push!(reasons, "ΔF=$(round(r.max_delta_f; sigdigits=3))")
            end
            if !r.ks_slope_pass
                push!(reasons, "slope=$(round(r.ks_slope; digits=3))")
            end
            println("  $(r.scenario_slug): $(join(reasons, ", "))")
        end
    else
        println("\n✓ ALL SCENARIOS PASS")
    end
    
    println("="^80)
end

function generate_all()
    results = DiagnosticResult[]
    
    for scenario in sort!(copy(SCENARIOS); by = s -> s.slug)
        println("\n--- $(scenario.label) ---")
        result = DiagnosticResult(scenario.slug)
        
        model, data = build_model(scenario)
        
        # Function panel
        try
            plot_function_panel(scenario, model, data)
            result.function_panel_saved = true
        catch e
            println("ERROR in function panel: $e")
        end
        
        # Simulation panel with KS analysis
        try
            sim_result = plot_distribution_panel(scenario, model)
            result.simulation_panel_saved = true
            result.max_delta_f = sim_result.max_abs_diff
            result.delta_f_pass = result.max_delta_f < DELTA_F_THRESHOLD
            
            # KS convergence plot
            ks_result = plot_ks_convergence(scenario, sim_result.eval_ns, sim_result.ks_at_n)
            result.ks_convergence_saved = true
            result.ks_slope = ks_result.slope
            result.ks_slope_pass = ks_result.pass
        catch e
            println("ERROR in simulation/KS panel: $e")
            @error "Full error" exception=(e, catch_backtrace())
        end
        
        push!(results, result)
    end
    
    print_summary(results)
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    generate_all()
end
