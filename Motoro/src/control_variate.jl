using Statistics

"""
    BetaMethod

Abstract type for control variate coefficient strategies.

Concrete subtypes: [`FixedBeta`](@ref), [`OptimalBeta`](@ref)
"""
abstract type BetaMethod end

"""
    FixedBeta(β) <: BetaMethod

Use a fixed, user-supplied control variate coefficient.

# Fields
- `β::Float64`: The control variate coefficient

See also: [`OptimalBeta`](@ref), [`ControlVariate`](@ref)
"""
struct FixedBeta <: BetaMethod
    β::Float64
end

"""
    OptimalBeta <: BetaMethod

Estimate the optimal control variate coefficient from the simulation itself.

The optimal coefficient `β* = Cov(V, C) / Var(C)` minimises the variance of the
adjusted estimator. It is computed from the same paths used for pricing, so it
introduces a small in-sample bias that vanishes as `reps → ∞`.

See also: [`FixedBeta`](@ref), [`ControlVariate`](@ref)
"""
struct OptimalBeta <: BetaMethod end

compute_beta(b::FixedBeta,  V, C) = b.β
compute_beta(::OptimalBeta, V, C) = cov(V, C) / var(C)


"""
    ControlVariate(option[, β])

Pairs a control variate option with a coefficient strategy.

The `option` must have a closed-form BSM price (i.e. support
`price(option, BlackScholes(), data)`). Its payoffs are evaluated on the same
simulated paths as the target option, and the known analytical price is used to
centre the correction.

# Arguments
- `option`: The control variate option (e.g. [`EuropeanCall`](@ref))
- `β`: Either a `Float64` (→ [`FixedBeta`](@ref)) or a [`BetaMethod`](@ref).
  Defaults to [`OptimalBeta`](@ref).

# Examples
```julia
atm = EuropeanCall(100.0, 1.0)

ControlVariate(atm)       # optimal β (default)
ControlVariate(atm, 1.0)  # fixed β = 1
```

See also: [`ControlVariateMonteCarlo`](@ref), [`BetaMethod`](@ref)
"""
struct ControlVariate{O, B<:BetaMethod}
    option::O
    β::B
end

ControlVariate(option)          = ControlVariate(option, OptimalBeta())
ControlVariate(option, β::Real) = ControlVariate(option, FixedBeta(Float64(β)))


"""
    ControlVariateMonteCarlo(steps, reps, control[, method])

Monte Carlo pricing model with a control variate for variance reduction.

Generates paths under the risk-neutral (Q) measure and evaluates both the
target option and the control variate on the same paths. The adjusted estimator

    V̂_cv = mean(V - β·(C - C_BSM))

has the same expectation as the plain estimator but lower variance. The degree
of reduction depends on the correlation between the two payoffs and the choice of β.

# Fields
- `steps::Int`: Number of time steps per path
- `reps::Int`: Number of simulation paths
- `method::VarianceReductionMethod`: Draw and pairing strategy
- `control::ControlVariate`: The control variate option and β method

# Examples
```julia
data   = MarketData(100.0, 0.05, 0.25, 0.0)
target = CashOrNothingCall(100.0, 1.0, 1.0)

price(target, ControlVariateMonteCarlo(1, 10_000, ControlVariate(EuropeanCall(100.0, 1.0))),      data)
price(target, ControlVariateMonteCarlo(1, 10_000, ControlVariate(EuropeanCall(100.0, 1.0), 1.0)), data)
```

See also: [`ControlVariate`](@ref), [`RiskNeutralMonteCarlo`](@ref)
"""
struct ControlVariateMonteCarlo{CV<:ControlVariate} <: MonteCarlo
    steps::Int
    reps::Int
    method::VarianceReductionMethod
    control::CV
end

ControlVariateMonteCarlo(steps::Int, reps::Int, control::ControlVariate) =
    ControlVariateMonteCarlo(steps, reps, VarianceReduction(PseudoRandom(), NoPairing()), control)


# Internal: collect discounted payoffs for a target option from simulated paths.
# Terminal-price options (vanilla, binary) use only paths[:, end].
# Path-dependent options (lookback, Asian) use the full row.
_target_payoffs(option::VanillaOption, paths, disc) =
    disc .* payoff.(option, paths[:, end])
_target_payoffs(option::BinaryOption,  paths, disc) =
    disc .* payoff.(option, paths[:, end])
_target_payoffs(option::ExoticOption,  paths, disc) =
    disc .* [payoff(option, row) for row in eachrow(paths)]


"""
    price(option, model::ControlVariateMonteCarlo, data::MarketData)

Price an option via Monte Carlo with a control variate.

Works for vanilla, binary, and path-dependent exotic options as targets.
The control variate option must support `price(cv, BlackScholes(), data)`.

# Returns
A [`SimulationResult`](@ref) with the CV-adjusted mean price and its standard error.

See also: [`ControlVariateMonteCarlo`](@ref), [`ControlVariate`](@ref)
"""
function price(option, model::ControlVariateMonteCarlo, data::MarketData)
    (; expiry) = option
    (; spot, rate, vol) = data
    (; control) = model

    c_bsm = price(control.option, BlackScholes(), data).price

    paths = asset_paths(model.method, model, spot, rate, vol, expiry)
    disc  = exp(-rate * expiry)

    V    = _target_payoffs(option, paths, disc)
    C    = disc .* payoff.(control.option, paths[:, end])
    β    = compute_beta(control.β, V, C)
    V_cv = V .- β .* (C .- c_bsm)

    return SimulationResult(mean(V_cv), std(V_cv) / sqrt(model.reps))
end
