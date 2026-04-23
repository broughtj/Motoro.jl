using Distributions
using LinearAlgebra
using Statistics
using Random

norm_cdf(x) = cdf(Normal(0.0, 1.0), x)

"""
    PricingResult

Abstract type for option pricing results.

See also: [`AnalyticResult`](@ref), [`SimulationResult`](@ref)
"""
abstract type PricingResult end

"""
    AnalyticResult(price)

Result from an analytic pricing model (e.g., [`BlackScholes`](@ref), [`Binomial`](@ref)).

# Fields
- `price::Float64`: Option price
"""
struct AnalyticResult <: PricingResult
    price::Float64
end

"""
    SimulationResult(price, std)

Result from a simulation-based pricing model (e.g., [`RiskNeutralMonteCarlo`](@ref), [`HedgedMonteCarlo`](@ref)).

# Fields
- `price::Float64`: Estimated option price (discounted mean payoff)
- `std::Float64`: Standard error of the price estimate (standard deviation of payoffs divided by √reps)
"""
struct SimulationResult <: PricingResult
    price::Float64
    std::Float64
end

"""
    Binomial(steps)

Cox-Ross-Rubinstein (CRR) binomial tree pricing model.

Works for both European and American options, including dividend-paying underlyings.
Accuracy increases with `steps`; prices converge to the Black-Scholes-Merton value
as `steps → ∞`.

# Fields
- `steps::Int`: Number of time steps in the tree

# Examples
```julia
data = MarketData(41.0, 0.08, 0.30, 0.0)

price(EuropeanCall(40.0, 1.0), Binomial(100), data)
price(AmericanPut(40.0, 1.0),  Binomial(100), data)
```

See also: [`price`](@ref), [`BlackScholes`](@ref)
"""
struct Binomial
    steps::Int
end

function price(option::EuropeanOption, model::Binomial, data::MarketData)
    (; strike, expiry) = option
    (; spot, rate, vol, div) = data
    steps = model.steps

    dt = expiry / steps
    u = exp((rate - div) * dt + vol * sqrt(dt))
    d = exp((rate - div) * dt - vol * sqrt(dt))
    pu = (exp((rate - div) * dt) - d) / (u - d)
    pd = 1 - pu
    disc = exp(-rate * dt)

    s = zeros(steps + 1)
    x = zeros(steps + 1)

    @inbounds for i in 1:steps+1
        s[i] = spot * u^(steps + 1 - i) * d^(i - 1)
        x[i] = payoff(option, s[i])
    end

    for j in steps:-1:1
        @inbounds for i in 1:j
            x[i] = disc * (pu * x[i] + pd * x[i + 1])
        end
    end

    return AnalyticResult(x[1])
end


function price(option::AmericanOption, model::Binomial, data::MarketData)
    (; strike, expiry) = option
    (; spot, rate, vol, div) = data
    steps = model.steps

    dt = expiry / steps
    u = exp((rate - div) * dt + vol * sqrt(dt))
    d = exp((rate - div) * dt - vol * sqrt(dt))
    pu = (exp((rate - div) * dt) - d) / (u - d)
    pd = 1 - pu
    disc = exp(-rate * dt)

    x = zeros(steps + 1)

    # Terminal payoffs
    @inbounds for i in 1:steps+1
        x[i] = payoff(option, spot * u^(steps + 1 - i) * d^(i - 1))
    end

    # Backward induction with early exercise
    for j in steps:-1:1
        @inbounds for i in 1:j
            continuation = disc * (pu * x[i] + pd * x[i + 1])
            exercise = payoff(option, spot * u^(j - i) * d^(i - 1))
            x[i] = max(continuation, exercise)
        end
    end

    return AnalyticResult(x[1])
end

"""
    BlackScholes()

Black-Scholes-Merton analytical pricing model for European options.

Provides exact closed-form solutions using the Black-Scholes-Merton formula. This is the
fastest and most accurate method for European options under the BSM assumptions
(constant volatility, log-normal price distribution, continuous dividend yield).

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
call = EuropeanCall(100.0, 1.0)
put = EuropeanPut(100.0, 1.0)

model = BlackScholes()
call_price = price(call, model, data)
put_price = price(put, model, data)
```

See also: [`price`](@ref), [`Binomial`](@ref)
"""
struct BlackScholes end

function price(option::EuropeanCall, model::BlackScholes, data::MarketData)
    (; strike, expiry) = option
    (; spot, rate, vol, div) = data

    d1 = (log(spot / strike) + (rate - div + 0.5 * vol^2) * expiry) / (vol * sqrt(expiry))
    d2 = d1 - vol * sqrt(expiry)

    p = spot * exp(-div * expiry) * norm_cdf(d1) - strike * exp(-rate * expiry) * norm_cdf(d2)

    return AnalyticResult(p)
end


"""
    delta(option::EuropeanCall, model::BlackScholes, data::MarketData) -> Float64

Black-Scholes-Merton delta for a European call option.

Delta measures the sensitivity of the option price to a unit change in the spot price,
i.e., `∂V/∂S`. For a call, delta is in `[0, 1]`.

# Arguments
- `option::EuropeanCall`: The call option contract
- `model::BlackScholes`: BSM analytical model
- `data::MarketData`: Current market parameters

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
call = EuropeanCall(100.0, 1.0)
delta(call, BlackScholes(), data)  # ≈ 0.637
```

See also: [`price`](@ref), [`BlackScholes`](@ref)
"""
function delta(option::EuropeanCall, model::BlackScholes, data::MarketData)
    (; strike, expiry) = option
    (; spot, rate, vol, div) = data

    d1 = (log(spot / strike) + (rate - div + 0.5 * vol^2) * expiry) / (vol * sqrt(expiry))
    delta = exp(-div * expiry) * norm_cdf(d1)

    return delta
end


function price(option::EuropeanPut, model::BlackScholes, data::MarketData)
    (; strike, expiry) = option
    (; spot, rate, vol, div) = data

    d1 = (log(spot / strike) + (rate - div + 0.5 * vol^2) * expiry) / (vol * sqrt(expiry))
    d2 = d1 - vol * sqrt(expiry)

    p = strike * exp(-rate * expiry) * norm_cdf(-d2) - spot * exp(-div * expiry) * norm_cdf(-d1)

    return AnalyticResult(p)
end

"""
    geom_asian_price(option, spot, rate, vol, div, steps; fixings_observed=0, running_geom_mean=spot, next_fixing_time=option.expiry/steps)

Closed-form Kemna-Vorst price for fixed-strike geometric Asian options.
"""
function geom_asian_price(
    option::Union{FixedStrikeGeometricAsianCall{T}, FixedStrikeGeometricAsianPut{T}},
    spot::Real, rate::Real, vol::Real, div::Real, steps::Integer;
    fixings_observed::Integer = 0,
    running_geom_mean::Real = spot,
    next_fixing_time::Real = option.expiry / steps,
) where {T<:AbstractFloat}
    if fixings_observed != 0
        throw(ArgumentError("`geom_asian_price` currently supports fresh options only (`fixings_observed == 0`)."))
    end

    (; strike, expiry) = option
    σg = vol * sqrt(((steps + 1) * (2 * steps + 1)) / (6 * steps^2))
    μg = (rate - div - 0.5 * vol^2) * ((steps + 1) / (2 * steps)) + 0.5 * σg^2

    sqrtT = sqrt(expiry)
    d1 = (log(spot / strike) + (μg + 0.5 * σg^2) * expiry) / (σg * sqrtT)
    d2 = d1 - σg * sqrtT
    normal = Normal()

    if option isa FixedStrikeGeometricAsianCall
        p = exp(-rate * expiry) * (spot * exp(μg * expiry) * cdf(normal, d1) - strike * cdf(normal, d2))
    else
        p = exp(-rate * expiry) * (strike * cdf(normal, -d2) - spot * exp(μg * expiry) * cdf(normal, -d1))
    end

    return max(zero(T), p)
end

"""
    geom_asian_price(option, data, steps; kwargs...)

Convenience overload that accepts [`MarketData`](@ref), mirroring package pricing APIs.
"""
function geom_asian_price(
    option::Union{FixedStrikeGeometricAsianCall{T}, FixedStrikeGeometricAsianPut{T}},
    data::MarketData,
    steps::Integer;
    fixings_observed::Integer = 0,
    running_geom_mean::Real = data.spot,
    next_fixing_time::Real = option.expiry / steps,
) where {T<:AbstractFloat}
    (; spot, rate, vol, div) = data
    return geom_asian_price(
        option, spot, rate, vol, div, steps;
        fixings_observed=fixings_observed,
        running_geom_mean=running_geom_mean,
        next_fixing_time=next_fixing_time,
    )
end

"""
    price(option, model::BlackScholes, data::MarketData; steps=252)

Black-Scholes-style API wrapper for fixed-strike geometric Asian options.
The number of monitoring dates is provided via `steps`.
"""
function price(
    option::Union{FixedStrikeGeometricAsianCall{T}, FixedStrikeGeometricAsianPut{T}},
    model::BlackScholes,
    data::MarketData;
    steps::Integer = 252,
    fixings_observed::Integer = 0,
    running_geom_mean::Real = data.spot,
    next_fixing_time::Real = option.expiry / steps,
) where {T<:AbstractFloat}
    p = geom_asian_price(
        option, data, steps;
        fixings_observed=fixings_observed,
        running_geom_mean=running_geom_mean,
        next_fixing_time=next_fixing_time,
    )
    return AnalyticResult(p)
end


"""
    delta(option::EuropeanPut, model::BlackScholes, data::MarketData) -> Float64

Black-Scholes-Merton delta for a European put option.

Delta measures the sensitivity of the option price to a unit change in the spot price,
i.e., `∂V/∂S`. For a put, delta is in `[-1, 0]`.

# Arguments
- `option::EuropeanPut`: The put option contract
- `model::BlackScholes`: BSM analytical model
- `data::MarketData`: Current market parameters

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
put = EuropeanPut(100.0, 1.0)
delta(put, BlackScholes(), data)  # ≈ -0.363
```

See also: [`price`](@ref), [`BlackScholes`](@ref)
"""
function delta(option::EuropeanPut, model::BlackScholes, data::MarketData)
    (; strike, expiry) = option
    (; spot, rate, vol, div) = data

    d1 = (log(spot / strike) + (rate - div + 0.5 * vol^2) * expiry) / (vol * sqrt(expiry))
    delta = -1.0 * exp(-div * expiry) * norm_cdf(-d1)

    return delta
end


"""
    VarianceReductionMethod

Abstract type for Monte Carlo variance reduction strategies.

See also: [`VarianceReduction`](@ref), [`DrawMethod`](@ref), [`PairingMethod`](@ref)
"""
abstract type VarianceReductionMethod end

"""
    DrawMethod

Abstract type for random draw generation strategies.

Concrete subtypes: [`PseudoRandom`](@ref), [`Stratified`](@ref)
"""
abstract type DrawMethod end

"""
    PseudoRandom <: DrawMethod

Standard pseudo-random normal draws via `randn`. Baseline draw method.
"""
struct PseudoRandom <: DrawMethod end

"""
    Stratified <: DrawMethod

Stratified sampling draws. Divides `[0,1]` into `n` equal strata and draws one
sample per stratum, ensuring coverage of the full distribution and reducing
variance compared to pure pseudo-random sampling.
"""
struct Stratified <: DrawMethod end

"""
    PairingMethod

Abstract type for antithetic pairing strategies.

Concrete subtypes: [`NoPairing`](@ref), [`Antithetic`](@ref)
"""
abstract type PairingMethod end

"""
    NoPairing <: PairingMethod

No antithetic pairing; draws are used as-is.
"""
struct NoPairing  <: PairingMethod end

"""
    Antithetic <: PairingMethod

Antithetic variates pairing. For each draw `z`, also simulates a path using `-z`.
The negative correlation between paired paths reduces estimator variance.
"""
struct Antithetic <: PairingMethod end

"""
    VarianceReduction{D<:DrawMethod, P<:PairingMethod}(draw, pairing)

Combines a draw method and a pairing method into a variance reduction strategy
for use with [`RiskNeutralMonteCarlo`](@ref) or [`HedgedMonteCarlo`](@ref).

# Fields
- `draw::DrawMethod`: How to generate random draws ([`PseudoRandom`](@ref) or [`Stratified`](@ref))
- `pairing::PairingMethod`: Pairing strategy ([`NoPairing`](@ref) or [`Antithetic`](@ref))

# Examples
```julia
VarianceReduction(PseudoRandom(), NoPairing())   # baseline
VarianceReduction(PseudoRandom(), Antithetic())  # antithetic variates
VarianceReduction(Stratified(),   NoPairing())   # stratified sampling
VarianceReduction(Stratified(),   Antithetic())  # both combined
```
"""
struct VarianceReduction{D<:DrawMethod, P<:PairingMethod} <: VarianceReductionMethod
    draw::D
    pairing::P
end

"""
    MonteCarlo

Abstract type for all Monte Carlo simulation models.

Concrete subtypes differ in the pricing measure and hedging strategy used:
- [`RiskNeutralMonteCarlo`](@ref): standard discounted expected payoff under the Q measure
- [`HedgedMonteCarlo`](@ref): real-world hedge cost simulation under the P measure

See also: [`RiskNeutralMonteCarlo`](@ref), [`HedgedMonteCarlo`](@ref)
"""
abstract type MonteCarlo end

abstract type AssetPaths end

"""
    RiskNeutralMonteCarlo(steps, reps[, method])

Monte Carlo simulation model for pricing options under the risk-neutral (Q) measure.

Generates asset price paths via geometric Brownian motion using the risk-free rate
as the drift and returns the discounted expected payoff.

# Fields
- `steps::Int`: Number of time steps per simulation path
- `reps::Int`: Number of simulation paths (replications)
- `method::VarianceReductionMethod`: Variance reduction strategy (default: `PseudoRandom` with `NoPairing`)

# Examples
```julia
data = MarketData(41.0, 0.08, 0.30, 0.0)
call = EuropeanCall(40.0, 1.0)

price(call, RiskNeutralMonteCarlo(100, 10_000), data)
price(call, RiskNeutralMonteCarlo(100, 10_000, VarianceReduction(PseudoRandom(), Antithetic())), data)
price(call, RiskNeutralMonteCarlo(100, 10_000, VarianceReduction(Stratified(), NoPairing())), data)
```

See also: [`MonteCarlo`](@ref), [`HedgedMonteCarlo`](@ref), [`asset_paths`](@ref)
"""
struct RiskNeutralMonteCarlo <: MonteCarlo
    steps::Int
    reps::Int
    method::VarianceReductionMethod
end

RiskNeutralMonteCarlo(steps::Int, reps::Int) = RiskNeutralMonteCarlo(steps, reps, VarianceReduction(PseudoRandom(), NoPairing()))

"""
    HedgeStrategy

Abstract type for hedging strategies used with [`HedgedMonteCarlo`](@ref).

Concrete subtypes: [`StopLoss`](@ref)
"""
abstract type HedgeStrategy end

"""
    HedgedMonteCarlo(steps, reps, strategy[, method])

Monte Carlo simulation model for estimating the cost of a hedging strategy under
the real-world (P) measure. Asset paths are simulated using the strategy's drift
rather than the risk-free rate.

# Fields
- `steps::Int`: Number of time steps per simulation path
- `reps::Int`: Number of simulation paths (replications)
- `method::VarianceReductionMethod`: Variance reduction strategy (default: `PseudoRandom` with `NoPairing`)
- `strategy::HedgeStrategy`: The hedging strategy to evaluate (e.g., [`StopLoss`](@ref))

# Examples
```julia
data = MarketData(41.0, 0.08, 0.30, 0.0)
call = EuropeanCall(40.0, 1.0)

price(call, HedgedMonteCarlo(100, 10_000, StopLoss(0.10)), data)
```

See also: [`MonteCarlo`](@ref), [`RiskNeutralMonteCarlo`](@ref), [`HedgeStrategy`](@ref), [`StopLoss`](@ref)
"""
struct HedgedMonteCarlo{S<:HedgeStrategy} <: MonteCarlo
    steps::Int
    reps::Int
    method::VarianceReductionMethod
    strategy::S
end

HedgedMonteCarlo(steps::Int, reps::Int, strategy::HedgeStrategy) =
    HedgedMonteCarlo(steps, reps, VarianceReduction(PseudoRandom(), NoPairing()), strategy)


function generate_draws(::PseudoRandom, n::Int)
    randn(n)
end

function generate_draws(::Stratified, n::Int)
    u = rand(n)
    d = Normal()
    draws = [quantile(d, (i - 1 + u[i]) / n) for i in 1:n]
    shuffle!(draws)
    return draws
end

function apply_pairing(::NoPairing, draws)
    draws
end

function apply_pairing(::Antithetic, draws)
    [draws; -draws]
end

# """
#     asset_paths(method::VarianceReduction, model::MonteCarlo, spot, rate, vol, expiry)

# Generate simulated asset price paths using geometric Brownian motion.

# Works with any [`MonteCarlo`](@ref) subtype (`RiskNeutralMonteCarlo` or `HedgedMonteCarlo`).

# # Arguments
# - `method::VarianceReduction`: Variance reduction strategy controlling draw generation and pairing
# - `model::MonteCarlo`: Any Monte Carlo model (provides `steps` and `reps`)
# - `spot`: Initial asset price
# - `rate`: Drift rate (risk-free rate for risk-neutral pricing; real-world drift for hedging)
# - `vol`: Volatility (annualized)
# - `expiry`: Time to expiration in years

# # Returns
# Matrix of size `(reps, steps+1)`. Each row is one simulated price path;
# column 1 is the initial spot price and column `steps+1` is the terminal price.

# # Examples
# ```julia
# data  = MarketData(41.0, 0.08, 0.30, 0.0)
# model = RiskNeutralMonteCarlo(100, 1_000)
# paths = asset_paths(model.method, model, data.spot, data.rate, data.vol, 1.0)
# size(paths)  # (1000, 101)
# ```

# See also: [`RiskNeutralMonteCarlo`](@ref), [`HedgedMonteCarlo`](@ref), [`price`](@ref)
# """
# function asset_paths(method::VarianceReduction, model::MonteCarlo, spot, rate, vol, expiry)
#     (; steps, reps) = model

#     dt = expiry / steps
#     nudt = (rate - 0.5 * vol^2) * dt
#     sidt = vol * sqrt(dt)
#     n = reps ÷ (method.pairing isa Antithetic ? 2 : 1)

#     paths = zeros(reps, steps + 1)
#     paths[:, 1] .= spot

#     @inbounds for j in 2:steps+1
#         z = generate_draws(method.draw, n)
#         z = apply_pairing(method.pairing, z)
#         paths[:, j] = paths[:, j-1] .* exp.(nudt .+ sidt .* z)
#     end

#     return paths
# end


function price(option::EuropeanOption, model::RiskNeutralMonteCarlo, data::MarketData)
    (; spot, rate, vol) = data
    process = GeometricBrownianMotion(model.method, model, spot, rate, vol, option.expiry)
    return price(option, model, data, process)
end

function price(option::EuropeanOption, model::RiskNeutralMonteCarlo, data::MarketData, process::AssetPaths)
    (; strike, expiry) = option
    (; rate) = data

    paths = asset_paths(process)
    disc_payoffs = exp(-rate * expiry) .* payoff.(option, paths[:, end])

    return SimulationResult(mean(disc_payoffs), std(disc_payoffs) / sqrt(model.reps))
end

function price(
    option::GeometricAsianOption,
    model::RiskNeutralMonteCarlo,
    data::MarketData;
    fixings_observed::Integer = 0,
    running_geom_mean::Real = data.spot,
)
    (; spot, rate, vol) = data
    process = GeometricBrownianMotion(model.method, model, spot, rate, vol, option.expiry)
    return price(
        option, model, data, process;
        fixings_observed=fixings_observed,
        running_geom_mean=running_geom_mean,
    )
end

function price(
    option::GeometricAsianOption,
    model::RiskNeutralMonteCarlo,
    data::MarketData,
    steps::Integer;
    fixings_observed::Integer = 0,
    running_geom_mean::Real = data.spot,
)
    if steps <= 0
        throw(ArgumentError("`steps` must be positive."))
    end

    model_with_steps = RiskNeutralMonteCarlo(steps, model.reps, model.method)
    return price(
        option, model_with_steps, data;
        fixings_observed=fixings_observed,
        running_geom_mean=running_geom_mean,
    )
end

function price(
    option::GeometricAsianOption,
    model::RiskNeutralMonteCarlo,
    data::MarketData,
    process::AssetPaths;
    fixings_observed::Integer = 0,
    running_geom_mean::Real = data.spot,
)
    if fixings_observed < 0
        throw(ArgumentError("`fixings_observed` must be non-negative."))
    end
    if running_geom_mean <= 0
        throw(ArgumentError("`running_geom_mean` must be positive."))
    end

    (; expiry) = option
    (; rate) = data
    paths = asset_paths(process)
    future_fixings = size(paths, 2)
    total_fixings = fixings_observed + future_fixings
    obs_weight = fixings_observed / total_fixings
    fut_weight = future_fixings / total_fixings

    disc_payoffs = exp(-rate * expiry) .* map(eachrow(paths)) do row
        future_geom_mean = geom_mean(row)
        combined_geom_mean = (running_geom_mean^obs_weight) * (future_geom_mean^fut_weight)

        if option isa FixedStrikeGeometricAsianCall
            max(0.0, combined_geom_mean - option.strike)
        elseif option isa FixedStrikeGeometricAsianPut
            max(0.0, option.strike - combined_geom_mean)
        elseif option isa FloatingStrikeGeometricAsianCall
            max(0.0, row[end] - combined_geom_mean)
        else
            max(0.0, combined_geom_mean - row[end])
        end
    end

    return SimulationResult(mean(disc_payoffs), std(disc_payoffs) / sqrt(model.reps))
end


"""
    price(option::ExoticOption, model::RiskNeutralMonteCarlo, data::MarketData)

Price a path-dependent exotic option via Monte Carlo simulation.

Each simulated path is passed in full to `payoff`, allowing the payoff to depend
on the entire price history (e.g., the running maximum, minimum, or average).

# Arguments
- `option::ExoticOption`: A lookback or Asian option contract
- `model::RiskNeutralMonteCarlo`: Simulation parameters (steps, reps, variance reduction method)
- `data::MarketData`: Market parameters (spot, rate, vol, div)

# Returns
A [`SimulationResult`](@ref) with the mean discounted payoff and its standard error.

# Examples
```julia
data  = MarketData(100.0, 0.05, 0.2, 0.0)
model = RiskNeutralMonteCarlo(252, 10_000)

price(FloatingStrikeLookbackCall(1.0),              model, data)
price(FloatingPriceLookbackPut(100.0, 1.0),         model, data)
price(FloatingPriceArithmeticAsianCall(100.0, 1.0), model, data)
```

See also: [`ExoticOption`](@ref), [`asset_paths`](@ref), [`SimulationResult`](@ref)
"""
function price(option::ExoticOption, model::RiskNeutralMonteCarlo, data::MarketData)
    (; spot, rate, vol) = data
    process = GeometricBrownianMotion(model.method, model, spot, rate, vol, option.expiry)
    return price(option, model, data, process)
end

function price(option::ExoticOption, model::RiskNeutralMonteCarlo, data::MarketData, process::AssetPaths)
    (; expiry) = option
    (; rate) = data

    paths = asset_paths(process)
    disc_payoffs = exp(-rate * expiry) .* [payoff(option, row) for row in eachrow(paths)]

    return SimulationResult(mean(disc_payoffs), std(disc_payoffs) / sqrt(model.reps))
end

function price(option::BinaryOption, model::RiskNeutralMonteCarlo, data::MarketData)
    (; spot, rate, vol) = data
    process = GeometricBrownianMotion(model.method, model, spot, rate, vol, option.expiry)
    return price(option, model, data, process)
end

function price(option::BinaryOption, model::RiskNeutralMonteCarlo, data::MarketData, process::AssetPaths)
    (; expiry) = option
    (; rate) = data

    paths = asset_paths(process)
    disc_payoffs = exp(-rate * expiry) .* payoff.(option, paths[:, end])

    return SimulationResult(mean(disc_payoffs), std(disc_payoffs) / sqrt(model.reps))
end





"""
    StopLoss(mu) <: HedgeStrategy

Stop-loss hedging strategy. Holds the underlying whenever the spot is at or above
the strike and holds cash otherwise. The drift `mu` is the real-world (P-measure)
expected return used when simulating asset paths.

# Fields
- `mu::Float64`: Expected drift of the underlying asset (annualized, as decimal)

# Examples
```julia
data  = MarketData(41.0, 0.08, 0.30, 0.0)
call  = EuropeanCall(40.0, 1.0)

price(call, HedgedMonteCarlo(100, 10_000, StopLoss(0.10)), data)
```

See also: [`HedgeStrategy`](@ref), [`HedgedMonteCarlo`](@ref)
"""
struct StopLoss <: HedgeStrategy
    mu::Float64
end


"""
    price(option::EuropeanOption, model::HedgedMonteCarlo{StopLoss}, data::MarketData)

Estimate the cost of a stop-loss hedging strategy for a European option via Monte Carlo.

Simulates asset paths under the real-world drift `model.strategy.mu` and tracks the
cash flows of a stop-loss hedge: buying the underlying when the spot crosses above the
strike and selling when it crosses below. The present value of all cash flows
(including terminal delivery) is averaged across paths.

# Arguments
- `option::EuropeanOption`: The option contract being hedged
- `model::HedgedMonteCarlo{StopLoss}`: Simulation model with stop-loss strategy
- `data::MarketData`: Market parameters (spot, rate, vol, div)

# Returns
A [`SimulationResult`](@ref) with the mean hedging cost and its standard error.

# Examples
```julia
data  = MarketData(41.0, 0.08, 0.30, 0.0)
call  = EuropeanCall(40.0, 1.0)

result = price(call, HedgedMonteCarlo(100, 10_000, StopLoss(0.10)), data)
result.price  # mean hedge cost
result.std    # standard error
```

See also: [`StopLoss`](@ref), [`HedgedMonteCarlo`](@ref), [`SimulationResult`](@ref)
"""
function price(option::EuropeanOption, model::HedgedMonteCarlo{StopLoss}, data::MarketData)
    (; strike, expiry) = option
    (; steps, reps, method, strategy) = model
    (; mu) = strategy
    (; spot, rate, vol, div) = data

    process = GeometricBrownianMotion(method, model, spot, mu, vol, expiry)
    return price(option, model, data, process)
end

function price(option::EuropeanOption, model::HedgedMonteCarlo{StopLoss}, data::MarketData, process::AssetPaths)
    (; strike, expiry) = option
    (; steps, reps, strategy) = model
    (; mu) = strategy
    (; rate, vol, div) = data

    dt = expiry / steps
    dfs = exp.(-rate * collect(0:1:steps) * dt)
    cost = zeros(reps)
    paths = asset_paths(process)

    for k in 1:reps
        cash_flows = zeros(steps + 1)

        if paths[k, 1] >= strike
            covered = 1
            cash_flows[1] = -paths[k, 1]
        else
            covered = 0
        end

        for t in 2:steps+1
            if (covered == 1) & (paths[k, t] < strike)
                covered = 0
                cash_flows[t] = paths[k, t]
            elseif (covered == 0) & (paths[k, t] > strike)
                covered = 1
                cash_flows[t] = -paths[k, t]
            end
        end

        if paths[k, end] >= strike
            cash_flows[end] += strike
        end

        cost[k] = -dot(dfs, cash_flows)
    end

    return SimulationResult(mean(cost), std(cost) / sqrt(reps))
end

"""
    DeltaHedge(mu) <: HedgeStrategy

Delta-hedging replication strategy for European options.

`mu` is the real-world drift used for path simulation in `HedgedMonteCarlo`.
"""

struct DeltaHedge <: HedgeStrategy
    mu::Float64   
end

"""
    price(option::EuropeanOption, model::HedgedMonteCarlo{DeltaHedge}, data::MarketData)

Estimate option value by simulating the discounted replication cost of a
discretely rebalanced Black-Scholes delta hedge.
"""
function price(option::EuropeanOption, model::HedgedMonteCarlo{DeltaHedge}, data::MarketData)
    (; strike, expiry) = option
    (; steps, reps, method, strategy) = model
    (; mu) = strategy
    (; spot, rate, vol, div) = data

    process = GeometricBrownianMotion(method, model, spot, mu, vol, expiry)
    return price(option, model, data, process)
end

function price(option::EuropeanOption, model::HedgedMonteCarlo{DeltaHedge}, data::MarketData, process::AssetPaths)
    (; strike, expiry) = option
    (; steps, reps) = model
    (; rate, vol, div) = data

    dt = expiry / steps
    dfs = exp.(-rate * collect(0:steps) * dt)
    cost = zeros(reps)
    paths = asset_paths(process)

    for k in 1:reps
        cash_flows = zeros(steps + 1)
        delta_prev = 0.0

        # Rebalance after each move up to (but not including) maturity.
        for t in 2:steps
            tau = expiry - (t - 1) * dt
            opt_t = typeof(option)(strike, tau)
            data_t = MarketData(paths[k, t], rate, vol, div)
            delta_now = delta(opt_t, BlackScholes(), data_t)

            cash_flows[t] += (delta_prev - delta_now) * paths[k, t]
            delta_prev = delta_now
        end

        st = paths[k, end]
        cash_flows[end] += delta_prev * st - payoff(option, st)

        cost[k] = -dot(dfs, cash_flows)
    end

    return SimulationResult(mean(cost), std(cost) / sqrt(reps))
end

"""
    compare_estimators_same_paths(option, paths, data)

Compute naive risk-neutral Monte Carlo and delta-hedged replication estimators
from the exact same simulated paths. Useful for variance comparisons in Q7(e).
"""
# function compare_estimators_same_paths(option::EuropeanOption, paths::AbstractMatrix, data::MarketData)
#     (; strike, expiry) = option
#     (; rate, vol, div) = data

#     reps = size(paths, 1)
#     steps = size(paths, 2) - 1
#     dt = expiry / steps
#     dfs = exp.(-rate * collect(0:steps) * dt)

#     # Naive discounted payoff estimator on the same paths.
#     disc_payoffs = exp(-rate * expiry) .* payoff.(option, paths[:, end])
#     naive = SimulationResult(mean(disc_payoffs), std(disc_payoffs) / sqrt(reps))

#     # Delta-hedged replication-cost estimator on the same paths.
#     cost = zeros(reps)
#     for k in 1:reps
#         cash_flows = zeros(steps + 1)
#         delta_prev = 0.0

#         for t in 2:steps
#             tau = expiry - (t - 1) * dt
#             opt_t = typeof(option)(strike, tau)
#             data_t = MarketData(paths[k, t], rate, vol, div)
#             delta_now = delta(opt_t, BlackScholes(), data_t)

#             cash_flows[t] += (delta_prev - delta_now) * paths[k, t]
#             delta_prev = delta_now
#         end

#         st = paths[k, end]
#         cash_flows[end] += delta_prev * st - payoff(option, st)
#         cost[k] = -dot(dfs, cash_flows)
#     end
#     hedged = SimulationResult(mean(cost), std(cost) / sqrt(reps))

#     return (naive=naive, hedged=hedged)
# end

struct ControlVariateMonteCarlo <: MonteCarlo
    steps::Int
    reps::Int
    method::VarianceReductionMethod
end

ControlVariateMonteCarlo(steps::Int, reps::Int) = ControlVariateMonteCarlo(steps, reps, VarianceReduction(PseudoRandom(), NoPairing()))
function price(
    option::EuropeanCall,
    model::ControlVariateMonteCarlo,
    data::MarketData
)
    (; strike, expiry) = option
    (; steps, reps) = model
    (; spot, rate, vol, div) = data

    # Precompute constants
    dt = expiry / steps
    nudt = (rate - div - 0.5 * vol^2) * dt
    sigsdt = vol * sqrt(dt)
    erddt = exp((rate - div) * dt)

    beta1 = -1.0
    disc_ct = zeros(reps)

    for j in 1:reps
        st = spot
        cv = 0.0

        for i in 1:steps
            t = (i - 1) * dt
            tau = expiry - t
            opt_t = EuropeanCall(strike, tau)
            data_t = MarketData(st, rate, vol, div)

            delta_t = delta(opt_t, BlackScholes(), data_t)
            z = randn()
            stn = st * exp(nudt + sigsdt * z)

            cv += delta_t * (stn - st * erddt)
            st = stn
        end

        ct = payoff(option, st) + beta1 * cv
        disc_ct[j] = exp(-rate * expiry) * ct
    end

    price = mean(disc_ct)
    se = std(disc_ct) / sqrt(reps)

    return SimulationResult(price, se)
end