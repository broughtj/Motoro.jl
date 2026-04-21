using Distributions
using Statistics
using Random

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

RiskNeutralMonteCarlo(steps::Int, reps::Int) =
    RiskNeutralMonteCarlo(steps, reps, VarianceReduction(PseudoRandom(), NoPairing()))


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

"""
    asset_paths(method::VarianceReduction, model::MonteCarlo, spot, rate, vol, expiry)

Generate simulated asset price paths using geometric Brownian motion.

Works with any [`MonteCarlo`](@ref) subtype (`RiskNeutralMonteCarlo` or `HedgedMonteCarlo`).

# Arguments
- `method::VarianceReduction`: Variance reduction strategy controlling draw generation and pairing
- `model::MonteCarlo`: Any Monte Carlo model (provides `steps` and `reps`)
- `spot`: Initial asset price
- `rate`: Drift rate (risk-free rate for risk-neutral pricing; real-world drift for hedging)
- `vol`: Volatility (annualized)
- `expiry`: Time to expiration in years

# Returns
Matrix of size `(reps, steps+1)`. Each row is one simulated price path;
column 1 is the initial spot price and column `steps+1` is the terminal price.

# Examples
```julia
data  = MarketData(41.0, 0.08, 0.30, 0.0)
model = RiskNeutralMonteCarlo(100, 1_000)
paths = asset_paths(model.method, model, data.spot, data.rate, data.vol, 1.0)
size(paths)  # (1000, 101)
```

See also: [`RiskNeutralMonteCarlo`](@ref), [`HedgedMonteCarlo`](@ref), [`price`](@ref)
"""
function asset_paths(method::VarianceReduction, model::MonteCarlo, spot, rate, vol, expiry)
    (; steps, reps) = model

    dt = expiry / steps
    nudt = (rate - 0.5 * vol^2) * dt
    sidt = vol * sqrt(dt)
    n = reps ÷ (method.pairing isa Antithetic ? 2 : 1)

    paths = zeros(reps, steps + 1)
    paths[:, 1] .= spot

    @inbounds for j in 2:steps+1
        z = generate_draws(method.draw, n)
        z = apply_pairing(method.pairing, z)
        paths[:, j] = paths[:, j-1] .* exp.(nudt .+ sidt .* z)
    end

    return paths
end


function price(option::EuropeanOption, model::RiskNeutralMonteCarlo, data::MarketData)
    (; strike, expiry) = option
    (; spot, rate, vol) = data

    paths = asset_paths(model.method, model, spot, rate, vol, expiry)
    disc_payoffs = exp(-rate * expiry) .* payoff.(option, paths[:, end])

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
function price(option::BinaryOption, model::RiskNeutralMonteCarlo, data::MarketData)
    (; expiry) = option
    (; spot, rate, vol) = data

    paths = asset_paths(model.method, model, spot, rate, vol, expiry)
    disc_payoffs = exp(-rate * expiry) .* payoff.(option, paths[:, end])

    return SimulationResult(mean(disc_payoffs), std(disc_payoffs) / sqrt(model.reps))
end

function price(option::ExoticOption, model::RiskNeutralMonteCarlo, data::MarketData)
    (; expiry) = option
    (; spot, rate, vol) = data

    paths = asset_paths(model.method, model, spot, rate, vol, expiry)
    disc_payoffs = exp(-rate * expiry) .* [payoff(option, row) for row in eachrow(paths)]

    return SimulationResult(mean(disc_payoffs), std(disc_payoffs) / sqrt(model.reps))
end
