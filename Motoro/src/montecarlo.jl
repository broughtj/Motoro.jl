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
struct NoPairing <: PairingMethod end

"""
    Antithetic <: PairingMethod

Antithetic variates pairing. For each draw `z`, also simulates a path using `-z`.
The negative correlation between paired paths reduces estimator variance.
"""
struct Antithetic <: PairingMethod end

"""
    VarianceReduction{D<:DrawMethod, P<:PairingMethod}(draw, pairing)

Combines a draw method and a pairing method into a variance reduction strategy
for use with any [`MonteCarlo`](@ref) subtype.

# Fields
- `draw::DrawMethod`: How to generate random draws ([`PseudoRandom`](@ref) or
  [`Stratified`](@ref))
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

Concrete subtypes differ in the pricing measure and variance reduction approach used:
- [`RiskNeutralMonteCarlo`](@ref): discounted expected payoff under the Q measure
- [`HedgedMonteCarlo`](@ref): real-world hedge cost simulation under the P measure
- [`ControlVariateMonteCarlo`](@ref): Q-measure pricing with a control variate adjustment

See also: [`RiskNeutralMonteCarlo`](@ref), [`HedgedMonteCarlo`](@ref),
[`ControlVariateMonteCarlo`](@ref)
"""
abstract type MonteCarlo end

"""
    RiskNeutralMonteCarlo(steps, reps[, method][, dynamics])

Monte Carlo simulation model for pricing options under the risk-neutral (Q) measure.

Generates asset price paths using the risk-free rate as the drift and returns the
discounted expected payoff. The path dynamics default to [`GeometricBrownianMotion`](@ref)
but can be swapped for any [`AssetDynamics`](@ref) subtype.

# Fields
- `steps::Int`: Number of time steps per simulation path
- `reps::Int`: Number of simulation paths (replications)
- `method::VarianceReductionMethod`: Variance reduction strategy (default: `PseudoRandom`
  with `NoPairing`)
- `dynamics::AssetDynamics`: Asset price process (default:
  [`GeometricBrownianMotion`](@ref))

# Examples
```julia
data = MarketData(41.0, 0.08, 0.30, 0.0)
call = EuropeanCall(40.0, 1.0)

price(call, RiskNeutralMonteCarlo(100, 10_000), data)

av = VarianceReduction(PseudoRandom(), Antithetic())
price(call, RiskNeutralMonteCarlo(100, 10_000, av), data)

jd  = JumpDiffusion(3.0, -0.02, 0.05)
price(call, RiskNeutralMonteCarlo(100, 10_000, jd), data)
```

See also: [`MonteCarlo`](@ref), [`GeometricBrownianMotion`](@ref), [`JumpDiffusion`](@ref),
[`asset_paths`](@ref)
"""
struct RiskNeutralMonteCarlo <: MonteCarlo
    steps::Int
    reps::Int
    method::VarianceReductionMethod
    dynamics::AssetDynamics
end

RiskNeutralMonteCarlo(steps::Int, reps::Int) =
    RiskNeutralMonteCarlo(steps, reps, VarianceReduction(PseudoRandom(), NoPairing()),
        GeometricBrownianMotion())

RiskNeutralMonteCarlo(steps::Int, reps::Int, method::VarianceReductionMethod) =
    RiskNeutralMonteCarlo(steps, reps, method, GeometricBrownianMotion())

RiskNeutralMonteCarlo(steps::Int, reps::Int, dynamics::AssetDynamics) =
    RiskNeutralMonteCarlo(steps, reps, VarianceReduction(PseudoRandom(), NoPairing()),
        dynamics)


# Internal: generate n standard-normal draws using method.
# PseudoRandom: randn(n). Stratified: one quantile per stratum, shuffled.
generate_draws(::PseudoRandom, n::Int) = randn(n)

function generate_draws(::Stratified, n::Int)
    u = rand(n)
    d = Normal()
    draws = [quantile(d, (i - 1 + u[i]) / n) for i in 1:n]
    shuffle!(draws)
    return draws
end

# Internal: apply antithetic pairing to draws.
# NoPairing: identity. Antithetic: [draws; -draws].
apply_pairing(::NoPairing, draws) = draws

apply_pairing(::Antithetic, draws) = [draws; -draws]

"""
    asset_paths(model::MonteCarlo, spot, rate, vol, expiry)

Generate simulated asset price paths.

Dispatches on `model.dynamics` to select the path generation model
([`GeometricBrownianMotion`](@ref) or [`JumpDiffusion`](@ref)) and on `model.method` for
variance reduction. Works with any [`MonteCarlo`](@ref) subtype.

# Arguments
- `model::MonteCarlo`: Simulation model (provides `steps`, `reps`, `method`, `dynamics`)
- `spot`: Initial asset price
- `rate`: Drift rate (risk-free rate for Q-measure pricing; real-world drift for hedging)
- `vol`: Volatility (annualized)
- `expiry`: Time to expiration in years

# Returns
Matrix of size `(reps, steps+1)`. Each row is one simulated path; column 1 is
the initial spot price and column `steps+1` is the terminal price.

# Examples
```julia
data  = MarketData(100.0, 0.05, 0.20, 0.0)
model = RiskNeutralMonteCarlo(252, 1_000)
paths = asset_paths(model, data.spot, data.rate, data.vol, 1.0)
size(paths)   # (1000, 253)

jd    = JumpDiffusion(3.0, -0.02, 0.05)
model = RiskNeutralMonteCarlo(252, 1_000, jd)
paths = asset_paths(model, data.spot, data.rate, data.vol, 1.0)
```

See also: [`GeometricBrownianMotion`](@ref), [`JumpDiffusion`](@ref),
[`RiskNeutralMonteCarlo`](@ref)
"""
asset_paths(model::MonteCarlo, spot, rate, vol, expiry) =
    asset_paths(model.method, model.dynamics, model, spot, rate, vol, expiry)

# Geometric Brownian Motion path generation
function asset_paths(method::VarianceReduction, ::GeometricBrownianMotion,
        model::MonteCarlo, spot, rate, vol, expiry)
    (; steps, reps) = model

    dt = expiry / steps
    nudt = (rate - 0.5 * vol^2) * dt
    sidt = vol * sqrt(dt)
    n = reps ÷ (method.pairing isa Antithetic ? 2 : 1)

    paths = zeros(reps, steps + 1)
    paths[:, 1] .= spot

    @inbounds for j in 2:(steps + 1)
        z = generate_draws(method.draw, n)
        z = apply_pairing(method.pairing, z)
        paths[:, j] = paths[:, j - 1] .* exp.(nudt .+ sidt .* z)
    end

    return paths
end

# Jump diffusion (Merton 1976) path generation
function asset_paths(method::VarianceReduction, jd::JumpDiffusion, model::MonteCarlo,
        spot, rate, vol, expiry)
    (; steps, reps) = model
    (; lambda, alpha_j, sigma_j) = jd

    dt = expiry / steps
    k = exp(alpha_j) - 1.0
    nudtJ = (rate - lambda * k - 0.5 * vol^2) * dt
    sidt = vol * sqrt(dt)
    n = reps ÷ (method.pairing isa Antithetic ? 2 : 1)
    poisson_dist = Poisson(lambda * dt)

    paths = zeros(reps, steps + 1)
    paths[:, 1] .= spot

    @inbounds for j in 2:(steps + 1)
        z = generate_draws(method.draw, n)
        z = apply_pairing(method.pairing, z)
        m = rand(poisson_dist, reps)
        W = sqrt.(float.(m)) .* randn(reps)
        jump = exp.(m .* (alpha_j - 0.5 * sigma_j^2) .+ sigma_j .* W)
        paths[:, j] = paths[:, j - 1] .* exp.(nudtJ .+ sidt .* z) .* jump
    end

    return paths
end


"""
    price(option::EuropeanOption, model::RiskNeutralMonteCarlo, data::MarketData)

Price a European option via risk-neutral Monte Carlo simulation.

Simulates asset paths under the Q measure and returns the discounted mean payoff.
Only the terminal price of each path is used.

# Arguments
- `option::EuropeanOption`: A [`EuropeanCall`](@ref) or [`EuropeanPut`](@ref)
- `model::RiskNeutralMonteCarlo`: Simulation parameters (steps, reps, variance reduction)
- `data::MarketData`: Market parameters (spot, rate, vol, div)

# Returns
A [`SimulationResult`](@ref) with the mean discounted payoff and its standard error.

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
price(EuropeanCall(100.0, 1.0), RiskNeutralMonteCarlo(100, 10_000), data)
```

See also: [`RiskNeutralMonteCarlo`](@ref), [`asset_paths`](@ref), [`SimulationResult`](@ref)
"""
function price(option::EuropeanOption, model::RiskNeutralMonteCarlo, data::MarketData)
    (; strike, expiry) = option
    (; spot, rate, vol) = data

    paths = asset_paths(model, spot, rate, vol, expiry)
    disc_payoffs = exp(-rate * expiry) .* payoff.(option, paths[:, end])

    return SimulationResult(mean(disc_payoffs), std(disc_payoffs) / sqrt(model.reps))
end


"""
    price(option::BinaryOption, model::RiskNeutralMonteCarlo, data::MarketData)

Price a binary option via risk-neutral Monte Carlo simulation.

Only the terminal price of each path is used; the full path is not needed.

# Arguments
- `option::BinaryOption`: A [`CashOrNothingCall`](@ref) or [`CashOrNothingPut`](@ref)
- `model::RiskNeutralMonteCarlo`: Simulation parameters (steps, reps, variance reduction)
- `data::MarketData`: Market parameters (spot, rate, vol, div)

# Returns
A [`SimulationResult`](@ref) with the mean discounted payoff and its standard error.

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
price(CashOrNothingCall(100.0, 1.0, 1.0), RiskNeutralMonteCarlo(1, 10_000), data)
```

See also: [`BinaryOption`](@ref), [`RiskNeutralMonteCarlo`](@ref),
[`SimulationResult`](@ref)
"""
function price(option::BinaryOption, model::RiskNeutralMonteCarlo, data::MarketData)
    (; expiry) = option
    (; spot, rate, vol) = data

    paths = asset_paths(model, spot, rate, vol, expiry)
    disc_payoffs = exp(-rate * expiry) .* payoff.(option, paths[:, end])

    return SimulationResult(mean(disc_payoffs), std(disc_payoffs) / sqrt(model.reps))
end

"""
    price(option::ExoticOption, model::RiskNeutralMonteCarlo, data::MarketData)

Price a path-dependent exotic option via risk-neutral Monte Carlo simulation.

Each full simulated path is passed to `payoff`, allowing the payoff to depend on the
entire price history (e.g., the running maximum, minimum, or arithmetic mean).

# Arguments
- `option::ExoticOption`: A lookback or Asian option contract
- `model::RiskNeutralMonteCarlo`: Simulation parameters (steps, reps, variance reduction)
- `data::MarketData`: Market parameters (spot, rate, vol, div)

# Returns
A [`SimulationResult`](@ref) with the mean discounted payoff and its standard error.

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
model = RiskNeutralMonteCarlo(252, 10_000)

price(FloatingStrikeLookbackCall(1.0), model, data)
price(FloatingPriceLookbackPut(100.0, 1.0), model, data)
price(FloatingPriceArithmeticAsianCall(100.0, 1.0), model, data)
```

See also: [`ExoticOption`](@ref), [`asset_paths`](@ref), [`SimulationResult`](@ref)
"""
function price(option::ExoticOption, model::RiskNeutralMonteCarlo, data::MarketData)
    (; expiry) = option
    (; spot, rate, vol) = data

    paths = asset_paths(model, spot, rate, vol, expiry)
    disc_payoffs = exp(-rate * expiry) .* [payoff(option, row) for row in eachrow(paths)]

    return SimulationResult(mean(disc_payoffs), std(disc_payoffs) / sqrt(model.reps))
end
