using Distributions
using Statistics
using Random

norm_cdf(x) = cdf(Normal(0.0, 1.0), x)

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

    return x[1]
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

    return x[1]
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

    price = spot * exp(-div * expiry) * norm_cdf(d1) - strike * exp(-rate * expiry) * norm_cdf(d2)

    return price
end


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

    price = strike * exp(-rate * expiry) * norm_cdf(-d2) - spot * exp(-div * expiry) * norm_cdf(-d1)

    return price
end


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
for use with [`MonteCarlo`](@ref).

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
    MonteCarlo(steps, reps[, method])

Monte Carlo simulation model for European option pricing.

Generates random asset price paths via geometric Brownian motion and returns
the discounted expected payoff.

# Fields
- `steps::Int`: Number of time steps per simulation path
- `reps::Int`: Number of simulation paths (replications)
- `method::VarianceReductionMethod`: Variance reduction strategy (default: `PseudoRandom` with `NoPairing`)

# Examples
```julia
data = MarketData(41.0, 0.08, 0.30, 0.0)
call = EuropeanCall(40.0, 1.0)

# Default (pseudo-random, no pairing)
price(call, MonteCarlo(100, 10_000), data)

# Antithetic variates
price(call, MonteCarlo(100, 10_000, VarianceReduction(PseudoRandom(), Antithetic())), data)

# Stratified sampling
price(call, MonteCarlo(100, 10_000, VarianceReduction(Stratified(), NoPairing())), data)
```

See also: [`asset_paths`](@ref), [`VarianceReduction`](@ref)
"""
struct MonteCarlo
    steps::Int
    reps::Int
    method::VarianceReductionMethod
end

MonteCarlo(steps::Int, reps::Int) = MonteCarlo(steps, reps, VarianceReduction(PseudoRandom(), NoPairing()))


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

# Arguments
- `method::VarianceReduction`: Variance reduction strategy controlling draw generation and pairing
- `model::MonteCarlo`: Monte Carlo model (provides `steps` and `reps`)
- `spot`: Initial asset price
- `rate`: Risk-free interest rate (annualized)
- `vol`: Volatility (annualized)
- `expiry`: Time to expiration in years

# Returns
Matrix of size `(reps, steps+1)`. Each row is one simulated price path;
column 1 is the initial spot price and column `steps+1` is the terminal price.

# Examples
```julia
data   = MarketData(41.0, 0.08, 0.30, 0.0)
model = MonteCarlo(100, 1_000)
paths  = asset_paths(model.method, model, data.spot, data.rate, data.vol, 1.0)
size(paths)  # (1000, 101)
```

See also: [`MonteCarlo`](@ref), [`price`](@ref)
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


function price(option::EuropeanOption, model::MonteCarlo, data::MarketData)
    (; strike, expiry) = option
    (; spot, rate, vol) = data
    (; steps, reps) = model

    paths = asset_paths(model.method, model, spot, rate, vol, expiry)
    payoffs = payoff.(option, paths[:, end])

    return exp(-rate * expiry) * mean(payoffs)
end





struct StopLoss
    mu::Float64
end


function price(option::EuropeanOption, model::MonteCarlo, hedge::StopLoss, data::MarketData)
    (; strike, expiry) = option
    (; steps, reps, method) = model
    (; mu) = hedge
    (; spot, rate, vol, div) = data

    dt = expiry / steps
    dfs = exp.(-rate * collect(0:1:steps) * dt)
    cost = zeros(reps)
    paths  = asset_paths(method, model, spot, mu, vol, expiry)

    for k in 1:reps
        cash_flows = zeros(steps+1)

        if paths[k,1] >= strike
            covered = 1
            cash_flows[1] = -paths[k,1]
        else
            covered = 0
        end

        for t in 2:steps+1
            if (covered == 1) & (paths[k,t] < strike)
                covered = 0
                cash_flows[t] = paths[k,t]
            elseif (covered == 0) & (paths[k,t] > strike)
                covered = 1
                cash_flows[t] = -paths[k,t]
            end
        end

        if paths[k, end] >= strike
            cash_flows[end] += strike
        end

        cost[k] = -dot(dfs, cash_flows)
    end

    return mean(cost)

end
