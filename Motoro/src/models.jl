using Distributions
using Statistics

norm_cdf(x) = cdf(Normal(0.0, 1.0), x)

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


function price(option::AmericanOption, engine::Binomial, data::MarketData)
    (; strike, expiry) = option
    (; spot, rate, vol, div) = data
    steps = engine.steps

    dt = expiry / steps
    u = exp((rate - div) * dt + vol * sqrt(dt))
    d = exp((rate - div) * dt - vol * sqrt(dt))
    pu = (exp((rate - div) * dt) - d) / (u - d)
    pd = 1 - pu
    disc = exp(-rate * dt)

    # Asset prices at each node
    s = zeros(steps + 1, steps + 1)
    # Option values at each node
    x = zeros(steps + 1, steps + 1)

    # Initialize asset prices
    for i in 0:steps
        for j in 0:i
            s[j + 1, i + 1] = spot * u^(i - j) * d^j
        end
    end

    # Terminal payoffs
    for j in 1:steps+1
        x[j, steps + 1] = payoff(option, s[j, steps + 1])
    end

    # Backward induction with early exercise
    for i in steps:-1:1
        for j in 1:i
            # Continuation value
            continuation = disc * (pu * x[j, i + 1] + pd * x[j + 1, i + 1])
            # Immediate exercise value
            exercise = payoff(option, s[j, i])
            # Take maximum for American option
            x[j, i] = max(continuation, exercise)
        end
    end

    return x[1, 1]
end

"""
    BlackScholes()

Black-Scholes-Merton analytical pricing engine for European options.

Provides exact closed-form solutions using the Black-Scholes formula. This is the
fastest and most accurate method for European options under the Black-Scholes assumptions
(constant volatility, log-normal price distribution, no dividends).

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
call = EuropeanCall(100.0, 1.0)
put = EuropeanPut(100.0, 1.0)

engine = BlackScholes()
call_price = price(call, engine, data)
put_price = price(put, engine, data)
```

See also: [`price`](@ref), [`Binomial`](@ref)
"""
struct BlackScholes end

function price(option::EuropeanCall, engine::BlackScholes, data::MarketData)
    (; strike, expiry) = option
    (; spot, rate, vol, div) = data

    d1 = (log(spot / strike) + (rate - div + 0.5 * vol^2) * expiry) / (vol * sqrt(expiry))
    d2 = d1 - vol * sqrt(expiry)

    price = spot * exp(-div * expiry) * norm_cdf(d1) - strike * exp(-rate * expiry) * norm_cdf(d2)

    return price
end

function price(option::EuropeanPut, engine::BlackScholes, data::MarketData)
    (; strike, expiry) = option
    (; spot, rate, vol, div) = data

    d1 = (log(spot / strike) + (rate - div + 0.5 * vol^2) * expiry) / (vol * sqrt(expiry))
    d2 = d1 - vol * sqrt(expiry)

    price = strike * exp(-rate * expiry) * norm_cdf(-d2) - spot * exp(-div * expiry) * norm_cdf(-d1)

    return price
end



"""
    MonteCarlo(steps, reps)

Monte Carlo simulation engine for European option pricing.

Generates random price paths and computes the discounted expected payoff.
Useful for path-dependent options and provides visualization capabilities.

# Fields
- `steps::Int`: Number of time steps per simulation path
- `reps::Int`: Number of simulation paths (replications)

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
option = EuropeanCall(100.0, 1.0)

# Quick estimate with 1,000 paths
engine = MonteCarlo(100, 1000)
price(option, engine, data)

# Production quality with 100,000 paths
engine = MonteCarlo(100, 100000)
price(option, engine, data)

# Visualize sample paths
paths = asset_paths(engine, 100.0, 0.05, 0.2, 1.0)
```

See also: [`asset_paths`](@ref), [`asset_paths_col`](@ref), [`plot_paths`](@ref)
"""
struct MonteCarlo
    steps::Int
    reps::Int
end

"""
    asset_paths(model::MonteCarlo, spot, rate, vol, expiry)

Generate asset price paths using geometric Brownian motion.

Returns a matrix of simulated asset prices with dimensions `(reps, steps+1)`.

# Arguments
- `engine::MonteCarlo`: Monte Carlo engine with steps and reps
- `spot`: Initial spot price
- `rate`: Risk-free interest rate
- `vol`: Volatility
- `expiry`: Time to expiration

# Returns
Matrix of size `(reps, steps+1)` where each row is a simulated path.

# Examples
```julia
engine = MonteCarlo(100, 1000)
paths = asset_paths(engine, 100.0, 0.05, 0.2, 1.0)
size(paths)  # (1000, 101)
```

See also: [`asset_paths_col`](@ref), [`asset_paths_ax`](@ref)
"""
function asset_paths(model::MonteCarlo, spot, rate, vol, expiry)
    (; steps, reps) = model 

    dt = expiry / steps
    nudt = (rate - 0.5 * vol^2) * dt
    sidt = vol * sqrt(dt)
    paths = zeros(reps, steps + 1)
    paths[:, 1] .= spot

    @inbounds for i in 1:reps
        @inbounds for j in 2:steps + 1
            #z = rand(Normal(0.0, 1.0))
            z = randn()
            paths[i, j] = paths[i, j - 1] * exp(nudt + sidt * z)
        end
    end

    return paths
end


function price(option::EuropeanOption, engine::MonteCarlo, data::MarketData)
    (; strike, expiry) = option
    (; spot, rate, vol) = data
    (; steps, reps) = engine

    paths = asset_paths(engine, spot, rate, vol, expiry)
    payoffs = payoff.(option, paths[:, end])

    return exp(-rate * expiry) * mean(payoffs)
end

"""
    price(option::LookbackCall, engine::MonteCarlo, data::SVMarketData)

Price a fixed-strike lookback call under stochastic volatility (Heston-style)
using Monte Carlo with antithetic variates.

Evolves two coupled SDEs per step:
  Variance:  Vt+dt = Vt + α(V̄ - Vt)dt + ξ√Vt √dt εᵥ
  Asset:     St+dt = St exp((r - δ - ½Vt)dt ± √(Vt dt) εₛ)

Antithetic pairs (±εₛ on the asset, shared εᵥ on variance) cut
standard error significantly with minimal extra cost.
"""
function price(option::LookbackCall, engine::MonteCarlo, data::SVMarketData)
    (; strike, expiry) = option
    (; spot, rate, vol, div, alpha, vbar, xi) = data
    (; steps, reps) = engine

    dt      = expiry / steps
    alphadt = alpha * dt
    xisdt   = xi * sqrt(dt)
    disc    = exp(-rate * expiry)

    sum_CT = 0.0

    @inbounds for _ in 1:(reps ÷ 2)
        St1 = spot;  St2 = spot
        Vt  = vol^2
        maxSt1 = spot;  maxSt2 = spot

        @inbounds for _ in 1:steps
            εV = randn()
            εS = randn()
            Vt    = max(Vt + alphadt * (vbar - Vt) + xisdt * sqrt(Vt) * εV, 0.0)
            drift = (rate - div - 0.5 * Vt) * dt
            move  = sqrt(Vt * dt) * εS
            St1  *= exp(drift + move)
            St2  *= exp(drift - move)
            maxSt1 = max(maxSt1, St1)
            maxSt2 = max(maxSt2, St2)
        end

        sum_CT += 0.5 * (max(0.0, maxSt1 - strike) + max(0.0, maxSt2 - strike))
    end

    return disc * sum_CT / (reps ÷ 2)
end