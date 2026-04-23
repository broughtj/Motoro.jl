using Distributions

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

"""
    price(option::EuropeanOption, model::Binomial, data::MarketData)

Price a European option using the Cox-Ross-Rubinstein binomial tree.

Builds the tree forward and applies backward induction from terminal payoffs.
No early exercise check is performed.

# Arguments
- `option::EuropeanOption`: A [`EuropeanCall`](@ref) or [`EuropeanPut`](@ref)
- `model::Binomial`: Tree model (number of steps)
- `data::MarketData`: Market parameters (spot, rate, vol, div)

# Returns
An [`AnalyticResult`](@ref) with the option price.

# Examples
```julia
data = MarketData(41.0, 0.08, 0.30, 0.0)
price(EuropeanCall(40.0, 1.0), Binomial(100), data)
```

See also: [`Binomial`](@ref), [`BlackScholes`](@ref)
"""
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

    @inbounds for i in 1:(steps + 1)
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


"""
    price(option::AmericanOption, model::Binomial, data::MarketData)

Price an American option using the Cox-Ross-Rubinstein binomial tree.

At each interior node the early exercise value is compared against the continuation
value; the higher of the two is retained. This is the key difference from European
pricing.

# Arguments
- `option::AmericanOption`: An [`AmericanCall`](@ref) or [`AmericanPut`](@ref)
- `model::Binomial`: Tree model (number of steps)
- `data::MarketData`: Market parameters (spot, rate, vol, div)

# Returns
An [`AnalyticResult`](@ref) with the option price.

# Examples
```julia
data = MarketData(41.0, 0.08, 0.30, 0.0)
price(AmericanPut(40.0, 1.0), Binomial(100), data)
```

See also: [`Binomial`](@ref), [`BlackScholes`](@ref)
"""
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
    @inbounds for i in 1:(steps + 1)
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

"""
    price(option::EuropeanCall, model::BlackScholes, data::MarketData)

Price a European call using the Black-Scholes-Merton closed-form formula.

# Returns
An [`AnalyticResult`](@ref) with the BSM call price.

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
price(EuropeanCall(100.0, 1.0), BlackScholes(), data)
```

See also: [`BlackScholes`](@ref), [`delta`](@ref), [`price`](@ref)
"""
function price(option::EuropeanCall, model::BlackScholes, data::MarketData)
    (; strike, expiry) = option
    (; spot, rate, vol, div) = data

    d1 = (log(spot / strike) + (rate - div + 0.5 * vol^2) * expiry) / (vol * sqrt(expiry))
    d2 = d1 - vol * sqrt(expiry)

    p = spot * exp(-div * expiry) * norm_cdf(d1) -
        strike * exp(-rate * expiry) * norm_cdf(d2)

    return AnalyticResult(p)
end

"""
    price(option::EuropeanPut, model::BlackScholes, data::MarketData)

Price a European put using the Black-Scholes-Merton closed-form formula.

# Returns
An [`AnalyticResult`](@ref) with the BSM put price.

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
price(EuropeanPut(100.0, 1.0), BlackScholes(), data)
```

See also: [`BlackScholes`](@ref), [`delta`](@ref), [`price`](@ref)
"""
function price(option::EuropeanPut, model::BlackScholes, data::MarketData)
    (; strike, expiry) = option
    (; spot, rate, vol, div) = data

    d1 = (log(spot / strike) + (rate - div + 0.5 * vol^2) * expiry) / (vol * sqrt(expiry))
    d2 = d1 - vol * sqrt(expiry)

    p = strike * exp(-rate * expiry) * norm_cdf(-d2) -
        spot * exp(-div * expiry) * norm_cdf(-d1)

    return AnalyticResult(p)
end

"""
    price(option::CashOrNothingCall, model::BlackScholes, data::MarketData)

Price a cash-or-nothing call using the Black-Scholes-Merton formula.

The analytical price is `cash * exp(-r*T) * N(d2)`.

# Returns
An [`AnalyticResult`](@ref) with the BSM binary call price.

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
price(CashOrNothingCall(100.0, 1.0, 1.0), BlackScholes(), data)
```

See also: [`CashOrNothingCall`](@ref), [`BlackScholes`](@ref)
"""
function price(option::CashOrNothingCall, model::BlackScholes, data::MarketData)
    (; strike, expiry, cash) = option
    (; spot, rate, vol, div) = data
    d2 = (log(spot / strike) + (rate - div - 0.5 * vol^2) * expiry) / (vol * sqrt(expiry))
    return AnalyticResult(cash * exp(-rate * expiry) * norm_cdf(d2))
end

"""
    price(option::CashOrNothingPut, model::BlackScholes, data::MarketData)

Price a cash-or-nothing put using the Black-Scholes-Merton formula.

The analytical price is `cash * exp(-r*T) * N(-d2)`.

# Returns
An [`AnalyticResult`](@ref) with the BSM binary put price.

# Examples
```julia
data = MarketData(100.0, 0.05, 0.2, 0.0)
price(CashOrNothingPut(100.0, 1.0, 1.0), BlackScholes(), data)
```

See also: [`CashOrNothingPut`](@ref), [`BlackScholes`](@ref)
"""
function price(option::CashOrNothingPut, model::BlackScholes, data::MarketData)
    (; strike, expiry, cash) = option
    (; spot, rate, vol, div) = data
    d2 = (log(spot / strike) + (rate - div - 0.5 * vol^2) * expiry) / (vol * sqrt(expiry))
    return AnalyticResult(cash * exp(-rate * expiry) * norm_cdf(-d2))
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
    return exp(-div * expiry) * norm_cdf(d1)
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
    return -exp(-div * expiry) * norm_cdf(-d1)
end
