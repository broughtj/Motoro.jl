using Statistics

"""
    VanillaOption

Abstract base type for all vanilla option contracts.

See also: [`EuropeanOption`](@ref), [`AmericanOption`](@ref)
"""
abstract type VanillaOption end

"""
    EuropeanOption <: VanillaOption

Abstract type for European-style options that can only be exercised at expiration.

See also: [`EuropeanCall`](@ref), [`EuropeanPut`](@ref)
"""
abstract type EuropeanOption <: VanillaOption end

"""
    AmericanOption <: VanillaOption

Abstract type for American-style options that can be exercised at any time up to
and including expiration.

See also: [`AmericanCall`](@ref), [`AmericanPut`](@ref)
"""
abstract type AmericanOption <: VanillaOption end

## European Options

"""
    EuropeanCall(strike, expiry)

A European call option that gives the holder the right (but not the obligation)
to buy the underlying asset at the strike price at expiration.

# Fields
- `strike::AbstractFloat`: Strike price (exercise price)
- `expiry::AbstractFloat`: Time to expiration in years

# Payoff
The payoff at expiration is `max(0, S - K)` where `S` is the spot price and `K`
is the strike.

# Examples
```julia
# At-the-money call with 1 year to expiration
call = EuropeanCall(100.0, 1.0)

# Out-of-the-money call with 6 months to expiration
call = EuropeanCall(110.0, 0.5)
```

See also: [`EuropeanPut`](@ref), [`payoff`](@ref)
"""
struct EuropeanCall{T<:AbstractFloat} <: EuropeanOption
    strike::T
    expiry::T
end

Base.broadcastable(x::EuropeanCall) = Ref(x)

function payoff(option::EuropeanCall, spot)
    return max(0.0, spot - option.strike)
end


"""
    EuropeanPut(strike, expiry)

A European put option that gives the holder the right (but not the obligation)
to sell the underlying asset at the strike price at expiration.

# Fields
- `strike::AbstractFloat`: Strike price (exercise price)
- `expiry::AbstractFloat`: Time to expiration in years

# Payoff
The payoff at expiration is `max(0, K - S)` where `S` is the spot price and `K`
is the strike.

# Examples
```julia
# At-the-money put with 1 year to expiration
put = EuropeanPut(100.0, 1.0)

# In-the-money put with 6 months to expiration
put = EuropeanPut(110.0, 0.5)
```

See also: [`EuropeanCall`](@ref), [`payoff`](@ref)
"""
struct EuropeanPut{T<:AbstractFloat} <: EuropeanOption
    strike::T
    expiry::T
end

Base.broadcastable(x::EuropeanPut) = Ref(x)

"""
    payoff(option::VanillaOption, spot)

Calculate the intrinsic value (payoff) of an option at a given spot price.

# Arguments
- `option::VanillaOption`: The option contract
- `spot`: Spot price of the underlying asset (scalar)

# Returns
The intrinsic value of the option. For calls: `max(0, S - K)`, for puts: `max(0, K - S)`.

# Examples
```julia
call = EuropeanCall(100.0, 1.0)
payoff(call, 110.0)  # Returns 10.0

put = EuropeanPut(100.0, 1.0)
payoff(put, 90.0)    # Returns 10.0
payoff(put, 110.0)   # Returns 0.0

# For a vector of spot prices, use broadcast syntax:
payoff.(put, [85.0, 90.0, 95.0, 100.0, 105.0])
```
"""
function payoff(option::EuropeanPut, spot)
    return max(0.0, option.strike - spot)
end


## American Options

"""
    AmericanCall(strike, expiry)

An American call option that can be exercised at any time up to and including expiration.

# Fields
- `strike::AbstractFloat`: Strike price (exercise price)
- `expiry::AbstractFloat`: Time to expiration in years

# Notes
For non-dividend paying stocks, American calls have the same value as European calls
since early exercise is never optimal.

# Examples
```julia
call = AmericanCall(100.0, 1.0)
```

See also: [`AmericanPut`](@ref), [`EuropeanCall`](@ref)
"""
struct AmericanCall{T<:AbstractFloat} <: AmericanOption
    strike::T
    expiry::T
end

Base.broadcastable(x::AmericanCall) = Ref(x)

function payoff(option::AmericanCall, spot)
    return max(0.0, spot - option.strike)
end


"""
    AmericanPut(strike, expiry)

An American put option that can be exercised at any time up to and including expiration.

# Fields
- `strike::AbstractFloat`: Strike price (exercise price)
- `expiry::AbstractFloat`: Time to expiration in years

# Notes
American puts always trade at a premium to European puts due to the early exercise feature.
This early exercise option is particularly valuable when interest rates are high or
the option is deep in-the-money.

# Examples
```julia
put = AmericanPut(100.0, 1.0)
```

See also: [`AmericanCall`](@ref), [`EuropeanPut`](@ref)
"""
struct AmericanPut{T<:AbstractFloat} <: AmericanOption
    strike::T
    expiry::T
end

Base.broadcastable(x::AmericanPut) = Ref(x)

function payoff(option::AmericanPut, spot)
    return max(0.0, option.strike - spot)
end



"""
    ExoticOption

Abstract base type for all exotic option contracts.

See also: [`LookbackOption`](@ref), [`ArithmeticAsianOption`](@ref)
"""
abstract type ExoticOption end

"""
    LookbackOption <: ExoticOption

Abstract type for lookback options, whose payoff depends on the extremum
(maximum or minimum) of the asset price over the life of the contract.

See also: [`FloatingStrikeLookbackCall`](@ref), [`FloatingStrikeLookbackPut`](@ref),
[`FloatingPriceLookbackCall`](@ref), [`FloatingPriceLookbackPut`](@ref)
"""
abstract type LookbackOption <: ExoticOption end

"""
    ArithmeticAsianOption <: ExoticOption

Abstract type for arithmetic Asian options, whose payoff depends on the arithmetic
mean of the asset price over the life of the contract.

See also: [`FloatingStrikeArithmeticAsianCall`](@ref), [`FloatingStrikeArithmeticAsianPut`](@ref),
[`FloatingPriceArithmeticAsianCall`](@ref), [`FloatingPriceArithmeticAsianPut`](@ref)
"""
abstract type ArithmeticAsianOption <: ExoticOption end


## Lookback Options

"""
    FloatingStrikeLookbackCall(expiry)

A lookback call with a floating strike equal to the minimum asset price over the
life of the option. At expiration the holder effectively buys at the lowest observed price.

# Fields
- `expiry::AbstractFloat`: Time to expiration in years

# Payoff
`S_T - S_min`  (always non-negative; no `max` needed)

# Examples
```julia
opt = FloatingStrikeLookbackCall(1.0)
```

See also: [`FloatingStrikeLookbackPut`](@ref), [`payoff`](@ref)
"""
struct FloatingStrikeLookbackCall{T<:AbstractFloat} <: LookbackOption
    expiry::T
end

"""
    FloatingStrikeLookbackPut(expiry)

A lookback put with a floating strike equal to the maximum asset price over the
life of the option. At expiration the holder effectively sells at the highest observed price.

# Fields
- `expiry::AbstractFloat`: Time to expiration in years

# Payoff
`S_max - S_T`  (always non-negative; no `max` needed)

# Examples
```julia
opt = FloatingStrikeLookbackPut(1.0)
```

See also: [`FloatingStrikeLookbackCall`](@ref), [`payoff`](@ref)
"""
struct FloatingStrikeLookbackPut{T<:AbstractFloat} <: LookbackOption
    expiry::T
end

"""
    FloatingPriceLookbackCall(strike, expiry)

A lookback call with a fixed strike and a floating price equal to the maximum asset
price over the life of the option.

# Fields
- `strike::AbstractFloat`: Fixed strike price
- `expiry::AbstractFloat`: Time to expiration in years

# Payoff
`max(0, S_max - K)`

# Examples
```julia
opt = FloatingPriceLookbackCall(100.0, 1.0)
```

See also: [`FloatingPriceLookbackPut`](@ref), [`payoff`](@ref)
"""
struct FloatingPriceLookbackCall{T<:AbstractFloat} <: LookbackOption
    strike::T
    expiry::T
end

"""
    FloatingPriceLookbackPut(strike, expiry)

A lookback put with a fixed strike and a floating price equal to the minimum asset
price over the life of the option.

# Fields
- `strike::AbstractFloat`: Fixed strike price
- `expiry::AbstractFloat`: Time to expiration in years

# Payoff
`max(0, K - S_min)`

# Examples
```julia
opt = FloatingPriceLookbackPut(100.0, 1.0)
```

See also: [`FloatingPriceLookbackCall`](@ref), [`payoff`](@ref)
"""
struct FloatingPriceLookbackPut{T<:AbstractFloat} <: LookbackOption
    strike::T
    expiry::T
end

"""
    payoff(option::LookbackOption, path)

Compute the payoff of a lookback option from a simulated asset price path.

# Arguments
- `option::LookbackOption`: The lookback option contract
- `path`: Vector of asset prices along one simulated path (first element is spot at inception)

# Returns
- `FloatingStrikeLookbackCall`: `S_T - S_min`
- `FloatingStrikeLookbackPut`: `S_max - S_T`
- `FloatingPriceLookbackCall`: `max(0, S_max - K)`
- `FloatingPriceLookbackPut`: `max(0, K - S_min)`
"""
function payoff(option::FloatingStrikeLookbackCall, path)
    return path[end] - minimum(path)
end

function payoff(option::FloatingStrikeLookbackPut, path)
    return maximum(path) - path[end]
end

function payoff(option::FloatingPriceLookbackCall, path)
    return max(0.0, maximum(path) - option.strike)
end

function payoff(option::FloatingPriceLookbackPut, path)
    return max(0.0, option.strike - minimum(path))
end


## Arithmetic Asian Options

"""
    FloatingStrikeArithmeticAsianCall(expiry)

An arithmetic Asian call with a floating strike equal to the arithmetic mean of the
asset price over the life of the option.

# Fields
- `expiry::AbstractFloat`: Time to expiration in years

# Payoff
`max(0, S_T - Ā)`  where `Ā` is the arithmetic mean of the path

# Examples
```julia
opt = FloatingStrikeArithmeticAsianCall(1.0)
```

See also: [`FloatingStrikeArithmeticAsianPut`](@ref), [`payoff`](@ref)
"""
struct FloatingStrikeArithmeticAsianCall{T<:AbstractFloat} <: ArithmeticAsianOption
    expiry::T
end

"""
    FloatingStrikeArithmeticAsianPut(expiry)

An arithmetic Asian put with a floating strike equal to the arithmetic mean of the
asset price over the life of the option.

# Fields
- `expiry::AbstractFloat`: Time to expiration in years

# Payoff
`max(0, Ā - S_T)`  where `Ā` is the arithmetic mean of the path

# Examples
```julia
opt = FloatingStrikeArithmeticAsianPut(1.0)
```

See also: [`FloatingStrikeArithmeticAsianCall`](@ref), [`payoff`](@ref)
"""
struct FloatingStrikeArithmeticAsianPut{T<:AbstractFloat} <: ArithmeticAsianOption
    expiry::T
end

"""
    FloatingPriceArithmeticAsianCall(strike, expiry)

An arithmetic Asian call with a fixed strike and a floating price equal to the
arithmetic mean of the asset price over the life of the option.

# Fields
- `strike::AbstractFloat`: Fixed strike price
- `expiry::AbstractFloat`: Time to expiration in years

# Payoff
`max(0, Ā - K)`  where `Ā` is the arithmetic mean of the path

# Examples
```julia
opt = FloatingPriceArithmeticAsianCall(100.0, 1.0)
```

See also: [`FloatingPriceArithmeticAsianPut`](@ref), [`payoff`](@ref)
"""
struct FloatingPriceArithmeticAsianCall{T<:AbstractFloat} <: ArithmeticAsianOption
    strike::T
    expiry::T
end

"""
    FloatingPriceArithmeticAsianPut(strike, expiry)

An arithmetic Asian put with a fixed strike and a floating price equal to the
arithmetic mean of the asset price over the life of the option.

# Fields
- `strike::AbstractFloat`: Fixed strike price
- `expiry::AbstractFloat`: Time to expiration in years

# Payoff
`max(0, K - Ā)`  where `Ā` is the arithmetic mean of the path

# Examples
```julia
opt = FloatingPriceArithmeticAsianPut(100.0, 1.0)
```

See also: [`FloatingPriceArithmeticAsianCall`](@ref), [`payoff`](@ref)
"""
struct FloatingPriceArithmeticAsianPut{T<:AbstractFloat} <: ArithmeticAsianOption
    strike::T
    expiry::T
end

"""
    payoff(option::ArithmeticAsianOption, path)

Compute the payoff of an arithmetic Asian option from a simulated asset price path.

# Arguments
- `option::ArithmeticAsianOption`: The Asian option contract
- `path`: Vector of asset prices along one simulated path

# Returns
- `FloatingStrikeArithmeticAsianCall`: `max(0, S_T - Ā)`
- `FloatingStrikeArithmeticAsianPut`: `max(0, Ā - S_T)`
- `FloatingPriceArithmeticAsianCall`: `max(0, Ā - K)`
- `FloatingPriceArithmeticAsianPut`: `max(0, K - Ā)`
"""
function payoff(option::FloatingStrikeArithmeticAsianCall, path)
    return max(0.0, path[end] - mean(path))
end

function payoff(option::FloatingStrikeArithmeticAsianPut, path)
    return max(0.0, mean(path) - path[end])
end

function payoff(option::FloatingPriceArithmeticAsianCall, path)
    return max(0.0, mean(path) - option.strike)
end

function payoff(option::FloatingPriceArithmeticAsianPut, path)
    return max(0.0, option.strike - mean(path))
end


"""
    CashOrNothingCall(K, Threshold, Expiry) <: ExoticOption

Cash-or-nothing call option. Pays a fixed cash amount (`Threshold`) if the terminal
stock price exceeds the strike (`K`), and zero otherwise.

# Fields
- `K::T`: Strike price
- `Threshold::T`: Fixed cash payout
- `expiry::T`: Time to expiration in years

# Examples
```julia
data  = MarketData(100.0, 0.05, 0.25, 0.0)
model = RiskNeutralMonteCarlo(252, 1_000_000)

price(CashOrNothingCall(100.0, 1.0, 1.0), model, data)
```

See also: [`ExoticOption`](@ref), [`price`](@ref), [`SimulationResult`](@ref)
"""
abstract type BinaryOption <: ExoticOption end
struct CashOrNothingCall{T<:AbstractFloat} <: BinaryOption
    strike::T
    expiry::T
end

Base.broadcastable(x::CashOrNothingCall) = Ref(x)

function payoff(option::CashOrNothingCall, spot)
    return spot > option.K ? option.Threshold : 0.0
end

struct CashOrNothingPut{T<:AbstractFloat} <: BinaryOption
    strike::T
    expiry::T
end

Base.broadcastable(x::CashOrNothingPut) = Ref(x)

function payoff(option::CashOrNothingCall, spot)
    return spot > option.K ? option.Threshold : 0.0
end
