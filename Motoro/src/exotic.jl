using Statistics

"""
    ExoticOption

Abstract base type for all exotic option contracts.

See also: [`BinaryOption`](@ref), [`LookbackOption`](@ref), [`ArithmeticAsianOption`](@ref)
"""
abstract type ExoticOption end

"""
    BinaryOption <: ExoticOption

Abstract type for binary (digital) options whose payoff is a fixed cash amount
contingent on the terminal asset price relative to the strike.

Unlike [`LookbackOption`](@ref) and [`ArithmeticAsianOption`](@ref), binary option
payoffs depend only on the terminal spot price, not the full path.

See also: [`CashOrNothingCall`](@ref), [`CashOrNothingPut`](@ref)
"""
abstract type BinaryOption <: ExoticOption end

"""
    CashOrNothingCall(strike, expiry, cash)

A cash-or-nothing call that pays a fixed amount `cash` if the asset price
finishes above `strike` at expiration, and zero otherwise.

# Fields
- `strike::AbstractFloat`: Strike price
- `expiry::AbstractFloat`: Time to expiration in years
- `cash::AbstractFloat`: Fixed payout if the option finishes in the money (default: 1.0)

# Payoff
`cash  if  S_T > K,  else  0`

# Examples
```julia
opt = CashOrNothingCall(100.0, 1.0, 1.0)   # pays \$1 if S_T > 100
```

See also: [`CashOrNothingPut`](@ref), [`payoff`](@ref)
"""
struct CashOrNothingCall{T<:AbstractFloat} <: BinaryOption
    strike::T
    expiry::T
    cash::T
end

CashOrNothingCall(strike::T, expiry::T) where {T<:AbstractFloat} =
    CashOrNothingCall{T}(strike, expiry, one(T))

Base.broadcastable(x::CashOrNothingCall) = Ref(x)

"""
    CashOrNothingPut(strike, expiry, cash)

A cash-or-nothing put that pays a fixed amount `cash` if the asset price
finishes below `strike` at expiration, and zero otherwise.

# Fields
- `strike::AbstractFloat`: Strike price
- `expiry::AbstractFloat`: Time to expiration in years
- `cash::AbstractFloat`: Fixed payout if the option finishes in the money (default: 1.0)

# Payoff
`cash  if  S_T < K,  else  0`

# Examples
```julia
opt = CashOrNothingPut(100.0, 1.0, 1.0)   # pays \$1 if S_T < 100
```

See also: [`CashOrNothingCall`](@ref), [`payoff`](@ref)
"""
struct CashOrNothingPut{T<:AbstractFloat} <: BinaryOption
    strike::T
    expiry::T
    cash::T
end

CashOrNothingPut(strike::T, expiry::T) where {T<:AbstractFloat} =
    CashOrNothingPut{T}(strike, expiry, one(T))

Base.broadcastable(x::CashOrNothingPut) = Ref(x)

"""
    payoff(option::BinaryOption, spot)

Compute the payoff of a binary option at a given terminal spot price.

# Returns
- `CashOrNothingCall`: `option.cash` if `spot > option.strike`, else `0.0`
- `CashOrNothingPut`:  `option.cash` if `spot < option.strike`, else `0.0`
"""
function payoff(option::CashOrNothingCall, spot)
    return spot > option.strike ? option.cash : 0.0
end

function payoff(option::CashOrNothingPut, spot)
    return spot < option.strike ? option.cash : 0.0
end


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
    return max(path[end] - minimum(path), 0.0)
end

function payoff(option::FloatingStrikeLookbackPut, path)
    return max(maximum(path) - path[end], 0.0)
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
