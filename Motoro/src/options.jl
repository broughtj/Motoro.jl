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

EuropeanCall(strike::Real, expiry::Real) =
    EuropeanCall(promote(float(strike), float(expiry))...)

"""
    payoff(option::VanillaOption, spot)

Compute the intrinsic payoff of a vanilla option at a given terminal spot price.

# Arguments
- `option::VanillaOption`: The option contract
- `spot`: Terminal spot price of the underlying asset (scalar)

# Returns
The non-negative payoff: `max(0, S - K)` for calls, `max(0, K - S)` for puts.

# Examples
```julia
call = EuropeanCall(100.0, 1.0)
payoff(call, 110.0)  # 10.0
payoff(call, 90.0)   # 0.0

put = EuropeanPut(100.0, 1.0)
payoff(put, 90.0)    # 10.0

# Broadcast over a vector of spot prices:
payoff.(put, [85.0, 90.0, 95.0, 100.0, 105.0])
```

See also: [`price`](@ref)
"""
payoff(option::EuropeanCall, spot) = max(0.0, spot - option.strike)


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

EuropeanPut(strike::Real, expiry::Real) =
    EuropeanPut(promote(float(strike), float(expiry))...)

payoff(option::EuropeanPut, spot) = max(0.0, option.strike - spot)


## American Options

"""
    AmericanCall(strike, expiry)

An American call option that can be exercised at any time up to and including expiration.

# Fields
- `strike::AbstractFloat`: Strike price (exercise price)
- `expiry::AbstractFloat`: Time to expiration in years

# Payoff
The payoff at exercise is `max(0, S - K)` where `S` is the spot price and `K` is the strike.

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

AmericanCall(strike::Real, expiry::Real) =
    AmericanCall(promote(float(strike), float(expiry))...)

payoff(option::AmericanCall, spot) = max(0.0, spot - option.strike)


"""
    AmericanPut(strike, expiry)

An American put option that can be exercised at any time up to and including expiration.

# Fields
- `strike::AbstractFloat`: Strike price (exercise price)
- `expiry::AbstractFloat`: Time to expiration in years

# Payoff
The payoff at exercise is `max(0, K - S)` where `S` is the spot price and `K` is the strike.

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

AmericanPut(strike::Real, expiry::Real) =
    AmericanPut(promote(float(strike), float(expiry))...)

payoff(option::AmericanPut, spot) = max(0.0, option.strike - spot)
