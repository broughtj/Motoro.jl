using Statistics
using Distributions
using Motoro

# ─────────────────────────────────────────────────────────────
# Abstract type (parallel to ArithmeticAsianOption)
# ─────────────────────────────────────────────────────────────

abstract type GeometricAsianOption end

# ─────────────────────────────────────────────────────────────
## Option structs
# ─────────────────────────────────────────────────────────────

"""
    FixedStrikeGeometricAsianCall(strike, expiry)

A geometric Asian call with a fixed strike `K` and a floating price equal to the
geometric mean of the asset price over the monitoring window.

# Fields
- `strike::AbstractFloat`: Fixed strike price K
- `expiry::AbstractFloat`: Time to expiration T (years)

# Payoff
`max(0, G̃ - K)`  where `G̃ = exp(mean(log.(path)))` is the geometric mean of the path

# Examples
```julia
opt = FixedStrikeGeometricAsianCall(100.0, 1.0)
```
See also: [`FixedStrikeGeometricAsianPut`](@ref), [`geom_asian_price`](@ref)
"""
struct FixedStrikeGeometricAsianCall{T<:AbstractFloat} <: GeometricAsianOption
    strike::T
    expiry::T
end

"""
    FixedStrikeGeometricAsianPut(strike, expiry)

A geometric Asian put with a fixed strike `K` and a floating price equal to the
geometric mean of the asset price over the monitoring window.

# Fields
- `strike::AbstractFloat`: Fixed strike price K
- `expiry::AbstractFloat`: Time to expiration T (years)

# Payoff
`max(0, K - G̃)`  where `G̃ = exp(mean(log.(path)))` is the geometric mean of the path

# Examples
```julia
opt = FixedStrikeGeometricAsianPut(100.0, 1.0)
```
See also: [`FixedStrikeGeometricAsianCall`](@ref), [`geom_asian_price`](@ref)
"""
struct FixedStrikeGeometricAsianPut{T<:AbstractFloat} <: GeometricAsianOption
    strike::T
    expiry::T
end

"""
    FloatingStrikeGeometricAsianCall(expiry)

A geometric Asian call with a floating strike equal to the geometric mean of the
asset price over the monitoring window.

# Fields
- `expiry::AbstractFloat`: Time to expiration T (years)

# Payoff
`max(0, S_T - G̃)`  where `G̃` is the geometric mean of the path

# Examples
```julia
opt = FloatingStrikeGeometricAsianCall(1.0)
```
"""
struct FloatingStrikeGeometricAsianCall{T<:AbstractFloat} <: GeometricAsianOption
    expiry::T
end

"""
    FloatingStrikeGeometricAsianPut(expiry)

A geometric Asian put with a floating strike equal to the geometric mean of the
asset price over the monitoring window.

# Fields
- `expiry::AbstractFloat`: Time to expiration T (years)

# Payoff
`max(0, G̃ - S_T)`  where `G̃` is the geometric mean of the path

# Examples
```julia
opt = FloatingStrikeGeometricAsianPut(1.0)
```
"""
struct FloatingStrikeGeometricAsianPut{T<:AbstractFloat} <: GeometricAsianOption
    expiry::T
end

# ─────────────────────────────────────────────────────────────
## Payoff functions (Monte Carlo use)
# ─────────────────────────────────────────────────────────────

"""
    geom_mean(path) -> Float64

Geometric mean of a price path: exp(mean(log.(path))).
"""
function geom_mean(path::AbstractVector{T}) where T<:AbstractFloat
    isempty(path) && throw(ArgumentError("path must be non-empty"))
    return exp(mean(log.(path)))
end

"""
    payoff(option::GeometricAsianOption, path)

Compute the payoff of a geometric Asian option from a simulated asset price path.

# Returns
- `FixedStrikeGeometricAsianCall`:    `max(0, G̃ - K)`
- `FixedStrikeGeometricAsianPut`:     `max(0, K - G̃)`
- `FloatingStrikeGeometricAsianCall`: `max(0, S_T - G̃)`
- `FloatingStrikeGeometricAsianPut`:  `max(0, G̃ - S_T)`
"""
function payoff(opt::FixedStrikeGeometricAsianCall, path::AbstractVector)
    return max(zero(eltype(path)), geom_mean(path) - opt.strike)
end

function payoff(opt::FixedStrikeGeometricAsianPut, path::AbstractVector)
    return max(zero(eltype(path)), opt.strike - geom_mean(path))
end

function payoff(opt::FloatingStrikeGeometricAsianCall, path::AbstractVector)
    return max(zero(eltype(path)), last(path) - geom_mean(path))
end

function payoff(opt::FloatingStrikeGeometricAsianPut, path::AbstractVector)
    return max(zero(eltype(path)), geom_mean(path) - last(path))
end

# ─────────────────────────────────────────────────────────────
## Closed-form pricer  (Kemna & Vorst 1990)
#
#  Notation from the formula image:
#    N   = number of monitoring dates
#    m   = number of monitoring dates already past (0 for a fresh option)
#    t_{m+1} = time of the next (first future) monitoring date
#    G_t = running geometric mean of observed prices so far
#          (= S_0 when m = 0, i.e. no past fixings)
#
#  Parameters
#    S   current spot price
#    K   strike
#    T   expiry (years)
#    r   risk-free rate (continuously compounded)
#    σ   volatility
#    δ   continuous dividend / convenience yield
#    N   total monitoring steps
#    m   fixings already observed  (default 0)
#    G_t geometric mean of observed prices (default S, irrelevant when m=0)
#    t_{m+1} time of next fixing (default T/N for m=0)
# ─────────────────────────────────────────────────────────────

"""
    geom_asian_price(opt, S, r, σ, δ, N; m=0, G_t=S, t_next=opt.expiry/N)

Closed-form price for a `FixedStrikeGeometricAsianCall` or `FixedStrikeGeometricAsianPut`
using the modified Black–Scholes formula of Kemna & Vorst (1990).

# Arguments
- `opt`    : `FixedStrikeGeometricAsianCall` or `FixedStrikeGeometricAsianPut`
- `S`      : Current spot price
- `r`      : Risk-free rate (continuously compounded)
- `σ`      : Volatility
- `δ`      : Continuous dividend / convenience yield
- `N`      : Total number of monitoring dates

# Keyword arguments
- `m`      : Number of fixings already observed (default 0)
- `G_t`    : Running geometric mean of observed prices (default `S`, ignored when `m=0`)
- `t_next` : Time to the next (first future) fixing (default `T/N`)

# Returns
Scalar option price.

# Formula
```
C_GA = e^{-rT} (e^{a + ½b} N(x) - K N(x - √b))

a = ln(G_t) + (N-m)/N * (ln(S) + v(t_{m+1} - t) + ½v(T - t_{m+1}))
b = (N-m)²/N² * σ²(t_{m+1} - t) + σ²(T - t_{m+1})/(6N²) * (N-m)(2(N-m)-1)
v = r - δ - ½σ²
x = (a - ln(K) + b) / √b
```
"""
function geom_asian_price(
        opt::Union{FixedStrikeGeometricAsianCall{T}, FixedStrikeGeometricAsianPut{T}},
        S::Real, r::Real, σ::Real, δ::Real, N::Integer;
        m::Integer = 0,
        G_t::Real  = S,
        t_next::Real = opt.expiry / N
    ) where T<:AbstractFloat

    K  = opt.strike
    TT = opt.expiry          # T in the formula
    t  = zero(T)             # current time (pricing date), set to 0

    # adjusted counts
    Nm  = N - m              # remaining monitoring steps
    Nm2 = Nm^2

    # intermediate drift-like term
    v = r - δ - σ^2 / 2

    # a  (log of the forward geometric mean, adjusted for past fixings)
    a = log(G_t) +
        (Nm / N) * (log(S) + v * (t_next - t) + v / 2 * (TT - t_next))

    # b  (variance of the log geometric mean)
    b = (Nm2 / N^2) * σ^2 * (t_next - t) +
        σ^2 * (TT - t_next) / (6 * N^2) * Nm * (2Nm - 1)

    sqrtb = sqrt(b)

    # standardised moneyness
    x = (a - log(K) + b) / sqrtb

    Φ = Normal()

    if opt isa FixedStrikeGeometricAsianCall
        price = exp(-r * TT) * (exp(a + b / 2) * cdf(Φ, x) - K * cdf(Φ, x - sqrtb))
    else  # Put via put-call parity for geometric Asian
        price = exp(-r * TT) * (K * cdf(Φ, -(x - sqrtb)) - exp(a + b / 2) * cdf(Φ, -x))
    end

    return max(zero(T), price)
end

# ─────────────────────────────────────────────────────────────
## Monte Carlo pricer (reference / validation)
# ─────────────────────────────────────────────────────────────

"""
    mc_geom_asian_price(opt, S, r, σ, δ, N; n_paths=500_000, seed=42)

Monte Carlo estimate of a geometric Asian option price under GBM.

Useful for validating `geom_asian_price` against the closed-form result.
"""
function mc_geom_asian_price(
        opt::GeometricAsianOption,
        S::Real, r::Real, σ::Real, δ::Real, N::Integer;
        n_paths::Integer = 500_000,
        seed::Integer    = 42
    )
    rng  = MersenneTwister(seed)
    TT   = opt.expiry
    dt   = TT / N
    disc = exp(-r * TT)

    total = 0.0
    for _ in 1:n_paths
        path  = Vector{Float64}(undef, N)
        St    = Float64(S)
        for i in 1:N
            z  = randn(rng)
            St = St * exp((r - δ - σ^2/2)*dt + σ*sqrt(dt)*z)
            path[i] = St
        end
        total += payoff(opt, path)
    end
    return disc * total / n_paths
end

# ─────────────────────────────────────────────────────────────
## Quick smoke-test
# ─────────────────────────────────────────────────────────────

function _demo()
    S, K, T_  = 100.0, 100.0, 1.0
    r, σ, δ   = 0.05, 0.20, 0.02
    N         = 252

    call = FixedStrikeGeometricAsianCall(K, T_)
    put  = FixedStrikeGeometricAsianPut(K, T_)

    cf_call = geom_asian_price(call, S, r, σ, δ, N)
    cf_put  = geom_asian_price(put,  S, r, σ, δ, N)

    mc_call = mc_geom_asian_price(call, S, r, σ, δ, N)
    mc_put  = mc_geom_asian_price(put,  S, r, σ, δ, N)

    println("─── Geometric Asian Option Prices ───")
    println("  S=$S  K=$K  T=$T_  r=$r  σ=$σ  δ=$δ  N=$N")
    println()
    @printf "  %-30s  closed-form: %7.4f   MC: %7.4f\n" "FixedStrikeGeomAsianCall" cf_call mc_call
    @printf "  %-30s  closed-form: %7.4f   MC: %7.4f\n" "FixedStrikeGeomAsianPut"  cf_put  mc_put

    # floating-strike: MC only (no simple closed form)
    fs_call = mc_geom_asian_price(FloatingStrikeGeometricAsianCall(T_), S, r, σ, δ, N)
    fs_put  = mc_geom_asian_price(FloatingStrikeGeometricAsianPut(T_),  S, r, σ, δ, N)
    @printf "  %-30s  MC: %7.4f\n" "FloatingStrikeGeomAsianCall" fs_call
    @printf "  %-30s  MC: %7.4f\n" "FloatingStrikeGeomAsianPut"  fs_put
end

_demo()