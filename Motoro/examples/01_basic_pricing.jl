# # Basic Option Pricing
#
# This example introduces the core pricing API: constructing market data,
# defining vanilla option contracts, computing payoffs, and pricing with
# both the Black-Scholes-Merton formula and the binomial tree.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Motoro

# ## Market data
#
# All pricing functions take a `MarketData` struct that bundles the four
# parameters common to every model: spot price, risk-free rate, volatility,
# and continuous dividend yield. Rates and volatility are annualised decimals.

data = MarketData(100.0, 0.05, 0.20, 0.0)

# ## Option contracts
#
# Options are plain structs parameterised by strike and time-to-expiry (in years).

call = EuropeanCall(100.0, 1.0)   # at-the-money call, 1 year
put  = EuropeanPut(100.0, 1.0)    # at-the-money put,  1 year

# ## Payoffs
#
# `payoff` evaluates the terminal intrinsic value at a given spot price.
# It broadcasts cleanly over vectors of spot prices.

payoff(call, 110.0)   # 10.0  (in-the-money)
payoff(call,  90.0)   # 0.0   (out-of-the-money)
payoff(put,   90.0)   # 10.0

spot_grid = 80.0:5.0:120.0
payoff.(call, spot_grid)

# ## Black-Scholes-Merton pricing
#
# `BlackScholes()` is the exact closed-form model for European options.
# All `price` calls return a `PricingResult` subtype; `.price` extracts
# the scalar value.

bsm_call = price(call, BlackScholes(), data)
bsm_put  = price(put,  BlackScholes(), data)

bsm_call.price   # ≈ 10.45
bsm_put.price    # ≈  5.57

# ### Put-call parity check
#
# For a non-dividend-paying stock:
#   C - P = S - K * exp(-r * T)

S, K, r, T = data.spot, call.strike, data.rate, call.expiry
parity_lhs = bsm_call.price - bsm_put.price
parity_rhs = S - K * exp(-r * T)

isapprox(parity_lhs, parity_rhs; atol=1e-10)   # true

# ## BSM delta
#
# `delta` gives the first-order sensitivity of the option price to the spot.
# Call delta ∈ [0, 1]; put delta ∈ [-1, 0].

delta(call, BlackScholes(), data)   # ≈  0.637
delta(put,  BlackScholes(), data)   # ≈ -0.363

# ## Binomial tree pricing
#
# The Cox-Ross-Rubinstein binomial tree converges to the BSM price as the
# number of steps increases. It also handles American options, where
# early exercise can add value.

price(call, Binomial(10),   data).price   # coarse — noticeable error
price(call, Binomial(100),  data).price   # ≈ BSM
price(call, Binomial(1000), data).price   # very close to BSM

# American puts carry an early exercise premium over their European counterparts.

am_put = AmericanPut(100.0, 1.0)
eu_put = EuropeanPut(100.0, 1.0)

price(am_put, Binomial(200), data).price   # > European put price
price(eu_put, Binomial(200), data).price

# ### Binomial convergence
#
# The oscillatory convergence pattern of CRR trees is visible when stepping
# through a range of tree sizes.

[price(call, Binomial(n), data).price for n in [10, 25, 50, 100, 200, 500]]
