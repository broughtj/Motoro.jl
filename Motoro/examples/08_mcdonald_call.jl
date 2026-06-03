# # Pricing the McDonald European Call Three Ways
#
# This example prices the standard "McDonald" European call — the canonical
# example from McDonald, *Derivatives Markets* (S=41, K=40, r=8%, σ=30%, T=1)
# — using three different machineries in `Motoro`:
#
#   1. Black-Scholes-Merton closed form          (exact, Q measure)
#   2. Risk-neutral Monte Carlo                  (simulated, Q measure)
#   3. Stationary bootstrap delta hedge          (empirical, P measure)
#
# The first two price the option directly under the risk-neutral measure. The
# third is conceptually different: `StationaryBootstrap` resamples *historical*
# returns under the real-world (P) measure, so it cannot be used with
# `RiskNeutralMonteCarlo` (that pairing raises a `MethodError` by design). The
# bootstrap reaches the option value indirectly, through the cost of a delta
# hedge — which converges to the BSM price as the rebalancing grid is refined,
# independent of the drift.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Motoro
using Random
using Statistics

# Reproducible draws for the synthetic history and the Monte Carlo runs.
Random.seed!(20260603)

# ## The McDonald contract and market
#
# Spot 41, strike 40, 8% risk-free rate, 30% volatility, one year, no dividend.

data = MarketData(41.0, 0.08, 0.30, 0.0)
call = EuropeanCall(40.0, 1.0)

# ## 1. Black-Scholes-Merton
#
# The exact closed-form benchmark. Every other method is judged against this.

bsm = price(call, BlackScholes(), data)
println("1. Black-Scholes-Merton")
println("   price = $(round(bsm.price, digits=4))")

# ## 2. Risk-neutral Monte Carlo
#
# Simulate GBM paths under the risk-free drift and average the discounted
# terminal payoff. For a path-independent European option a single time step
# is sufficient, but we use a finer grid here to mirror the bootstrap setup.
# `SimulationResult` carries the standard error alongside the estimate.

rnmc = price(call, RiskNeutralMonteCarlo(252, 100_000), data)
println("\n2. Risk-neutral Monte Carlo (steps=252, reps=100_000)")
println("   price = $(round(rnmc.price, digits=4))  ± $(round(rnmc.std, digits=4))")

# ## 3. Stationary bootstrap delta hedge
#
# `StationaryBootstrap` draws paths from a historical return series rather than
# from a parametric model. We have no real price file for this synthetic
# contract, so we manufacture five years of daily "history" from GBM at the
# same 30% volatility. Because delta-hedge replication cost is drift-independent,
# the real-world drift `mu` used to build the history (and passed to
# `DeltaHedge`) does not bias the result — it only needs the volatility to match.

mu    = 0.10                 # real-world (P-measure) drift — NOT the risk-free rate
sigma = data.vol

# Build the history with the package's own GBM path generator rather than
# hand-rolling the Euler step. `asset_paths` runs on any `MonteCarlo` model and
# treats its drift argument as whatever measure you feed it — here the real-world
# `mu`. We ask for a single path (`reps = 1`) covering `n_years` of daily steps;
# with `expiry = n_years` the internal time step is exactly 1/252.

n_years   = 5
n_hist    = 252 * n_years
generator = RiskNeutralMonteCarlo(n_hist, 1)
hist_path = asset_paths(generator, data.spot, mu, sigma, float(n_years))[1, :]

hist = HistoricalData(log_returns(hist_path))
bs   = StationaryBootstrap(hist, 20)   # ~20-day mean block length

# The bootstrap is restricted to `HedgedMonteCarlo`. The reported price is the
# mean discounted cost of running a continuously rebalanced BSM delta hedge
# along bootstrapped paths; it converges to the option's fair value.

boot = price(call, HedgedMonteCarlo(252, 20_000, DeltaHedge(mu), bs), data)
println("\n3. Stationary bootstrap delta hedge (steps=252, reps=20_000)")
println("   price = $(round(boot.price, digits=4))  ± $(round(boot.std, digits=4))")

# ## Summary
#
# All three should agree with the BSM benchmark to within Monte Carlo error.

println("\n--- Summary (McDonald call) ---")
println("  Black-Scholes-Merton:        $(round(bsm.price, digits=4))")
println("  Risk-neutral Monte Carlo:    $(round(rnmc.price, digits=4))  ± $(round(rnmc.std, digits=4))")
println("  Stationary bootstrap hedge:  $(round(boot.price, digits=4))  ± $(round(boot.std, digits=4))")
