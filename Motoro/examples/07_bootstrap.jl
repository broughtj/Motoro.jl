# # Stationary Bootstrap for Hedged Monte Carlo
#
# This example demonstrates `StationaryBootstrap` as an empirical alternative
# to parametric GBM path simulation for delta hedging.
#
# Part 1 uses synthetic "historical" prices simulated from GBM. Because the
# history is drawn from GBM with the same volatility as our market data, the
# bootstrap and parametric hedge costs should be similar — this is the logic
# check. Deviations reflect finite-sample noise in the historical return
# distribution.
#
# Part 2 uses real SPY closing prices loaded from `examples/data/SPY_close.csv`
# to illustrate the file-based `HistoricalData` constructor and the difference
# between empirical and parametric hedge costs under realistic market dynamics.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Motoro
using Statistics

data = MarketData(100.0, 0.05, 0.20, 0.0)
call = EuropeanCall(100.0, 1.0)
mu   = 0.10    # real-world (P-measure) drift; not the risk-free rate

bsm = price(call, BlackScholes(), data).price
println("BSM reference: $(round(bsm, digits=4))")

# ## Generating synthetic historical data
#
# Simulate 5 years of daily prices under the real-world measure using drift mu
# and the same volatility as in `data`. The price path is used only to compute
# log-returns; the initial spot level does not affect the returns.

sigma = data.vol
dt    = 1.0 / 252

n_hist = 252 * 5
hist_prices = Vector{Float64}(undef, n_hist + 1)
hist_prices[1] = data.spot
for t in 2:(n_hist + 1)
    hist_prices[t] = hist_prices[t - 1] *
        exp((mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * randn())
end

hist = HistoricalData(log_returns(hist_prices))
println("History: $(length(hist.returns)) daily returns ($(length(hist.returns) ÷ 252) years)")

# ## Bootstrap vs parametric GBM delta hedge
#
# Delta hedging under GBM converges to the BSM price as `steps → ∞` regardless
# of drift (as shown in example 04). The bootstrap should give a similar result
# when the history is GBM with the same σ — the resampled return distribution
# approximates the true GBM distribution. Remaining differences are finite-sample
# noise from the historical draw.

bs = StationaryBootstrap(hist, 20)

gbm_dh = price(call, HedgedMonteCarlo(252, 20_000, DeltaHedge(mu)),     data)
bs_dh  = price(call, HedgedMonteCarlo(252, 20_000, DeltaHedge(mu), bs), data)

println("\nDelta hedge comparison (steps=252, reps=20_000):")
println("  BSM:              $(round(bsm, digits=4))")
println("  GBM delta hedge:  $(round(gbm_dh.price, digits=4))  ± $(round(gbm_dh.std, digits=4))")
println("  Bootstrap:        $(round(bs_dh.price, digits=4))  ± $(round(bs_dh.std, digits=4))")

# ## Effect of mean block length
#
# The mean block length controls how much serial dependence the bootstrap
# preserves. GBM returns are serially uncorrelated, so shorter blocks are
# more accurate — block_length = 1 reduces to plain iid resampling and
# recovers the true distribution exactly in expectation. Longer blocks
# introduce spurious autocorrelation and add variance without benefit.
#
# For real market data with genuine volatility clustering, a longer block
# length is appropriate.

println("\nEffect of mean block length (steps=252, reps=10_000):")
println("  BSM reference: $(round(bsm, digits=4))")
let
    for bl in [1, 5, 20, 50, 100]
        r = price(call, HedgedMonteCarlo(252, 10_000, DeltaHedge(mu),
            StationaryBootstrap(hist, bl)), data)
        println("  block_length=$(lpad(bl, 3)):  $(round(r.price, digits=4))  ± $(round(r.std, digits=4))")
    end
end

# ## Effect of history length
#
# With more historical data the resampled return distribution converges to the
# true GBM distribution and the bootstrap hedge cost approaches the parametric
# result. Short histories add sampling noise — the bootstrap can only draw
# returns it has observed.

println("\nEffect of history length (steps=252, reps=10_000, block_length=20):")
println("  BSM reference: $(round(bsm, digits=4))")
let
    for n_years in [1, 2, 5, 10, 20]
        n = 252 * n_years
        p = Vector{Float64}(undef, n + 1)
        p[1] = data.spot
        for t in 2:(n + 1)
            p[t] = p[t - 1] * exp((mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * randn())
        end
        h = HistoricalData(log_returns(p))
        r = price(call, HedgedMonteCarlo(252, 10_000, DeltaHedge(mu),
            StationaryBootstrap(h, 20)), data)
        println("  years=$(lpad(n_years, 2)):  $(round(r.price, digits=4))  ± $(round(r.std, digits=4))")
    end
end

# ## Stop-loss hedge with bootstrap
#
# `StationaryBootstrap` works with any `HedgeStrategy`. Note that the drift
# parameter in `StopLoss(mu)` has no effect on the simulated paths when using
# the bootstrap — paths are determined entirely by the resampled historical
# returns, not by a parametric drift. The `mu` argument is still required for
# the `StopLoss` constructor but is ignored by the bootstrap sampler.

gbm_sl = price(call, HedgedMonteCarlo(252, 20_000, StopLoss(mu)),     data)
bs_sl  = price(call, HedgedMonteCarlo(252, 20_000, StopLoss(mu), bs), data)

println("\nStop-loss comparison (steps=252, reps=20_000):")
println("  BSM:              $(round(bsm, digits=4))")
println("  GBM stop-loss:    $(round(gbm_sl.price, digits=4))  ± $(round(gbm_sl.std, digits=4))")
println("  Bootstrap:        $(round(bs_sl.price, digits=4))  ± $(round(bs_sl.std, digits=4))")

# ## Real market data: SPY closing prices
#
# `HistoricalData` can load prices directly from a CSV file. The constructor
# reads one column of prices, computes log-returns internally, and stores them.
# Column 2 here because column 1 is a date string.
#
# The file `examples/data/SPY_close.csv` contains 5 years of daily SPY closing
# prices fetched from Yahoo Finance. Two things distinguish it from the
# synthetic GBM history above:
#
#  1. The realized volatility (≈17%) differs from the 20% assumed in `data`.
#     The bootstrap hedge cost therefore diverges from the GBM reference — it
#     reflects what actual hedging would have cost, not a model's prediction.
#
#  2. The returns exhibit volatility clustering (GARCH-like behaviour). A
#     longer mean block length preserves more of this dependence structure.

spy_csv  = joinpath(@__DIR__, "data", "SPY_close.csv")
spy_hist = HistoricalData(spy_csv; col=2)

# Estimate realized volatility from the historical returns.
# Daily std × √252 annualizes under the standard iid assumption.
spy_vol_realized = std(spy_hist.returns) * sqrt(252)

println("\n--- Real SPY data ---")
println("Returns loaded:   $(length(spy_hist.returns)) daily log-returns")
println("Realized vol (σ̂): $(round(spy_vol_realized, digits=4)) annualized")

spy_call = EuropeanCall(562.0, 1.0)
spy_mu   = 0.10

# Two market data objects: one with assumed σ=0.20, one with realized σ̂.
# This lets us separate two sources of model error:
#   assumed_data → GBM mis-specified (wrong vol)
#   realized_data → GBM correctly calibrated to historical vol
spy_assumed  = MarketData(562.0, 0.045, 0.20, 0.0)
spy_realized = MarketData(562.0, 0.045, spy_vol_realized, 0.0)

bsm_assumed  = price(spy_call, BlackScholes(), spy_assumed).price
bsm_realized = price(spy_call, BlackScholes(), spy_realized).price

println("BSM (σ=0.20):     $(round(bsm_assumed, digits=4))")
println("BSM (σ=σ̂):        $(round(bsm_realized, digits=4))")

# Delta hedge comparison:
#   GBM (assumed)  — parametric model with σ=0.20
#   GBM (realized) — parametric model re-calibrated to historical vol
#   Bootstrap      — empirical paths drawn from the actual SPY return history
spy_bs = StationaryBootstrap(spy_hist, 20)

spy_gbm_assumed  = price(spy_call, HedgedMonteCarlo(252, 20_000, DeltaHedge(spy_mu)), spy_assumed)
spy_gbm_realized = price(spy_call, HedgedMonteCarlo(252, 20_000, DeltaHedge(spy_mu)), spy_realized)
spy_bs_dh        = price(spy_call, HedgedMonteCarlo(252, 20_000, DeltaHedge(spy_mu), spy_bs), spy_assumed)

println("\nDelta hedge comparison (steps=252, reps=20_000):")
println("  BSM (σ=0.20):             $(round(bsm_assumed, digits=4))")
println("  BSM (σ=σ̂):                $(round(bsm_realized, digits=4))")
println("  GBM delta hedge (σ=0.20): $(round(spy_gbm_assumed.price, digits=4))  ± $(round(spy_gbm_assumed.std, digits=4))")
println("  GBM delta hedge (σ=σ̂):    $(round(spy_gbm_realized.price, digits=4))  ± $(round(spy_gbm_realized.std, digits=4))")
println("  SPY bootstrap:            $(round(spy_bs_dh.price, digits=4))  ± $(round(spy_bs_dh.std, digits=4))")
