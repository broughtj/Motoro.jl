# # Hedging Strategies
#
# This example compares two hedging strategies for a European call:
# stop-loss and delta hedging. Both are simulated under the real-world
# (P) measure using `HedgedMonteCarlo`. The expected discounted hedge cost
# converges to the BSM price as the hedge becomes more refined.
#
# Key insight: the BSM price is not just a theoretical construct — it equals
# the expected cost of replicating the option payoff through continuous
# delta hedging. Stop-loss hedging is simpler but never fully replicates
# the option, leaving residual hedging error regardless of step frequency.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Motoro

data = MarketData(50.0, 0.05, 0.40, 0.0)   # Hull (2014) Example 17.1
call = EuropeanCall(52.0, 5 / 12)

bsm = price(call, BlackScholes(), data)
bsm.price   # ≈ 5.18  — the theoretical replication cost

# ## Stop-loss hedging
#
# The stop-loss strategy holds the underlying whenever S ≥ K and holds cash
# otherwise. It switches position whenever the spot crosses the strike.
# The drift parameter `mu` is the real-world expected return of the stock.
#
# Despite its simplicity, stop-loss hedging is not self-financing in continuous
# time: the cost of crossing the strike is always positive (buy high, sell low),
# so the expected hedge cost is strictly greater than the BSM price.

mu = 0.13   # assumed real-world drift

result_sl = price(call, HedgedMonteCarlo(500, 50_000, StopLoss(mu)), data)
result_sl.price   # > BSM price — stop-loss carries a hedging error premium
result_sl.std

# ### Stop-loss does not converge to BSM with finer hedging
#
# Unlike delta hedging, refining the time grid does not eliminate the
# stop-loss hedging error. The discrete-time error persists because the
# strategy structure itself is sub-optimal.

let
    for steps in [10, 50, 100, 500, 1_000]
        r = price(call, HedgedMonteCarlo(steps, 20_000, StopLoss(mu)), data)
        println("steps=$steps:  cost=$(round(r.price, digits=4))  (BSM=$(round(bsm.price, digits=4)))")
    end
end

# ## Delta hedging
#
# The delta hedge continuously rebalances a position of Δ = ∂V/∂S shares.
# At each step the BSM delta is recomputed for the current spot and remaining
# time, and the cash flows from rebalancing are tracked. At expiry, the option
# delivery is settled against the final hedge position.
#
# In the limit of continuous rebalancing (steps → ∞), the expected discounted
# hedge cost converges exactly to the BSM price — regardless of the real-world
# drift `mu`. This is the fundamental result underlying risk-neutral pricing.

result_dh = price(call, HedgedMonteCarlo(100, 50_000, DeltaHedge(mu)), data)
result_dh.price   # close to BSM price
result_dh.std

# ### Delta hedge convergence to BSM
#
# As the hedge is rebalanced more frequently, the expected cost approaches the
# BSM price. The convergence rate is approximately 1/√steps.

println("\nDelta hedge convergence (BSM = $(round(bsm.price, digits=4))):")
let
    for steps in [5, 10, 25, 50, 100, 250, 500]
        r = price(call, HedgedMonteCarlo(steps, 20_000, DeltaHedge(mu)), data)
        println("  steps=$steps:  cost=$(round(r.price, digits=4)),  std=$(round(r.std, digits=4))")
    end
end

# ## Comparing the two strategies
#
# At the same step count, delta hedging is much closer to the BSM price.
# Stop-loss consistently overshoots; delta hedging brackets the BSM price
# with an error that shrinks with finer rebalancing.

steps = 100
reps  = 50_000

r_sl = price(call, HedgedMonteCarlo(steps, reps, StopLoss(mu)),  data)
r_dh = price(call, HedgedMonteCarlo(steps, reps, DeltaHedge(mu)), data)

println("\nsteps = $steps, reps = $reps:")
println("  BSM price:        $(round(bsm.price, digits=4))")
println("  Stop-loss cost:   $(round(r_sl.price, digits=4))  ± $(round(r_sl.std, digits=4))")
println("  Delta-hedge cost: $(round(r_dh.price, digits=4))  ± $(round(r_dh.std, digits=4))")

# ## Drift invariance of delta hedging
#
# The expected delta-hedge cost is independent of the real-world drift `mu`.
# This is the economic content of risk-neutral pricing: replication cost is
# determined by volatility, not by expected return.

println("\nDrift invariance of delta hedge (steps=100, reps=20_000):")
let
    for mu_test in [0.05, 0.10, 0.15, 0.20, 0.30]
        r = price(call, HedgedMonteCarlo(100, 20_000, DeltaHedge(mu_test)), data)
        println("  mu=$(mu_test):  cost=$(round(r.price, digits=4))")
    end
end
