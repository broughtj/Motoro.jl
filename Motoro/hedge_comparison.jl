using Motoro
using Statistics
using Printf
using Plots

# ── Parameters ───────────────────────────────────────────────────────────────
spot   = 50.0
strike = 52.0
rate   = 0.05
vol    = 0.40
div    = 0.0
expiry = 5/12
mu     = 0.10      # real-world drift for delta hedge paths

steps  = 100       # steps per inner simulation
reps   = 1_000     # paths per inner simulation
trials = 2_000     # number of repeated experiments

data = MarketData(spot, rate, vol, div)
call = EuropeanCall(strike, expiry)

bsm_price = price(call, BlackScholes(), data).price

# ── Repeated experiments ──────────────────────────────────────────────────────
rn_estimates = zeros(trials)
dh_estimates = zeros(trials)

for i in 1:trials
    rn_estimates[i] = price(call, RiskNeutralMonteCarlo(steps, reps), data).price
    dh_estimates[i] = price(call, HedgedMonteCarlo(steps, reps, DeltaHedge(mu)), data).price
end

# ── Summary table ─────────────────────────────────────────────────────────────
rn_mean, rn_std = mean(rn_estimates), std(rn_estimates)
dh_mean, dh_std = mean(dh_estimates), std(dh_estimates)

println()
println("=" ^ 65)
println("   Sampling Distribution of the Price Estimator")
@printf("   BSM price: %.4f   steps: %d   reps: %d   trials: %d\n",
        bsm_price, steps, reps, trials)
println("=" ^ 65)
@printf("   %-26s  %8s  %8s  %8s\n", "Method", "Mean", "Std Dev", "vs BSM")
println("-" ^ 65)
@printf("   %-26s  %8.4f  %8s  %8s\n",  "Black-Scholes-Merton", bsm_price, "—", "—")
@printf("   %-26s  %8.4f  %8.4f  %+8.4f\n", "Risk-Neutral MC",  rn_mean, rn_std, rn_mean - bsm_price)
@printf("   %-26s  %8.4f  %8.4f  %+8.4f\n", "Delta Hedge MC",   dh_mean, dh_std, dh_mean - bsm_price)
println("-" ^ 65)
@printf("   Variance reduction: %.1fx\n", rn_std / dh_std)
println("=" ^ 65)
println()

# ── Histograms ────────────────────────────────────────────────────────────────
# Use shared x-axis limits centered on BSM price so scales are comparable
half_range = 4 * rn_std
xlims = (bsm_price - half_range, bsm_price + half_range)

p1 = histogram(rn_estimates,
    bins      = 60,
    normalize = :pdf,
    color     = :steelblue,
    alpha     = 0.7,
    label     = "Estimate",
    xlabel    = "Price estimate",
    ylabel    = "Density",
    title     = "Risk-Neutral MC",
    xlims     = xlims)
vline!(p1, [bsm_price], color = :red,   lw = 2, label = "BSM price")
vline!(p1, [rn_mean],   color = :black, lw = 2, ls = :dash, label = "Mean")

p2 = histogram(dh_estimates,
    bins      = 60,
    normalize = :pdf,
    color     = :darkorange,
    alpha     = 0.7,
    label     = "Estimate",
    xlabel    = "Price estimate",
    ylabel    = "Density",
    title     = "Delta Hedge MC",
    xlims     = xlims)
vline!(p2, [bsm_price], color = :red,   lw = 2, label = "BSM price")
vline!(p2, [dh_mean],   color = :black, lw = 2, ls = :dash, label = "Mean")

fig = plot(p1, p2,
    layout     = (1, 2),
    size       = (950, 420),
    margin     = 5Plots.mm,
    plot_title = "steps=$steps, reps=$reps, trials=$trials, μ=$mu")

savefig(fig, "hedge_comparison.png")
println("  Plot saved → hedge_comparison.png")
println()
