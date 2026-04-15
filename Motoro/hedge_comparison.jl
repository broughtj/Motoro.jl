using Motoro
using LinearAlgebra
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
mu     = 0.10        # real-world drift for delta hedge paths
reps   = 50_000

data = MarketData(spot, rate, vol, div)
call = EuropeanCall(strike, expiry)

# ── BSM reference ─────────────────────────────────────────────────────────────
bsm_price = price(call, BlackScholes(), data).price

# ── Collect raw payoffs for a given step count ────────────────────────────────
function rn_payoffs(steps)
    model = RiskNeutralMonteCarlo(steps, reps)
    paths = asset_paths(model.method, model, spot, rate, vol, expiry)
    disc  = exp(-rate * expiry)
    return disc .* payoff.(call, paths[:, end])
end

function dh_costs(steps)
    model      = HedgedMonteCarlo(steps, reps, DeltaHedge(mu))
    paths      = asset_paths(model.method, model, spot, mu, vol, expiry)
    dt         = expiry / steps
    dfs        = exp.(-rate * collect(0:steps) * dt)
    costs      = zeros(reps)

    for k in 1:reps
        path       = paths[k, :]
        position   = 0.0
        cash_flows = zeros(steps + 1)

        for j in 1:steps
            τ             = expiry - (j - 1) * dt
            Δ             = delta(EuropeanCall(strike, τ), BlackScholes(), MarketData(path[j], rate, vol, div))
            cash_flows[j] = (position - Δ) * path[j]
            position      = Δ
        end

        if path[end] > strike
            cash_flows[end] = strike - (1.0 - position) * path[end]
        else
            cash_flows[end] = position * path[end]
        end

        costs[k] = -dot(dfs, cash_flows)
    end

    return costs
end

# ── Convergence table ─────────────────────────────────────────────────────────
step_counts = [10, 50, 100, 500]

results = [(steps, rn_payoffs(steps), dh_costs(steps)) for steps in step_counts]

println()
println("=" ^ 75)
println("   Convergence: Risk-Neutral MC vs Delta Hedge MC")
@printf("   BSM price: %.4f    reps: %d    μ: %.2f\n", bsm_price, reps, mu)
println("=" ^ 75)
@printf("   %-6s  %-22s  %-22s  %s\n",
        "Steps", "Risk-Neutral MC", "Delta Hedge MC", "Var. Red.")
@printf("   %-6s  %-10s %-10s  %-10s %-10s  %s\n",
        "", "Mean", "Std Err", "Mean", "Std Err", "(SE ratio)")
println("-" ^ 75)
for (steps, rn, dh) in results
    rn_m, rn_se = mean(rn), std(rn) / sqrt(reps)
    dh_m, dh_se = mean(dh), std(dh) / sqrt(reps)
    @printf("   %-6d  %-10.4f %-10.5f  %-10.4f %-10.5f  %.1fx\n",
            steps, rn_m, rn_se, dh_m, dh_se, rn_se / dh_se)
end
println("-" ^ 75)
@printf("   %-6s  %-10.4f %-10s  %-10.4f %-10s\n",
        "BSM", bsm_price, "—", bsm_price, "—")
println("=" ^ 75)
println()

# ── Histograms at steps = 100 ─────────────────────────────────────────────────
_, rn_100, dh_100 = results[findfirst(r -> r[1] == 100, results)]

xlims_rn = (-1.0, maximum(rn_100) * 1.05)
xlims_dh = (bsm_price - 4 * std(dh_100), bsm_price + 4 * std(dh_100))

p1 = histogram(rn_100,
    bins       = 80,
    normalize  = :pdf,
    color      = :steelblue,
    alpha      = 0.7,
    label      = "Discounted payoff",
    xlabel     = "Value",
    ylabel     = "Density",
    title      = "Risk-Neutral MC  (steps=100)",
    xlims      = xlims_rn)
vline!(p1, [bsm_price], color = :red,   lw = 2, label = "BSM price")
vline!(p1, [mean(rn_100)], color = :black, lw = 2, ls = :dash, label = "MC mean")

p2 = histogram(dh_100,
    bins       = 80,
    normalize  = :pdf,
    color      = :darkorange,
    alpha      = 0.7,
    label      = "Hedge cost",
    xlabel     = "Value",
    ylabel     = "Density",
    title      = "Delta Hedge MC  (steps=100)",
    xlims      = xlims_dh)
vline!(p2, [bsm_price], color = :red,   lw = 2, label = "BSM price")
vline!(p2, [mean(dh_100)], color = :black, lw = 2, ls = :dash, label = "DH mean")

fig = plot(p1, p2,
    layout     = (1, 2),
    size       = (950, 420),
    margin     = 5Plots.mm,
    plot_title = "Risk-Neutral vs Delta Hedge MC  (reps=$(reps÷1_000)k, μ=$(mu))")

savefig(fig, "hedge_comparison.png")
println("  Plot saved → hedge_comparison.png")
println()
