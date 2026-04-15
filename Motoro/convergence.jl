using Motoro
using Statistics
using Printf

# ── Parameters ───────────────────────────────────────────────────────────────
spot   = 50.0
strike = 52.0
rate   = 0.05
vol    = 0.40
div    = 0.0
expiry = 5/12
mu     = 0.10

reps   = 1_000     # paths per inner simulation
trials = 1_000     # repeated experiments per step count

step_counts = [5, 10, 25, 50, 100, 250, 500]

data = MarketData(spot, rate, vol, div)
call = EuropeanCall(strike, expiry)

bsm_price = price(call, BlackScholes(), data).price

# ── Run trials ────────────────────────────────────────────────────────────────
println()
println("=" ^ 72)
println("   Pathwise Convergence: Risk-Neutral vs Delta Hedge MC")
@printf("   BSM price: %.4f   reps: %d   trials: %d   μ: %.2f\n",
        bsm_price, reps, trials, mu)
println("=" ^ 72)
@printf("   %-6s  %-22s  %-22s  %10s\n",
        "Steps", "Risk-Neutral MC", "Delta Hedge MC", "Var. Red.")
@printf("   %-6s  %-10s  %-10s  %-10s  %-10s  %10s\n",
        "", "Mean", "Std Dev", "Mean", "Std Dev", "(σ_RN/σ_DH)")
println("-" ^ 72)

for steps in step_counts
    rn_est = [price(call, RiskNeutralMonteCarlo(steps, reps), data).price
              for _ in 1:trials]
    dh_est = [price(call, HedgedMonteCarlo(steps, reps, DeltaHedge(mu)), data).price
              for _ in 1:trials]

    rn_mean, rn_std = mean(rn_est), std(rn_est)
    dh_mean, dh_std = mean(dh_est), std(dh_est)

    @printf("   %-6d  %-10.4f  %-10.4f  %-10.4f  %-10.4f  %10.1fx\n",
            steps, rn_mean, rn_std, dh_mean, dh_std, rn_std / dh_std)
end

println("-" ^ 72)
@printf("   %-6s  %-10.4f  %-10s  %-10.4f  %-10s\n",
        "BSM", bsm_price, "—", bsm_price, "—")
println("=" ^ 72)
println()
println("   Note: σ_RN is roughly constant across steps (path count drives")
println("   variance, not resolution). σ_DH shrinks as O(1/√steps) since")
println("   finer rebalancing leaves a smaller residual hedging error.")
println()
