using Motoro
using Printf

# ── Setup ─────────────────────────────────────────────────────────────────────
S0   = 100.0
K    = 100.0
r    = 0.05
σ    = 0.25
δ    = 0.0
T    = 1.0
reps = 1_000

data         = MarketData(S0, r, σ, δ)
con_call     = CashOrNothingCall(K, T, 1.0)
vanilla_call = EuropeanCall(K, T)

true_price = 0.5040
bsm_check  = price(con_call, BlackScholes(), data).price

println()
println("=" ^ 58)
println("  Exam Q3: Cash-or-Nothing Call via Monte Carlo")
@printf("  True price: %.4f   BSM check: %.4f   reps: %d\n",
        true_price, bsm_check, reps)
println("=" ^ 58)

# ── Part a & b: plain Monte Carlo ─────────────────────────────────────────────
plain = price(con_call, RiskNeutralMonteCarlo(1, reps), data)
ci_lo = plain.price - 1.96 * plain.std
ci_hi = plain.price + 1.96 * plain.std

println()
println("a. Basic Monte Carlo estimate")
@printf("   Price estimate : %.4f\n",  plain.price)
@printf("   True price     : %.4f\n",  true_price)
@printf("   Error          : %+.4f\n", plain.price - true_price)
println()
println("b. Precision of the estimate")
@printf("   Std error      : %.4f\n", plain.std)
@printf("   95%% CI         : (%.4f, %.4f)\n", ci_lo, ci_hi)

# ── Part c: control variate with β = 1 ───────────────────────────────────────
cv1 = price(con_call, ControlVariateMonteCarlo(1, reps, ControlVariate(vanilla_call, 1.0)), data)
println()
println("c. Control variate, β = 1")
@printf("   Price estimate : %.4f\n", cv1.price)
@printf("   Std error      : %.4f\n", cv1.std)
if plain.std / cv1.std < 1.0
    @printf("   SE change      : %.1fx WORSE  (β=1 inappropriate: scales differ)\n", cv1.std / plain.std)
else
    @printf("   SE reduction   : %.1fx\n", plain.std / cv1.std)
end

# ── Part d: control variate with optimal β ────────────────────────────────────
cv_opt = price(con_call, ControlVariateMonteCarlo(1, reps, ControlVariate(vanilla_call)), data)
println()
println("d. Control variate, optimal β")
@printf("   Price estimate : %.4f\n", cv_opt.price)
@printf("   Std error      : %.4f\n", cv_opt.std)
@printf("   SE reduction   : %.1fx\n", plain.std / cv_opt.std)

println()
println("=" ^ 58)
@printf("  Summary: SE without CV : %.4f\n", plain.std)
if plain.std / cv1.std < 1.0
    @printf("           SE with β=1  : %.4f  (%.1fx worse)\n",  cv1.std,    cv1.std / plain.std)
else
    @printf("           SE with β=1  : %.4f  (%.1fx better)\n", cv1.std,    plain.std / cv1.std)
end
@printf("           SE with β*   : %.4f  (%.1fx better)\n",     cv_opt.std, plain.std / cv_opt.std)
println("=" ^ 58)
println()
