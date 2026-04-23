# # Control Variate Monte Carlo
#
# This example demonstrates `ControlVariateMonteCarlo`, which prices a target
# option by correcting a plain Monte Carlo estimator using a control variate —
# a second option whose analytical price is known.
#
# The adjusted estimator is:
#
#   V̂_cv = mean(V - beta * (C - C_BSM))
#
# where V is the simulated target payoff, C is the simulated control payoff,
# C_BSM is the known analytical control price, and `beta` is a coefficient
# chosen to minimise the estimator variance.
#
# When the target and control payoffs are highly correlated, the variance
# reduction can be dramatic — often 10–100× compared to plain Monte Carlo.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Motoro

data = MarketData(100.0, 0.05, 0.20, 0.0)

# ## Pricing a binary option with a European call control
#
# A cash-or-nothing call is difficult to price accurately by plain Monte Carlo
# because its payoff is binary — the estimator variance is high. A European
# call on the same underlying is an effective control: both pay off when S_T > K,
# so their simulated payoffs are strongly correlated.

target  = CashOrNothingCall(100.0, 1.0, 1.0)
control = ControlVariate(EuropeanCall(100.0, 1.0))   # uses OptimalBeta by default

bsm_target = price(target, BlackScholes(), data).price   # ≈ 0.532

# ### Optimal beta (estimated from the simulation)
#
# `OptimalBeta` estimates beta* = Cov(V, C) / Var(C) from the same paths
# used for pricing. This introduces a small in-sample bias that vanishes
# as reps → ∞, but in practice it is negligible for reps ≥ 1000.

plain = price(target, RiskNeutralMonteCarlo(1, 10_000), data)
cv    = price(target, ControlVariateMonteCarlo(1, 10_000, control), data)

println("BSM (reference):  $(round(bsm_target, digits=5))")
println("Plain MC:         $(round(plain.price, digits=5))  ± $(round(plain.std, digits=5))")
println("Control variate:  $(round(cv.price, digits=5))  ± $(round(cv.std, digits=5))")
println("Variance ratio:   $(round((plain.std / cv.std)^2, digits=1))×")

# ### Fixed beta
#
# If the correlation structure is known in advance, a fixed beta can be
# supplied directly. `beta = 1` is a natural starting point when the target
# and control have similar payoff magnitudes.

control_fixed = ControlVariate(EuropeanCall(100.0, 1.0), 1.0)
cv_fixed = price(target, ControlVariateMonteCarlo(1, 10_000, control_fixed), data)

cv_fixed.price
cv_fixed.std   # may be better or worse than optimal depending on the true beta

# ## Pricing a path-dependent exotic option
#
# Control variates are most valuable when the target is expensive to simulate
# accurately. Arithmetic Asian options are a good example: their payoff depends
# on the path mean, and a vanilla call on the same underlying (with a roughly
# matching strike) is an effective control.

asian_target = FloatingPriceArithmeticAsianCall(100.0, 1.0)
asian_control = ControlVariate(EuropeanCall(100.0, 1.0))

plain_asian = price(asian_target, RiskNeutralMonteCarlo(252, 10_000), data)
cv_asian    = price(asian_target, ControlVariateMonteCarlo(252, 10_000, asian_control), data)

println("\nAsian call:")
println("Plain MC:         $(round(plain_asian.price, digits=5))  ± $(round(plain_asian.std, digits=5))")
println("Control variate:  $(round(cv_asian.price, digits=5))  ± $(round(cv_asian.std, digits=5))")
println("Variance ratio:   $(round((plain_asian.std / cv_asian.std)^2, digits=1))×")

# ## Effect of simulation budget
#
# Control variates reduce variance at every budget level. The relative
# improvement is roughly constant across reps, so the variance ratio
# is a good summary statistic.

println("\nConvergence with reps (binary call target):")
let
    for reps in [1_000, 5_000, 10_000, 50_000]
        plain_r = price(target, RiskNeutralMonteCarlo(1, reps), data)
        cv_r    = price(target, ControlVariateMonteCarlo(1, reps, control), data)
        ratio   = round((plain_r.std / cv_r.std)^2, digits=1)
        println("  reps=$reps:  plain std=$(round(plain_r.std, digits=5)),  cv std=$(round(cv_r.std, digits=5)),  ratio=$(ratio)×")
    end
end

# ## Combining control variates with other variance reduction
#
# `ControlVariateMonteCarlo` accepts a `VarianceReduction` method, so control
# variates can be stacked with stratified sampling or antithetic variates.

av_method = VarianceReduction(PseudoRandom(), Antithetic())
cv_av = price(target,
    ControlVariateMonteCarlo(1, 10_000, control, av_method),
    data)

println("\nCombined (control variate + antithetic):")
println("  price=$(round(cv_av.price, digits=5)),  std=$(round(cv_av.std, digits=5))")
