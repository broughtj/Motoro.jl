# # Monte Carlo Pricing and Variance Reduction
#
# This example covers risk-neutral Monte Carlo pricing for European options
# and demonstrates how variance reduction techniques — stratified sampling
# and antithetic variates — reduce estimator standard error for a fixed
# simulation budget.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Motoro

data = MarketData(100.0, 0.05, 0.20, 0.0)
call = EuropeanCall(100.0, 1.0)

bsm_price = price(call, BlackScholes(), data).price   # reference value ≈ 10.45

# ## Baseline Monte Carlo
#
# `RiskNeutralMonteCarlo(steps, reps)` simulates `reps` paths each of `steps`
# time steps under the risk-neutral (Q) measure. The result is a
# `SimulationResult` with a `.price` estimate and `.std` standard error.
#
# For a European option only the terminal price matters, so `steps = 1`
# is exact (one GBM step to expiry). More steps are needed for path-dependent
# options.

mc = RiskNeutralMonteCarlo(1, 10_000)
result = price(call, mc, data)

result.price   # close to BSM
result.std     # standard error of the estimate

# ## Variance reduction
#
# `VarianceReduction` composes a draw method with a pairing method.
# The default (used when no method is supplied) is pseudo-random with no pairing.

# ### Antithetic variates
#
# For each draw z, also simulate a path using -z. The negative correlation
# between paired paths reduces variance. The `reps` budget is split evenly
# between the two sets, so the effective path count is unchanged.

av = VarianceReduction(PseudoRandom(), Antithetic())
result_av = price(call, RiskNeutralMonteCarlo(1, 10_000, av), data)

result_av.price
result_av.std    # lower than baseline

# ### Stratified sampling
#
# Divides [0, 1] into `reps` equal strata and draws one standard-normal
# quantile per stratum (then shuffles). This ensures uniform coverage of
# the draw distribution and further reduces variance.

strat = VarianceReduction(Stratified(), NoPairing())
result_strat = price(call, RiskNeutralMonteCarlo(1, 10_000, strat), data)

result_strat.price
result_strat.std

# ### Combined
#
# Stratified sampling and antithetic variates are independent and can be
# used together.

combined = VarianceReduction(Stratified(), Antithetic())
result_combined = price(call, RiskNeutralMonteCarlo(1, 10_000, combined), data)

result_combined.price
result_combined.std   # typically the lowest standard error of the four

# ## Comparing standard errors
#
# Running each method at the same budget makes the variance reduction
# clearly visible.

methods = [
    ("Pseudo-random",              VarianceReduction(PseudoRandom(), NoPairing())),
    ("Antithetic",                 VarianceReduction(PseudoRandom(), Antithetic())),
    ("Stratified",                 VarianceReduction(Stratified(),   NoPairing())),
    ("Stratified + antithetic",    VarianceReduction(Stratified(),   Antithetic())),
]

let reps = 10_000
    for (label, method) in methods
        r = price(call, RiskNeutralMonteCarlo(1, reps, method), data)
        println("$label:  price = $(round(r.price, digits=4)),  std = $(round(r.std, digits=4))")
    end
end

# ## Convergence with simulation budget
#
# Standard error scales as 1/√reps. Doubling reps halves the standard error.

let
    for reps in [500, 1_000, 5_000, 10_000, 50_000, 100_000]
        r = price(call, RiskNeutralMonteCarlo(1, reps), data)
        println("reps = $reps:  std = $(round(r.std, digits=5))")
    end
end

# ## Simulating asset paths directly
#
# `asset_paths` returns a matrix of size (reps, steps+1). Each row is one
# simulated price path; column 1 is the initial spot price.

model = RiskNeutralMonteCarlo(52, 1_000)   # 52 weekly steps, 1000 paths
paths = asset_paths(model, data.spot, data.rate, data.vol, call.expiry)

size(paths)        # (1000, 53)
paths[1, :]        # first simulated path
paths[:, end]      # terminal prices across all paths
