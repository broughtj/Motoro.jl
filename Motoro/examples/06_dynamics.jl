# # Asset Price Dynamics
#
# This example compares two asset price models available in Motoro:
# geometric Brownian motion (`GeometricBrownianMotion`) and Merton (1976) jump
# diffusion (`JumpDiffusion`). Both can be used with any Monte Carlo pricing model.
#
# The key question is: when does the choice of dynamics matter, and how much?

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Motoro

data = MarketData(100.0, 0.05, 0.20, 0.0)

# ## Vanilla option prices: GBM vs jump diffusion
#
# For at-the-money options the two models give similar prices — most of the
# terminal distribution is away from the jump region. The difference grows for
# deep out-of-the-money options where the tails dominate.

jd = JumpDiffusion(3.0, -0.02, 0.05)   # ~3 jumps/year, small negative mean jump

call_atm  = EuropeanCall(100.0, 1.0)   # ATM
call_otm  = EuropeanCall(115.0, 1.0)   # 15% OTM
call_dotm = EuropeanCall(130.0, 1.0)   # 30% deep OTM

bsm_atm  = price(call_atm,  BlackScholes(), data).price
bsm_otm  = price(call_otm,  BlackScholes(), data).price
bsm_dotm = price(call_dotm, BlackScholes(), data).price

reps = 100_000

println("Call price comparison (reps=$reps):")
println("  Strike  BSM     GBM-MC   JD-MC")
let
    for (label, option, bsm) in [
            ("ATM  K=100", call_atm,  bsm_atm),
            ("OTM  K=115", call_otm,  bsm_otm),
            ("DOTM K=130", call_dotm, bsm_dotm),
        ]
        gbm = price(option, RiskNeutralMonteCarlo(252, reps),       data).price
        jdp = price(option, RiskNeutralMonteCarlo(252, reps, jd),   data).price
        println("  $label:  $(round(bsm, digits=3))   $(round(gbm, digits=3))   $(round(jdp, digits=3))")
    end
end

# ## Jump parameter sensitivity
#
# `JumpDiffusion` has three parameters:
#   - `lambda`: jump intensity (average jumps per year). Higher lambda means
#     more frequent, smaller jumps. At very high lambda, the process approaches
#     a diffusion with higher effective volatility.
#   - `alpha_j`: mean log-jump size. Negative values create a downward bias in
#     jumps (common in equity markets — crashes are more frequent than rallies).
#   - `sigma_j`: dispersion of log-jump sizes. Higher sigma_j fattens the tails
#     of the return distribution.
#
# We price an OTM call under each variation to see which parameters matter most.

call_otm = EuropeanCall(115.0, 1.0)

println("\nParameter sensitivity (OTM call, K=115, reps=50_000):")
println("  BSM reference: $(round(bsm_otm, digits=4))")

println("\n  Varying lambda (alpha_j=-0.02, sigma_j=0.05):")
let
    for lam in [0.5, 1.0, 3.0, 5.0, 10.0]
        r = price(call_otm, RiskNeutralMonteCarlo(252, 50_000, JumpDiffusion(lam, -0.02, 0.05)), data)
        println("    lambda=$(lam):  $(round(r.price, digits=4))")
    end
end

println("\n  Varying alpha_j (lambda=3, sigma_j=0.05):")
let
    for aj in [-0.10, -0.05, -0.02, 0.0, 0.05]
        r = price(call_otm, RiskNeutralMonteCarlo(252, 50_000, JumpDiffusion(3.0, aj, 0.05)), data)
        println("    alpha_j=$(aj):  $(round(r.price, digits=4))")
    end
end

println("\n  Varying sigma_j (lambda=3, alpha_j=-0.02):")
let
    for sj in [0.01, 0.05, 0.10, 0.20, 0.30]
        r = price(call_otm, RiskNeutralMonteCarlo(252, 50_000, JumpDiffusion(3.0, -0.02, sj)), data)
        println("    sigma_j=$(sj):  $(round(r.price, digits=4))")
    end
end

# ## The implied volatility smile
#
# Under GBM, Black-Scholes prices are exact for every strike — inverting BSM
# always recovers the same flat volatility σ. Under jump diffusion, the
# terminal return distribution has fat tails and negative skewness (with
# downward-biased jumps). When JD prices are inverted through BSM, the implied
# volatility varies by strike — this is the "smile" or "skew" that practitioners
# observe in option markets.
#
# We compute JD prices across strikes and back out the BSM-implied vol for each.

# Simple BSM implied vol via bisection
function implied_vol(option::EuropeanCall, target_price, data::MarketData;
        lo=1e-4, hi=5.0, tol=1e-6)
    for _ in 1:100
        mid = (lo + hi) / 2
        p = price(option, BlackScholes(), MarketData(data.spot, data.rate, mid, data.div)).price
        p > target_price ? (hi = mid) : (lo = mid)
        hi - lo < tol && break
    end
    return (lo + hi) / 2
end

jd_smile = JumpDiffusion(5.0, -0.05, 0.10)   # more pronounced parameters for visibility
strikes = [80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0]

println("\nImplied volatility smile (JumpDiffusion(5.0, -0.05, 0.10), reps=200_000):")
println("  Strike   JD price   Implied vol")
let
    for K in strikes
        opt = EuropeanCall(K, 1.0)
        jd_price = price(opt, RiskNeutralMonteCarlo(252, 200_000, jd_smile), data).price
        iv = implied_vol(opt, jd_price, data)
        println("  K=$(lpad(K, 5)):   $(rpad(round(jd_price, digits=4), 8))   $(round(iv * 100, digits=2))%")
    end
end

# ## Path-dependent options under jump diffusion
#
# Lookback and Asian payoffs depend on the full price path. Jumps affect these
# options differently than vanillas: a large downward jump creates a new minimum
# that the floating-strike lookback call benefits from, but that also depresses
# the arithmetic mean used by Asian options.

model_gbm = RiskNeutralMonteCarlo(252, 50_000)                  # GeometricBrownianMotion default
model_jd  = RiskNeutralMonteCarlo(252, 50_000, jd)

lb_call  = FloatingStrikeLookbackCall(1.0)
as_call  = FloatingPriceArithmeticAsianCall(100.0, 1.0)

println("\nPath-dependent options:")
let
    lb_gbm = price(lb_call, model_gbm, data).price
    lb_jd  = price(lb_call, model_jd,  data).price
    as_gbm = price(as_call, model_gbm, data).price
    as_jd  = price(as_call, model_jd,  data).price

    println("  Floating-strike lookback call:  GBM=$(round(lb_gbm, digits=3)),  JD=$(round(lb_jd, digits=3))")
    println("  Arithmetic Asian call (K=100):  GBM=$(round(as_gbm, digits=3)),  JD=$(round(as_jd, digits=3))")
end

# ## Combining jump diffusion with variance reduction
#
# `JumpDiffusion` composes freely with `VarianceReduction`. Antithetic variates
# apply to the diffusion component; the jump draws are always independent.

av = VarianceReduction(PseudoRandom(), Antithetic())
call_atm = EuropeanCall(100.0, 1.0)

plain = price(call_atm, RiskNeutralMonteCarlo(252, 20_000, jd),     data)
anti  = price(call_atm, RiskNeutralMonteCarlo(252, 20_000, av, jd), data)

println("\nJD + antithetic variates (ATM call, reps=20_000):")
println("  Plain JD:    price=$(round(plain.price, digits=4)),  std=$(round(plain.std, digits=5))")
println("  JD+antithetic: price=$(round(anti.price, digits=4)),  std=$(round(anti.std, digits=5))")
