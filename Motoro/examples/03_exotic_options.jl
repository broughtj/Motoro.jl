# # Exotic Options
#
# This example covers the three families of exotic options in Motoro:
# binary (cash-or-nothing), lookback, and arithmetic Asian. Binary options
# have a closed-form BSM price; lookback and Asian options are path-dependent
# and require Monte Carlo simulation.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Motoro

data  = MarketData(100.0, 0.05, 0.20, 0.0)
model = RiskNeutralMonteCarlo(252, 50_000)   # daily steps, 50k paths

# ## Binary (cash-or-nothing) options
#
# A cash-or-nothing call pays a fixed `cash` amount if S_T > K, and zero
# otherwise. The put pays if S_T < K. Unlike vanilla options, the payoff
# is discontinuous at the strike.

con_call = CashOrNothingCall(100.0, 1.0, 1.0)   # pays $1 if S_T > 100
con_put  = CashOrNothingPut(100.0, 1.0, 1.0)    # pays $1 if S_T < 100

# Binary payoffs at a grid of terminal spot prices:
payoff.(con_call, [90.0, 99.0, 100.0, 101.0, 110.0])   # 0, 0, 0, 1, 1
payoff.(con_put,  [90.0, 99.0, 100.0, 101.0, 110.0])   # 1, 1, 0, 0, 0

# ### Analytical BSM prices
#
# Binary options have a closed-form price: cash * exp(-r*T) * N(±d2).

bsm_con_call = price(con_call, BlackScholes(), data)
bsm_con_put  = price(con_put,  BlackScholes(), data)

bsm_con_call.price   # ≈ 0.532  (risk-neutral probability of finishing ITM)
bsm_con_put.price    # ≈ 0.420

# Digital put-call parity: C + P = exp(-r*T) (present value of $1 certain)
bsm_con_call.price + bsm_con_put.price   # ≈ exp(-0.05 * 1.0) ≈ 0.951

# ### Monte Carlo check
#
# Only the terminal price is needed for binary options (steps = 1 is exact).

mc_binary = RiskNeutralMonteCarlo(1, 50_000)
price(con_call, mc_binary, data)   # should agree with BSM

# ## Lookback options
#
# Lookback payoffs depend on the running maximum or minimum of the asset
# price over the life of the contract.

# ### Floating-strike lookback
#
# The floating-strike call pays S_T - S_min (always non-negative — the holder
# effectively buys at the lowest price observed). The put pays S_max - S_T.

lb_call = FloatingStrikeLookbackCall(1.0)
lb_put  = FloatingStrikeLookbackPut(1.0)

price(lb_call, model, data)   # expensive — lookback premium over vanilla
price(lb_put,  model, data)

# ### Fixed-price lookback
#
# The fixed-price call pays max(0, S_max - K); the put pays max(0, K - S_min).
# These resemble vanilla options but with the best possible terminal price.

lb_fixed_call = FloatingPriceLookbackCall(100.0, 1.0)
lb_fixed_put  = FloatingPriceLookbackPut(100.0, 1.0)

price(lb_fixed_call, model, data)
price(lb_fixed_put,  model, data)

# Lookback options are strictly more valuable than their vanilla counterparts
# because the holder is guaranteed the best possible exercise decision in hindsight.

vanilla_call = EuropeanCall(100.0, 1.0)
price(vanilla_call, BlackScholes(), data).price         # ≈ 10.45
price(lb_fixed_call, model, data).price                 # > vanilla call price

# ## Arithmetic Asian options
#
# Asian payoffs depend on the arithmetic mean of the asset price over the
# simulation period, Ā. Averaging smooths out terminal price fluctuations,
# making Asian options cheaper than vanilla options.

# ### Floating-strike Asian
#
# The floating-strike call pays max(0, S_T - Ā). The holder benefits if the
# terminal price exceeds the historical average.

as_call = FloatingStrikeArithmeticAsianCall(1.0)
as_put  = FloatingStrikeArithmeticAsianPut(1.0)

price(as_call, model, data)
price(as_put,  model, data)

# ### Fixed-price Asian
#
# The fixed-price call pays max(0, Ā - K). This is the most common Asian
# variant in practice (e.g. commodity markets with average-price settlement).

as_fixed_call = FloatingPriceArithmeticAsianCall(100.0, 1.0)
as_fixed_put  = FloatingPriceArithmeticAsianPut(100.0, 1.0)

price(as_fixed_call, model, data)
price(as_fixed_put,  model, data)

# Asian options are cheaper than vanilla options due to averaging:
price(as_fixed_call, model, data).price    # < vanilla call price
price(vanilla_call, BlackScholes(), data).price

# ## Effect of time steps on path-dependent pricing
#
# Lookback and Asian payoffs depend on the full path, so the number of steps
# (monitoring frequency) materially affects the price. More steps = finer
# monitoring = higher lookback prices, lower Asian prices (more averaging).

let
    for steps in [12, 52, 252]
        m = RiskNeutralMonteCarlo(steps, 50_000)
        lb = price(lb_call, m, data).price
        as = price(as_fixed_call, m, data).price
        println("steps=$steps:  lookback_call=$(round(lb, digits=3)),  asian_call=$(round(as, digits=3))")
    end
end
