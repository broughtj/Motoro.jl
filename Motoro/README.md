# Motoro.jl

A Julia package for teaching computational options pricing. Motoro implements classical
pricing methods — binomial trees, Black-Scholes-Merton, and Monte Carlo simulation —
alongside exotic options, hedging strategies, variance reduction techniques, and
empirical path simulation from historical data, with a focus on clarity and
pedagogical value.

## Installation

From the Julia REPL:

```julia
using Pkg
Pkg.add(url="https://github.com/broughtj/Motoro.jl", subdir="Motoro")
```

Or clone the repository and activate the local environment:

```julia
using Pkg
Pkg.activate("Motoro")
Pkg.instantiate()
```

## Quick Start

```julia
using Motoro

# Define market data: spot, risk-free rate, volatility, dividend yield
data = MarketData(41.0, 0.08, 0.30, 0.0)

# Define options
call = EuropeanCall(40.0, 1.0)   # ATM call, 1 year to expiry
put  = EuropeanPut(40.0, 1.0)    # ATM put,  1 year to expiry
```

## Examples

Annotated example scripts are in `examples/`. Each file is self-contained and can be
run directly or `include`d from the REPL:

| File | Topic |
|---|---|
| `01_basic_pricing.jl` | BSM, Binomial, put-call parity, delta, convergence |
| `02_monte_carlo.jl` | `RiskNeutralMonteCarlo`, variance reduction, convergence with budget |
| `03_exotic_options.jl` | Binary, lookback, and Asian options; monitoring frequency |
| `04_hedging.jl` | Stop-loss vs delta hedging; BSM convergence; drift invariance |
| `05_control_variate.jl` | `ControlVariateMonteCarlo`; optimal vs fixed beta; stacking with antithetic |
| `06_dynamics.jl` | `GeometricBrownianMotion` vs `JumpDiffusion`; parameter sensitivity; implied vol smile |
| `07_bootstrap.jl` | `StationaryBootstrap` with `HedgedMonteCarlo`; block length and history length effects; real SPY data; specification vs distribution error |

```julia
include("examples/01_basic_pricing.jl")
```

## Pricing Methods

### Black-Scholes-Merton (analytical)

Exact closed-form solution for European and binary options under the BSM assumptions.

```julia
bsm = BlackScholes()
price(call, bsm, data)   # ≈ 6.96
price(put,  bsm, data)   # ≈ 2.89
```

### Binomial Tree (CRR)

Cox-Ross-Rubinstein binomial lattice. Works for both European and American options.

```julia
# European
binom = Binomial(200)
price(call, binom, data)
price(put,  binom, data)

# American (supports early exercise)
am_put = AmericanPut(40.0, 1.0)
price(am_put, Binomial(200), data)   # premium > European put
```

Increase `steps` for higher accuracy; the binomial price converges to BSM as `steps → ∞`.

### Monte Carlo

Motoro has three Monte Carlo model types, all subtypes of the abstract `MonteCarlo`:

#### `RiskNeutralMonteCarlo` — standard option pricing

Simulates asset paths under the risk-neutral (Q) measure and averages discounted payoffs.

```julia
mc = RiskNeutralMonteCarlo(100, 10_000)   # 100 steps, 10,000 paths
price(call, mc, data)
```

#### Variance Reduction

All Monte Carlo types accept an optional `VarianceReduction` argument:

| Draw method  | Pairing method | Effect |
|---|---|---|
| `PseudoRandom` | `NoPairing`  | Baseline Monte Carlo |
| `PseudoRandom` | `Antithetic` | Antithetic variates |
| `Stratified`   | `NoPairing`  | Stratified sampling |
| `Stratified`   | `Antithetic` | Both combined |

```julia
# Antithetic variates
mc_anti = RiskNeutralMonteCarlo(100, 10_000, VarianceReduction(PseudoRandom(), Antithetic()))
price(call, mc_anti, data)

# Stratified sampling
mc_strat = RiskNeutralMonteCarlo(100, 10_000, VarianceReduction(Stratified(), NoPairing()))
price(call, mc_strat, data)
```

#### Asset Dynamics

All Monte Carlo types accept an optional `AssetDynamics` argument that controls how
paths are generated. The default is `GeometricBrownianMotion()`.

`JumpDiffusion(lambda, alpha_j, sigma_j)` adds a compound Poisson jump process
(Merton 1976) to the GBM diffusion. Jumps are most significant for out-of-the-money
options and produce a volatility smile when prices are inverted through BSM.

```julia
jd = JumpDiffusion(3.0, -0.02, 0.05)   # ~3 jumps/year, negative mean jump

# Jump diffusion pricing (dynamics as third argument)
price(call, RiskNeutralMonteCarlo(252, 10_000, jd), data)

# Combined with variance reduction (method before dynamics)
av = VarianceReduction(PseudoRandom(), Antithetic())
price(call, RiskNeutralMonteCarlo(252, 10_000, av, jd), data)
```

`AssetDynamics` and `VarianceReduction` compose freely — any combination is supported
across all three Monte Carlo model types.

`StationaryBootstrap` is a third dynamics type that resamples from historical
log-returns (Politis & Romano 1994). Because the resampled paths reflect the
real-world (P) return distribution, it can only be used with `HedgedMonteCarlo`.
See [Historical Data](#historical-data) below.

#### `ControlVariateMonteCarlo` — variance reduction via a correlated control

Uses a second option with a known analytical price to reduce estimator variance. The
adjusted estimator is `V̂_cv = mean(V - β·(C - C_BSM))`.

```julia
# Optimal β (estimated from the same paths)
cv = ControlVariateMonteCarlo(1, 10_000, ControlVariate(EuropeanCall(40.0, 1.0)))
price(target, cv, data)

# Fixed β = 1
cv1 = ControlVariateMonteCarlo(1, 10_000, ControlVariate(EuropeanCall(40.0, 1.0), 1.0))
price(target, cv1, data)
```

Works for vanilla, binary, and path-dependent exotic option targets. The control variate
option must support `price(option, BlackScholes(), data)`.

## Exotic Options

### Binary Options

Pay a fixed cash amount contingent on the terminal asset price.

| Type | Fields | Payoff |
|---|---|---|
| `CashOrNothingCall` | `strike, expiry, cash` | `cash` if `S_T > K`, else `0` |
| `CashOrNothingPut`  | `strike, expiry, cash` | `cash` if `S_T < K`, else `0` |

The `cash` argument defaults to `1.0` when omitted.

```julia
data = MarketData(100.0, 0.05, 0.25, 0.0)
con  = CashOrNothingCall(100.0, 1.0)          # pays $1 if S_T > 100

price(con, BlackScholes(), data)              # analytical BSM price
price(con, RiskNeutralMonteCarlo(1, 50_000), data)   # Monte Carlo
```

### Lookback Options

Payoffs depend on the maximum or minimum asset price over the life of the contract.

| Type | Fields | Payoff |
|---|---|---|
| `FloatingStrikeLookbackCall` | `expiry` | `S_T - S_min` |
| `FloatingStrikeLookbackPut`  | `expiry` | `S_max - S_T` |
| `FloatingPriceLookbackCall`  | `strike, expiry` | `max(0, S_max - K)` |
| `FloatingPriceLookbackPut`   | `strike, expiry` | `max(0, K - S_min)` |

```julia
mc = RiskNeutralMonteCarlo(252, 50_000)

price(FloatingStrikeLookbackCall(1.0),          mc, data)
price(FloatingStrikeLookbackPut(1.0),           mc, data)
price(FloatingPriceLookbackCall(40.0, 1.0),     mc, data)
price(FloatingPriceLookbackPut(40.0, 1.0),      mc, data)
```

### Arithmetic Asian Options

Payoffs depend on the arithmetic mean of the asset price over the life of the contract.

| Type | Fields | Payoff |
|---|---|---|
| `FloatingStrikeArithmeticAsianCall` | `expiry` | `max(0, S_T - Ā)` |
| `FloatingStrikeArithmeticAsianPut`  | `expiry` | `max(0, Ā - S_T)` |
| `FloatingPriceArithmeticAsianCall`  | `strike, expiry` | `max(0, Ā - K)` |
| `FloatingPriceArithmeticAsianPut`   | `strike, expiry` | `max(0, K - Ā)` |

```julia
mc = RiskNeutralMonteCarlo(252, 50_000)

price(FloatingStrikeArithmeticAsianCall(1.0),          mc, data)
price(FloatingStrikeArithmeticAsianPut(1.0),           mc, data)
price(FloatingPriceArithmeticAsianCall(40.0, 1.0),     mc, data)
price(FloatingPriceArithmeticAsianPut(40.0, 1.0),      mc, data)
```

## Hedging Strategies

`HedgedMonteCarlo` simulates hedge costs under the real-world (P) measure. It takes
a `HedgeStrategy` that specifies the strategy and its real-world drift `mu`.

### Stop-Loss Hedge

A naive strategy that holds the underlying whenever the spot is above the strike and
holds cash otherwise. The expected hedge cost is strictly greater than the BSM price
at any step frequency — unlike delta hedging, stop-loss does not converge to BSM as
`steps → ∞` because the strategy itself is sub-optimal (it always buys high and sells
low at the strike boundary).

```julia
model = HedgedMonteCarlo(100, 50_000, StopLoss(0.10))   # mu = 10%
result = price(call, model, data)
result.price   # mean hedge cost
result.std     # standard error
```

### Delta Hedge

Continuously rebalances a BSM delta hedge under the real-world (P) measure. At each
time step the BSM delta is recomputed and the portfolio is rebalanced; cash flows are
discounted and averaged. As `steps → ∞` the mean cost converges to the BSM price with
variance that shrinks as `O(1/steps)`.

```julia
model = HedgedMonteCarlo(100, 50_000, DeltaHedge(0.10))   # mu = 10%
result = price(call, model, data)
result.price   # mean hedge cost (→ BSM as steps → ∞)
result.std     # standard error
```

Variance reduction works with both hedge strategies — pass the strategy before the
optional `VarianceReduction` method:

```julia
model = HedgedMonteCarlo(100, 50_000, StopLoss(0.10), VarianceReduction(PseudoRandom(), Antithetic()))
price(call, model, data)
```

## Historical Data

`HistoricalData` stores a time series of continuously compounded log-returns and
is the data source for `StationaryBootstrap` path simulation.

```julia
# From a price vector — compute returns explicitly
prices = [100.0, 102.3, 101.8, 103.5]
hist   = HistoricalData(log_returns(prices))

# From a CSV file of prices (one price per row)
hist = HistoricalData("SPY.csv")              # prices in column 1, header row
hist = HistoricalData("data.csv"; col=2)      # prices in second column
hist = HistoricalData("raw.csv"; header=false)
```

`log_returns(prices)` is also exported as a standalone helper for computing
`log(S_t / S_{t-1})` from any price vector.

`examples/data/SPY_close.csv` contains 5 years of daily SPY closing prices and
is used in example `07_bootstrap.jl` to demonstrate the file-based constructor
and the difference between assumed, realized, and empirical hedge costs.

### Stationary Bootstrap

`StationaryBootstrap(data, mean_block_length)` resamples blocks of historical
returns with geometrically distributed lengths, preserving the serial dependence
structure of the original series.

```julia
hist   = HistoricalData("SPY.csv")
bs     = StationaryBootstrap(hist, 20)   # ~20-day mean block length

data   = MarketData(450.0, 0.05, 0.20, 0.0)
call   = EuropeanCall(450.0, 1.0)

# Only valid with HedgedMonteCarlo (P-measure simulation)
result = price(call, HedgedMonteCarlo(252, 10_000, DeltaHedge(0.10), bs), data)
result.price   # empirical hedge cost
result.std     # standard error
```

Using `StationaryBootstrap` with `RiskNeutralMonteCarlo` raises a `MethodError` —
the restriction is enforced by the type system.

## Type Hierarchy

```
VanillaOption
├── EuropeanOption
│   ├── EuropeanCall(strike, expiry)
│   └── EuropeanPut(strike, expiry)
└── AmericanOption
    ├── AmericanCall(strike, expiry)
    └── AmericanPut(strike, expiry)

ExoticOption
├── BinaryOption
│   ├── CashOrNothingCall(strike, expiry[, cash])
│   └── CashOrNothingPut(strike, expiry[, cash])
├── LookbackOption
│   ├── FloatingStrikeLookbackCall(expiry)
│   ├── FloatingStrikeLookbackPut(expiry)
│   ├── FloatingPriceLookbackCall(strike, expiry)
│   └── FloatingPriceLookbackPut(strike, expiry)
└── ArithmeticAsianOption
    ├── FloatingStrikeArithmeticAsianCall(expiry)
    ├── FloatingStrikeArithmeticAsianPut(expiry)
    ├── FloatingPriceArithmeticAsianCall(strike, expiry)
    └── FloatingPriceArithmeticAsianPut(strike, expiry)

AssetDynamics (abstract)
├── GeometricBrownianMotion
├── JumpDiffusion(lambda, alpha_j, sigma_j)
└── StationaryBootstrap(data, mean_block_length)   # HedgedMonteCarlo only

HistoricalData(returns)                            # also: HistoricalData(filepath)

MonteCarlo (abstract)
├── RiskNeutralMonteCarlo(steps, reps[, method][, dynamics])
├── HedgedMonteCarlo(steps, reps, strategy[, method][, dynamics])
└── ControlVariateMonteCarlo(steps, reps, control[, method][, dynamics])

HedgeStrategy (abstract)
├── StopLoss(mu)
└── DeltaHedge(mu)

BetaMethod (abstract)
├── FixedBeta(beta)
└── OptimalBeta
```

## Payoff Function

`payoff(option, spot)` computes the intrinsic value of a vanilla or binary option at a
given terminal spot price:

```julia
payoff(EuropeanCall(100.0, 1.0), 110.0)       # 10.0
payoff(EuropeanPut(100.0, 1.0),   90.0)       # 10.0
payoff(CashOrNothingCall(100.0, 1.0), 110.0)  #  1.0
payoff(CashOrNothingCall(100.0, 1.0),  90.0)  #  0.0
```

`payoff(option, path)` computes the payoff of a path-dependent exotic option from a
full simulated price path:

```julia
path = [100.0, 95.0, 102.0, 98.0, 105.0]
payoff(FloatingStrikeLookbackCall(1.0), path)        # 105.0 - 95.0 = 10.0
payoff(FloatingPriceLookbackPut(100.0, 1.0), path)   # max(0, 100.0 - 95.0) = 5.0
```

## Code Style

All source files follow [BlueStyle](https://github.com/JuliaDiff/BlueStyle) with a
92-character line limit.

## Dependencies

- [Distributions.jl](https://github.com/JuliaStats/Distributions.jl)
- [DelimitedFiles](https://docs.julialang.org/en/v1/stdlib/DelimitedFiles/) (standard library)
- [LinearAlgebra](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/) (standard library)
- [Statistics](https://docs.julialang.org/en/v1/stdlib/Statistics/) (standard library)

## Author

Tyler Brough — Utah State University
Caleb Dissel — Utah State University
