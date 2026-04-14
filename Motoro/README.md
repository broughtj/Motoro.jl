# Motoro.jl

A Julia package for teaching computational options pricing. Motoro implements three classical pricing methods — binomial trees, Black-Scholes-Merton, and Monte Carlo simulation — with a focus on clarity and pedagogical value.

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

## Pricing Methods

### Black-Scholes-Merton (analytical)

Exact closed-form solution for European options under the BSM assumptions.

```julia
bsm = BlackScholes()
price(call, bsm, data)   # ≈ 9.96
price(put, bsm, data)    # ≈ 2.89
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

Simulates asset paths via geometric Brownian motion and averages discounted payoffs.

```julia
mc = MonteCarlo(100, 10_000)   # 100 steps, 10,000 paths
price(call, mc, data)
```

#### Variance Reduction

Motoro includes two variance reduction techniques that can be combined:

| Draw method  | Pairing method | Effect |
|---|---|---|
| `PseudoRandom` | `NoPairing`  | Baseline Monte Carlo |
| `PseudoRandom` | `Antithetic` | Antithetic variates |
| `Stratified`   | `NoPairing`  | Stratified sampling |
| `Stratified`   | `Antithetic` | Both combined |

```julia
# Antithetic variates
mc_anti = MonteCarlo(100, 10_000, VarianceReduction(PseudoRandom(), Antithetic()))
price(call, mc_anti, data)

# Stratified sampling
mc_strat = MonteCarlo(100, 10_000, VarianceReduction(Stratified(), NoPairing()))
price(call, mc_strat, data)
```

## Type Hierarchy

```
VanillaOption
├── EuropeanOption
│   ├── EuropeanCall(strike, expiry)
│   └── EuropeanPut(strike, expiry)
└── AmericanOption
    ├── AmericanCall(strike, expiry)
    └── AmericanPut(strike, expiry)
```

## Payoff Function

`payoff(option, spot)` computes the intrinsic value of an option at any spot price:

```julia
payoff(EuropeanCall(100.0, 1.0), 110.0)   # 10.0
payoff(EuropeanPut(100.0, 1.0),   90.0)   # 10.0
payoff(EuropeanPut(100.0, 1.0),  110.0)   #  0.0
```

## Dependencies

- [Distributions.jl](https://github.com/JuliaStats/Distributions.jl)
- [LinearAlgebra](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/) (standard library)
- [Statistics](https://docs.julialang.org/en/v1/stdlib/Statistics/) (standard library)

## Author

Tyler Brough — Utah State University
