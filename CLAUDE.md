# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

The Julia package lives in `Motoro/`. The root directory contains only a minimal `Manifest.toml` for top-level script use. All substantive work happens inside `Motoro/`.

```
Motoro/
  src/
    data.jl       # MarketData struct
    options.jl    # All option types and payoff methods
    models.jl     # All pricing models, result types, and price() methods
    Motoro.jl     # Module entry point and all exports
  Project.toml
```

Instructor-only scripts (`hedge_comparison.jl`, `convergence.jl`, `exam_q3.jl`) sit in `Motoro/` but are not part of the package.

## Common commands

All commands run from `Motoro/` with the package environment active.

```julia
# Activate and load
using Pkg; Pkg.activate(".")
using Motoro

# Quick sanity check after changes
data = MarketData(100.0, 0.05, 0.2, 0.0)
call = EuropeanCall(100.0, 1.0)
price(call, BlackScholes(), data)
price(call, RiskNeutralMonteCarlo(100, 10_000), data)

# Add a dependency
Pkg.add("PackageName")   # never edit Project.toml directly
```

There is no test suite. Verification is done by loading the package and checking prices against BSM values or known analytical results.

## Branch structure

- `main` — student-facing. Contains vanilla options, exotic options, `RiskNeutralMonteCarlo`, `StopLoss`, and BSM/Binomial pricing.
- `instructor` — extends `main` with `DeltaHedge`, `ControlVariateMonteCarlo`, and teaching scripts. Not merged to main intentionally.

## Architecture

### Type hierarchies

**Options** (`options.jl`):
```
VanillaOption
├── EuropeanOption → EuropeanCall, EuropeanPut
└── AmericanOption → AmericanCall, AmericanPut

ExoticOption
├── BinaryOption      → CashOrNothingCall, CashOrNothingPut
├── LookbackOption    → FloatingStrike{Call,Put}, FloatingPrice{Call,Put}
└── ArithmeticAsianOption → FloatingStrike{Call,Put}, FloatingPrice{Call,Put}
```

**Models** (`models.jl`):
```
MonteCarlo (abstract)
├── RiskNeutralMonteCarlo(steps, reps[, method])
├── HedgedMonteCarlo{S<:HedgeStrategy}(steps, reps, method, strategy)
└── ControlVariateMonteCarlo{CV}(steps, reps, method, control)    [instructor]

HedgeStrategy → StopLoss(mu), DeltaHedge(mu)                     [instructor]
BetaMethod    → FixedBeta(β), OptimalBeta                         [instructor]
ControlVariate{O, B<:BetaMethod}                                  [instructor]

PricingResult → AnalyticResult(price), SimulationResult(price, std)
```

Analytical models (`BlackScholes`, `Binomial`) are plain structs, not subtypes of `MonteCarlo`.

### Dispatch pattern

The primary API is `price(option, model, data)`. Julia multiple dispatch routes to the correct implementation based on all three argument types. All `price` methods return a `PricingResult` subtype, never a raw number.

### Payoff signatures — critical distinction

- **Vanilla and Binary options**: `payoff(option, spot::Number)` — takes a scalar terminal price
- **Lookback and Asian options**: `payoff(option, path::AbstractVector)` — takes a full simulated path

`BinaryOption <: ExoticOption`, but binary payoffs use the terminal-spot signature. The internal `_target_payoffs` helper in `ControlVariateMonteCarlo`'s `price` method handles this dispatch correctly — do not collapse it.

### VarianceReduction

`VarianceReduction{D<:DrawMethod, P<:PairingMethod}` composes independently along two axes: draw method (`PseudoRandom`, `Stratified`) and pairing (`NoPairing`, `Antithetic`). It is the `method` field on all `MonteCarlo` subtypes. Default is `VarianceReduction(PseudoRandom(), NoPairing())`.

### Adding new option types

1. Add struct and `payoff` to `options.jl` under the appropriate abstract parent
2. Add `price` method(s) to `models.jl`
3. Export from `Motoro.jl`

If the new option is path-dependent, use the `payoff(option, path::AbstractVector)` signature and make it a subtype of an appropriate `ExoticOption` child. If it is terminal-price-only (like `BinaryOption`), add a `_target_payoffs` dispatch for it alongside the existing `BinaryOption` and `VanillaOption` dispatches in `models.jl`.

### Parametric structs and constructors

All option structs are parametric `{T<:AbstractFloat}`. When defining convenience constructors (e.g. a 2-argument version that fills in a default), always call the inner constructor explicitly (`StructName{T}(...)`) rather than the outer one to avoid method overwrite warnings during precompilation. Also add `Base.broadcastable(x::MyType) = Ref(x)` so `payoff.(option, spots)` works correctly.
