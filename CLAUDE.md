# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

The Julia package lives in `Motoro/`. The root directory contains only a minimal `Manifest.toml` for top-level script use. All substantive work happens inside `Motoro/`.

```
Motoro/
  src/
    Motoro.jl          # Module entry point and all exports
    data.jl            # MarketData struct
    results.jl         # PricingResult, AnalyticResult, SimulationResult
    options.jl         # Vanilla option types and payoff methods
    exotic.jl          # Exotic option types and payoff methods (Binary, Lookback, Asian)
    analytical.jl      # Binomial, BlackScholes + price/delta methods
    montecarlo.jl      # VarianceReduction infrastructure, RiskNeutralMonteCarlo, asset_paths
    hedging.jl         # HedgeStrategy, HedgedMonteCarlo, StopLoss, DeltaHedge
    control_variate.jl # BetaMethod, ControlVariate, ControlVariateMonteCarlo
  Project.toml
```

Include order in `Motoro.jl` is load-order dependent: `results` before pricing methods, `options` and `exotic` before analytical models, `analytical` before `hedging` (DeltaHedge calls `delta()`).

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

- `main` — student-facing. Vanilla options, exotic options, `RiskNeutralMonteCarlo`, `StopLoss`, BSM/Binomial pricing. Still uses the older two-file layout (`options.jl` + `models.jl`).
- `instructor` — extends `main` with `DeltaHedge`, `ControlVariateMonteCarlo`, and teaching scripts (`hedge_comparison.jl`, `convergence.jl`, `exam_q3.jl`). Uses the refactored split-file layout described above. Not merged to main intentionally.

## Architecture

### Type hierarchies

**Options** (`options.jl`, `exotic.jl`):
```
VanillaOption
├── EuropeanOption → EuropeanCall, EuropeanPut
└── AmericanOption → AmericanCall, AmericanPut

ExoticOption
├── BinaryOption      → CashOrNothingCall, CashOrNothingPut
├── LookbackOption    → FloatingStrike{Call,Put}, FloatingPrice{Call,Put}
└── ArithmeticAsianOption → FloatingStrike{Call,Put}, FloatingPrice{Call,Put}
```

**Models** (`analytical.jl`, `montecarlo.jl`, `hedging.jl`, `control_variate.jl`):
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

`BinaryOption <: ExoticOption`, but binary payoffs use the terminal-spot signature. The internal `_collect_payoffs` helper in `ControlVariateMonteCarlo`'s `price` method handles this dispatch correctly — do not collapse it.

### VarianceReduction

`VarianceReduction{D<:DrawMethod, P<:PairingMethod}` composes independently along two axes: draw method (`PseudoRandom`, `Stratified`) and pairing (`NoPairing`, `Antithetic`). It is the `method` field on all `MonteCarlo` subtypes. Default is `VarianceReduction(PseudoRandom(), NoPairing())`.

### Adding new option types

1. Add struct and `payoff` to `options.jl` (vanilla) or `exotic.jl` (exotic) under the appropriate abstract parent
2. Add `price` method(s) to the relevant model file (`analytical.jl`, `montecarlo.jl`, etc.)
3. Export from `Motoro.jl`

If the new option is path-dependent, use the `payoff(option, path::AbstractVector)` signature and make it a subtype of an appropriate `ExoticOption` child. If it is terminal-price-only (like `BinaryOption`), add a `_collect_payoffs` dispatch for it alongside the existing `BinaryOption` and `VanillaOption` dispatches in `control_variate.jl`.

## Code style

All source files follow [BlueStyle](https://github.com/JuliaDiff/BlueStyle). Key rules to enforce on every change:

- **Line length**: 92-character limit. Split long lines at operators or with an intermediate variable.
- **No alignment padding**: do not add spaces to vertically align `=`, `::`, or argument lists across lines.
- **Short-form functions**: use `f(x) = ...` for single-expression methods; `function ... end` only when the body spans multiple lines.
- **Operators**: use `&&` / `||`, never `&` / `|` for boolean logic.
- **Arithmetic spacing**: spaces around binary operators (`j - 1`, `steps + 1`), not `j-1`.
- **Range literals**: parenthesise non-trivial endpoints — `2:(steps + 1)`, not `2:steps+1`.
- **Variable names**: ASCII only — no Greek letters (`beta` not `β`, `tau` not `τ`, `delta_t` not `Δ`).
- **Example scripts**: wrap every `for` loop in a `let` block to avoid Julia soft-scope warnings when the file is `include`d from the REPL.

### Parametric structs and constructors

All option structs are parametric `{T<:AbstractFloat}`. When defining convenience constructors (e.g. a 2-argument version that fills in a default), always call the inner constructor explicitly (`StructName{T}(...)`) rather than the outer one to avoid method overwrite warnings during precompilation. Also add `Base.broadcastable(x::MyType) = Ref(x)` so `payoff.(option, spots)` works correctly.
