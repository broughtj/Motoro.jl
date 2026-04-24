module Motoro

include("data.jl")
include("results.jl")
include("options.jl")
include("exotic.jl")
include("analytical.jl")
include("historical.jl")
include("dynamics.jl")
include("montecarlo.jl")
include("hedging.jl")
include("control_variate.jl")

# Market data and pricing results
export MarketData
export PricingResult, AnalyticResult, SimulationResult

# Vanilla options
export VanillaOption, EuropeanOption, AmericanOption
export EuropeanCall, EuropeanPut, AmericanCall, AmericanPut
export payoff

# Exotic options
export ExoticOption, BinaryOption, LookbackOption, ArithmeticAsianOption
export CashOrNothingCall, CashOrNothingPut
export FloatingStrikeLookbackCall, FloatingStrikeLookbackPut
export FloatingPriceLookbackCall, FloatingPriceLookbackPut
export FloatingStrikeArithmeticAsianCall, FloatingStrikeArithmeticAsianPut
export FloatingPriceArithmeticAsianCall, FloatingPriceArithmeticAsianPut

# Pricing models and primary API
export price, delta
export Binomial, BlackScholes
export MonteCarlo, RiskNeutralMonteCarlo
export VarianceReductionMethod, VarianceReduction
export DrawMethod, PseudoRandom, Stratified
export PairingMethod, NoPairing, Antithetic
export HistoricalData, log_returns
export AssetDynamics, GeometricBrownianMotion, JumpDiffusion, StationaryBootstrap
export asset_paths

# Hedging
export HedgeStrategy, StopLoss, DeltaHedge
export HedgedMonteCarlo

# Control variate
export BetaMethod, FixedBeta, OptimalBeta
export ControlVariate, ControlVariateMonteCarlo

end # module Motoro
