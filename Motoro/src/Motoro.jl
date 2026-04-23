module Motoro

include("data.jl")
include("options.jl")
include("models.jl")
include("paths.jl")

export VanillaOption, EuropeanOption, AmericanOption
export EuropeanCall, EuropeanPut, AmericanCall, AmericanPut
export payoff

export ExoticOption, LookbackOption, ArithmeticAsianOption
export GeometricAsianOption
export BinaryOption, CashOrNothingCall, CashOrNothingPut
export FloatingStrikeLookbackCall, FloatingStrikeLookbackPut
export FloatingPriceLookbackCall, FloatingPriceLookbackPut
export FloatingStrikeArithmeticAsianCall, FloatingStrikeArithmeticAsianPut
export FloatingPriceArithmeticAsianCall, FloatingPriceArithmeticAsianPut
export FixedStrikeGeometricAsianCall, FixedStrikeGeometricAsianPut
export FloatingStrikeGeometricAsianCall, FloatingStrikeGeometricAsianPut

export VarianceReductionMethod, VarianceReduction
export DrawMethod, PseudoRandom, Stratified
export PairingMethod, NoPairing, Antithetic
export generate_draws, apply_pairing
export PricingResult, AnalyticResult, SimulationResult
export Binomial, BlackScholes, asset_paths, price, delta, compare_estimators_same_paths
export geom_asian_price
export MonteCarlo, RiskNeutralMonteCarlo, HedgedMonteCarlo
export HedgeStrategy, StopLoss, DeltaHedge
export ControlVariateMonteCarlo
export AssetPaths, GeometricBrownianMotion, JumpDiffusion

export MarketData

end # module Motoro
