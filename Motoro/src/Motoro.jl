module Motoro

include("data.jl")
include("options.jl")
include("models.jl")

export VanillaOption, EuropeanOption, AmericanOption
export EuropeanCall, EuropeanPut, AmericanCall, AmericanPut
export payoff

export ExoticOption, LookbackOption, ArithmeticAsianOption
export BinaryOption, CashOrNothingCall, CashOrNothingPut
export FloatingStrikeLookbackCall, FloatingStrikeLookbackPut
export FloatingPriceLookbackCall, FloatingPriceLookbackPut
export FloatingStrikeArithmeticAsianCall, FloatingStrikeArithmeticAsianPut
export FloatingPriceArithmeticAsianCall, FloatingPriceArithmeticAsianPut

export VarianceReductionMethod, VarianceReduction
export DrawMethod, PseudoRandom, Stratified
export PairingMethod, NoPairing, Antithetic
export generate_draws, apply_pairing
export PricingResult, AnalyticResult, SimulationResult
export Binomial, BlackScholes, asset_paths, price, delta
export MonteCarlo, RiskNeutralMonteCarlo, HedgedMonteCarlo
export HedgeStrategy, StopLoss

export MarketData

end # module Motoro
