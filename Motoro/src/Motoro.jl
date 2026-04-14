module Motoro

include("data.jl")
include("options.jl")
include("models.jl")

export VanillaOption, EuropeanOption, AmericanOption
export EuropeanCall, EuropeanPut, AmericanCall, AmericanPut
export payoff

export VarianceReductionMethod, VarianceReduction
export DrawMethod, PseudoRandom, Stratified
export PairingMethod, NoPairing, Antithetic
export generate_draws, apply_pairing
export Binomial, BlackScholes, MonteCarlo, asset_paths, price, delta

export MarketData

end # module Motoro
