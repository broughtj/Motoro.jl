"""
    PricingResult

Abstract type for option pricing results.

See also: [`AnalyticResult`](@ref), [`SimulationResult`](@ref)
"""
abstract type PricingResult end

"""
    AnalyticResult(price)

Result from an analytic pricing model (e.g., [`BlackScholes`](@ref), [`Binomial`](@ref)).

# Fields
- `price::Float64`: Option price
"""
struct AnalyticResult <: PricingResult
    price::Float64
end

"""
    SimulationResult(price, std)

Result from a simulation-based pricing model (e.g., [`RiskNeutralMonteCarlo`](@ref),
[`HedgedMonteCarlo`](@ref)).

# Fields
- `price::Float64`: Estimated option price (discounted mean payoff)
- `std::Float64`: Standard error of the price estimate (standard deviation of
  payoffs divided by √reps)
"""
struct SimulationResult <: PricingResult
    price::Float64
    std::Float64
end
