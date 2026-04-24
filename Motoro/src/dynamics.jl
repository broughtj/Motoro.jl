"""
    AssetDynamics

Abstract type for asset price simulation models.

Concrete subtypes determine how asset paths are generated inside all
[`MonteCarlo`](@ref) simulation models.

Concrete subtypes: [`GeometricBrownianMotion`](@ref), [`JumpDiffusion`](@ref)
"""
abstract type AssetDynamics end


"""
    GeometricBrownianMotion <: AssetDynamics

Geometric Brownian Motion (GBM) dynamics. The asset price follows:

    dS = μ S dt + σ S dW

where μ is the drift (risk-free rate under Q, real-world return under P) and σ
is the volatility. This is the default dynamics for all Monte Carlo model types.

See also: [`JumpDiffusion`](@ref), [`RiskNeutralMonteCarlo`](@ref)
"""
struct GeometricBrownianMotion <: AssetDynamics end


"""
    JumpDiffusion(lambda, alpha_j, sigma_j) <: AssetDynamics

Merton (1976) jump diffusion dynamics. Extends geometric Brownian motion with a
compound Poisson jump process:

    dS/S = (μ - λk) dt + σ dW + (e^J - 1) dN

where N is a Poisson process with intensity λ, J ~ N(alpha_j, sigma_j²) is the
log-jump size, and k = exp(alpha_j) - 1 is the mean fractional jump. The drift
is compensated by λk so that the process remains a martingale under Q.

# Fields
- `lambda::Float64`: Jump intensity (expected number of jumps per year)
- `alpha_j::Float64`: Mean log-jump size (negative for downward-biased jumps)
- `sigma_j::Float64`: Standard deviation of log-jump size

# Examples
```julia
jd = JumpDiffusion(3.0, -0.02, 0.05)   # ~3 jumps/year, small negative mean jump

data = MarketData(100.0, 0.05, 0.20, 0.0)
call = EuropeanCall(100.0, 1.0)
price(call, RiskNeutralMonteCarlo(252, 10_000, jd), data)
```

See also: [`GeometricBrownianMotion`](@ref), [`RiskNeutralMonteCarlo`](@ref)
"""
struct JumpDiffusion <: AssetDynamics
    lambda::Float64
    alpha_j::Float64
    sigma_j::Float64
end


"""
    StationaryBootstrap(data, mean_block_length) <: AssetDynamics

Stationary bootstrap dynamics (Politis & Romano 1994). Resamples blocks of
historical log-returns with geometrically distributed block lengths to generate
artificial price paths that preserve the serial dependence structure of the
historical data.

Because paths are drawn from the historical (real-world) return distribution,
this dynamics type can only be used with [`HedgedMonteCarlo`](@ref), which
simulates under the P measure. Using it with `RiskNeutralMonteCarlo` will raise
a method error. [`VarianceReduction`](@ref) settings have no effect — the
bootstrap has its own resampling structure.

# Fields
- `data::HistoricalData`: Historical return series to resample from
- `mean_block_length::Int`: Average block length (controls autocorrelation preservation)

# Examples
```julia
hist  = HistoricalData("SPY.csv")
bs    = StationaryBootstrap(hist, 20)   # ~20-day mean block length
call  = EuropeanCall(450.0, 1.0)
data  = MarketData(450.0, 0.05, 0.20, 0.0)

price(call, HedgedMonteCarlo(252, 10_000, DeltaHedge(0.10), bs), data)
```

See also: [`HistoricalData`](@ref), [`HedgedMonteCarlo`](@ref)
"""
struct StationaryBootstrap <: AssetDynamics
    data::HistoricalData
    mean_block_length::Int
end
