"""
    MarketData(spot, rate, vol, div)

Market parameters for option pricing.

# Fields
- `spot::Float64`: Current spot price of the underlying asset
- `rate::Float64`: Risk-free interest rate (annualized, as decimal, e.g., 0.05 for 5%)
- `vol::Float64`: Volatility (annualized standard deviation, as decimal, e.g., 0.2 for 20%)
- `div::Float64`: Dividend yield (continuous, annualized, as decimal)

# Examples
```julia
# Non-dividend paying stock
data = MarketData(100.0, 0.05, 0.2, 0.0)

# Stock with 2% dividend yield
data = MarketData(100.0, 0.05, 0.2, 0.02)
```
"""
struct MarketData
    spot::AbstractFloat
    rate::AbstractFloat
    vol::AbstractFloat
    div::AbstractFloat
end

"""
    SVMarketData(spot, rate, vol, div, alpha, vbar, xi)

Market parameters for stochastic volatility option pricing (Heston-style).

# Fields
- `spot`: Current spot price S
- `rate`: Risk-free interest rate r
- `vol`: Initial volatility σ (initial variance V₀ = vol²)
- `div`: Dividend yield δ
- `alpha`: Mean reversion speed α of the variance process
- `vbar`: Long-run mean variance V̄
- `xi`: Volatility of volatility ξ

# Examples
```julia
# S=K=100, T=1, σ=20%, r=6%, δ=3%, α=5, ξ=0.02
data = SVMarketData(100.0, 0.06, 0.20, 0.03, 5.0, 0.04, 0.02)
```
"""
struct SVMarketData
    spot::AbstractFloat
    rate::AbstractFloat
    vol::AbstractFloat
    div::AbstractFloat
    alpha::AbstractFloat
    vbar::AbstractFloat
    xi::AbstractFloat
end
