"""
    MarketData(spot, rate, vol, div)

Market parameters for option pricing.

# Fields
- `spot::T`: Current spot price of the underlying asset
- `rate::T`: Risk-free interest rate (annualized, as decimal, e.g., 0.05 for 5%)
- `vol::T`: Volatility (annualized standard deviation, as decimal, e.g., 0.2 for 20%)
- `div::T`: Continuous dividend yield (annualized, as decimal; use 0.0 for
  non-dividend paying assets)

# Examples
```julia
# Non-dividend paying stock
data = MarketData(100.0, 0.05, 0.2, 0.0)

# Stock with 2% dividend yield
data = MarketData(100.0, 0.05, 0.2, 0.02)
```
"""
struct MarketData{T<:AbstractFloat}
    spot::T
    rate::T
    vol::T
    div::T
end
