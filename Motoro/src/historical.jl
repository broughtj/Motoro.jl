using DelimitedFiles

"""
    HistoricalData

Stores a time series of continuously compounded log-returns computed from
historical asset prices.

# Fields
- `returns::Vector{Float64}`: Log-returns `log(S_t / S_{t-1})`

# Examples
```julia
# From a price vector — compute returns first
hist = HistoricalData(log_returns([100.0, 102.0, 101.5, 103.0]))

# From a CSV file of prices
hist = HistoricalData("AAPL.csv")
```

See also: [`log_returns`](@ref), [`StationaryBootstrap`](@ref)
"""
struct HistoricalData
    returns::Vector{Float64}
end

"""
    log_returns(prices::AbstractVector{<:Real}) -> Vector{Float64}

Compute continuously compounded log-returns from a price series.

Returns `log(S_t / S_{t-1})` for each consecutive pair, producing a vector of
length `length(prices) - 1`.

# Examples
```julia
r = log_returns([100.0, 102.0, 101.5, 103.0])
hist = HistoricalData(r)
```

See also: [`HistoricalData`](@ref)
"""
log_returns(prices::AbstractVector{<:Real}) = diff(log.(float.(prices)))

"""
    HistoricalData(filepath::String; col=1, header=true)

Load historical prices from a delimited (CSV) file and construct a `HistoricalData`.

Prices are read from the specified column and converted to log-returns via
[`log_returns`](@ref). Expects one price observation per row.

# Arguments
- `filepath::String`: Path to the CSV file
- `col::Int`: Column index containing prices (default: `1`)
- `header::Bool`: Whether the file has a header row to skip (default: `true`)

# Examples
```julia
hist = HistoricalData("AAPL.csv")              # prices in column 1, with header
hist = HistoricalData("prices.csv"; col=2)     # prices in second column
hist = HistoricalData("raw.csv"; header=false) # no header row
```

See also: [`log_returns`](@ref)
"""
function HistoricalData(filepath::String; col::Int=1, header::Bool=true)
    raw = readdlm(filepath, ',', Any; header=header)
    data = header ? first(raw) : raw
    return HistoricalData(log_returns(Float64.(data[:, col])))
end
