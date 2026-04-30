using LinearAlgebra
using Statistics

"""
    HedgeStrategy

Abstract type for hedging strategies used with [`HedgedMonteCarlo`](@ref).

Concrete subtypes: [`StopLoss`](@ref), [`DeltaHedge`](@ref)
"""
abstract type HedgeStrategy end

"""
    HedgedMonteCarlo(steps, reps, strategy[, method])

Monte Carlo simulation model for estimating the cost of a hedging strategy under
the real-world (P) measure. Asset paths are simulated using the strategy's drift
rather than the risk-free rate.

# Fields
- `steps::Int`: Number of time steps per simulation path
- `reps::Int`: Number of simulation paths (replications)
- `method::VarianceReductionMethod`: Variance reduction strategy (default: `PseudoRandom`
  with `NoPairing`)
- `strategy::HedgeStrategy`: The hedging strategy to evaluate (e.g., [`StopLoss`](@ref))

# Examples
```julia
data = MarketData(41.0, 0.08, 0.30, 0.0)
call = EuropeanCall(40.0, 1.0)

price(call, HedgedMonteCarlo(100, 10_000, StopLoss(0.10)), data)
```

See also: [`MonteCarlo`](@ref), [`RiskNeutralMonteCarlo`](@ref), [`HedgeStrategy`](@ref),
[`StopLoss`](@ref)
"""
struct HedgedMonteCarlo{S<:HedgeStrategy} <: MonteCarlo
    steps::Int
    reps::Int
    method::VarianceReductionMethod
    strategy::S
    dynamics::AssetDynamics
end

function HedgedMonteCarlo(steps::Int, reps::Int, strategy::HedgeStrategy)
    return HedgedMonteCarlo(
        steps,
        reps,
        VarianceReduction(PseudoRandom(), NoPairing()),
        strategy,
        GeometricBrownianMotion(),
    )
end

function HedgedMonteCarlo(
    steps::Int, reps::Int, strategy::HedgeStrategy, method::VarianceReductionMethod
)
    return HedgedMonteCarlo(steps, reps, method, strategy, GeometricBrownianMotion())
end

function HedgedMonteCarlo(
    steps::Int, reps::Int, strategy::HedgeStrategy, dynamics::AssetDynamics
)
    return HedgedMonteCarlo(
        steps, reps, VarianceReduction(PseudoRandom(), NoPairing()), strategy, dynamics
    )
end

"""
    StopLoss(mu) <: HedgeStrategy

Stop-loss hedging strategy. Holds the underlying whenever the spot is at or above
the strike and holds cash otherwise. The drift `mu` is the real-world (P-measure)
expected return used when simulating asset paths.

# Fields
- `mu::Float64`: Expected drift of the underlying asset (annualized, as decimal)

# Examples
```julia
data = MarketData(41.0, 0.08, 0.30, 0.0)
call = EuropeanCall(40.0, 1.0)

price(call, HedgedMonteCarlo(100, 10_000, StopLoss(0.10)), data)
```

See also: [`HedgeStrategy`](@ref), [`HedgedMonteCarlo`](@ref)
"""
struct StopLoss <: HedgeStrategy
    mu::Float64
end

"""
    price(option::EuropeanOption, model::HedgedMonteCarlo{StopLoss}, data::MarketData)

Estimate the cost of a stop-loss hedging strategy for a European option via Monte Carlo.

Simulates asset paths under the real-world drift `model.strategy.mu` and tracks the
cash flows of a stop-loss hedge: buying the underlying when the spot crosses above the
strike and selling when it crosses below. The present value of all cash flows
(including terminal delivery) is averaged across paths.

# Arguments
- `option::EuropeanOption`: The option contract being hedged
- `model::HedgedMonteCarlo{StopLoss}`: Simulation model with stop-loss strategy
- `data::MarketData`: Market parameters (spot, rate, vol, div)

# Returns
A [`SimulationResult`](@ref) with the mean hedging cost and its standard error.

# Examples
```julia
data = MarketData(41.0, 0.08, 0.30, 0.0)
call = EuropeanCall(40.0, 1.0)

result = price(call, HedgedMonteCarlo(100, 10_000, StopLoss(0.10)), data)
result.price  # mean hedge cost
result.std    # standard error
```

See also: [`StopLoss`](@ref), [`HedgedMonteCarlo`](@ref), [`SimulationResult`](@ref)
"""
function price(option::EuropeanOption, model::HedgedMonteCarlo{StopLoss}, data::MarketData)
    (; strike, expiry) = option
    (; steps, reps, method, strategy) = model
    (; mu) = strategy
    (; spot, rate, vol, div) = data

    dt = expiry / steps
    dfs = exp.(-rate * collect(0:steps) * dt)
    cost = zeros(reps)
    paths = asset_paths(model, spot, mu, vol, expiry)

    for k in 1:reps
        cash_flows = zeros(steps + 1)

        if paths[k, 1] >= strike
            covered = 1
            cash_flows[1] = -paths[k, 1]
        else
            covered = 0
        end

        for t in 2:(steps + 1)
            if (covered == 1) && (paths[k, t] < strike)
                covered = 0
                cash_flows[t] = paths[k, t]
            elseif (covered == 0) && (paths[k, t] > strike)
                covered = 1
                cash_flows[t] = -paths[k, t]
            end
        end

        if paths[k, end] >= strike
            cash_flows[end] += strike
        end

        cost[k] = -dot(dfs, cash_flows)
    end

    return SimulationResult(mean(cost), std(cost) / sqrt(reps))
end

"""
    DeltaHedge(mu) <: HedgeStrategy

Delta hedging strategy that simulates the cost of continuously rebalancing a
BSM delta hedge under the real-world (P) measure.

At each time step the BSM delta is computed for the current spot and remaining
time, and the cash flows from rebalancing the hedge portfolio are tracked. The
expected discounted cost converges to the BSM price as `steps → ∞`, analogous to
[`StopLoss`](@ref) but with a smoother position that rebalances continuously.

# Fields
- `mu::Float64`: Real-world drift of the underlying asset (annualized, as decimal)

# Examples
```julia
data = MarketData(50.0, 0.05, 0.40, 0.0)
call = EuropeanCall(52.0, 5/12)

price(call, HedgedMonteCarlo(100, 10_000, DeltaHedge(0.10)), data)
```

See also: [`HedgeStrategy`](@ref), [`HedgedMonteCarlo`](@ref), [`StopLoss`](@ref)
"""
struct DeltaHedge <: HedgeStrategy
    mu::Float64
end

"""
    price(option::EuropeanCall, model::HedgedMonteCarlo{DeltaHedge}, data::MarketData)

Estimate the cost of a delta hedging strategy for a European call via Monte Carlo.

Simulates asset paths under the real-world drift `model.strategy.mu` and tracks
the cash flows of a continuously rebalanced delta hedge. At each step the BSM
delta is computed and the portfolio is rebalanced; at expiry the option delivery
is settled against the final hedge position.

# Arguments
- `option::EuropeanCall`: The call option being hedged
- `model::HedgedMonteCarlo{DeltaHedge}`: Simulation model with delta hedge strategy
- `data::MarketData`: Market parameters (spot, rate, vol, div)

# Returns
A [`SimulationResult`](@ref) with the mean hedging cost and its standard error.

# Examples
```julia
data = MarketData(50.0, 0.05, 0.40, 0.0)
call = EuropeanCall(52.0, 5/12)

result = price(call, HedgedMonteCarlo(100, 10_000, DeltaHedge(0.10)), data)
result.price   # mean hedge cost (converges to BSM price as steps → ∞)
result.std     # standard error
```

See also: [`DeltaHedge`](@ref), [`HedgedMonteCarlo`](@ref), [`SimulationResult`](@ref)
"""
function price(option::EuropeanCall, model::HedgedMonteCarlo{DeltaHedge}, data::MarketData)
    (; strike, expiry) = option
    (; steps, reps, method, strategy) = model
    (; mu) = strategy
    (; spot, rate, vol, div) = data

    dt = expiry / steps
    dfs = exp.(-rate * collect(0:steps) * dt)
    cost = zeros(reps)

    paths = asset_paths(model, spot, mu, vol, expiry)

    for k in 1:reps
        path = paths[k, :]
        position = 0.0
        cash_flows = zeros(steps + 1)

        for j in 1:steps
            tau = expiry - (j - 1) * dt
            mkt = MarketData(path[j], data.rate, data.vol, data.div)
            delta_t = delta(EuropeanCall(option.strike, tau), BlackScholes(), mkt)
            cash_flows[j] = (position - delta_t) * path[j]
            position = delta_t
        end

        # Settle delivery at expiry
        if path[end] > strike
            cash_flows[end] = strike - (1.0 - position) * path[end]
        else
            cash_flows[end] = position * path[end]
        end

        cost[k] = -dot(dfs, cash_flows)
    end

    return SimulationResult(mean(cost), std(cost) / sqrt(reps))
end

# Internal: apply the stationary bootstrap resampling scheme to `indices` in-place.
# On each step, the block continues (indices[i] = indices[i-1] + 1) with
# probability 1-p, or a new block starts at a fresh random index with
# probability p. Indices wrap around at n_hist to stay within the history.
function _stationary_bootstrap_sample!(
    indices::Vector{Int}, u::Vector{Float64}, p::Float64, n_hist::Int
)
    for i in 2:length(indices)
        if u[i] > p
            indices[i] = indices[i - 1] + 1
            if indices[i] > n_hist
                indices[i] = 1
            end
        end
    end
    return indices
end

"""
    asset_paths(method, dynamics::StationaryBootstrap, model::HedgedMonteCarlo, ...)

Generate asset price paths by resampling from historical log-returns using the
stationary bootstrap.

Restricted to [`HedgedMonteCarlo`](@ref) because the resampled paths reflect the
real-world (P) return distribution. The `method` argument is accepted for dispatch
consistency but has no effect — variance reduction does not apply to bootstrap
resampling.

See also: [`StationaryBootstrap`](@ref), [`HistoricalData`](@ref)
"""
function asset_paths(
    method::VarianceReduction,
    bs::StationaryBootstrap,
    model::HedgedMonteCarlo,
    spot,
    rate,
    vol,
    expiry,
)
    (; steps, reps) = model
    returns = bs.data.returns
    n_hist = length(returns)
    p = 1.0 / bs.mean_block_length

    paths = zeros(reps, steps + 1)
    paths[:, 1] .= spot

    for k in 1:reps
        indices = rand(1:n_hist, steps)
        u = rand(steps)
        _stationary_bootstrap_sample!(indices, u, p, n_hist)
        for j in 1:steps
            paths[k, j + 1] = paths[k, j] * exp(returns[indices[j]])
        end
    end

    return paths
end
