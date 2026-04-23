struct GeometricBrownianMotion <: AssetPaths
    method::VarianceReduction
    model::MonteCarlo
    spot::Float64
    rate::Float64
    vol::Float64
    expiry::Float64
end

struct JumpDiffusion <: AssetPaths
    method::VarianceReduction
    model::MonteCarlo
    spot::Float64
    rate::Float64
    vol::Float64
    expiry::Float64
    lambda::Float64
    alphaJ::Float64
    sigmaJ::Float64
end


"""
    asset_paths(method::VarianceReduction, model::MonteCarlo, spot, rate, vol, expiry)

Generate simulated asset price paths using geometric Brownian motion.

Works with any [`MonteCarlo`](@ref) subtype (`RiskNeutralMonteCarlo` or `HedgedMonteCarlo`).

# Arguments
- `method::VarianceReduction`: Variance reduction strategy controlling draw generation and pairing
- `model::MonteCarlo`: Any Monte Carlo model (provides `steps` and `reps`)
- `spot`: Initial asset price
- `rate`: Drift rate (risk-free rate for risk-neutral pricing; real-world drift for hedging)
- `vol`: Volatility (annualized)
- `expiry`: Time to expiration in years

# Returns
Matrix of size `(reps, steps+1)`. Each row is one simulated price path;
column 1 is the initial spot price and column `steps+1` is the terminal price.

# Examples
```julia
data  = MarketData(41.0, 0.08, 0.30, 0.0)
model = RiskNeutralMonteCarlo(100, 1_000)
paths = asset_paths(model.method, model, data.spot, data.rate, data.vol, 1.0)
size(paths)  # (1000, 101)
```

See also: [`RiskNeutralMonteCarlo`](@ref), [`HedgedMonteCarlo`](@ref), [`price`](@ref)
"""
function asset_paths(paths::GeometricBrownianMotion)
    (; method, model, spot, rate, vol, expiry) = paths
    (; steps, reps) = model

    dt = expiry / steps
    nudt = (rate - 0.5 * vol^2) * dt
    sidt = vol * sqrt(dt)
    n = reps ÷ (method.pairing isa Antithetic ? 2 : 1)

    out = zeros(reps, steps + 1)
    out[:, 1] .= spot

    @inbounds for j in 2:steps+1
        z = generate_draws(method.draw, n)
        z = apply_pairing(method.pairing, z)
        out[:, j] = out[:, j-1] .* exp.(nudt .+ sidt .* z)
    end

    return out
end


# E[J - 1], where J = exp(Y), Y ~ N(alphaJ, sigmaJ^2).
_jump_compensator(alphaJ::Float64, sigmaJ::Float64) = exp(alphaJ + 0.5 * sigmaJ^2) - 1.0

function asset_paths(paths::JumpDiffusion)
    (; method, model, spot, rate, vol, expiry, lambda, alphaJ, sigmaJ) = paths
    (; steps, reps) = model

    dt = expiry / steps
    sqrt_dt = sqrt(dt)
    κ = _jump_compensator(alphaJ, sigmaJ)
    drift_dt = (rate - lambda * κ - 0.5 * vol^2) * dt

    out = zeros(reps, steps + 1)
    out[:, 1] .= spot
    log_s = fill(log(spot), reps)
    jump_count_dist = Poisson(lambda * dt)
    jump_log_dist = Normal(alphaJ, sigmaJ)
    if (method.pairing isa Antithetic) && isodd(reps)
        throw(ArgumentError("Antithetic pairing requires an even number of reps; got reps=$reps"))
    end
    n = reps ÷ (method.pairing isa Antithetic ? 2 : 1)

    @inbounds for j in 2:steps+1
        z = generate_draws(method.draw, n)
        z = apply_pairing(method.pairing, z)
        n_jumps = rand(jump_count_dist, reps)
        jump_terms = zeros(reps)

        for i in eachindex(jump_terms)
            n_jump_i = n_jumps[i]
            if n_jump_i > 0
                jump_terms[i] = sum(rand(jump_log_dist, n_jump_i))
            end
        end

        @. log_s = log_s + drift_dt + vol * sqrt_dt * z + jump_terms
        out[:, j] = exp.(log_s)
    end

    return out
end

function asset_paths(method::VarianceReduction, model::MonteCarlo, spot, rate, vol, expiry)
    process = GeometricBrownianMotion(method, model, spot, rate, vol, expiry)
    return asset_paths(process)
end