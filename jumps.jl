using Distributions
using Random
using Plots

function simulate_jump_diffusion(; S0=100.0, N=3650, α=0.08, σ=0.30,
                                   αⱼ=-0.02, σⱼ=0.05, λ=3.0, q=0.0,
                                   seed=nothing)
    rng = isnothing(seed) ? Random.default_rng() : MersenneTwister(seed)

    h   = 1.0 / 365.0
    k   = exp(αⱼ) - 1.0

    nudt  = (α - q - 0.5 * σ^2) * h
    nudtJ = (α - q - λ * k - 0.5 * σ^2) * h
    sigsdt = σ * sqrt(h)

    S  = Vector{Float64}(undef, N)
    Sj = Vector{Float64}(undef, N)
    S[1] = Sj[1] = S0

    poisson = Poisson(λ * h)

    for t in 2:N
        Z = randn(rng)
        m = rand(rng, poisson)
        W = sum(randn(rng) for _ in 1:m; init=0.0)

        S[t]  = S[t-1]  * exp(nudt  + sigsdt * Z)
        Sj[t] = Sj[t-1] * exp(nudtJ + sigsdt * Z) *
                           exp(m * (αⱼ - 0.5 * σⱼ^2) + σⱼ * W)
    end

    return S, Sj
end

S, Sj = simulate_jump_diffusion(seed=42)

days = 1:length(S)
years = days ./ 365.0

p = plot(years, S;
    label="GBM",
    color=:steelblue,
    linewidth=1.5,
    linealpha=0.85,
    xlabel="Time (years)",
    ylabel="Asset Price",
    title="Geometric Brownian Motion vs. Jump Diffusion",
    legend=:topleft,
    grid=true,
    gridalpha=0.3,
    size=(900, 500),
)

plot!(p, years, Sj;
    label="Jump Diffusion",
    color=:crimson,
    linewidth=1.5,
    linealpha=0.85,
)

hline!(p, [S[1]]; label="S₀", color=:black, linestyle=:dash, linewidth=1)

savefig(p, "jumps.png")
display(p)
