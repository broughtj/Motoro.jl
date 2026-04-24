"""
    stationary_bootstrap_sample!(indices, u, p)

Generate indices for sampling from the stationary bootstrap (in-place).

# Arguments
- `indices::Vector{Int}`: Initial draw indices in the range `1:n` (1-based).
- `u::Vector{Float64}`: Array of standard uniform draws, same length as `indices`.
- `p::Float64`: Probability that a new block starts. Equal to `1 / mean_block_length`.

# Returns
The modified `indices` vector.
"""
function stationary_bootstrap_sample!(indices::Vector{Int}, u::Vector{Float64}, p::Float64)
    n = length(indices)
    for i in 2:n
        if u[i] > p
            indices[i] = indices[i-1] + 1
            if indices[i] > n
                indices[i] = 1
            end
        end
    end
    return indices
end
