module MovingGaussianMixtures
export MovingGaussianMixture, params, nconverged, converged_pct

include("Utils.jl")
include("Kmeans.jl")
include("GaussianMixture.jl")
include("Plot.jl")

import StatsBase
using StatsBase: params, fit!
using Distributions: UnivariateGMM, probs
using ProgressMeter

mutable struct MovingGaussianMixture{T <: Real}
    range::StepRange
    gm::GaussianMixture{T}

    n_iter::Vector{Int}
    converged::BitVector
    distributions::Vector{UnivariateGMM}
end

function MovingGaussianMixture(K::Integer, win_size::Integer, step_size::Integer, ::Type{T}=Float64;
    warm_start::Bool=false
) where T <: Real
    @assert step_size > 0

    gm = GaussianMixture(K, win_size, T, warm_start=warm_start)

    MovingGaussianMixture(gm.N:step_size:1, gm, Int[], BitVector(), UnivariateGMM[])
end

function StatsBase.fit!(
    mgm::MovingGaussianMixture{T}, data::AbstractVector{T};
    quiet::Bool=false, gm_params...
) where T <: Real
    mgm.range = mgm.range.start:mgm.range.step:length(data)

    mgm.n_iter = Vector{eltype(mgm.n_iter)}(undef, length(mgm.range))
    mgm.converged = BitVector(undef, length(mgm.range))
    mgm.distributions = Vector{UnivariateGMM}(undef, length(mgm.range))

    progress = Progress(
        length(mgm.range),
        barglyphs=BarGlyphs("[=> ]"), showspeed=true
    )

    # CANNOT be parallelized because it modifies itself!
    @inbounds for (i, n) ∈ enumerate(mgm.range)
        window = @view data[n - mgm.gm.N + 1:n]

        # Modifies `mgm.gm`!
        fit!(mgm.gm, window; gm_params...)

        mgm.n_iter[i] = mgm.gm.n_iter
        mgm.converged[i] = mgm.gm.converged
        mgm.distributions[i] = distribution(mgm.gm)

        (!quiet) && next!(progress)
    end

    mgm
end

"Number of converged mixtures"
nconverged(mgm::MovingGaussianMixture) = sum(mgm.converged)

"Percent of converged mixtures ∈ [0, 100]"
function converged_pct(mgm::MovingGaussianMixture)
    l = length(mgm.converged)

    (l == 0) ? 0.0 : nconverged(mgm) / l * 100
end

struct MovingGaussianMixtureParams{T <: Real}
    # Range over which the mixtures were fitted
    range::StepRange

    # Number of components
    K::Int8

    # (K x N) matrices:
    # each column represents
    # parameters estimated for window n ∈ 1:N
    P::Matrix{T}
    M::Matrix{T}
    Σ::Matrix{T}

    function MovingGaussianMixtureParams(
        range::StepRange, K::Integer,
        P::AbstractMatrix{T}, M::AbstractMatrix{T}, Σ::AbstractMatrix{T}
    ) where T <: Real
        @assert K > 0
        @assert size(P) == size(M) == size(Σ) == (K, length(range))
        @assert all(sum(P, dims=1) .≈ one(T))
        @assert all(Σ .> zero(T))

        new{T}(range, K, P, M, Σ)
    end
end

function MovingGaussianMixtureParams(mgm::MovingGaussianMixture{T}) where T <: Real
    K = mgm.gm.K
    N = length(mgm.range)

    P = zeros(T, K, N)
    M = copy(P)
    Σ = copy(P)

    for (n, distr) ∈ enumerate(mgm.distributions)
        μ, σ, cat = params(distr)

        P[:, n] .= probs(cat)
        M[:, n] .= μ
        Σ[:, n] .= σ
    end

    MovingGaussianMixtureParams(mgm.range, K, P, M, Σ)
end

StatsBase.params(gm::MovingGaussianMixture) = MovingGaussianMixtureParams(gm)

include("SaveSQL.jl")

end # module
