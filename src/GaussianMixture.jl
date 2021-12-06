mutable struct GaussianMixture{T<:AbstractFloat}
    K::Integer
    N::Integer

    p::Vector{T}
    mu::Vector{T}
    var::Vector{T}

    G::Matrix{T}

    n_iter::UInt
    history_ELBO::Vector{T}

    function GaussianMixture(n_components::Integer, T::{<:AbstractFloat})
        (n_components > 0) || throw(ArgumentError(
            "Number of components must be positive, got $n_components"
        ))

        K, N = n_components, 1
        p = zeros(T, K)

        new{T}(
            n_components, N,
            p, copy(p), copy(p),
            zeros(T, K, N),
            0x00, T[]
        )
    end
end

const GM = GaussianMixture{T} where T

has_zeros(x::AV{<:Real})::Bool = any(≈(0), x)

step_M!(gmm::GM, data::AV{<:Real}, reg::MaybeRegularization) =
    step_M!(gmm.G, gmm.p, gmm.mu, gmm.var, data, reg)
step_E!(gmm::GM, data::AV{<:Real}, reg::MaybeRegularization) =
    step_E!(gmm.G, gmm.p, gmm.mu, gmm.var, data, reg)
ELBO_1(gmm::GM, data::AV{<:Real}, reg::MaybeRegularization) =
    ELBO_1(gmm.G, gmm.p, gmm.mu, gmm.var, data, reg)

# ===== Initialization =====
"""
$(TYPEDSIGNATURES)

Initialize Gaussian mixture using
given initialization `strategy`
and maybe regularization `reg`
"""
function initialize!(
    gmm::GM, data::AV{<:Real},
    strategy::InitRandomPosterior, reg::MaybeRegularization
)::Nothing
    gmm.G .= rand(Dirichlet(gmm.K, strategy.a), gmm.N)
    step_M!(gmm, data, reg)

    !has_zeros(gmm.var) || throw(ZeroVarianceException(gmm.var))

    nothing
end

# ===== Stopping =====
function should_stop(gmm::GM, criterion::StoppingELBO)::Bool
    (length(gmm.history_ELBO) < 2) && return false

    abs(gmm.history_ELBO[end] - gmm.history_ELBO[end-1]) < criterion.tol
end

"""
$(TYPEDSIGNATURES)

Fit Gaussian mixture model `gmm` to `data`.
"""
function fit!(
    gmm::GM{T}, data::AV{<:Real};
    init_strategy::AbstractInitStrategy=InitRandomPosterior(200),
    stopping_criterion::AbstractStopping=StoppingELBO(1e-10),
    regularization::MaybeRegularization=nothing
) where T<:AbstractFloat
    if gmm.N != length(data)
        gmm.N = length(data)
        gmm.G = zeros(T, gmm.K, gmm.N)
    end

    gmm.n_iter = 0x00
    gmm.history_ELBO = T[]
    initialize!(gmm, data, init_strategy, regularization)
    !has_zeros(gmm.var) || throw(ZeroVarianceException(gmm.var))

    push!(gmm.history_ELBO, ELBO_1(gmm, data, regularization))
    while !should_stop(gmm, stopping_criterion)
        step_E!(gmm, data, regularization)
        step_M!(gmm, data, regularization)
        !has_zeros(gmm.var) || throw(ZeroVarianceException(gmm.var))

        push!(gmm.history_ELBO, ELBO_1(gmm, data, regularization))
        gmm.n_iter += 0x01
    end

    gmm
end
