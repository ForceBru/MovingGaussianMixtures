"""
$(TYPEDEF)

Gaussian mixture models fit on time-series data.

$(TYPEDFIELDS)
"""
mutable struct DynamicMixture{T<:Real, U<:AbstractMixtureUpdate{T}}
    updater::U

    """
    Components' weights.
    `n`th _column_ containts estimates for the `n`th moving window
    """
    P::Matrix{T}
    """
    Components' means.
    `n`th _column_ containts estimates for the `n`th moving window
    """
    M::Matrix{T}
    """
    Components' variances.
    `n`th _column_ containts estimates for the `n`th moving window
    """
    V::Matrix{T}

    "Convergence indicators for each window"
    converged::BitVector
    ELBO::Vector{T}

    function DynamicMixture(updater::AbstractMixtureUpdate{T}) where T<:Real
        P = zeros(T, n_components(updater), 1)
        M = copy(P)
        V = copy(P)
        converged = falses(1)
        ELBO = zeros(T, 1)

        new{T, typeof(updater)}(updater, P, M, V, converged, ELBO)
    end
end

n_components(dmm::DynamicMixture) = n_components(dmm.updater)
get_ELBO(dmm::DynamicMixture) = dmm.ELBO
get_params(dmm::DynamicMixture) = (p=dmm.P, mu=dmm.M, var=dmm.V)
has_converged(dmm::DynamicMixture) = dmm.converged

"""
$(TYPEDSIGNATURES)

Fit Gaussian mixture model on moving windows of width `win_size`
across given data `stream`. `kwargs` are passed to `update!(::AbstractMixtureUpdate, ...)`
"""
function fit!(
    dmm::DynamicMixture{T}, stream::AV{<:Real}, init_size::Integer;
    init_strategy::Settings.AbstractInitStrategy=Settings.InitKeepParameters(),
    regularization::Settings.MaybeRegularization=nothing,
    kwargs...
) where T<:Real
    @assert init_size > 0

    N = length(stream)
    K = n_components(dmm)

    if N != size(dmm.P, 2)
        dmm.P = zeros(T, K, N)
        dmm.M = copy(dmm.P)
        dmm.V = copy(dmm.P)
        dmm.ELBO = zeros(T, N)
    end

    dmm.converged = falses(N)

    initialize!(dmm.updater, @view(stream[1:init_size]); init_strategy=Settings.InitRandomPosterior(200), kwargs...)

    @showprogress 0.5 "Fitting mixtures..." for (t, x) in enumerate(stream)
        update!(dmm.updater, x; init_strategy, regularization, kwargs...)

        p, mu, var = get_params(dmm.updater)
        dmm.P[:, t] .= p
        dmm.M[:, t] .= mu
        dmm.V[:, t] .= var
        dmm.converged[t] = has_converged(dmm.updater)
        dmm.ELBO[t] = get_ELBO(dmm.updater)
    end

    dmm
end
