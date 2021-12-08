"""
$(TYPEDEF)

Gaussian mixture model (finite mixture of normal distributions)
with `K` components and probability density function:
```math
p(x) = \\sum_{k=1}^K p[k] * N(x; mu[k], var[k])
```

Here `N(x; mu[k], var[k])` is a normal distribution
with location `mu[k]` and scale `sqrt(var[k])`.

$(TYPEDFIELDS)
"""
mutable struct GaussianMixture{T<:Real}
    "Number of components"
    K::Integer
    "Length of data used to fit the model"
    N::Integer

    "Components' weights"
    p::Vector{T}
    "Components' means"
    mu::Vector{T}
    "Components' variances"
    var::Vector{T}

    """
    Matrix of posterior distributions `q(z)` of latent variables `Z`.
    Each _column_ is a probability distribution of `Z`
    that corresponds to an element of the training data.
    For example, `G[:, 1]` is the distribution of `z[1]`, which corresponds to `training_data[1]`.
    """
    G::Matrix{T}

    "Number of EM iterations"
    n_iter::UInt
    "Values of `(ELBO - posterior entropy)` for each EM iteration"
    history_ELBO::Vector{T}

    function GaussianMixture(n_components::Integer, T::Type{<:Real}=Float64)
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

function Base.show(io::IO, ::MIME"text/plain", gmm::GM; digits::Integer=3)
    print(io, typeof(gmm), " with $(gmm.K) components:\n")
    print(io, "\tp  = ", round.(gmm.p; digits), '\n')
    print(io, "\tmu = ", round.(gmm.mu; digits), '\n')
    print(io, "\tvar= ", round.(gmm.var; digits))
end

"""
$(TYPEDSIGNATURES)

Retrieve a `Distributions.jl` distribution of the mixture,
namely - `UnivariateGMM`.
"""
distribution(gmm::GM) = UnivariateGMM(
    gmm.mu, sqrt.(gmm.var), Categorical(gmm.p)
)

is_almost_zero(x::Real)::Bool = isnan(1 / x)
has_zeros(x::AV{<:Real})::Bool = any(is_almost_zero, x)

step_M!(gmm::GM, data::AV{<:Real}, reg::Settings.MaybeRegularization) =
    step_M!(gmm.G, gmm.p, gmm.mu, gmm.var, data, reg)
step_E!(gmm::GM, data::AV{<:Real}, reg::Settings.MaybeRegularization) =
    step_E!(gmm.G, gmm.p, gmm.mu, gmm.var, data, reg)
ELBO_1(gmm::GM, data::AV{<:Real}, reg::Settings.MaybeRegularization) =
    ELBO_1(gmm.G, gmm.p, gmm.mu, gmm.var, data, reg)

# ===== Initialization =====
"""
$(TYPEDSIGNATURES)

Initialize Gaussian mixture from random posterior:
1. For each observation `x_n` the posterior `q(z_n)`
is sampled from `Dirichlet(strategy.a)`
2. The mixture parameters are computed by the usual M-step
"""
function initialize!(
    gmm::GM, data::AV{<:Real},
    strategy::Settings.InitRandomPosterior, reg::Settings.MaybeRegularization
)::Nothing
    gmm.G .= rand(Dirichlet(gmm.K, strategy.a), gmm.N)

    initialize!(gmm, data, Settings.InitKeepPosterior(), reg)

    nothing
end

"""
$(TYPEDSIGNATURES)

Initialize Gaussian mixture from the current posterior:
1. The posteriors `q(z_n)` are left unchanged
2. The mixture parameters are computed by the usual M-step
"""
function initialize!(
    gmm::GM, data::AV{<:Real},
    strategy::Settings.InitKeepPosterior, reg::Settings.MaybeRegularization
)::Nothing
    step_M!(gmm, data, reg)

    !has_zeros(gmm.var) || throw(ZeroVarianceException(gmm.var))

    nothing
end

"""
$(TYPEDSIGNATURES)

Initialize Gaussian mixture from current parameter estimates:
1. The parameters are left unchanged
2. For each observation `x_n` the posterior `q(z_n)`
is computed by the usual E-step
"""
function initialize!(
    gmm::GM, data::AV{<:Real},
    strategy::Settings.InitKeepParameters, reg::Settings.MaybeRegularization
)::Nothing
    !has_zeros(gmm.var) || throw(ZeroVarianceException(gmm.var))

    step_E!(gmm, data, reg)

    nothing
end

# ===== Fitting =====
function should_stop(gmm::GM, criterion::Settings.StoppingELBO)::Bool
    (length(gmm.history_ELBO) < criterion.n_iter + 2) && return false

    metric = abs.(
        gmm.history_ELBO[end-criterion.n_iter:end]
        .- gmm.history_ELBO[end-criterion.n_iter-1:end-1]
    ) |> maximum

    metric < criterion.tol
end

"""
$(TYPEDSIGNATURES)

Fit Gaussian mixture model `gmm` to `data`.
"""
function fit!(
    gmm::GM{T}, data::AV{<:Real};
    init_strategy::Settings.AbstractInitStrategy=Settings.InitRandomPosterior(200),
    stopping_criterion::Settings.AbstractStopping=Settings.StoppingELBO(1e-10, 20),
    regularization::Settings.MaybeRegularization=nothing
) where T<:Real
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
