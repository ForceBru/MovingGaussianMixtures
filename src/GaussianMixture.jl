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
    "Values of ELBO for each EM iteration"
    history_ELBO::Vector{T}
    "Convergence indicator"
    converged::Bool
    has_valid_params::Bool

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
            0x00, T[],
            false, false
        )
    end
end

const GM = GaussianMixture{T} where T

function Base.show(io::IO, ::MIME"text/plain", gmm::GM; digits::Integer=3)
    state = gmm.has_valid_params ? "valid" : "invalid"

    print(io, typeof(gmm), " with $(gmm.K) components ($state; converged: $(gmm.converged)):\n")
    print(io, "\tp  = ", round.(gmm.p; digits), '\n')
    print(io, "\tmu = ", round.(gmm.mu; digits), '\n')
    print(io, "\tvar= ", round.(gmm.var; digits))
end

n_components(gmm::GaussianMixture) = gmm.K
get_ELBO(gmm::GaussianMixture) = gmm.ELBO[end]
get_params(gmm::GaussianMixture) = (p=gmm.p, mu=gmm.mu, var=gmm.var)
has_converged(gmm::GaussianMixture) = gmm.converged


"""
$(TYPEDSIGNATURES)

Retrieve a `Distributions.jl` distribution of the mixture,
namely - `UnivariateGMM`.
"""
distribution(gmm::GM) = UnivariateGMM(
    gmm.mu, sqrt.(gmm.var), Categorical(gmm.p)
)

# Can't use `is_almost_zero(x::T)::Bool where T<:Real = x ≈ zero(T)`
# See https://github.com/JuliaLang/julia/issues/21847
is_almost_zero(x::Real)::Bool = x ≈ zero(x)
has_zeros(x::AV{<:Real})::Bool = any(is_almost_zero, x)

step_M!(gmm::GM, data::AV{<:Real}, reg::Settings.MaybeRegularization) =
    step_M!(gmm.G, gmm.p, gmm.mu, gmm.var, data, reg)
step_E!(gmm::GM, data::AV{<:Real}, reg::Settings.MaybeRegularization) =
    step_E!(gmm.G, gmm.p, gmm.mu, gmm.var, data, reg)
ELBO(gmm::GM, data::AV{<:Real}, reg::Settings.MaybeRegularization) =
    ELBO(gmm.G, gmm.p, gmm.mu, gmm.var, data, reg)
log_likelihood(gmm::GM, data::AV{<:Real}, reg::Settings.MaybeRegularization) =
    log_likelihood(gmm.G, gmm.p, gmm.mu, gmm.var, data, reg)
    

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
    gmm::GaussianMixture{T}, data::AV{<:Real};
    init_strategy::Settings.AbstractInitStrategy=Settings.InitRandomPosterior(200),
    stopping_criterion::Settings.AbstractStopping=Settings.StoppingELBO(1e-10, 20),
    regularization::Settings.MaybeRegularization=nothing,
    max_iter::Unsigned=UInt(5_000), quiet::Bool=true
) where T<:Real
    @assert length(data) > 0
    @assert max_iter > 0
    
    if gmm.N != length(data) || gmm.N != size(gmm.G, 2)
        gmm.N = length(data)
        gmm.G = zeros(T, gmm.K, gmm.N)
    end

    @assert size(gmm.G) == (gmm.K, gmm.N)

    # Ensure that the regularization is computed
    # for the correct number of components
    if regularization isa Settings.RegPosteriorNonzero
        regularization = Settings.RegPosteriorNonzero(regularization.eps, gmm.K)
    end

    progr = Progress(
        max_iter, dt=1, desc="Fitting:",
        barglyphs=BarGlyphs("[=> ]"), showspeed=true,
        enabled=!quiet
    )

    gmm.n_iter = 0x01
    gmm.history_ELBO = Vector{T}(undef, max_iter)
    gmm.history_ELBO .= NaN
    gmm.converged = false
    gmm.has_valid_params = false

    # Set up initial parameters
    initialize!(gmm, data, init_strategy, regularization)
    !has_zeros(gmm.var) || throw(ZeroVarianceException(gmm.var))

    gmm.history_ELBO[gmm.n_iter] = ELBO(gmm, data, regularization)
    while gmm.n_iter < max_iter && !should_stop(gmm, stopping_criterion)
        step_E!(gmm, data, regularization)
        step_M!(gmm, data, regularization)
        !has_zeros(gmm.var) || throw(ZeroVarianceException(gmm.var))

        gmm.n_iter += 0x01
        gmm.history_ELBO[gmm.n_iter] = ELBO(gmm, data, regularization)
        next!(progr)
    end

    ProgressMeter.finish!(progr)

    gmm.converged = should_stop(gmm, stopping_criterion)
    gmm.has_valid_params = !has_zeros(gmm.var)
    gmm.history_ELBO = gmm.history_ELBO[.~isnan.(gmm.history_ELBO)]

    gmm
end
