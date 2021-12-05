@enum ConvergenceStatus C_SUCCESS C_RUNNING C_ZEROSTD

mutable struct GaussianMixture{T<:AbstractFloat}
    K::Int
    n_iter::UInt
    converged::Bool
    convergence_status::ConvergenceStatus

    p::Vector{T}
    mu::Vector{T}
    sigma::Vector{T}

    G::Matrix{T}

    loglikelihood_history::Vector{T}

    function GaussianMixture(n_components::Integer, T::Type{<:AbstractFloat}=Float64)
        @assert n_components > 0

        K = n_components
        p = Vector{T}(undef, K)
        mu = similar(p)
        sigma = similar(p)
        G = Matrix{T}(undef, K, 1)

        new{T}(
            K, 0, false, C_RUNNING,
            p, mu, sigma, G,
            T[]
        )
    end
end

function Base.show(io::IO, ::MIME"text/plain", gmm::GaussianMixture)
    print(io, typeof(gmm), '\n')
    print(io, '\t', "pi=", round.(gmm.p, digits=4), '\n')
    print(io, '\t', "mu=", round.(gmm.mu, digits=4), '\n')
    print(io, '\t', "sigma=", round.(gmm.sigma, digits=4), '\n')
end

function get_distribution(gmm::GaussianMixture)
    UnivariateGMM(
        gmm.mu, gmm.sigma, Categorical(gmm.p)
    )
end

function step_E!(gmm::GaussianMixture, data::AV{<:Real}, reg::Settings.AbstractRegularization)
    step_E!(gmm.G, gmm.p, gmm.mu, gmm.sigma, data, reg)
end

function step_M!(gmm::GaussianMixture, data::AV{<:Real}, reg::Settings.AbstractRegularization)
    step_M!(gmm.G, gmm.p, gmm.mu, gmm.sigma, data, reg)
end

function init!(gmm::GaussianMixture)::Nothing
    gmm.n_iter = 0
    gmm.converged = true
    gmm.convergence_status = C_RUNNING
    gmm.p .= gmm.mu .= gmm.sigma .= 0

    nothing
end

function init!(
    gmm::GaussianMixture{T},
    data::AV{<:Real},
    init_type::Settings.InitRandomPosteriors,
    regularization::Settings.AbstractRegularization
)::Nothing where T<:AbstractFloat
    init!(gmm)

    g = rand(Dirichlet(gmm.K, init_type.a))
    @show g

    @inbounds for n in 1:size(gmm.G, 2)
        gmm.G[:, n] .= g
    end

    @assert !any(isnan, gmm.G)

    step_M!(gmm, data, regularization)
end

function should_stop(gmm::GaussianMixture, stopping::Settings.StoppingLogLikelihood)::Bool
    length(gmm.loglikelihood_history) < 2 && return false

    abs(gmm.loglikelihood_history[end] - gmm.loglikelihood_history[end-1]) < stopping.tol
end

function fit!(
    gmm::GaussianMixture{T}, data::AbstractVector{<:Real};
    init_type::Settings.AbstractInitialization=Settings.InitRandomPosteriors(100),
    stopping::Settings.AbstractStoppingCriterion=Settings.StoppingLogLikelihood(1e-10),
    regularization::Settings.AbstractRegularization=Settings.NoRegularization(),
    min_iter::Integer=100
) where T<:AbstractFloat
    if regularization isa Settings.RegPosteriorSimple
        @assert 1/regularization.eps - 1/gmm.K > 0
    end

    N = length(data)

    if size(gmm.G, 2) != N
        gmm.G = zeros(T, gmm.K, N)
    end

    init!(gmm, data, init_type, regularization)
    @show gmm.p gmm.mu gmm.sigma
    push!(gmm.loglikelihood_history, log_likelihood(gmm, data, regularization))

    while gmm.n_iter < min_iter || !should_stop(gmm, stopping)
        step_E!(gmm, data, regularization)
        step_M!(gmm, data, regularization)

        if any(s->(s ≈ zero(T) || s < zero(T)), gmm.sigma)
            gmm.converged = false
            gmm.convergence_status = C_ZEROSTD
            break
        end

        push!(gmm.loglikelihood_history, log_likelihood(gmm, data, regularization))
        gmm.n_iter += 1
    end

    if gmm.converged
        gmm.convergence_status = C_SUCCESS
    end

    gmm
end
