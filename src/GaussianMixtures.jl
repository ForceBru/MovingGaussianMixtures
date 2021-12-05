mutable struct GaussianMixture{T<:AbstractFloat}
    K::Int

    pi::Vector{T}
    mu::Vector{T}
    sigma::Vector{T}

    G::Matrix{T}

    loglikelihood_history::Vector{T}

    function GaussianMixture(n_components::Integer, T::Type{<:AbstractFloat}=Float64)
        @assert n_components > 0

        K = n_components
        pi = Vector{T}(undef, K)
        mu = similar(pi)
        sigma = similar(pi)
        G = Matrix{T}(undef, K, 1)

        new{T}(K, pi, mu, sigma, G, T[])
    end
end

function step_E!(gmm::GaussianMixture, data::AV{<:Real}, reg::Settings.AbstractRegularization)
    step_E!(gmm.G, gmm.pi, gmm.mu, gmm.sigma, data, reg)
end

function step_M!(gmm::GaussianMixture, data::AV{<:Real}, reg::Settings.AbstractRegularization)
    step_M!(gmm.G, gmm.pi, gmm.mu, gmm.sigma, data, reg)
end

function init!(gmm::GaussianMixture)::Nothing
    gmm.pi .= gmm.mu .= gmm.sigma .= 0

    nothing
end

function init!(
    gmm::GaussianMixture{T},
    data::AV{<:Real},
    init_type::Settings.InitRandomPosteriors,
    regularization::Settings.AbstractRegularization
)::Nothing where T<:AbstractFloat
    init!(gmm)

    g = rand(Dirichlet(init_type.a, gmm.K))

    @inbounds for n in size(gmm.G, 2)
        gmm.G[:, n] .= g
    end

    step_M!(gmm, data, regularization)
end

function should_stop(gmm:GaussianMixture, data::AbstractVector, stopping::Settings.StoppingLogLikelihood)::Bool
    length(gmm.loglikelihood_history) < 2 && return false

    abs(gmm.loglikelihood_history[end] - gmm.loglikelihood_history[end-1]) < stopping.tol
end

function fit!(
    gmm::GaussianMixture{T}, data::AbstractVector{<:Real};
    init_type::Settings.AbstractInitialization=Settings.InitRandomPosteriors(200),
    stopping::Settings.AbstractStoppingCriterion=Settings.StoppingLogLikelihood(1e-6),
    regularization::Settings.Regularization=Settings.NoRegularization()
) where T<:AbstractFloat
    N = length(data)

    if size(gmm.G, 2) != N
        gmm.G = zeros(T, gmm.K, N)
    end

    init!(gmm, init_type, data)
    push!(gmm.loglikelihood_history, log_likelihood(gmm, data, regularization))

    while !should_stop(gmm, stopping)
        step_E!(gmm, data, regularization)
        step_M!(gmm, data, regularization)

        push!(gmm.loglikelihood_history, log_likelihood(gmm, data, regularization))
    end

    gmm
end
