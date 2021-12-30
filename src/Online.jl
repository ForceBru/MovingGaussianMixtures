mutable struct Online{T<:Real} <: AbstractMixtureUpdate{T}
    "Fitted Gaussian mixture model"
    mix::GaussianMixture{T}

    """
    Leaky integrator parameters.

    One parameter `α ∈ [0, 1]` for each mixture component.
    """
    αs::Vector{<:Real}

    # Hidden state
    A::Vector{T}
    B::Vector{T}
    C::Vector{T}

    function Online(mix::GaussianMixture{T}, α::AbstractVector{<:Real}) where T<:Real
        (length(α) == mix.K) || throw(ArgumentError(
            "Must have the same number of integrator params α (got $(length(α))) as mixture components (got $(mix.K))"
        ))
        all(0 .< α .< 1) || throw(ArgumentError("All integrator params α must be in range [0, 1]"))

        # Matrix with ONE column (for one observation)
        mix.G = [mix.G[:, end];;]

        A = copy(mix.p)
        B = A .* mix.mu
        C = @. A * (mix.var + mix.mu^2)

        new{T}(mix, copy(α), A, B, C)
    end
end

"""
$(TYPEDSIGNATURES)

Create an online mixture with equal leaky integrator parameters `α`
"""
Online(mix::GaussianMixture{<:Real}, α::Real) =
    Online(mix, fill(α, mix.K))

n_components(online::Online) = online.mix.K
get_ELBO(online::Online) = online.mix.history_ELBO[end]
get_params(online::Online) = (p=online.mix.p, mu=online.mix.mu, var=online.mix.var)
has_converged(::Online) = true # convergence not applicable

function initialize!(online::Online, stream::AV{<:Real}; init_strategy::Settings.AbstractInitStrategy, kwargs...)
    fit!(online.mix, stream; init_strategy, kwargs...)
    online.mix.G = [online.mix.G[:, end];;]

    p, mu, var = get_params(online.mix)
    online.A = copy(p)
    online.B = online.A .* mu
    online.C = @. online.A * (var + mu^2)

    nothing
end

function update!(online::Online, x::Real; regularization::Settings.MaybeRegularization=nothing, kwargs...)
    mix = online.mix

    @assert size(mix.G) == (mix.K, 1)
    #step_E!(online.mix, [x], regularization)
    #g = @view online.mix.G[:, begin]

    g = @. mix.p * normal_pdf(x, mix.mu, mix.var)
    g ./= sum(g)
    @assert length(g) == mix.K

    @. begin
        online.A = online.αs * online.A + (1 - online.αs) * g
        online.B = online.αs * online.B + (1 - online.αs) * g * x
        online.C = online.αs * online.C + (1 - online.αs) * g * x^2

        mix.p = online.A
        mix.mu = online.B / online.A
        mix.var = online.C / online.A - mix.mu^2
    end

    mix.history_ELBO[end] = ELBO(mix, [x], regularization)

    all(>(0), mix.var) || throw(ZeroVarianceException(mix.var))

    online
end
