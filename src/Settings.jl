module Settings

using DocStringExtensions
abstract type AbstractInitStrategy end

"""
$(TYPEDEF)

Initialize posteriors randomly from Dirichlet(a)
"""
struct InitRandomPosterior{T<:Real} <: AbstractInitStrategy
    "Parameter of the Dirichlet distribution"
    a::T

    function InitRandomPosterior(a::T) where T<:Real
        @assert a > 0

        new{T}(a)
    end
end

struct InitKeepPosterior <: AbstractInitStrategy end


abstract type AbstractStopping end

"""
$(TYPEDEF)
$(TYPEDFIELDS)

Consider EM converged when the absolute difference in ELBO
becomes less than `tol`
"""
struct StoppingELBO{T<:Real} <: AbstractStopping
    "Minimum required change in ELBO"
    tol::T

    function StoppingELBO(tol::T) where T<:Real
        @assert tol > 0

        new{T}(tol)
    end
end

abstract type AbstractRegularization end
abstract type AbstractRegPosterior <: AbstractRegularization end
abstract type AbstractRegPrior <: AbstractRegularization end

const MaybeRegularization = Union{AbstractRegularization, Nothing}

"""
$(TYPEDEF)

Regularize posterior q(z) such that `q(z) >= eps` for any `z`.
"""
struct RegPosteriorSimple{T<:Real} <: AbstractRegPosterior
    "Minimum probability in the posterior distribution q(z)"
    eps::T

    function RegPosteriorSimple(eps::T) where T<:Real
        @assert eps >= 0

        new{T}(eps)
    end
end

end
