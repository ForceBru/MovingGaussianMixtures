module Settings

abstract type AbstractInitialization end

struct InitRandomPosteriors{T<:Real} <: AbstractInitialization
    # Parameter of Dirichlet(a, a, a, ...)
    a::T

    function InitRandomPosteriors(a::T) where T<:Real
        @assert a > 0

        new{T}(a)
    end
end

struct InitGivenPosteriors{T<:Real}
    G::AbstractMatrix{T}
end

# ===== Stopping criteria =====

abstract type AbstractStoppingCriterion end

struct StoppingLogLikelihood{T<:Real} <: AbstractStoppingCriterion
    tol::T

    function StoppingLogLikelihood(tol::T) where T<:Real
        @assert tol > 0

        new{T}(tol)
    end
end

# ===== Regularization types =====

abstract type AbstractRegularization end

struct NoRegularization <: AbstractRegularization end

struct RegPosteriorSimple{T<:Real} <: AbstractRegularization
    eps::T

    function RegPosteriorSimple(eps::T) where T <: Real
        @assert eps >= 0

        new{T}(eps)
    end
end

end
