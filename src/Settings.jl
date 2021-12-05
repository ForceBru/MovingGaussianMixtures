module Settings

abstract type Initialization end

struct InitRandomPosteriors{T<:Real}
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

abstract type StoppingCriterion end

struct StoppingLogLikelihood{T<:Real} <: StoppingCriterion
    tol::T

    function StoppingLogLikelihood(tol::T) where T<:Real
        @assert tol > 0

        new{T}(tol)
    end
end

end
