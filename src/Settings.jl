module Settings

abstract type AbstractInitStrategy end

"""
    InitRandomPosterior(a::T) where T<:Real

Initialize posteriors from Dirichlet(a)
"""
struct InitRandomPosterior{T<:Real} <: AbstractInitStrategy
    a::T

    function InitRandomPosterior(a::T) where T<:Real
        @assert a > 0

        new{T}(a)
    end
end


abstract type AbstractStopping end

struct StoppingELBO{T<:Real}
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

struct RegPosteriorSimple{T<:Real} <: AbstractRegPosterior
    eps::T

    function RegPosteriorSimple(eps::T) where T<:Real
        @assert eps >= 0

        new{T}(eps)
    end
end

end
