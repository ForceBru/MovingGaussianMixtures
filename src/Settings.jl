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
Consider EM converged when the absolute difference in ELBO
stays less than `tol` during the last `n_iter` iterations.

$(TYPEDFIELDS)
"""
struct StoppingELBO{T<:Real} <: AbstractStopping
    "Minimum required change in ELBO"
    tol::T
    "Stop the algorithm if ELBO hasn't changed much during the last `n_iter` iterations"
    n_iter::Int

    function StoppingELBO(tol::T, n_iter::Integer) where T<:Real
        @assert tol > 0
        @assert n_iter > 0

        new{T}(tol, n_iter)
    end
end

abstract type AbstractRegularization end
abstract type AbstractRegPosterior <: AbstractRegularization end
abstract type AbstractRegPrior <: AbstractRegularization end

const MaybeRegularization = Union{AbstractRegularization, Nothing}

"""
$(TYPEDEF)
Regularize posterior q(z) such that `q(z) >= eps` for any `z`.

$(TYPEDFIELDS)
"""
struct RegPosteriorSimple{T<:Real} <: AbstractRegPosterior
    "Minimum probability in the posterior distribution q(z)"
    eps::T

    function RegPosteriorSimple(eps::T) where T<:Real
        @assert eps >= 0

        new{T}(eps)
    end
end

"""
$(TYPEDEF)
Keep components' variances away from zero
by adding a small positive number to them.

$(TYPEDFIELDS)
"""
struct RegVarianceSimple{T<:Real} <: AbstractRegPrior
    "Small value to add to components' variances"
    eps::T

    function RegVarianceSimple(eps::T) where T<:Real
        @assert eps > 0
        new{T}(eps)
    end
end

end
