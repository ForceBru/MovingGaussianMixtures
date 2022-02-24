"""
Settings to control fitting of Gaussian mixture models.

- Initialization is controlled by derivatives of `Settings.AbstractInitStrategy`,
types whose names look like `Settings.Init[something]`.
- Stopping criteria are controlled by `Settings.AbstractStopping`
- Regularization (to prevent zero variance, for example) is set up by `Settings.AbstractRegularization`.

Details are available in documentation of each type.
"""
module Settings

using DocStringExtensions

"""
$(TYPEDEF)

Controls the initialization strategy of the EM algorithm.
"""
abstract type AbstractInitStrategy end

"""
$(TYPEDEF)

Initialize posteriors randomly from `Dirichlet(a)`
"""
struct InitRandomPosterior{T<:Real} <: AbstractInitStrategy
    "Parameter of the Dirichlet distribution"
    a::T

    function InitRandomPosterior(a::T) where T<:Real
        @assert a > 0

        new{T}(a)
    end
end

"""
$(TYPEDEF)

Keep the posteriors unchanged.
Initialize parameter estimates using these posteriors in the usual M-step.
"""
struct InitKeepPosterior <: AbstractInitStrategy end

"""
$(TYPEDEF)

Keep the parameter estimates unchanged.
Initialize posteriors using these parameter estimates in the usual E-step.
"""
struct InitKeepParameters <: AbstractInitStrategy end

"""
$(TYPEDEF)

Controls the ctopping criterion of the EM algorithm.
"""
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

"""
$(TYPEDEF)

Controls the regularization of the EM algorithm.
Regularization is needed mainly to prevent the algorithm
from zeroing-out components' variances.
"""
abstract type AbstractRegularization end
abstract type AbstractRegAdHoc <: AbstractRegularization end

const MaybeRegularization = Union{AbstractRegularization, Nothing}

"""
$(TYPEDEF)
Regularize posterior q(z) such that `q(z) > 0` for any `z`
by adding `eps > 0` to the numerator in posterior computation.

This is equivalent to modifying the mixture PDF `p(x)`
to be `p(x) + eps`. It won't integrate to one,
but this guarantees that posteriors will be strictly positive.
"""
struct RegPosteriorAddEps{T<:Real} <: AbstractRegAdHoc
    "Small number to add to numerator of each posterior prob."
    eps::T

    function RegPosteriorAddEps(eps::T) where T<:Real
        @assert eps ≥ 0

        new{T}(eps)
    end
end

"""
$(TYPEDEF)
Regularize posterior q(z) such that `q(z) >= eps >= 0` for any `z`.
This simple version does it by using this as `q(z)`:

```math
q(z) = [p(z|x, \\theta) + s] / [1 + s * K]
```

Here `s` is a number computed from `eps`
and `K` is the number of mixture components.

$(TYPEDFIELDS)
"""
struct RegPosteriorNonzero{T<:Real} <: AbstractRegAdHoc
    "Minimum probability in the posterior distribution q(z)"
    eps::T

    s::T

    "Number of mixture components"
    K::Integer

    function RegPosteriorNonzero(eps::T, K::Integer) where T<:Real
        @assert eps ≥ 0
        @assert K > 0

        s::T = if iszero(eps)
            eps
        else
            1/eps > 1/K || throw(InvalidMinPosteriorProbException(eps, K))
            1 / (1/eps - 1/K)
        end

        new{T}(eps, s, K)
    end
end

"""
$(TYPEDSIGNATURES)

Regularize posterior q(z) such that `q(z) >= eps >= 0` for any `z`.
"""
function RegPosteriorNonzero(eps::T) where T<:Real
    RegPosteriorNonzero(eps, 1)
end

"""
$(TYPEDEF)
Keep components' variances away from zero
by adding a small positive number to them:

```math
var[k] = var[k] + \\epsilon
```

$(TYPEDFIELDS)
"""
struct RegVarianceAddEps{T<:Real} <: AbstractRegAdHoc
    "Small value to add to components' variances"
    eps::T
    #FIXME: this results in a really high KL divergence
    # How bad is this??? VI also does this - is it "bad"?..

    function RegVarianceAddEps(eps::T) where T<:Real
        @assert eps > 0
        new{T}(eps)
    end
end

"""
$(TYPEDEF)

If the variance of a component is too close to zero,
set this component's variance to equal the maximum estimate
and the component's mean - to a randomly scaled element of the training data.
"""
struct RegVarianceReset <: AbstractRegAdHoc end

end
