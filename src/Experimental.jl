module Experimental

export RiskMetrics, OnlineEM
export online!

import Statistics: var
using ..DocStringExtensions, ..Distributions

# ===== RiskMetrics =====

"""
$(TYPEDEF)

RiskMetrics exponential moving average estimate of variance.

$(TYPEDFIELDS)
"""
mutable struct RiskMetrics{T<:Real}
    "Estimated _variance_ (not standard deviation or volatility)"
	var::T
end

function update!(rm::RiskMetrics, x::Real, γ::Real)
	@assert 0 < γ < 1
	rm.var = (1 - γ) * rm.var + γ * x^2
	rm
end

initialize!(rm::RiskMetrics, subseries::AbstractVector{<:Real}) = (rm.var = var(subseries))

"""
$(TYPEDSIGNATURES)

Estimate variance of `series` with RiskMetrics.

- `γ ∈ [0, 1]` - smoothing parameter.
  Greater `γ` - greater emphasis on most recent observation.
- `init` - whether to initialize model from first `n_init` elements of data

__Returns__: vector of estimated variance for each observation
"""
function online!(rm::RiskMetrics, series::AbstractVector{<:Real}, γ::Real; init::Bool=true, n_init::Integer=10)
    @assert n_init > 0
    @assert length(series) > n_init

    init && initialize!(rm, variances[1:n_init])

    variances = similar(series)
	for (i, x) ∈ enumerate(series)
		update!(rm, x, γ)
		variances[i] = rm.var
	end
	variances
end

# ===== OnlineEM =====

"""
$(TYPEDEF)

Online EM algorithm inspired by Cappé & Moulines.
Basically an exponential moving average over sufficient statistics
of the complete-data distribution that estimates Gaussian mixtures on-the-fly.

$(TYPEDFIELDS)
"""
mutable struct OnlineEM{T<:Real}
    "Number of components"
    K::Integer
    
    "Components' weights"
    p::Vector{T}
    "Components' means"
    mu::Vector{T}
    "Components' variances"
    var::Vector{T}

    "First set of sufficient statistics"
    A::Vector{T}
    "Second set of sufficient statistics"
    B::Vector{T}
    "Third set of sufficient statistics"
    C::Vector{T}
end

function OnlineEM{T}(K::Integer; α::Real=50) where T<:Real
    @assert K > 0

    p = rand(Dirichlet(K, α))
    mu = zeros(T, K)
    var = ones(T, K)

    OnlineEM{T}(
        K,
        p, mu, var,
        copy(p), zeros(T, K), zeros(T, K)
    )
end

OnlineEM(K::Integer) = OnlineEM{Float64}(K)

@inline normal(x::Real, mu::Real, var::Real) = exp(-(x-mu)^2 / (2var)) / sqrt(2π * var)

@inline function calc_g(x::Real, p::AbstractVector{<:Real}, mu::AbstractVector{<:Real}, var::AbstractVector{<:Real})
	unnorm = @. p * normal(x, mu, var)
	unnorm ./ sum(unnorm)
end

function update!(onl::OnlineEM, x::Real, γ::Real, M_step::Bool=true)
	# 1. Estimate posteriors
	gs = calc_g(x, onl.p, onl.mu, onl.var)

	# 2. Update sufficient statistics
	@. begin
		onl.A = (1 - γ) * onl.A + γ * gs
		onl.B = (1 - γ) * onl.B + γ * gs * x
		onl.C = (1 - γ) * onl.C + γ * gs * x^2
	end

	if M_step
		# 3. Update estimates
		@. begin
			onl.p = onl.A
			onl.mu = onl.B / onl.A
			onl.var = onl.C / onl.A - onl.mu^2
		end
	end
	onl
end

function initialize!(onl::OnlineEM, xs::AbstractVector{<:Real}, γ::Real)
	for (i, x) ∈ enumerate(xs)
        # Don't update mixture params!
		update!(onl, x, γ/i^0.1, false)
	end
end

"""
$(TYPEDSIGNATURES)

Estimate mixture parameters of `series` with online EM.

- `γ ∈ [0, 1]` - smoothing parameter.
  Greater `γ` - greater emphasis on most recent observation.
- `init` - whether to initialize model from first `n_init` elements of data

__Returns__: matrices of estimated parameters for each observation
"""
function online!(online::OnlineEM, series::AbstractVector{<:Real}, γ::Real; init::Bool=true, n_init::Integer=100)
    @assert n_init > 0
    @assert length(series) > n_init

    init && initialize!(online, series[1:n_init], γ)

    P = zeros(online.K, length(series))
	M, V = similar(P), similar(P)
    for (i, x) ∈ enumerate(series)
		update!(online, x, γ)
		P[:, i] .= online.p
		M[:, i] .= online.mu
		V[:, i] .= online.var
	end
	(; P, M, V)
end

end