export GaussianMixture, distribution, predict_proba, nconverged, converged_pct
export log_likelihood, logpdf # reexport from Distributions

using Statistics

using LoopVectorization
import StatsBase
using StatsBase: params, fit!
using Distributions: UnivariateGMM, Categorical, ncomponents, probs, logpdf

"""
A finite K-component Gaussian mixture model with density

    p(x) = ∑ₖ (p[k] / σ[k]) N( (x - μ[k]) / σ[k] )

Where:
- `k ∈ {1, …, K}` is the index of the current mixture component
- `p[k]` is the weight of that component
- `μ[k]` is its mean
- `σ[k]` is its standard deviation
- `N(⋅)` is the density of the standard normal distribution

See also: [`fit!`](@ref)
"""
mutable struct GaussianMixture{T <: Real}
    # Number of components
	K::Int8
    # Length of input vector
	N::Integer
	
	converged::Bool
	warm_start::Bool
	first_call::Bool
	n_iter::Int
	
    # Temporary variables to reduce
    # memory allocation during estimation
	tmp_m::Vector{T}
	tmp_ln_s::Vector{T}
	G_tmp::Matrix{T}

    # (K x N) matrix of probabilities
    # of the latent variable being equal
    # to `k` given the observation `x_n`:
    # `P(z_n = k | x_n)`
	G_prev::Matrix{T}
	G::Matrix{T}
	
    # Estimated mixture parameters
	p::Vector{T}
	μ::Vector{T}
	τ::Vector{T} # = 1/σ
	
	mask::BitVector

	kmeans::Union{Missing, KMeans{T}}
	
	"""
    ```
    function GaussianMixture(K::Integer, N::Integer, ::Type{T}=Float64;
        warm_start::Bool=false
    ) where T <: Real
    ```

	- `K` - number of components
	- `N` - length of vector
	"""
	function GaussianMixture(K::Integer, N::Integer, ::Type{T}=Float64;
			warm_start::Bool=false
	) where T <: Real
		@assert K > 0
		@assert N > 0
		
		mask = BitVector(undef, N)
		mask .= false
		
		new{T}(
			Int8(K), N,
			false, warm_start, true, 0,
			zeros(T, N), zeros(T, N),
			zeros(T, K, N), zeros(T, K, N), zeros(T, K, N), # G
			zeros(T, K), zeros(T, K), zeros(T, K), # p,μ,τ
			mask,
			missing # KMeans instance
		)
	end
end

"""
__INTERNAL!__

For each element of `data` calculate
probabilities for it to come
from `k`th mixture component.

Essentially does this:
```
for (n, x) ∈ enumerate(data)
	num = p/σ * ϕ((x - μ) / σ)
	G[:, n] = num / sum(num)
end
```

...but in a numerically stable way
(<https://leimao.github.io/blog/LogSumExp/>)
"""
function update_G!(
	G::AbstractMatrix{T}, data::AbstractVector,
	p::AbstractVector{T}, μ::AbstractVector{T}, τ::AbstractVector{T}
) where T <: Real
	N = length(data)
	K = length(p)
	@assert size(G) == (K, N)
	@assert K == length(μ) == length(τ)

	# ln_G = ln(p * τ * ϕ((x - μ) * τ))
	#   = ( ln(p) + ln(τ) - ln(2π)/2 ) - ((x - μ) * τ)^2 / 2
	@tturbo @. G = (
		log(p) + log(τ) - log(2π)/2
		- ((data' - μ) * τ)^2 / 2
	)
	
	# For each n ∈ 1:N calculate
	# ln_s = ln(sum(G[:, n])), so that
	# G[:, n] = exp(ln_G[:, n]) / exp(ln_s)
	m = maximum(G, dims=1)[1, :]
	
	ln_s = sum((@tturbo @. exp(G - m')), dims=1)[1, :]
	ln_s = @turbo @. m + log(ln_s)

	@tturbo @. G = exp(G) / exp(ln_s')
end

function update_G!(G::Matrix, data::AbstractVector, gmm::UnivariateGMM)
	μ, σ, cat = params(gmm)

	update_G!(G, data, probs(cat), μ, 1 ./ σ)
end

function update_G!(gm::GaussianMixture{T}, data::AbstractVector{T}) where T <: Real
	# ln_G = ln(p * τ * ϕ((x - μ) * τ))
	#   = ( ln(p) + ln(τ) - ln(2π)/2 ) - ((x - μ) * τ)^2 / 2
	@tturbo @. gm.G = (
		log(gm.p) + log(gm.τ) - log(2π)/2
		- ((data' - gm.μ) * gm.τ)^2 / 2
	)
	
	# For each n ∈ 1:N calculate
	# ln_s = ln(sum(G[:, n])), so that
	# G[:, n] = exp(ln_G[:, n]) / exp(ln_s)
	maximum!(gm.tmp_m, gm.G')
	@tturbo @. gm.G_tmp = exp(gm.G - gm.tmp_m')
	
	sum!(gm.tmp_ln_s, gm.G_tmp')
	@turbo @. gm.tmp_ln_s = gm.tmp_m + log(gm.tmp_ln_s)
	@tturbo @. gm.G = exp(gm.G) / exp(gm.tmp_ln_s')
end

"""
```
function StatsBase.fit!(
    gm::GaussianMixture{T}, data::AbstractVector{T};
	sort_by=:μ,
    tol=1e-3, eps=1e-10, maxiter::Integer=1000
) where T <: Real
```

Fit the Gaussian mixture model `gm` to `data`.

- Will sort the estimated parameters by the values of prameter `sort_by` (`:μ` or `:σ`) to ensure identifiability of the mixture
- `tol > 0` is the tolerance used in convergence checking
- `eps > 0` is a very small number used to avoid division by zero
- `maxiter > 0` is the maximum number of iterations to perform

### Convergence criteria

Currently convergence is declared when the L1 norm
between the latest and the previous distributions
of the latent variable becomes less than `tol`.
"""
function StatsBase.fit!(
	gm::GaussianMixture{T}, data::AbstractVector{T};
	sort_by::Symbol=:μ,
    tol=1e-3, eps=1e-10, maxiter::Integer=1000
) where T <: Real
	(sort_by ∉ (:μ, :σ)) && 
		throw(ArgumentError("Parameters can only be sorted by `sort_by ∈ (:μ, :σ)` (got $sort_by)"))
	(tol ≤ 0) &&
		throw(ArgumentError("Tolerance `tol` must be strictly greater than zero (got $tol)"))
	(eps ≤ 0) &&
		throw(ArgumentError("Epsilon `eps` must be a very small strictly positive number (got $eps)"))
	(maxiter ≤ 0) &&
		throw(ArgumentError("The maximum number of iterations `maxiter` must be strictly positive (got $maxiter)"))

	N = length(data)
	(N ≠ gm.N) &&
		throw(ArgumentError("Expected data of length $(gm.N) (got $N)"))
	
	gm.converged = false
	gm.n_iter = 0
	
	# Initialize mixture parameters
	if gm.first_call || !gm.warm_start
		gm.G_prev .= zero(T)

		if gm.kmeans === missing
			gm.kmeans = KMeans(gm.K, gm.N)
		end
		fit!(gm.kmeans, data)
		
		gm.μ .= gm.kmeans.μ
		@inbounds for k ∈ 1:gm.K
			@turbo gm.mask .= gm.kmeans.labels .== k
			gm.p[k] = sum(gm.mask) / gm.N
			
			gm.τ[k] = if !any(gm.mask)
				# `k`th cluster is empty
				zero(T)
			else
				the_std = std(data[gm.mask], corrected=false)

				(the_std ≈ zero(T)) ? (1 / eps) : (1 / the_std)
			end
		end
	end
	
	# @assert !any(isnan.(gm.σ)) "Got NaN: $(gm.σ)"
	
	for _ ∈ 1:maxiter
		update_G!(gm, data)
		
		# @assert !any(isnan.(gm.G))
		
		# Update weights `p`
		mean!(gm.p, gm.G)
		
		# Update means `μ`
		@tturbo @. gm.G_tmp = gm.G .* data'
		mean!(gm.μ, gm.G_tmp)
		@turbo @. gm.μ /= clamp(gm.p, eps, one(T))
		
		# Update precisions `τ`
		@tturbo @. gm.G_tmp = gm.G * (data' - gm.μ)^2
		mean!(gm.τ, gm.G_tmp)
		@turbo @. gm.τ = sqrt(gm.p / clamp(gm.τ, eps, Inf))

		gm.n_iter += 1
		
		# Check for convergence
		@tturbo @. gm.G_prev = abs(gm.G - gm.G_prev)
		if maximum(gm.G_prev) < tol
			gm.converged = true
			break
		end
		
		@tturbo gm.G_prev .= gm.G
	end
	
	# Sort the parameters
	# in INCREASING order of `sort_by`
	# to ensure identifiability
	sort_idx = if sort_by == :μ
		sortperm(gm.μ)
	elseif sort_by == :σ
		# τ = 1/σ, so sort in reverse order!
		sortperm(gm.τ, rev=true)
	else
		@assert false "BUG: sort_by=$sort_by not handled!"
	end
	gm.p .= gm.p[sort_idx]
	gm.μ .= gm.μ[sort_idx]
	gm.τ .= gm.τ[sort_idx]

	# Now the first call is done,
	# and it's safe to query results
	gm.first_call = false
	
	gm
end

_not_fit_error() = ArgumentError(
	"The mixture hasn't been estimated yet. Call `StatsBase.fit!(the_mixture, your_data)` first"
) |> throw

"""
    nconverged(gm::GaussianMixture)

Number of converged mixtures.
Either `0` (this mixture didn't converge) or `1` otherwise
"""
nconverged(gm::GaussianMixture) = Int(gm.converged)

"""
    converged_pct(gm::GaussianMixture)

Percent of converged mixtures. Either 0.0 or 100.0.
"""
converged_pct(gm::GaussianMixture) = Float64(nconverged(gm)) * 100

"""
    distribution(gm::GaussianMixture; eps=1e-10)

Get the Distributions.jl `UnivariateGMM` of this `GaussianMixture`.
"""
distribution(gm::GaussianMixture; eps=1e-10) =
	if gm.first_call
		_not_fit_error()
	else
		UnivariateGMM(
			# Copy everything! Otherwise the params will be SHARED!
			copy(gm.μ), 1 ./ clamp.(gm.τ, eps, Inf), Categorical(copy(gm.p))
		)
	end

log_likelihood(gmm::UnivariateGMM, x::Real) = logpdf(gmm, x)

"""
    log_likelihood(gmm::UnivariateGMM, x::AbstractVector)

Compute log-likelihood of the mixture
"""
log_likelihood(gmm::UnivariateGMM, x::AbstractVector{T}) where T <: Real = sum(logpdf.(gmm, x))

"""
    predict_proba(gmm::UnivariateGMM, data::AbstractVector{T})::Matrix{T} where T <: Real

Return (K x N) matrix, where each _column_ is
the probability mass function of the latent variable `z`.
- `K` - number of mixture components
- `N` - length of input `data`
"""
function predict_proba(gmm::UnivariateGMM, data::AbstractVector{T})::Matrix{T} where T <: Real
	N = length(data)
	K = ncomponents(gmm)

	G = Matrix{T}(undef, K, N)
	update_G!(G, data, gmm)

	G
end

"""
    StatsBase.predict(gmm::UnivariateGMM, data::AbstractVector)

Return most probable value of latent variable `z` for each element of `data`.
"""
function StatsBase.predict(gmm::UnivariateGMM, data::AbstractVector)
	G = predict_proba(gmm, data)
	N = size(G, 2)

	ret = Vector{UInt8}(undef, N)
	@inbounds for n ∈ 1:N
		ret[n] = argmax(@view G[:, N])
	end

	ret
end
