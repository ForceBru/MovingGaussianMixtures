export GaussianMixture, fit!, distribution, predict, predict_proba

using Statistics

using LoopVectorization
using StatsBase: params
using Distributions: UnivariateGMM, Categorical, ncomponents, probs

"""
The state of the gaussian mixture.
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
	σ::Vector{T}
	
	mask::BitVector
	
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
			zeros(T, K), zeros(T, K), zeros(T, K), # p,μ,σ
			mask
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
	p::AbstractVector{T}, μ::AbstractVector{T}, σ::AbstractVector{T}
) where T <: Real
	N = length(data)
	K = length(p)
	@assert size(G) == (K, N)
	@assert K == length(μ) == length(σ)

	# ln_G = ln(p/σ * ϕ((x - μ) / σ))
	#   = ( ln(p) - ln(σ) - ln(2π)/2 ) - ((x - μ) / σ)^2 / 2
	@tturbo @. G = (
		log(p) - log(σ) - log(2π)/2
		- ((data' - μ) / σ)^2 / 2
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

	update_G!(G, data, probs(cat), μ, σ)
end

function update_G!(gm::GaussianMixture{T}, data::AbstractVector{T}) where T <: Real
	# ln_G = ln(p/σ * ϕ((x - μ) / σ))
	#   = ( ln(p) - ln(σ) - ln(2π)/2 ) - ((x - μ) / σ)^2 / 2
	@tturbo @. gm.G = (
		log(gm.p) - log(gm.σ) - log(2π)/2
		- ((data' - gm.μ) / gm.σ)^2 / 2
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
function fit!(
    gm::GaussianMixture{T}, data::AbstractVector{T};
    tol=1e-6, eps=1e-10, maxiter::Integer=100
) where T <: Real
```

Fit a Gaussian mixture model to `data`.
"""
function fit!(
    gm::GaussianMixture{T}, data::AbstractVector{T};
    tol=1e-6, eps=1e-10, maxiter::Integer=100
) where T <: Real
	@assert tol > 0
	@assert eps > 0
	
	@assert maxiter > 0
	N = length(data)
	@assert N == gm.N "data must be of length $(gm.N), got $N"
	
	gm.converged = false
	gm.n_iter = 0
	
	# Initialize mixture parameters
	if gm.first_call || !gm.warm_start
		gm.G_prev .= zero(T)
		km = KMeans(gm.K, gm.N)
		fit!(km, data)
		
		gm.μ .= km.μ
		@inbounds for k ∈ 1:gm.K
			@turbo gm.mask .= km.labels .== k
			gm.p[k] = sum(gm.mask) / gm.N
			
			gm.σ[k] = if !any(gm.mask)
				1/eps
			else
				std(data[gm.mask], corrected=false)
			end
			
			if gm.σ[k] ≈ zero(T)
				gm.σ[k] = 1/eps
			end
		end
	end
	
	# @assert !any(isnan.(gm.σ)) "Got NaN: $(gm.σ)"
	
	gm.first_call = false
	
	for i ∈ 1:maxiter
		# @assert !any(gm.σ .≈ zero(T)) "Got zeros: $(gm.σ) $(gm.p)"
		# @assert !any(isnan.(gm.σ)) "[$i] Got NaN: $(gm.σ) $(gm.p)"
		
		update_G!(gm, data)
		
		# @assert !any(isnan.(gm.G))
		
		# Update weights `p`
		mean!(gm.p, gm.G)
		
		# Update means `μ`
		mean!(gm.μ, @turbo gm.G .* data')
		@turbo gm.μ ./= clamp.(gm.p, eps, one(T))
		
		# Update standard deviations `σ`
		mean!(gm.σ, @turbo gm.G .* (data' .- gm.μ).^2)
		@turbo gm.σ ./= clamp.(gm.p, eps, one(T))
		@turbo gm.σ .= sqrt.(gm.σ)
		
		# Make sure `σ` doesn't contain zeros
		for k ∈ 1:gm.K
			if gm.σ[k] ≈ zero(T)
				gm.σ[k] = 1 / eps
				gm.μ[k] = minimum(data) * rand()
			end
		end
		
		# Check for convergence
		@tturbo @. gm.G_tmp = abs(gm.G - gm.G_prev)
		if maximum(gm.G_tmp) < tol
			gm.converged = true
			break
		end
		
		@tturbo gm.G_prev .= gm.G
			
		gm.n_iter += 1
	end
	
	# Sort the parameters
	# to ensure identifiability
	sort_idx = sortperm(gm.μ)
	gm.p .= gm.p[sort_idx]
	gm.μ .= gm.μ[sort_idx]
	gm.σ .= gm.σ[sort_idx]
	
	gm
end

distribution(gm::GaussianMixture) = UnivariateGMM(gm.μ, gm.σ, Categorical(gm.p))

"""
```
function predict_proba(gmm::UnivariateGMM, data::AbstractVector{T})::Matrix{T} where T <: Real
```

Return (K x N) matrix, where each _column_ is the PMF of the latent variable `z`.
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
    function predict(gmm::UnivariateGMM, data::AbstractVector)

Return most probable value of latent variable `z` for each element of `data`.
"""
function predict(gmm::UnivariateGMM, data::AbstractVector)
	G = predict_proba(gmm, data)
	N = size(G, 2)

	ret = Vector{UInt8}(undef, N)
	@inbounds for n ∈ 1:N
		ret[n] = argmax(@view G[:, N])
	end

	ret
end
