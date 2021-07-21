export KMeans, inverse_transform

import StatsBase
using Statistics: mean, quantile

using LoopVectorization

"""
The state of the k-means algorithm.
"""
mutable struct KMeans{T <: Real, U <: Unsigned}
	# Number of clusters
	K::U
	# Length of input vector
	N::UInt

	converged::Bool
	warm_start::Bool
	_first_call::Bool
	n_iter::UInt

	# Estimated centers
	μ::Vector{T}
	μ_old::Vector{T}

    # For each element of data,
    # the index of the cluster it belongs to
	labels::Vector{U}

	mask::BitVector

    """
    Initialize the k-means algorithm to find `K` clusters
    in vectors of `N` elements.
    """
	function KMeans(
		K::Integer, N::Integer, ::Type{T}=Float64, ::Type{U}=UInt8;
		warm_start::Bool=false
	) where { T <: Real, U <: Unsigned }
		@assert K > 0
		@assert N > 0

		μ = zeros(T, K)

		new{T, U}(
			K, N,
			false, warm_start, true, 0,
			μ, μ,
			zeros(U, N), zeros(Bool, N)
		)
	end
end

function Base.copy(km::KMeans{T, U})::KMeans{T, U} where {T, U}
	ret = KMeans(km.K, km.N, T, U; warm_start=km.warm_start)

	ret.converged = km.converged
	ret._first_call = km._first_call
	ret.n_iter = km.n_iter

	ret.μ .= km.μ
	ret.μ_old .= km.μ_old

	ret.labels .= km.labels
	ret.mask .= km.mask

	ret
end

l2_norm(x, y) = maximum(abs.(x .- y))

"""
```
function StatsBase.fit!(
	km::KMeans{T, U}, data::AbstractVector{T};
	maxiter::Integer=100, tol=1e-6,
	init=:quantile, metric::Function=l2_norm
)::KMeans{T, U} where { T <: Real, U <: Unsigned }
```

Find `km.K` clusters in vector `data` of length `km.N` using the k-means algorithm.

- `init` - initialization for cluster centers
    - `:quantile` - `k`th center will be initialized with `k/km.K`th quantile of `data`
    - `:random` - each center is a random element of `data`
"""
function StatsBase.fit!(
	km::KMeans{T, U}, data::AbstractVector{T};
	maxiter::Integer=100, tol=1e-6,
	init=:quantile, metric::Function=l2_norm
)::KMeans{T, U} where { T <: Real, U <: Unsigned }
	N = length(data)
	(N ≠ km.N) &&
		throw(ArgumentError("KMeans was set up for use with data of length $(km.N) (got $N)"))
	(tol ≤ 0) &&
		throw(ArgumentError("Tolerance `tol` must be strictly greater than zero (got $tol)"))
	(maxiter ≤ 0) &&
		throw(ArgumentError("The maximum number of iterations `maxiter` must be strictly positive (got $maxiter)"))
	(init ∉ (:quantile, :random)) &&
		throw(ArgumentError("`init` must be one of :quantile, :random (got $init)"))

	# Initialize everything
	km.converged = false
	km.n_iter = 0

	if km._first_call || !km.warm_start
		# Init centers
		km.μ .= if init == :quantile
			quantile(data, range(zero(T), one(T), length=km.K))
		elseif init == :random
			rand(data, km.K)
		else
			@assert false "BUG while initializing centers"
		end
	end

	# Main loop
	for i ∈ 1:maxiter
		km.n_iter += 1

		# Update labels
		@inbounds for (i, x) ∈ enumerate(data)
			km.labels[i] = argmin(@turbo abs.(km.μ .- x))
		end

		# Update centers
		@inbounds for k ∈ 1:km.K
			@turbo km.mask .= km.labels .== k

			km.μ[k] = if !any(km.mask)
				# The cluster is empty!
				rand(data)
			else
				mean(data[km.mask])
			end
		end

		# Check convergence
		if metric(km.μ, km.μ_old) < tol
			km.converged = true
			break
		end

		km.μ_old .= km.μ
	end

	# Sort centers to ensure identifiability
	sort!(km.μ)
	sort!(km.μ_old)

	km._first_call = false

	km
end

function inverse_transform(scaler::StandardScaler, km::KMeans)
	ret = copy(km)
	
	ret.centers_prev .= inverse_transform(scaler, ret.centers_prev)
	ret.centers .= inverse_transform(scaler, ret.centers)
	
	ret
end
