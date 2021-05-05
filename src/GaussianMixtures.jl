using Printf
using LinearAlgebra: norm
using Statistics: quantile
using StaticArrays

export log_likelihood, GaussianMixture, GaussianMixtureEstimate, kmeans, em

const EPS = 1e-10

"Density of standard normal distribution"
@inline ϕ(x::Number) = exp(-x^2 / 2) / sqrt(2π)

"Density of mixture component (unweighted)"
@inline pdf(x::AbstractVector, μ::Number, σ::Number) = @. ϕ((x - μ) / σ) / σ

"""
    log_likelihood(x, p, μ, σ)

Log likelihood to be optimized:

    ∑ₜlog(∑ᵢpᵢ * ϕ((xₜ - μᵢ) / σᵢ))
"""
@inline log_likelihood(x, p, μ, σ) = sum(log.((p ./ σ)' * ϕ.((x' .- μ) ./ σ)))

# Termination criteria
@inline metric_l1(θ₀::AbstractVector, θ₁::AbstractVector) = maximum(abs.(θ₀ - θ₁))
@inline metric_l2(θ₀::AbstractVector, θ₁::AbstractVector) = norm(θ₀ - θ₁)

"""
All parameters are stored into a single vector θ,
but sometimes individual parameters (p, μ, σ) need to be accessed.
`get_pμσ` implements this access.
"""
@inline function get_pμσ(θ::AbstractVector, k::Integer)
	p = @view θ[1:k]
	μ = @view θ[k + 1:2k]
	σ = @view θ[2k + 1:end]
	
	p, μ, σ
end

@inline function get_pμσ(θ::MVector{_3k}) where _3k
	k = Int(_3k / 3)
	p = @view θ[1:k]
	μ = @view θ[k + 1:2k]
	σ = @view θ[2k + 1:end]
	
	p, μ, σ
end

"""
Storage for all data needed for EM algorithm.

- `k` - integer that specifies the number of components
 in the mixture;
- `_3k` - `3 * k`, needed for `StaticArrays`
"""
mutable struct GaussianMixture{k, T <: Real, _3k}
    # Parameters to be estimated
	θ::MVector{_3k, T}
	θ_old::MVector{_3k, T}
	θ_tmp::MVector{_3k, T}
	
    # Probabilities for each component
	probs::MVector{k, T}
	probs_tmp::MVector{k, T}
	distances::MVector{k, T}

    # Vector to store temporary `p ./ σ`
	pσ::MVector{k, T}
	
	"""
	    GaussianMixture(x::AbstractVector{T}, k::UInt) where T <: Real

	- `x` is an example window the model will be optimized for;
	- `k` is the number of mixture components
	"""
	function GaussianMixture(x::AbstractVector{T}, k::UInt) where T <: Real
		@assert k > 1
		k = Int(k)
		
        # Initialize parameters
		θ = MVector([
			# pᵢ = 1/k
			ones(k) ./ k;
            # μᵢ = quantile(i)
			quantile(x, range(zero(T), one(T), length=k));
            # σᵢ = 1
			ones(k)
		]...)
		θ_old = @MVector zeros(T, 3k)
		θ_tmp = MVector{3k, T}(undef)
		
		probs = MVector{k, T}(undef)
		probs_tmp = MVector{k, T}(undef)
		distances = MVector{k, T}(undef)
		
		new{k, T, 3k}(
			θ, θ_old, θ_tmp,
			probs, probs_tmp, distances,
			MVector{k, T}(undef)
		)
	end
end

"""
Stores estimates:

- `p` - weights of mixture components;
- `μ` - expected values of components;
- `σ` - standard deviations of components;
"""
struct GaussianMixtureEstimate{k, T <: Real}
	algorithm::String
	n_iter::UInt

	p::MVector{k, T}
	μ::MVector{k, T}
	σ::MVector{k, T}

	function GaussianMixtureEstimate(algo::String, n_iter::UInt, p::AbstractVector{T}, μ::AbstractVector{T}, σ::AbstractVector{T}) where T <: Real
		@assert length(p) == length(μ) == length(σ)

		k = length(p)
		new{k, T}(algo, n_iter, MVector{k, T}(p), MVector{k, T}(μ), MVector{k, T}(σ))
	end
end

function Base.sort(est::GaussianMixtureEstimate{k, T}; by=:μ, rev=true) where {k, T}
	@assert by ∈ (:p, :μ, :σ)

	order = sortperm(
		if by == :μ est.μ
		elseif by == :p est.p
		else est.σ
		end,
		rev=rev
	)

	GaussianMixtureEstimate(
		est.algorithm, est.n_iter,
		est.p[order], est.μ[order], est.σ[order]
	)
end

"""
    log_likelihood(x, est::GaussianMixtureEstimate{k, T}) where {k, T}

Evaluate log-likelihood obtained by this fit
"""
log_likelihood(x, est::GaussianMixtureEstimate{k, T}) where {k, T} = log_likelihood(x, est.p, est.μ, est.σ)


"""
    kmeans!(data::GaussianMixture{k, T}, x::AbstractVector{T}, n_steps::Unsigned; eps::T=1e-6) where {k, T <: Real}

Fit gaussian mixture model with `k` components to data in `x` using k-means.
Used mainly as initialization for EM.
"""
function kmeans!(
	data::GaussianMixture{k, T}, x::AbstractVector{T}; maxiter::Unsigned=UInt(50), eps=EPS, metric=metric_l1,
	raw=false
) where {k, T <: Real}
	p, μ, σ = get_pμσ(data.θ)
	p_tmp, μ_tmp, σ_tmp = get_pμσ(data.θ_tmp)

	for _ ∈ 1:maxiter
		@. data.probs = μ_tmp = σ_tmp = zero(T)

		@inbounds for x_ ∈ x
			# Get index of the centroid closest
			# to the current data point `x_`
			@. data.distances = abs(μ - x_)
			idx = argmin(data.distances)
			
			# Temporary computations for
			# new centroids and standard deviations
			# for each cluster
			μ_tmp[idx] += x_
			σ_tmp[idx] += (x_ - μ[idx])^2

			# Temporary computations for probabilities
			# for a data point to be part of this cluster.
			# These aren't really probabilities,
			# but number of items in `x` that
			# are closest to the centroid number `idx`.
			data.probs[idx] += 1
		end

		# We'll divide by `data.probs` later,
        # so make sure there are no zeros
        clamp!(data.probs, eps, Inf)
        clamp!(σ_tmp, eps, Inf)

        # Probabilities to choose each mixture component
		p .= data.probs ./ sum(data.probs)

        # Update centers and standard deviations of mixture components
		@. μ = μ_tmp / data.probs
		@. σ = sqrt(σ_tmp / data.probs)
	end

	(raw ? (maxiter, p, μ, σ, p_tmp, μ_tmp, σ_tmp)
	    : GaussianMixtureEstimate("KMeans", maxiter, p, μ, σ))
end

"""
    em!(
		data::GaussianMixture{k, T}, x::AbstractVector{T};
		tol::T=3e-4, eps::T=1e-4, kmeans_steps::Unsigned=UInt(4), metric=metric_l1
	)

EM algorithm. Fit the model to `x`, modifying `data`.

It's probably better to use `metric_l1` because `metric_l2`
will converge too quickly in high dimensions (curse of dimensionality?)
"""
function em!(
		data::GaussianMixture{k, T}, x::AbstractVector{T};
		tol::T=3e-4, maxiter::Unsigned=UInt(500), eps=EPS, kmeans_steps::Unsigned=UInt(4), metric::Union{Missing, Function}=missing, raw=false
	) where {k, T <: Real}
    # All of these are "pointers" into `θ`
	_, p, μ, σ, p_tmp, μ_tmp, σ_tmp = kmeans!(data, x; maxiter=kmeans_steps, raw=true)

	lik_old = metric === missing ? log_likelihood(x, p, μ, σ) : missing

	i = UInt(0)
	while i < maxiter
		@. data.probs = μ_tmp = σ_tmp = zero(T)
		
		@. data.pσ = p / σ
		for x_ ∈ x
			# These are basically the heights of each Gaussian at `x_`
			# Loops are faster than `@. data.distances = data.pσ * ϕ((x_ - μ) / σ)`
			the_sum = zero(T)
			@inbounds for i ∈ 1:k
				d = data.pσ[i] * ϕ((x_ - μ[i]) / σ[i])
				the_sum += d
				data.distances[i] = d
			end
			# Probabilities that `x_` belongs to each component
			@. data.probs_tmp = data.distances / the_sum

			@. μ_tmp += x_ * data.probs_tmp
			@. σ_tmp += (x_ - μ)^2 * data.probs_tmp
			data.probs .+= data.probs_tmp
		end
		
		if metric !== missing
			data.θ_old .= data.θ
		end
		
        # We'll divide by `data.probs` later,
        # so make sure there are no zeros
        clamp!(data.probs, eps, Inf)
        # `σ_tmp` are actually variances;
        # they must not be zero
        clamp!(σ_tmp, eps, Inf)

        # Probabilities to choose each mixture component
		p .= data.probs ./ sum(data.probs)

        # Update centers and standard deviations of mixture components
		@. μ = μ_tmp / data.probs
		@. σ = sqrt(σ_tmp / data.probs)
		
		i += 1
		should_stop = if metric === missing
			lik_new = log_likelihood(x, p, μ, σ)

			abs(lik_new - lik_old) < tol
		else
			metric(data.θ_old, θ)
		end

		should_stop && break
	end
	
	raw ? (i, p, μ, σ) : GaussianMixtureEstimate("EM", i, p, μ, σ)
end

"""
    em(
		x::AbstractVector{T}, k::Integer;
		tol::T=3e-4, eps=EPS, kmeans_steps::Unsigned=UInt(4), metric=metric_l2
	) where T <: Real

EM algorithm. Fit the model to `x`.

- `k` - number of mixture components
- Accepts same keyword arguments as `em!`
"""
em(x::AbstractVector, k::Integer; kwargs...) = em!(GaussianMixture(x, UInt(k)), x; raw=false, kwargs...)

"""
    kmeans(x::AbstractVector, k::Integer, n_steps::Unsigned=UInt(20); kwargs...)

Fit gaussian mixture model with `k` components to data in `x` using k-means.

- `k` - number of mixture components
- `n_steps` - number of k-mens steps
- For keyword arguments see `em!`
"""
kmeans(x::AbstractVector, k::Integer, n_steps::Unsigned=UInt(20); kwargs...) = kmeans!(GaussianMixture(x, UInt(k)), x; maxiter=n_steps, kwargs...)
