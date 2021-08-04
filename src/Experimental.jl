module Experimental

using Statistics
import Clustering
import Distributions: UnivariateGMM, Categorical

using LoopVectorization

import ..AbstractGaussianMixture, .._not_fit_error

# To be overriden
import ..fit!, ..distribution, ..nconverged, ..converged_pct

mutable struct GaussianMixture{T, U} <: AbstractGaussianMixture{T, U}
    K::U
    N::U

    converged::Bool
	warm_start::Bool
	first_call::Bool
	n_iter::U

    # Intermediate values, as in Hathaway
    # (K x 1) vectors
    a::Vector{T}
    b::Vector{T}
    tmp::Vector{T}

    # Mixture parameters - (K x 1) vectors
    old::NamedTuple{(:π, :μ, :σ), Tuple{Vector{T}, Vector{T}, Vector{T}}}
    new::NamedTuple{(:π, :μ, :σ), Tuple{Vector{T}, Vector{T}, Vector{T}}}
    evidence::Vector{T} # (N x 1)

    function GaussianMixture(K::U, N::U, ::Type{T}=Float64; warm_start::Bool=false) where { T <: Real, U <: Unsigned}
        @assert K > 0
        @assert N > 0

        d = zeros(T, K)
        new{T, U}(
            K, N,
            false, warm_start, true, zero(U),
            similar(d), similar(d), similar(d), # intermediates
            (π=similar(d), μ=similar(d), σ=similar(d)),
            (π=similar(d), μ=similar(d), σ=similar(d)),
            zeros(T, N) # evidence
        )
    end
end

function GaussianMixture(K::Signed, N::Signed, TYP::Type{T}=Float64; warm_start::Bool=false) where T <: Real
    GaussianMixture(UInt(K), UInt(N), TYP; warm_start)
end

"""
    initialize!(gm::GaussianMixture{T, U}, data::AbstractVector{T}, init::Symbol) where {T, U}

Initialize the [`GaussianMixture`](@ref) using k-means.
"""
function initialize!(gm::GaussianMixture{T}, data::AbstractVector{T}, init::Symbol, eps) where T
    (init ∈ (:kmeans, :fuzzy_cmeans)) ||
		throw(ArgumentError("Supported initialization methods `init` are (:kmeans, :fuzzy_cmeans) (got $init)"))

    if init == :kmeans
		res = Clustering.kmeans(reshape(data, 1, :), Int(gm.K))

		gm.old.μ .= res.centers[1, :]
		gm.old.π .= Clustering.counts(res) ./ gm.N

		assignments = Clustering.assignments(res)
        mask::BitVector = BitVector(undef, gm.N)
		@inbounds for k ∈ 1:gm.K
			@. mask = assignments == k

			gm.old.σ[k] = if !any(mask)
				# `k`th cluster is empty
				eps
			else
				the_std = std(data[mask], corrected=false)

				(the_std ≈ zero(T)) ? eps : the_std
			end
		end
	elseif init == :fuzzy_cmeans
		res = Clustering.fuzzy_cmeans(reshape(data, 1, :), Int(gm.K), 2)

		gm.old.μ .= res.centers[1, :]
		gm.old.π .= sum(res.weights, dims=1)[1, :]
		gm.old.π ./= sum(gm.old.π)

		@tturbo for k ∈ 1:gm.K
			σ = zero(T)
			s = eps
			for n ∈ eachindex(data)
				# These are the same formulas as for EM
				s += res.weights[n, k]
				σ += (data[n] - gm.old.μ[k])^2 * res.weights[n, k]
			end
			gm.old.σ[k] = sqrt(σ / s)
		end
	else
		@assert false "BUG: unexpected init=$init"
	end
end

"Standard normal PDF"
ϕ(x) = exp(-x^2 / 2) / sqrt(2π)
	
"PDF of a single mixture component"
component_pdf(x, μ, σ) = ϕ((x - μ) / σ) / σ

"""
Computes evidence:

```math
e_n = ∑_{k=1}^K π_k p(x_n, μ_k, σ_k)
n = 1,N
```
"""
function evidence!(
	ev::AbstractVector{T}, x::AbstractVector{T},
	π::AbstractVector{T}, μ::AbstractVector{T}, σ::AbstractVector{T},
	ε
) where T <: Real
	@tturbo for n ∈ eachindex(x)
		s = zero(T)
		for k ∈ eachindex(μ)
			s += π[k] * component_pdf(x[n], μ[k], σ[k])
		end
		ev[n] = s
	end
	
	@tturbo @. ev = clamp(ev, ε, Inf)
	
	nothing
end

"""
Intermediate computations of `a` and `b`, as in Hathaway:

```math
a_k = ∑_n W_{kn}^r
```

```math
b_k = ∑_n (x_n - μ_k^{r+1})^2 W_{kn}^r
    = ∑_n x_n^2 W_{kn}^r - (∑_n x_n W_{kn}^r)^2 / ∑_n W_{kn}^r
```

```math
tmp_k = ∑_n x_n W_{kn}^r
```

Last equality can be obtained by unrolling the square
and substituting the definition of ``μ_i^{r+1}``.
"""
function compute_intermediates!(
	gm::GaussianMixture{T}, x::AbstractVector{T}, ε
) where T <: Real
	evidence!(gm.evidence, x, gm.old.π, gm.old.μ, gm.old.σ, ε)
	
	@tturbo for k ∈ eachindex(gm.old.π)
	    gm.a[k] = zero(T)
		gm.b[k] = zero(T)
		gm.tmp[k] = zero(T)
		for n ∈ eachindex(x)
			g = gm.old.π[k] * component_pdf(x[n], gm.old.μ[k], gm.old.σ[k]) / gm.evidence[n]
			
			gm.a[k] += g
			gm.b[k] += x[n] * x[n] * g
			gm.tmp[k] += x[n] * g
		end
	end
	
	@turbo @. gm.b = clamp(gm.b - gm.tmp^2 / clamp(gm.a, ε, Inf), ε, Inf)
	
	nothing
end

function em_step!(
    gm::GaussianMixture{T, U}, data::AbstractVector{T}, eps
) where {T, U}
    if any(≈(zero(T)), gm.old.σ)
        gm.converged = false

        throw(DomainError(gm.old.σ, "One of the standard deviations is too close to zero"))
    end

    compute_intermediates!(gm, data, eps)
    
    @turbo @. gm.new.μ = gm.tmp / gm.a
    gm.new.π .= gm.a ./ sum(gm.a)
    @turbo @. gm.new.σ = sqrt(gm.b / gm.a)

    nothing
end

function fit!(
    gm::GaussianMixture{T, U}, data::AbstractVector{T};
    init::Symbol=:fuzzy_cmeans,
    maxiter::Integer=1000, tol=1e-3, eps=1e-10
)::GaussianMixture{T, U} where {T, U}
    @assert length(data) == gm.N
    @assert tol > 0
    @assert eps > 0

    r(x, y) = abs(x - y) / abs(y)
    function metric()
        # Don't allocate!
        @. gm.tmp = r(gm.new.π, gm.old.π)
        dπ = maximum(gm.tmp)

        @. gm.tmp = r(gm.new.μ, gm.old.μ)
        dμ = maximum(gm.tmp)

        @. gm.tmp = r(gm.new.σ, gm.old.σ)
        dσ = maximum(gm.tmp)

        max(dπ, dμ, dσ)
    end

    gm.converged = false
    gm.n_iter = zero(U)

    (gm.first_call || !gm.warm_start) && initialize!(gm, data, init, eps)

    gm.first_call = false
    
    em_step!(gm, data, eps)
    gm.old.π .= gm.new.π
	gm.old.μ .= gm.new.μ
	gm.old.σ .= gm.new.σ

    for i ∈ 1:maxiter
        em_step!(gm, data, eps)

        m = metric()
        if m < tol
            gm.converged = true
            gm.n_iter = convert(typeof(gm.n_iter), i)

            break
        end

        gm.old.π .= gm.new.π
        gm.old.μ .= gm.new.μ
        gm.old.σ .= gm.new.σ
    end

    if !gm.converged
        gm.n_iter = convert(typeof(gm.n_iter), maxiter)
    end

    gm
end

# ===== Obtain results =====
"""
    distribution(gm::GaussianMixture)

Get the Distributions.jl `UnivariateGMM` of this `GaussianMixture`.
"""
distribution(gm::GaussianMixture) =
	if gm.first_call
		_not_fit_error()
	else
		UnivariateGMM(
			# Copy everything! Otherwise the params will be SHARED!
			copy(gm.new.μ), copy(gm.new.σ), Categorical(copy(gm.new.π))
		)
	end

nconverged(gm::GaussianMixture) = Int(gm.converged)
converged_pct(gm::GaussianMixture) = Float64(nconverged(gm)) * 100

end

