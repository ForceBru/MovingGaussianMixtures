module Experimental

using Statistics
import Clustering

import ..ClusteringModel, ..fit!

mutable struct GaussianMixture{T, U} <: ClusteringModel{T, U}
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

"""
    initialize!(gm::GaussianMixture{T, U}, data::AbstractVector{T}, init::Symbol) where {T, U}

Initialize the [`GaussianMixture`](@ref) using k-means.
"""
function initialize!(gm::GaussianMixture{T}, data::AbstractVector{T}, init::Symbol) where T
    (init ∈ (:kmeans, :fuzzy_cmeans)) ||
		throw(ArgumentError("Supported initialization methods `init` are (:kmeans, :fuzzy_cmeans) (got $init)"))

    if init == :kmeans
		res = Clustering.kmeans(reshape(data, 1, :), Int(gm.K))

		gm.old.μ .= res.centers[1, :]
		gm.old.π .= Clustering.counts(res) ./ gm.N

		assignments = Clustering.assignments(res)
		@inbounds for k ∈ 1:gm.K
			@. gm.mask = assignments == k

			gm.old.σ[k] = if !any(gm.mask)
				# `k`th cluster is empty
				eps
			else
				the_std = std(data[gm.mask], corrected=false)

				(the_std ≈ zero(T)) ? eps : the_std
			end
		end
	elseif init == :fuzzy_cmeans
		res = Clustering.fuzzy_cmeans(reshape(data, 1, :), Int(gm.K), 2)

		gm.old.μ .= res.centers[1, :]
		gm.old.π .= sum(res.weights, dims=1)[1, :]
		gm.old.π ./= sum(gm.old.π)

		@inbounds for k ∈ 1:gm.K
			σ = zero(T)
			s = zero(T)
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
	@inbounds for n ∈ eachindex(x)
		s = zero(T)
		for k ∈ eachindex(μ)
			s += π[k] * component_pdf(x[n], μ[k], σ[k])
		end
		ev[n] = s
	end
	
	@. ev = clamp(ev, ε, Inf)
	
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
	
	@inbounds for k ∈ eachindex(gm.old.π)
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
	
	@. gm.b = clamp(gm.b - gm.tmp^2 / clamp(gm.a, ε, Inf), ε, Inf)
	
	nothing
end

function em_step!(
    gm::GaussianMixture{T, U}, data::AbstractVector{T}, eps
) where {T, U}
    compute_intermediates!(gm, data, eps)
    
    @. gm.new.μ = gm.tmp / gm.a
    gm.new.π .= gm.a ./ sum(gm.a)
    @. gm.new.σ = sqrt(gm.b / gm.a)

    nothing
end

function fit!(
    gm::GaussianMixture{T, U}, data::AbstractVector{T};
    init::Symbol=:fuzzy_cmeans,
    maxiter::Integer=1000, tol=1e-5, eps=1e-10
)::GaussianMixture{T, U} where {T, U}
    @assert length(data) == gm.N
    @assert tol > 0
    @assert eps > 0

    r(x, y) = abs(x - y) / abs(y)
    metric() = max(
        maximum(r.(gm.new.π, gm.old.π)),
        maximum(r.(gm.new.μ, gm.old.μ)),
        maximum(r.(gm.new.σ, gm.old.σ)),
    )

    gm.converged = false
    gm.n_iter = zero(U)

    (gm.first_call || !gm.warm_start) && initialize!(gm, data, init)

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

end

