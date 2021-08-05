module Experimental

using Statistics
import Clustering
import Distributions: UnivariateGMM, Categorical

using LoopVectorization

import ..AbstractGaussianMixture, .._not_fit_error, ..initialize_kmeans!, ..initialize_fuzzy_cmeans!

# To be overriden
import ..fit!, ..distribution, ..nconverged, ..converged_pct

mutable struct GaussianMixture{T} <: AbstractGaussianMixture{T}
    K::Int
    N::Int

    converged::Bool
	warm_start::Bool
	first_call::Bool
	n_iter::Int

    # Intermediate values, as in Hathaway
    # (K x 1) vectors
    a::Vector{T}
    b::Vector{T}
    tmp::Vector{T}

    # Mixture parameters - (K x 1) vectors
    old::NamedTuple{(:π, :μ, :σ), Tuple{Vector{T}, Vector{T}, Vector{T}}}
    new::NamedTuple{(:π, :μ, :σ), Tuple{Vector{T}, Vector{T}, Vector{T}}}
    evidence::Vector{T} # (N x 1)

    # Posterior probabilities for each data point
    # to belong to each cluster
    G::Union{Nothing, Matrix{T}} # (K x N) - huge!

    function GaussianMixture(K::Integer, N::Integer, ::Type{T}=Float64; warm_start::Bool=false) where T <: Real
        @assert K > 0
        @assert N > 0

        K = Int(K)
        N = Int(N)

        d = zeros(T, K)
        new{T}(
            K, N,
            false, warm_start, true, 0,
            similar(d), similar(d), similar(d), # intermediates
            (π=similar(d), μ=similar(d), σ=similar(d)),
            (π=similar(d), μ=similar(d), σ=similar(d)),
            zeros(T, N), # evidence
            nothing # BIG matrix
        )
    end
end

"""
    initialize!(gm::GaussianMixture{T}, data::AbstractVector{T}, init::Symbol, eps) where T

Initialize the [`GaussianMixture`](@ref) using k-means.
"""
function initialize!(gm::GaussianMixture{T}, data::AbstractVector{T}, init::Symbol, eps) where T
    (init ∈ (:kmeans, :fuzzy_cmeans)) ||
		throw(ArgumentError("Supported initialization methods `init` are (:kmeans, :fuzzy_cmeans) (got $init)"))

    if init == :kmeans
        initialize_kmeans!(gm.old.π, gm.old.μ, gm.old.σ, data, gm.K, eps)
	elseif init == :fuzzy_cmeans
        initialize_fuzzy_cmeans!(gm.old.π, gm.old.μ, gm.old.σ, data, gm.K, eps)
	else
		@assert false "BUG: unexpected init=$init"
	end
end

function Base.sort!(gm::GaussianMixture; by::Symbol=:μ)
    the_order = sortperm(
        (π=gm.new.π, μ=gm.new.μ, σ=gm.new.σ)[by]
    )

    gm.new.π .= gm.new.π[the_order]
    gm.new.μ .= gm.new.μ[the_order]
    gm.new.σ .= gm.new.σ[the_order]

    gm.old.π .= gm.old.π[the_order]
    gm.old.μ .= gm.old.μ[the_order]
    gm.old.σ .= gm.old.σ[the_order]

    nothing
end

"Standard normal PDF"
ϕ(x) = exp(-x^2 / 2) / sqrt(2π)
	
"PDF of a single mixture component"
component_pdf(x, μ, σ) = ϕ((x - μ) / σ) / σ

"""
Computes evidence:

```math
∑_n ∑_{k=1}^K π_k p(x_n, μ_k, σ_k)
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

function log_likelihood(
    x::AbstractVector{T},
    π::AbstractVector{T}, μ::AbstractVector{T}, σ::AbstractVector{T}
) where T <: Real
    ev = similar(x)
    evidence!(ev, x, π, μ, σ, 1e-10)

    sum(log.(ev))
end

function vsum!(a::AbstractVector{T}, G::AbstractMatrix{T}) where T <: Real
    K, N = size(G)
    @tturbo for k ∈ 1:K
        s = zero(T)
        for n ∈ 1:N
            s += G[k, n]
        end
        a[k] = s
    end
end

function vsum_prod!(a::AbstractVector{T}, x::AbstractVector{T}, G::AbstractMatrix{T}) where T <: Real
    K, N = size(G)
    @tturbo for k ∈ 1:K
        s = zero(T)
        for n ∈ 1:N
            s += G[k, n] * x[n]
        end
        a[k] = s
    end
end

function vsum_prod!(
    a::AbstractVector{T},
    x1::AbstractVector{T}, x2::AbstractVector{T},
    G::AbstractMatrix{T}
) where T <: Real
    K, N = size(G)
    @tturbo for k ∈ 1:K
        s = zero(T)
        for n ∈ 1:N
            s += G[k, n] * x1[n] * x2[n]
        end
        a[k] = s
    end
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
    ev = gm.evidence
    a = gm.a
    b = gm.b
    tmp = gm.tmp
    old_π = gm.old.π
    old_μ = gm.old.μ
    old_σ = gm.old.σ

    if gm.G === nothing
        gm.G = Matrix{T}(undef, gm.K, gm.N)
    end

    # This fills a huge matrix, and it's still
    # as fast as the non-allocating loop below!
    @tturbo @. gm.G = old_π * component_pdf(x', old_μ, old_σ)
    vsum!(ev, gm.G')
    @tturbo gm.G ./= clamp.(ev', ε, Inf)

    vsum!(a, gm.G)
    vsum_prod!(b, x, x, gm.G)
    vsum_prod!(tmp, x, gm.G)
    
    # ===== Same, but using a loop =====
    # This, including the call to `evidence!`,
    # calculates `old_π[k] * component_pdf(x[n], old_μ[k], old_σ[k])` TWICE,
    # while the code with the matrix - only once

    # evidence!(ev, x, old_π, old_μ, old_σ, ε)
    
	# @tturbo for k ∈ eachindex(old_π)
	#     a[k] = zero(T)
	# 	b[k] = zero(T)
	# 	tmp[k] = zero(T)
	# 	for n ∈ eachindex(x)
	# 		g = old_π[k] * component_pdf(x[n], old_μ[k], old_σ[k]) / ev[n]
			
	# 		a[k] += g
	# 		b[k] += x[n] * x[n] * g
	# 		tmp[k] += x[n] * g
	# 	end
	# end
	
	@turbo @. b = clamp(b - tmp^2 / clamp(a, ε, Inf), ε, Inf)
	
	nothing
end

function em_step!(
    gm::GaussianMixture{T}, data::AbstractVector{T}, eps
) where T
    if any(≈(zero(T)), gm.old.σ)
        gm.converged = false

        throw(DomainError(gm.old.σ, "One of the standard deviations is too close to zero"))
    end

    compute_intermediates!(gm, data, eps)
    
    gm.new.π .= gm.a ./ sum(gm.a)
    @turbo @. gm.new.μ = gm.tmp / gm.a
    @turbo @. gm.new.σ = sqrt(gm.b / gm.a)

    nothing
end

function fit!(
    gm::GaussianMixture{T}, data::AbstractVector{T};
    init::Symbol=:kmeans, sort_by::Symbol=:μ,
    maxiter::Integer=1000, tol=1e-3, eps=1e-10
)::GaussianMixture{T} where T
    @assert length(data) == gm.N
    @assert tol > 0
    @assert eps > 0

    r(x, y) = abs(x - y) / abs(y)
    function convergence_metric()
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
    gm.n_iter = 0

    (gm.first_call || !gm.warm_start) && initialize!(gm, data, init, eps)

    gm.first_call = false
    
    em_step!(gm, data, eps)
    gm.old.π .= gm.new.π
	gm.old.μ .= gm.new.μ
	gm.old.σ .= gm.new.σ

    # This is here just to error out early
    # in case `sort_by` is invalid
    sort!(gm, by=sort_by)

    for i ∈ 1:maxiter
        em_step!(gm, data, eps)

        m = convergence_metric()
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

    sort!(gm, by=sort_by)

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

