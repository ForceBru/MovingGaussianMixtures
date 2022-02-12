@inline normal_pdf(x::Real, mu::Real, var::Real) = exp(-(x - mu)^2 / (2var)) / sqrt(2pi * var)

const LN2PI = log(2pi)

function log_likelihood(
    G::AM{<:Real}, p::AV{<:Real}, mu::AV{<:Real}, var::AV{<:Real}, x::AV{<:Real},
    ::Nothing
)
    K, N = size(G)
    ret = G |> eltype |> zero

    @tturbo for n in 1:N
        s = 0
        for k in 1:K
            s += p[k] * normal_pdf(x[n], mu[k], var[k])
        end
        ret += log(s)
        # FIXME: tests of log-likelihood fail when multithreading is used?
        # Possible data race?
        # Very siilar code in documentation:
        # https://juliasimd.github.io/LoopVectorization.jl/latest/examples/sum_of_squared_error
    end

    ret
end

@inline function log_likelihood(
    G::AM{<:Real}, p::AV{<:Real}, mu::AV{<:Real}, var::AV{<:Real}, x::AV{<:Real},
    ::Settings.AbstractRegAdHoc
)
    # AdHoc regularization does NOT affect log-likelihood
    log_likelihood(G, p, mu, var, x, nothing)
end

ELBO(G_kn::Real, p_k::Real, mu_k::Real, var_k::Real, x_n::Real) =
    G_kn * (
        log(p_k) - (log(2pi) + log(var_k) + (x_n - mu_k)^2 / var_k) / 2
        - log(G_kn + 1e-100) #FIXME: entropy calculation correct?
        # Need to add 1e-100 inside log to avoid NaN when `G[k, n] ≈ 0`
    )

"""
Compute ELBO at a given _point_ `x`.
"""
function ELBO(
    g::AV{<:Real}, p::AV{<:Real}, mu::AV{<:Real}, var::AV{<:Real}, x::Real,
    ::Nothing
)
    error("Why am I executed?")

    ret = g |> eltype |> zero
    @turbo for k in eachindex(g)
        ret += ELBO(g[k], p[k], mu[k], var[k], x)
    end

    ret
end

# Posterior regularization does NOT affect ELBO
@inline ELBO(
    g::AV{<:Real}, p::AV{<:Real}, mu::AV{<:Real}, var::AV{<:Real}, x::Real,
    ::Settings.AbstractRegAdHoc
) = ELBO(g, p, mu, var, x, nothing)

"""
Compute ELBO = E_q ( log[p(X, Z | THETA)] - log[q(Z)] )

No regularization
"""
function ELBO(
    G::AM{<:Real}, p::AV{<:Real}, mu::AV{<:Real}, var::AV{<:Real}, x::AV{<:Real},
    ::Nothing
)
    K, N = size(G)
    ret = G |> eltype |> zero

    @tturbo for n in 1:N, k in 1:K
        ret += G[k, n] * (
            log(p[k]) - (log(2pi) + log(var[k]) + (x[n] - mu[k])^2 / var[k]) / 2
            - log(G[k, n] + 1e-100) #FIXME: entropy calculation correct?
            # Need to add 1e-100 inside log to avoid NaN when `G[k, n] ≈ 0`
        )
        # Calling the function is slow (-25% exec speed)
        # ELBO(G[k, n], p[k], mu[k], var[k], x[n])
    end

    ret
end

@inline function ELBO(
    G::AM{<:Real}, p::AV{<:Real}, mu::AV{<:Real}, var::AV{<:Real}, x::AV{<:Real},
    ::Settings.AbstractRegAdHoc
)
    # Posterior regularization does NOT affect ELBO
    ELBO(G, p, mu, var, x, nothing)
end

@inline function regularize_posteriors!(G::AM{<:Real}, normalization::AM{<:Real}, ::Nothing)::Nothing
    K, N = size(G)
    @tturbo for n ∈ 1:N, k ∈ 1:K
        G[k, n] /= normalization[1, n]
    end
    # G ./= normalization
    nothing
end

@inline regularize_posteriors!(G::AM{<:Real}, normalization::AM{<:Real}, ::Settings.AbstractRegAdHoc) =
    regularize_posteriors!(G, normalization, nothing)

function regularize_posteriors!(G::AM{<:Real}, normalization::AM{<:Real}, reg::Settings.RegPosteriorNonzero)::Nothing
    regularize_posteriors!(G, normalization, nothing)
    K = size(G, 1)
    s = reg.s

    @. G = (G + s) / (1 + s * K)

    nothing
end

function regularize_posteriors!(G::AM{<:Real}, normalization::AM{<:Real}, reg::Settings.RegPosteriorAddEps)::Nothing
    K = size(G, 1)

    G .+= reg.eps
    normalization .+= K * reg.eps

    regularize_posteriors!(G, normalization, nothing)
end

function step_E!(
    G::AM{<:Real}, p::AV{<:Real}, mu::AV{<:Real}, var::AV{<:Real}, x::AV{<:Real},
    reg::Union{Settings.AbstractRegAdHoc, Nothing}
)::Nothing
    K, N = size(G)
    normalization = zeros(eltype(G), 1, N)
    @tturbo for n in 1:N, k in 1:K
        G[k, n] = p[k] * exp(-(x[n] - mu[k])^2 / (2var[k])) / sqrt(2pi * var[k])
        # Calling the function is slow (-20% exec speed)
        # normal_pdf(x[n], mu[k], var[k])
        normalization[1, n] += G[k, n]
    end
    all(>(0), normalization) || throw(ZeroNormalizationException())

    regularize_posteriors!(G, normalization, reg)
end

function calc_weights!(p::AV{<:Real}, G::AM{<:Real}, ev::AV{<:Real}, ::Nothing)::Nothing
    p .= ev ./ sum(ev)
    nothing
end

calc_weights!(p::AV{<:Real}, G::AM{<:Real}, ev::AV{<:Real}, ::Settings.AbstractRegAdHoc) =
    calc_weights!(p, G, ev, nothing)

function calc_means!(mu::AV{<:Real}, G::AM{<:Real}, ev::AV{<:Real}, x::AV{<:Real}, ::Nothing)::Nothing
    K, N = size(G)

    mu .= zero(eltype(mu))
    @tturbo for n in 1:N, k in 1:K
        mu[k] += G[k, n] / ev[k] * x[n]
    end
    nothing
end

calc_means!(mu::AV{<:Real}, G::AM{<:Real}, ev::AV{<:Real}, x::AV{<:Real}, ::Settings.AbstractRegAdHoc) =
    calc_means!(mu, G, ev, x, nothing)

function calc_variances!(
    var::AV{<:Real}, G::AM{<:Real}, ev::AV{<:Real}, x::AV{<:Real}, mu::AV{<:Real},
    ::Nothing
)::Nothing
    K, N = size(G)

    var .= zero(eltype(var))
    @tturbo for n in 1:N, k in 1:K
        var[k] += G[k, n] / ev[k] * (x[n] - mu[k])^2
    end
    nothing
end

function calc_variances!(
    var::AV{<:Real}, G::AM{<:Real}, ev::AV{<:Real}, x::AV{<:Real}, mu::AV{<:Real},
    reg::Settings.RegVarianceAddEps
)::Nothing
    calc_variances!(var, G, ev, x, mu, nothing)
    var .+= reg.eps

    nothing
end

function calc_variances!(
    var::AV{<:Real}, G::AM{<:Real}, ev::AV{<:Real}, x::AV{<:Real}, mu::AV{<:Real},
    ::Settings.RegVarianceReset
)::Nothing
    K, _ = size(G)
    # 1. Calculate variances as usual
    calc_variances!(var, G, ev, x, mu, nothing)

    # 2. Select largest variance
    valid_mask = .~is_almost_zero.(var)
    any(valid_mask) || throw(ZeroVarianceException(var))
    max_variance = maximum(var[valid_mask])

    # 3. Reposition Gaussians with too low variance
    @inbounds for k in 1:K
        if !valid_mask[k]
            var[k] = max_variance
            mu[k] = rand(x) * 2rand()
        end
    end

    nothing
end

calc_variances!(
    var::AV{<:Real}, G::AM{<:Real}, ev::AV{<:Real}, x::AV{<:Real}, mu::AV{<:Real},
    ::Settings.AbstractRegAdHoc
) = calc_variances!(var, G, ev, x, mu, nothing)

function step_M!(
    G::AM{<:Real}, p::AV{<:Real}, mu::AV{<:Real}, var::AV{<:Real}, x::AV{<:Real},
    reg::Union{Settings.AbstractRegAdHoc, Nothing}
)::Nothing
    K, N = size(G)
    evidences = zeros(K) # very small vector
    @tturbo for n ∈ 1:N, k ∈ 1:K
        evidences[k] += G[k, n]
    end
    all(>(0), evidences) || throw(ZeroNormalizationException())

    calc_weights!(p, G, evidences, reg)
    calc_means!(mu, G, evidences, x, reg)
    calc_variances!(var, G, evidences, x, mu, reg)
end
