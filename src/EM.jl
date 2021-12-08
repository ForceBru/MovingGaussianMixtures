@inline normal_pdf(x::Real, mu::Real, var::Real) = exp(-(x - mu)^2 / (2var)) / sqrt(2pi * var)

"""
Compute E_q log[ p(X, Z | THETA) ]

No regularization
"""
function ELBO_1(
    G::AM{<:Real}, p::AV{<:Real}, mu::AV{<:Real}, var::AV{<:Real}, x::AV{<:Real},
    ::Nothing
)
    K, N = size(G)
    ln2pi = log(2pi)
    ret = 0.0

    @tturbo for k in 1:K, n in 1:N
        ret += G[k, n] * (
            log(p[k]) - (ln2pi + log(var[k]) + (x[n] - mu[k])^2 / var[k]) / 2
        )
    end

    ret
end

@inline function ELBO_1(
    G::AM{<:Real}, p::AV{<:Real}, mu::AV{<:Real}, var::AV{<:Real}, x::AV{<:Real},
    ::Union{Settings.AbstractRegPosterior, Settings.RegVarianceSimple, Settings.RegVarianceReset}
)
    # Posterior regularization does NOT affect ELBO
    ELBO_1(G, p, mu, var, x, nothing)
end

@inline regularize_posteriors!(G::AM{<:Real}, ::Nothing)::Nothing = nothing

function regularize_posteriors!(G::AM{<:Real}, reg::Settings.RegPosteriorSimple)::Nothing
    K = size(G, 1)
    s = if iszero(reg.eps)
        0
    else
        1/reg.eps > 1/K || throw(InvalidMinPosteriorProbException(reg.eps, K))
        1 / (1/reg.eps - 1/K)
    end

    @. G = (G + s) / (1 + s*K)

    nothing
end

function step_E!(
    G::AM{<:Real}, p::AV{<:Real}, mu::AV{<:Real}, var::AV{<:Real}, x::AV{<:Real},
    reg::Union{Settings.AbstractRegPosterior, Nothing}
)::Nothing
    K, N = size(G)
    normalization = zeros(1, N)
    @tturbo for n in 1:N, k in 1:K
        G[k, n] = p[k] * normal_pdf(x[n], mu[k], var[k])
        normalization[1, n] += G[k, n]
    end
    all(>(0), normalization) || throw(ZeroNormalizationException())

    G ./= normalization

    regularize_posteriors!(G, reg)
end

@inline function step_E!(
    G::AM{<:Real}, p::AV{<:Real}, mu::AV{<:Real}, var::AV{<:Real}, x::AV{<:Real},
    reg::Settings.AbstractRegPrior
)::Nothing
    # Prior regularization does NOT affect Expectation step
    step_E!(G, p, mu, var, x, nothing)
end

function calc_weights!(p::AV{<:Real}, G::AM{<:Real}, ev::AV{<:Real}, ::Nothing)::Nothing
    p .= ev ./ sum(ev)
    nothing
end

@inline calc_weights!(p::AV{<:Real}, G::AM{<:Real}, ev::AV{<:Real}, ::Settings.AbstractRegPrior) =
    calc_weights!(p, G, ev, nothing)

function calc_means!(mu::AV{<:Real}, G::AM{<:Real}, ev::AV{<:Real}, x::AV{<:Real}, ::Nothing)::Nothing
    K, N = size(G)

    mu .= 0
    @tturbo for k in 1:K, n in 1:N
        mu[k] += G[k, n] / ev[k] * x[n]
    end
    nothing
end

@inline calc_means!(mu::AV{<:Real}, G::AM{<:Real}, ev::AV{<:Real}, x::AV{<:Real}, ::Settings.AbstractRegPrior) =
    calc_means!(mu, G, ev, x, nothing)

function calc_variances!(
    var::AV{<:Real}, G::AM{<:Real}, ev::AV{<:Real}, x::AV{<:Real}, mu::AV{<:Real},
    ::Nothing
)::Nothing
    K, N = size(G)

    var .= 0
    @tturbo for k in 1:K, n in 1:N
        var[k] += G[k, n] / ev[k] * (x[n] - mu[k])^2
    end
    nothing
end

function calc_variances!(
    var::AV{<:Real}, G::AM{<:Real}, ev::AV{<:Real}, x::AV{<:Real}, mu::AV{<:Real},
    reg::Settings.RegVarianceSimple
)::Nothing
    calc_variances!(var, G, ev, x, mu, nothing)
    var .+= reg.eps

    nothing
end

function calc_variances!(
    var::AV{<:Real}, G::AM{<:Real}, ev::AV{<:Real}, x::AV{<:Real}, mu::AV{<:Real},
    reg::Settings.RegVarianceReset
)::Nothing
    K, _ = size(G)
    calc_variances!(var, G, ev, x, mu, nothing)

    valid_vars = var[.~is_almost_zero.(var)]
    length(valid_vars) != 0 || throw(ZeroVarianceException(var))
    max_variance = maximum(valid_vars)
    @inbounds for k in 1:K
        if is_almost_zero(var[k])
            var[k] = max_variance
            mu[k] = rand(x) * rand()
        end
    end

    nothing
end

@inline calc_variances!(
    var::AV{<:Real}, G::AM{<:Real}, ev::AV{<:Real}, x::AV{<:Real}, mu::AV{<:Real},
    ::Settings.AbstractRegPrior
) = calc_variances!(var, G, ev, x, mu, nothing)

function step_M!(
    G::AM{<:Real}, p::AV{<:Real}, mu::AV{<:Real}, var::AV{<:Real}, x::AV{<:Real},
    reg::Union{Settings.AbstractRegPrior, Nothing}
)::Nothing
    K, N = size(G)
    evidences = sum(G, dims=2) |> vec
    all(>(0), evidences) || throw(ZeroNormalizationException())

    calc_weights!(p, G, evidences, reg)
    calc_means!(mu, G, evidences, x, reg)
    calc_variances!(var, G, evidences, x, mu, reg)
end

@inline function step_M!(
    G::AM{<:Real}, p::AV{<:Real}, mu::AV{<:Real}, var::AV{<:Real}, x::AV{<:Real},
    reg::Settings.AbstractRegPosterior
)::Nothing
    # Posterior regularization does NOT affect maximization step
    step_M!(G, p, mu, var, x, nothing)
end
