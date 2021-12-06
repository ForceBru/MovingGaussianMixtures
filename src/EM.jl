# ===== Expactation step =====
function _calc_unnorm_posteriors!(
    G::AM{<:Real}, p::AV{<:Real}, mu::AV{<:Real}, sigma::AV{<:Real}, data::AV{<:Real}
)::Nothing
    K, N = size(G)

    @tturbo for n in 1:N, k in 1:K
        G[k, n] = p[k] * normal_pdf(data[n], mu[k], sigma[k])
    end

    nothing
end

function step_E!(
    G::AM{<:Real}, p::AV{<:Real}, mu::AV{<:Real}, sigma::AV{<:Real}, data::AV{<:Real},
    ::Settings.NoRegularization
)::Nothing
    _calc_unnorm_posteriors!(G, p, mu, sigma, data)

    # Normalize posteriors
    G ./= sum(G, dims=1)

    nothing
end

function step_E!(
    G::AM{<:Real}, p::AV{<:Real}, mu::AV{<:Real}, sigma::AV{<:Real}, data::AV{<:Real},
    reg::Settings.RegPosteriorSimple
)::Nothing
    K = size(G, 1)
    step_E!(G, p, mu, sigma, data)

    s = isnan(1/reg.eps) ? 0 : (1 / (1/reg.eps - K))
    @. G = (G + s) / (1 + K * s)

    nothing
end

function step_E!(
    G::AM{<:Real}, p::AV{<:Real}, mu::AV{<:Real}, sigma::AV{<:Real}, data::AV{<:Real}
)::Nothing
    step_E!(G, p, mu, sigma, data, Settings.NoRegularization())
end

# ===== Maximization step =====
function step_M!(
    G::AM{<:Real}, p::AV{<:Real}, mu::AV{<:Real}, sigma::AV{<:Real}, data::AV{<:Real},
    ::Settings.NoRegularization
)::Nothing
    K, N = size(G)
    s = sum(G, dims=2)
    @assert all(>(0), s)

    # Weights
    p .= s ./ sum(s)

    # Means
    mu .= 0
    @tturbo for k in 1:K, n in 1:N
        mu[k] += G[k, n] / s[k] * data[n]
    end

    # Standard deviations
    sigma .= 0
    @tturbo for k in 1:K, n in 1:N
        sigma[k] += G[k, n] / s[k] * (data[n] - mu[k])^2
    end
    sigma .= sqrt.(sigma)

    nothing
end


function step_M!(
    G::AM{<:Real}, p::AV{<:Real}, mu::AV{<:Real}, sigma::AV{<:Real}, data::AV{<:Real}
)::Nothing
    step_M!(G, p, mu, sigma, data, Settings.NoRegularization())
end

function step_M!(
    G::AM{<:Real}, p::AV{<:Real}, mu::AV{<:Real}, sigma::AV{<:Real}, data::AV{<:Real},
    ::Settings.RegPosteriorSimple
)::Nothing
    # Posterior regularization doesn't affect M step
    step_M!(G, p, mu, sigma, data)
end
