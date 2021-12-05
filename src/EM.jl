# ===== Expactation step =====
function step_E!(
    G::AM{<:Real}, pi::AV{<:Real}, mu::AV{<:Real}, sigma::AV{<:Real}, data::AV{<:Real},
    ::Settings.NoRegularization
)::Nothing
    K, N = size(G)

    @tturbo for n in 1:N,  k in 1:K
        G[k, n] = pi[k] * normal_pdf(data[n], mu[k], sigma[k])
    end

    G ./= sum(G, dims=1)

    nothing
end

function step_E!(
    G::AM{<:Real}, pi::AV{<:Real}, mu::AV{<:Real}, sigma::AV{<:Real}, data::AV{<:Real}
)::Nothing
    step_E!(G, pi, mu, sigma, data, Settings.NoRegularization())
end

# ===== Maximization step =====
function step_M!(
    G::AM{<:Real}, pi::AV{<:Real}, mu::AV{<:Real}, sigma::AV{<:Real}, data::AV{<:Real},
    ::Settings.NoRegularization
)::Nothing
    K, N = size(G)
    pi .= mu .= sigma .= 0
    s = sum(G, dims=2)

    # Weights
    pi .= s ./ sum(s)

    # Means
    @tturbo for k in 1:K, n in 1:N
        mu[k] += (G[k, n] / s[k]) * data[n]
    end

    # Standard deviations
    @tturbo for k in 1:K, n in 1:N
        sigma[k] += (G[k, n] / s[k]) * (data[n] - mu[k])^2
    end
    sigma .= sqrt.(sigma)

    nothing
end
