function normal_pdf(x::Real, mu::Real, std::Real)::Real
    exp(-(x - mu)^2 / (2 * std^2)) / (std * sqrt(2*pi))
end

function log_likelihood(gmm::GaussianMixture{T}, data::AbstractVector, ::Settings.NoRegularization) where T<:AbstractFloat
    K, N = size(gmm.G)
    ret = zero(T)

    @tturbo for n in 1:N
        s = zero(T)
        for k in 1:K
            s += gmm.p[k] * normal_pdf(data[n], gmm.mu[k], gmm.sigma[k])
        end
        ret += log(s)
    end

    ret
end

log_likelihood(gmm::GaussianMixture, data::AbstractVector) = log_likelihood(gmm, data, Settings.NoRegularization())

log_likelihood(gmm::GaussianMixture, data::AbstractVector, ::Settings.RegPosteriorSimple) = log_likelihood(gmm, data)
