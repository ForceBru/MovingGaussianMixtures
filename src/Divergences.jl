"""
Compute divergences between Gaussian mixtures
"""
module Divergences

export cauchy_schwarz

import ..normal_pdf

using DocStringExtensions
using Distributions: UnivariateGMM

function cs_part(gmm::UnivariateGMM)
    0.5 * log(
        #FIXME: the paper DOESN'T divide by sqrt(2),
        # but if you don't do this, `cauchy_schwarz(gmm, gmm) ≠ 0`,
        # but the distance between the same GMM must be zero!
        sum(@. gmm.prior.p^2 / sqrt(2π) / gmm.stds / sqrt(2))
        + 2 * sum(
            sum(
                gmm.prior.p[m] * gmm.prior.p[m_] * normal_pdf(
                    gmm.means[m], gmm.means[m_],
                    gmm.stds[m]^2 + gmm.stds[m_]^2
                )
                for m_ ∈ 1:(m-1); init=0.0
                # Specifying `init=0` is needed because otherwise
                # Julia can't sum over an empty collection 1:0
            )
            for m ∈ 1:gmm.K; init=0.0
            # The semicolon (NOT a comma!) is also necessary:
            # https://discourse.julialang.org/t/error-reducing-over-an-empty-collection-is-not-allowed/47410/2
        )
    )
end

"""
$(TYPEDSIGNATURES)

Compute Cauchy-Schwarz divergence between two Gaussian mixtures.

See eq. 3 of the paper.

Kampa, Kittipat, Erion Hasanbelliu, and Jose C. Principe.
"Closed-Form Cauchy-Schwarz PDF Divergence for Mixture of Gaussians."
In The 2011 International Joint Conference on Neural Networks, 2578–85.
San Jose, CA, USA: IEEE, 2011. https://doi.org/10.1109/IJCNN.2011.6033555.
"""
function cauchy_schwarz(gmm1::UnivariateGMM, gmm2::UnivariateGMM)
    a = 0
    @inbounds for m in 1:gmm1.K, k in 1:gmm2.K
        a += gmm1.prior.p[m] * gmm2.prior.p[k] * normal_pdf(
            gmm1.means[m], gmm2.means[k],
            gmm1.stds[m]^2 + gmm2.stds[k]^2
        )
    end

    -log(a) + cs_part(gmm1) + cs_part(gmm2)
end

end