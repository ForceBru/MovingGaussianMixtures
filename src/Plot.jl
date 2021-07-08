using RecipesBase
using StatsBase: params
using Distributions: UnivariateGMM, components, probs, pdf

@recipe function f(gmm::UnivariateGMM, x; plot_components::Bool=true)
    p = probs(gmm)
    full_pdf = zeros(length(x))

    for (k, c) ∈ enumerate(components(gmm))
        one_pdf = p[k] .* pdf.(c, x)
        if plot_components
            @series begin
                (x, one_pdf)
            end
        end
        full_pdf .+= one_pdf
    end

    (x, full_pdf)
end
