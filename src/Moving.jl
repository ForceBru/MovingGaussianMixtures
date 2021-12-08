"""
$(TYPEDEF)

Gaussian mixture models fit on a moving window.
Used for time-series analysis.

$(TYPEDFIELDS)
"""
mutable struct Moving{T<:Real, G<:GaussianMixture{T}}
    "Mixture model instance"
    mix::G

    """
    Components' weights.
    `n`th _column_ containts estimates for the `n`th moving window
    """
    P::Matrix{T}
    """
    Components' means.
    `n`th _column_ containts estimates for the `n`th moving window
    """
    M::Matrix{T}
    """
    Components' variances.
    `n`th _column_ containts estimates for the `n`th moving window
    """
    V::Matrix{T}
end

function Moving(gmm::GaussianMixture{T}) where T<:Real
    P = zeros(T, gmm.K, 1)
    M = copy(P)
    V = copy(P)

    Moving(gmm, P, M, V)
end

"""
$(TYPEDSIGNATURES)

Fit Gaussian mixture model on moving windows of width `win_size`
across given data `stream`. `kwargs` are passed to `fit!(::GaussianMixture, ...)`
"""
function fit!(
    mov::Moving{T}, stream::AV{<:Real}, win_size::Integer;
    first_init::Settings.AbstractInitStrategy=Settings.InitRandomPosterior(200),
    init_strategy::Settings.AbstractInitStrategy=Settings.InitKeepPosterior(),
    kwargs...
) where T<:Real
    N = length(stream)
    @assert win_size > 0 && win_size < N

    if N != size(mov.P, 2)
        mov.P = zeros(T, mov.mix.K, N)
        mov.M = copy(mov.P)
        mov.V = copy(mov.P)
    end

    the_range = win_size:N

    # Initialize the mixture
    window = @view stream[1:the_range[1]]
    fit!(mov.mix, window; init_strategy=first_init, kwargs...)

    @showprogress for off in the_range
        window = @view stream[off - the_range[1] + 1 : off]

        fit!(mov.mix, window; init_strategy, kwargs...)

        if init_strategy isa Settings.InitKeepPosterior
            mov.mix.G[:, 1:end-1] .= mov.mix.G[:, 2:end]
            mov.mix.G[:, end] .= 1/mov.mix.K
        end

        mov.P[:, off] .= mov.mix.p
        mov.M[:, off] .= mov.mix.mu
        mov.V[:, off] .= mov.mix.var
    end

    mov
end
