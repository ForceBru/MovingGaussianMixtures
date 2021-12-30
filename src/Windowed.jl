mutable struct Windowed{T<:Real} <: AbstractMixtureUpdate{T}
    "Mixture model instance"
    mix::GaussianMixture{T}

    "Window along the time series"
    window::Vector{T}
    unfilled_entries::UInt

    function Windowed(mix::GaussianMixture{T}, window_size::Integer, ::Type{T}=Float64) where T<:Real
        (window_size > 0) || throw(ArgumentError("Window size must be positive (got $window_size < 0)"))

        new{T}(mix, zeros(T, window_size), UInt(window_size))
    end
end

n_components(win::Windowed) = win.mix.K
get_ELBO(win::Windowed) = win.mix.history_ELBO[end]
get_params(win::Windowed) = get_params(win.mix)
has_converged(win::Windowed) = has_converged(win.mix)

function initialize!(win::Windowed, stream::AV{<:Real}; init_strategy::Settings.AbstractInitStrategy, kwargs...)
    fit!(win.mix, stream; init_strategy, kwargs...)
    for x in stream # order is important, so don't use broadcasting!
        slide_window!(win, x)
    end
    nothing
end

"""
$(TYPEDSIGNATURES)

Slide window one element in the direction of time.
"""
function slide_window!(mov::Windowed{T}, x::T) where T<:Real
    mov.window[1:end-1] .= mov.window[2:end]
    mov.window[end] = x

    if mov.unfilled_entries > 0
        mov.unfilled_entries -= 0x01
    end

    mov.window
end

function update!(
    mov::Windowed{T}, x::T;
    init_strategy::Settings.AbstractInitStrategy=Settings.InitKeepParameters(),
    kwargs...
) where T<:Real
    window = slide_window!(mov, x)

    (mov.unfilled_entries == 0) && fit!(mov.mix, window; init_strategy, kwargs...)

    if init_strategy isa Settings.InitKeepPosterior
        # Slide posterior distributions too
        mov.mix.G[:, 1:end-1] .= mov.mix.G[:, 2:end]
        mov.mix.G[:, end] .= 1/mov.mix.K
    end

    mov
end
