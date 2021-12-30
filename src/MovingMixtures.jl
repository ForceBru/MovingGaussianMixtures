module MovingMixtures

export GaussianMixture, Moving, fit!, distribution, Settings
export Online, Windowed, DynamicMixture
export cauchy_schwarz

using ProgressMeter
using DocStringExtensions
using LoopVectorization
using Distributions

const AV = AbstractVector{T} where T
const AM = AbstractMatrix{T} where T

abstract type AbstractMixtureUpdate{T<:Real} end

n_components(::AbstractMixtureUpdate)::Integer = 0

include("Exceptions.jl")
include("Settings.jl")

include("EM.jl")
include("GaussianMixture.jl")
include("Windowed.jl")
include("Online.jl")
include("DynamicMixture.jl")

include("Divergences.jl")
using .Divergences

end
