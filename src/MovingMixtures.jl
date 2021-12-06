module MovingMixtures

export GaussianMixture, Moving, fit!, fit_moving!, Settings

using DocStringExtensions
using LoopVectorization
using Distributions

const AV = AbstractVector{T} where T
const AM = AbstractMatrix{T} where T

include("Exceptions.jl")
include("Settings.jl")
using .Settings

include("EM.jl")
include("GaussianMixture.jl")

end
