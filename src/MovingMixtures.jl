module MovingMixtures

export GaussianMixture, Moving, fit!, distribution, Settings

using DocStringExtensions
using LoopVectorization
using Distributions

const AV = AbstractVector{T} where T
const AM = AbstractMatrix{T} where T

include("Exceptions.jl")
include("Settings.jl")

include("EM.jl")
include("GaussianMixture.jl")
include("Moving.jl")

end
