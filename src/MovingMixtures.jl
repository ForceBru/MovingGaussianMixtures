module MovingMixtures

export GaussianMixture, Moving, fit!, distribution, Settings
export cauchy_schwarz

using ProgressMeter
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

include("Divergences.jl")
using .Divergences

end
