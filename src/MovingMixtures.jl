module MovingMixtures

export GaussianMixture, fit!, Settings

using LoopVectorization
using Distributions

const AV = AbstractVector{T} where T
const AM = AbstractMatrix{T} where T

include("Settings.jl")
include("GaussianMixtures.jl")
include("EM.jl")
include("Utils.jl")

end
