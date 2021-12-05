module MovingMixtures

using Distributions

const AV = AbstractVector{T} where T
const AM = AbstractMatrix{T} where T

include("Settings.jl")
include("GaussianMixtures.jl")
include("Utils.jl")

end
