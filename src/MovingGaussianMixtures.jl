module MovingGaussianMixtures

export GaussianMixture, Moving, fit!, distribution, Settings
export Online, Windowed, DynamicMixture
export n_components, get_ELBO, get_params, has_converged
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

precompile(fit!, (GaussianMixture{Float64}, Vector{Float64}))

end
