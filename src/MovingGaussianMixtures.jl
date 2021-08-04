module MovingGaussianMixtures

export ClusteringModel, GaussianMixture, MovingGaussianMixture
export distribution, params # obtaining results
export nconverged, converged_pct # convergence monitoring
export fit!, predict, predict_proba
export log_likelihood, logpdf # reexport from Distributions
export StandardScaler, transform, inverse_transform # scaling data

# Statistics
using Statistics
import Clustering
# Must use `import` because these functions will be overriden
import StatsBase: StatisticalModel, params, fit!, predict
using Distributions

# Speed
using LoopVectorization

# User interface
using ProgressMeter


"""
    ClusteringModel{T <: Real} <: StatisticalModel

Abstract type of clustering models (`KMeans`, `GaussianMixture`, `MovingGaussianMixture`, etc)

- `T` is the type of data being processed (usually `AbstractFloat`)
- `U` is the type of the number of components (the default is something small like `UInt8`)
"""
abstract type ClusteringModel{T <: Real} <: StatisticalModel end
abstract type AbstractGaussianMixture{T} <: ClusteringModel{T} end

include("Utils.jl")

include("GaussianMixture.jl")
include("Moving.jl")
include("Experimental.jl")

include("Plot.jl")
include("Saving.jl")

end # module
