export StandardScaler, transform, inverse_transform

using StatsBase: params
using Distributions: UnivariateGMM

"""
Scales data to have zero mean and unit variance.
Stores original mean `μ` and standard deviation `σ`.
"""
struct StandardScaler{T <: Real}
	μ::T
	σ::T
	
	function StandardScaler(μ::T, σ::T) where T <: Real
		@assert σ > zero(T)
		
		new{T}(μ, σ)
	end
	
    """
    ```
    function StandardScaler(x::AbstractVector{T}) where T <: Real
    ```

    Initialize `StandardScaler` with the mean and
    standard deviation of the given vector
    """
	function StandardScaler(x::AbstractVector{T}) where T <: Real
		new{T}(mean(x), std(x))
	end
end

"""
    transform(sc::StandardScaler, x::Number)

Scale `x` using mean and standard deviation of `sc`.
"""
transform(sc::StandardScaler, x::Number) = (x - sc.μ) / sc.σ

transform(sc::StandardScaler, x::AbstractVector) = transform.(Ref(sc), x)

"""
    inverse_transform(sc::StandardScaler, x::Number)

Perform the inverse of `transform`.
"""
inverse_transform(sc::StandardScaler, x::Number) = x * sc.σ + sc.μ

inverse_transform(sc::StandardScaler, x::AbstractVector) = inverse_transform.(Ref(sc), x)

"""
    inverse_transform(scaler::StandardScaler, gmm::UnivariateGMM)

Assume that the given `UnivariateGMM` was obtained from data
transformed by `scaler`. Thus, these parameters are for the distribution
of that _scaled_ data.

"Undo" the scaling and return the `UnivariateGMM` of
the original unscaled data.
"""
function inverse_transform(scaler::StandardScaler, gmm::UnivariateGMM)
	μ, σ, cat = params(gmm)

	UnivariateGMM(
		inverse_transform(scaler, μ),
		σ .* scaler.σ,
		cat
	)
end
