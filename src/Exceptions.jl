abstract type MixException <: Exception end

struct ZeroVarianceException{T<:Real} <: MixException
    variances::Vector{T}

    function ZeroVarianceException(var::AbstractVector{T}) where T<:Real
        new{T}(copy(var))
    end
end

function Base.showerror(io::IO, exc::ZeroVarianceException)
    vars_rounded = round.(exc.variances, digits=3)
    print(io, "some of the variances are too close to zero: $vars_rounded")
end

struct ZeroNormalizationException <: MixException end

function Base.showerror(io::IO, exc::ZeroNormalizationException)
    print(io, "a normalization constant is zero")
end

struct InvalidMinPosteriorProbException{T<:Real} <: MixException
    eps::T
    K::Integer
end

function Base.showerror(io::IO, exc::InvalidMinPosteriorProbException)
    print(io, "minimum posterior probability must be in range (0, $(exc.K)), got $(exc.eps)")
end
