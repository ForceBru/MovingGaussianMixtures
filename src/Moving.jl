export MovingGaussianMixture, MGMKey, moving_em, moving_kmeans

using Dates, HDF5

"""
    MGMKey(k::Integer, w::Integer, s::Integer)

The identifier of current run of
the Moving Gaussian Mixture model (MGM)
"""
struct MGMKey
	# Number of components
	k::UInt8
	
	# Length of the sliding window
	win_size::UInt16

	# Size of step between windows
	step_size::UInt8
	
	function MGMKey(k::Integer, w::Integer, s::Integer)
		k, w, s = UInt8(k), UInt16(w), UInt8(s)
		@assert k > 0
		@assert w > 0
		@assert s > 0
		
		new(k, w, s)
	end
end

# Internal
Base.repr(key::MGMKey) = "k:$(key.k) w:$(key.win_size) s:$(key.step_size)"
Base.repr(::MIME, key::MGMKey) = repr(key)

function Base.parse(::Type{MGMKey}, str)::MGMKey
	kws = Dict(
		map(arr -> (arr[1], parse(UInt, arr[2])), split.(split(str), ':'))
	)
	
	MGMKey(kws["k"], kws["w"], kws["s"])
end

"""
Struct that holds the MGM results.

- `key` - `MGMKey` object
- `dates :: Vector{DateTime}` - dates the data apply to
- `P :: Matrix{T}` - weights for each mixture component
- `M :: Matrix{T}` - means for each component
- `Σ :: Matrix{T}` - standard deviations for each component

`P`, `M` and `Σ` are (`n_components` by `length(dates)`) matrices with columns stored in the same order as the `dates`.
"""
struct MovingGaussianMixture{T <: Real, D <: Union{DateTime, Number}}
	algorithm::String
	key::MGMKey
	
	dates::Vector{D}
	P::Matrix{T}
	M::Matrix{T}
	Σ::Matrix{T}

	function MovingGaussianMixture(
		algo::String,
		key::MGMKey, dates::Vector{D},
		P::Matrix{T}, M::Matrix{T}, Σ::Matrix{T}
	) where {T <: Real, D <: Union{DateTime, Number}}
		@assert size(P) == size(M) == size(Σ)
		@assert key.k == size(P, 1)
		@assert length(dates) == size(P, 2)
		
		new{T, D}(algo, key, dates, P, M, Σ)
	end
end

function MovingGaussianMixture(
	algo::String,
	win_size::UInt, step_size::UInt, dates::Vector{D},
	P::Matrix{T}, M::Matrix{T}, Σ::Matrix{T}
) where {T <: Real, D <: Union{DateTime, Number}}
	k = size(P, 1)
		
	MovingGaussianMixture(
		algo,
		MGMKey(k, win_size, step_size),
		dates, P, M, Σ
	)
end

function _write_data(
	fid::HDF5.File,
	grp_name::AbstractString, algo::String, dates::AbstractVector{T},
	P::AbstractMatrix, M::AbstractMatrix, Σ::AbstractMatrix
) where T <: Union{Number, AbstractString}
	g = try
		create_group(fid, grp_name)
	catch e
		if !(e isa ErrorException)
			throw(e)
		end
		@warn "Overwriting group '$grp_name'"
		delete_object(fid, grp_name)
		create_group(fid, grp_name)
	end
	
	write(g, "algorithm", algo)
	write(g, "dates", dates)
	write(g, "P", P)
	write(g, "M", M)
	write(g, "v", Σ)
end

"""
    Base.write(fid::HDF5.File, data::MovingGaussianMixture{T}) where T

Write `MovingGaussianMixture` to an open HDF5 file.
"""
function Base.write(fid::HDF5.File, data::MovingGaussianMixture{T, D}) where {T, D <: DateTime}
	grp_name = repr(MIME("text/plain"), data.key)
	_write_data(fid, grp_name, data.algorithm, datetime2unix.(data.dates), data.P, data.M, data.Σ)
end

function Base.write(fid::HDF5.File, data::MovingGaussianMixture)
	grp_name = repr(MIME("text/plain"), data.key)
	_write_data(fid, grp_name, data.algorithm, data.dates, data.P, data.M, data.Σ)
end

"""
    Base.read(fid::HDF5.File, ::Type{MovingGaussianMixture}, group_name::Union{Missing, AbstractString}=missing; parse_dates=true)::MovingGaussianMixture

Read `MovingGaussianMixture` from an open HDF5 file
"""
function Base.read(
	fid::HDF5.File, ::Type{MovingGaussianMixture},
	group_name::Union{Missing, AbstractString}=missing; parse_dates=true
)::MovingGaussianMixture
	the_keys = keys(fid)
	if length(the_keys) == 0
		error("Empty HDF5 file!")
	end

	if group_name === missing
		group_name = the_keys[1]
	end

	g = fid[group_name]
	algo_name = try
		read(g["algorithm"])
	catch x
		"<UNKNOWN ALGORITHM>"
	end

	MovingGaussianMixture(
		algo_name,
		parse(MGMKey, group_name),
		parse_dates ? unix2datetime.(read(g["dates"])) : read(g["dates"]),
		read(g["P"]), read(g["M"]), read(g["v"])
	)
end

"""
    Base.read(fid::HDF5.File, typ::Type{MovingGaussianMixture}, group_key::MGMKey)

Read `MovingGaussianMixture` from an open HDF5 file by group key
"""
Base.read(fid::HDF5.File, typ::Type{MovingGaussianMixture}, group_key::MGMKey) = read(fid, typ, repr(group_key))

"Moving Separation of Mixtures algorithm"
function run_MSM(
	estimator!,
	x::AbstractVector{T}, dates::AbstractVector{D}, k::UInt, win_size::UInt,
	step_size::UInt, verbose::Integer; reinit_kmeans::Bool=false, kwargs...
)::MovingGaussianMixture{T} where {T <: Real, D <: Union{DateTime, Number}}
	@assert size(dates) == size(x)
	
	the_range = win_size:step_size:length(x)
	the_range_length = length(the_range)
	M = Matrix{T}(undef, k, the_range_length)
	Σ = Matrix{T}(undef, k, the_range_length)
	P = Matrix{T}(undef, k, the_range_length)
	algo_name = ""

	data = GaussianMixture(x[1:win_size], k)
	init_kmeans = true
	
	# CANNOT parallelize this since `estimator!` is modifying state!
	for (i, off) ∈ enumerate(the_range)
		left, right = off - win_size + 1, off

		win = @view x[left:right]
		estimate = sort(estimator!(data, win; init_kmeans=init_kmeans, kwargs...))
		if !reinit_kmeans
			init_kmeans = false
		end

		algo_name = estimate.algorithm

		P[:, i] .= estimate.p
		M[:, i] .= estimate.μ
		Σ[:, i] .= estimate.σ

		if verbose > 0 && i % verbose == 0
			@info @sprintf("%4d / %4d (%6.2f%%)", i, the_range_length, i / the_range_length * 100)
		end
	end

	MovingGaussianMixture(algo_name, win_size, step_size, dates[the_range], P, M, Σ)
end

"""
    moving_em(
		x::AbstractVector{T}, dates::AbstractVector{D}, k::UInt, win_size::UInt;
		step_size::UInt=UInt(5), verbose::Integer=1000, kwargs...
    ) where {T <: Real, D <: Union{DateTime, Number}}

Moving Separation of Mixtures algorithm using EM.

- `x` - input data
- `dates` - dates at which the given data points were recorded
- `k` - number of mixture components
- `win_size` - length of the sliding window
- `step_size` - step between windows
- `kwargs...` - passed to `em!`
"""
function moving_em(
		x::AbstractVector{T}, dates::AbstractVector{D}, k::UInt, win_size::UInt;
		step_size::UInt=UInt(5), verbose::Integer=1000, kwargs...
) where {T <: Real, D <: Union{DateTime, Number}}
	run_MSM(em!, x, dates, k, win_size, step_size, verbose; kwargs...)
end

"""
    moving_kmeans(
		x::AbstractVector{T}, dates::AbstractVector{D}, k::UInt, win_size::UInt;
		step_size::UInt=UInt(5), verbose::Integer=1000, kwargs...
    ) where {T <: Real, D <: Union{DateTime, Number}}

Moving Separation of Mixtures algorithm using EM.

- `x` - input data
- `dates` - dates at which the given data points were recorded
- `k` - number of mixture components
- `win_size` - length of the sliding window
- `step_size` - step between windows
- `kwargs...` - passed to `kmeans!`
"""
function moving_kmeans(
		x::AbstractVector{T}, dates::AbstractVector{D}, k::UInt, win_size::UInt;
		step_size::UInt=UInt(5), verbose::Integer=1000, kwargs...
) where {T <: Real, D <: Union{DateTime, Number}}
	run_MSM(kmeans!, x, dates, k, win_size, step_size, verbose; kwargs...)
end

function moving_em(
		x::AbstractVector{T}, k::UInt, win_size::UInt; kwargs...
)::MovingGaussianMixture{T} where T <: Real
	moving_em(x, collect(1:length(x)), k, win_size; kwargs...)
end

function moving_kmeans(
		x::AbstractVector{T}, k::UInt, win_size::UInt; kwargs...
)::MovingGaussianMixture{T} where T <: Real
	moving_kmeans(x, collect(1:length(x)), k, win_size; kwargs...)
end
