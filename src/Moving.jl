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
struct MovingGaussianMixture{T <: Real, I <:Number}
	key::MGMKey
	
	dates::Union{Vector{DateTime}, Vector{I}}
	P::Matrix{T}
	M::Matrix{T}
	Σ::Matrix{T}

	function MovingGaussianMixture(
		key::MGMKey, dates::Union{Vector{DateTime}, Vector{I}},
		P::Matrix{T}, M::Matrix{T}, Σ::Matrix{T}
	) where {T <: Real, I <: Number}
		@assert size(P) == size(M) == size(Σ)
		@assert key.k == size(P, 1)
		@assert length(dates) == size(P, 2)
		
		new{T, I}(key, dates, P, M, Σ)
	end
end

function MovingGaussianMixture(
	win_size::UInt, step_size::UInt, dates::Union{Vector{DateTime}, Vector{I}},
	P::Matrix{T}, M::Matrix{T}, Σ::Matrix{T}
) where {T <: Real, I <: Number}
	k = size(P, 1)
		
	MovingGaussianMixture(
		MGMKey(k, win_size, step_size),
		dates, P, M, Σ
	)
end

"""
    Base.write(fid::HDF5.File, data::MovingGaussianMixture{T}) where T

Write `MovingGaussianMixture` to an open HDF5 file.
"""
function Base.write(fid::HDF5.File, data::MovingGaussianMixture{T}) where T
	grp_name = repr(MIME("text/plain"), data.key)
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
	
	write(g, "dates", datetime2unix.(data.dates))
	write(g, "P", data.P)
	write(g, "M", data.M)
	write(g, "v", data.Σ)
end

"""
    Base.read(fid::HDF5.File, ::Type{MovingGaussianMixture}, group_name::Union{Missing, AbstractString}=missing)::MovingGaussianMixture

Read `MovingGaussianMixture` from an open HDF5 file
"""
function Base.read(fid::HDF5.File, ::Type{MovingGaussianMixture}, group_name::Union{Missing, AbstractString}=missing)::MovingGaussianMixture
	the_keys = keys(fid)
	if length(the_keys) == 0
		error("Empty HDF5 file!")
	end

	if group_name === missing
		group_name = the_keys[1]
	end

	g = fid[group_name]

	MovingGaussianMixture(
		parse(MGMKey, group_name),
		unix2datetime.(read(g["dates"])),
		read(g["P"]), read(g["M"]), read(g["v"])
	)
end

"""
    Base.read(fid::HDF5.File, typ::Type{MovingGaussianMixture}, group_key::MGMKey)

Read `MovingGaussianMixture` from an open HDF5 file by group key
"""
Base.read(fid::HDF5.File, typ::Type{MovingGaussianMixture}, group_key::MGMKey) = read(fid, typ, repr(group_key))

"""
    em(
		x::AbstractVector{T}, dates::AbstractVector{DateTime}, k::UInt, win_size::UInt;
		step_size::UInt=UInt(5), tol::T=3e-4, verbose::Integer=1000
    ) where T <: Real

Moving Separation of Mixtures algorithm.

- `x` - input data
- `dates` - dates at which the given data points were recorded
- `k` - number of mixture components
- `win_size` - length of the sliding window
- `step_size` - step between windows
"""
function em(
		x::AbstractVector{T}, dates::Union{AbstractVector{DateTime}, AbstractVector{I}}, k::UInt, win_size::UInt;
		step_size::UInt=UInt(5), verbose::Integer=1000, kwargs...
)::MovingGaussianMixture{T} where {T <: Real, I <: Number}
	@assert size(dates) == size(x)
	
	the_range = win_size:step_size:length(x)
	the_range_length = length(the_range)
	M = Matrix{T}(undef, k, the_range_length)
	Σ = Matrix{T}(undef, k, the_range_length)
	P = Matrix{T}(undef, k, the_range_length)

	data = GaussianMixture(x[1:win_size], k)
	
	# CANNOT parallelize this since `em!` is modifying state!
	for (i, off) ∈ enumerate(the_range)
		left, right = off - win_size + 1, off

		win = @view x[left:right]
		estimate = sort(em!(data, win; kwargs...))

		P[:, i] .= estimate.p
		M[:, i] .= estimate.μ
		Σ[:, i] .= estimate.σ

		if verbose > 0 && i % verbose == 0
			@info @sprintf("%4d / %4d (%6.2f%%)", i, the_range_length, i / the_range_length * 100)
		end
	end

	MovingGaussianMixture(win_size, step_size, dates[the_range], P, M, Σ)
end

function em(
		x::AbstractVector{T}, k::UInt, win_size::UInt;
		step_size::UInt=UInt(5), verbose::Integer=1000, kwargs...
)::MovingGaussianMixture{T} where T <: Real
	em(x, collect(1:length(x)), k, win_size; step_size=step_size, verbose=verbose, kwargs...)
end
