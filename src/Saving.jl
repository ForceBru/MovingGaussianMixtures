using TypedTables

export to_table

"""
```
function to_table(
    params::MovingGaussianMixtureParams{T}, series_name::AbstractString,
    timestamps::Union{Missing, AbstractVector{DateType}}=missing
) where { T <: Real, DateType}
```

Return a `TypedTables.jl` table containing
moving mixture parameters and additional info.

Table columns:
- `series_name` - same as the `series_name` parameter
- `timestamp` - timestamps from the `timestamps` parameter
- `window_size` - width of the rolling window
- `n_components` - number of components in this mixture
- `component_id` - number of the current component (`component_id ∈ 1:n_components`)
- `p`, `mu` and `sigma` - weight `p`, mean `μ` and standard deviation `σ` for
mixture component with the current `component_id` at the current `timestamp`
for the given `series_name`

The resulting table can be saved to a CSV file, a database etc.
"""
function to_table(
    params::MovingGaussianMixtureParams{T}, series_name::AbstractString,
    timestamps::Union{Missing, AbstractVector{DateType}}=missing
) where { T <: Real, DateType}
    if timestamps === missing
        timestamps = collect(1:last(params.range))
    end

    @assert length(timestamps) ≥ last(params.range)

    win_size = params.range[1]

    TOTAL_SIZE = length(params.range) * params.K

    all_series_names = fill(series_name, TOTAL_SIZE)
    all_timestamps = eltype(timestamps)[]
    all_win_sizes = fill(win_size, TOTAL_SIZE)
    all_n_components = fill(params.K, TOTAL_SIZE)
    all_k = repeat(1:params.K, length(params.range))
    all_p = zeros(T, TOTAL_SIZE)
    all_μ = zeros(T, TOTAL_SIZE)
    all_σ = zeros(T, TOTAL_SIZE)
    
    idx = 1
    for (i, t) ∈ enumerate(params.range)
        for k ∈ 1:params.K
            p, μ, σ = @inbounds (params.P[k, i], params.M[k, i], params.Σ[k, i])

            push!(all_timestamps, timestamps[t])
            all_p[idx] = p
            all_μ[idx] = μ
            all_σ[idx] = σ

            idx += 1
        end
    end

    Table(
        series_name=all_series_names, timestamp=all_timestamps,
        window_size=all_win_sizes, n_components=all_n_components,
        component_id=all_k, p=all_p, mu=all_μ, sigma=all_σ
    )
end
