using TypedTables

export to_table

"""
```
function to_table(
    params::MovingGaussianMixtureParams{T},
    id::AbstractString, dates::Union{Missing, AbstractVector{DateType}}=missing
) where { T <: Real, DateType}
```

Return a `TypedTables.jl` table containing
moving mixture parameters and additional info.
"""
function to_table(
    params::MovingGaussianMixtureParams{T},
    id::AbstractString, dates::Union{Missing, AbstractVector{DateType}}=missing
) where { T <: Real, DateType}
    if dates === missing
        dates = collect(1:last(params.range))
    end

    @assert length(dates) ≥ last(params.range)

    win_size = params.range[1]

    TOTAL_SIZE = length(params.range) * params.K

    all_ids = fill(id, TOTAL_SIZE)
    all_dates = eltype(dates)[]
    all_win_sizes = fill(win_size, TOTAL_SIZE)
    all_k = repeat(1:params.K, length(params.range))
    all_p = zeros(T, TOTAL_SIZE)
    all_μ = zeros(T, TOTAL_SIZE)
    all_σ = zeros(T, TOTAL_SIZE)
    
    idx = 1
    for (i, t) ∈ enumerate(params.range)
        for k ∈ 1:params.K
            p, μ, σ = @inbounds (params.P[k, i], params.M[k, i], params.Σ[k, i])

            push!(all_dates, dates[t])
            all_p[idx] = p
            all_μ[idx] = μ
            all_σ[idx] = σ

            idx += 1
        end
    end

    Table(
        id=all_ids, date=all_dates, win_size=all_win_sizes,
        component_no=all_k, p=all_p, mu=all_μ, sigma=all_σ
    )
end
