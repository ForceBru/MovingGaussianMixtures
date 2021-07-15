import DBInterface

export save_sql

"""
```
function save_sql(
    conn, table_name::AbstractString,
    params::MovingGaussianMixtureParams{T},
    id::AbstractString, dates::Union{Missing, AbstractVector{DateType}}=missing
) where { T <: Real, DateType}
```

Save moving mixture parameters to an SQL database.

Required structure of table `table_name`:
1. [String] - for `id`
2. [DateType] - for elements of `dates`
3. [Integer] - for window size
4. [Integer] - for number of component (∈ 1:params.K)
5. [T] - for weights
6. [T] - for means
7. [T] - for standard deviations

So, exactly 7 columns in this particular order.
"""
function save_sql(
    conn, table_name::AbstractString,
    params::MovingGaussianMixtureParams{T},
    id::AbstractString, dates::Union{Missing, AbstractVector{DateType}}=missing
) where { T <: Real, DateType}
    if dates === missing
        dates = collect(1:last(params.range))
    end

    N = length(dates)

    @assert length(dates) ≥ last(params.range)

    win_size = params.range[1]
    stmt = DBInterface.prepare(conn, "INSERT INTO $table_name VALUES (:id, :date, :win_size, :component_no, :p, :μ, :σ)")

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

    DBInterface.executemany(stmt, (
        id=all_ids, date=all_dates, win_size=all_win_sizes,
        component_no=all_k, p=all_p, μ=all_μ, σ=all_σ
    ))
    
    DBInterface.close!(stmt)
end
