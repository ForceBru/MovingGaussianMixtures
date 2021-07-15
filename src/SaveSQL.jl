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
    
    for (i, t) ∈ enumerate(params.range)
        for k ∈ 1:params.K
            p, μ, σ = @inbounds (params.P[k, i], params.M[k, i], params.Σ[k, i])
            DBInterface.execute(stmt, [id, dates[t], win_size, k, p, μ, σ])
        end
    end
    
    DBInterface.close!(stmt)
end
