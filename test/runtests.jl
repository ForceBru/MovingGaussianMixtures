using DelimitedFiles, Test
using Plots
using Distributions

using BenchmarkTools
import Random
using SQLite # for saving moving models

using MovingGaussianMixtures

function test_experimental_clearly_separable(π_true, μ_true, σ_true)
    distr = UnivariateGMM(μ_true, σ_true, Categorical(π_true))
    data = rand(distr, 1000)

    gm = MovingGaussianMixtures.Experimental.GaussianMixture(
        UInt(3), UInt(length(data))
    )
    fit!(gm, data)

    # Enforce order to ensure identifiability
    the_order = sortperm(gm.new.μ)

    (
        Estimate=(π=gm.new.π[the_order], μ=gm.new.μ[the_order], σ=gm.new.σ[the_order]),
        True=(π=π_true, μ=μ_true, σ=σ_true)
    )
end

@testset "Experimental GMM" begin
    tolerance = 0.1

    res = test_experimental_clearly_separable(
        [.3, .4, .3], [-1., 0., 1.], [.1, .2, .1]
    )
    @show res

    @test all(abs.(res.Estimate.π .- res.True.π) .< tolerance)
    @test all(abs.(res.Estimate.μ .- res.True.μ) .< tolerance)
    @test all(abs.(res.Estimate.σ .- res.True.σ) .< tolerance)
end

# Don't automatically show plots
default(show=false)
ENV["GKSwstype"]="nul"
gr()

const sample_data = readdlm("sample_data.csv")[:, 1]

const N_COMPONENTS = 7
const STEP_SIZE = 4
const WIN_SIZE = length(sample_data) - STEP_SIZE * 8
const DB_FILE = "test_save.sqlite"
const TABLE_NAME = "MovingMixture"

@assert WIN_SIZE > 0

# Fit Gaussian mixture model
begin
    gm = GaussianMixture(N_COMPONENTS, length(sample_data))
    fit!(gm, sample_data)

    distr = distribution(gm)
    x = range(minimum(sample_data), maximum(sample_data), length=500)
    plt = histogram(sample_data, normalized=true, linewidth=0, alpha=.5)
    log_lik = log_likelihood(distr, sample_data)
    plot!(plt, distr, x, title="Log-likelihood: $(round(log_lik, digits=5)); init: k-means")
    savefig(plt, "img/mixture_em.png")
end

# Fit the same model, but initialized with fuzzy C-means
let
    gm = GaussianMixture(N_COMPONENTS, length(sample_data))
    fit!(gm, sample_data, init=:fuzzy_cmeans)

    distr = distribution(gm)
    x = range(minimum(sample_data), maximum(sample_data), length=500)
    plt = histogram(sample_data, normalized=true, linewidth=0, alpha=.5)
    log_lik = log_likelihood(distr, sample_data)
    plot!(plt, distr, x, title="Log-likelihood: $(round(log_lik, digits=5)); init: C-means")
    savefig(plt, "img/mixture_em_cmeans.png")
end

@show rand(distr)

@show predict(distr, sample_data[1:10])

@info "Benchmarking..."
bench_res = let
    Random.seed!(42)
    data = rand(distr, 1000)
    gm = GaussianMixture(N_COMPONENTS, 1000)

    @benchmark fit!($gm, $data)
end
display(bench_res)
println()

# Fit moving Gaussian mixture
mgm = MovingGaussianMixture(N_COMPONENTS, WIN_SIZE, STEP_SIZE)
fit!(mgm, sample_data)

@show converged_pct(mgm)

par = MovingGaussianMixtures.params(mgm)
@show size(par.P)

@info "Saving moving model to $DB_FILE ..."
let
    table_data = to_table(par, "my_time_series")
    println(table_data)
    rm(DB_FILE, force=true)
    conn = SQLite.DB(DB_FILE)
    SQLite.load!(table_data, conn, TABLE_NAME)
end
@info "Model saved!"

println("\nWeights:")
display(round.(par.P, digits=4))

println("\n\nStandard deviations:")
display(round.(par.Σ, digits=4))
println()

savefig(
    plot(par.M', title="Moving means of components"),
    "img/moving_mixture.png"
)
