using DelimitedFiles
using Plots

using BenchmarkTools
import Random
using StatsBase # `fit!` function
using SQLite # for saving moving models

using MovingGaussianMixtures

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

# Fit regular k-means
km = KMeans(N_COMPONENTS, WIN_SIZE)
fit!(km, @view sample_data[1:WIN_SIZE])

@show km.μ

# Fit Gaussian mixture model
gm = GaussianMixture(N_COMPONENTS, length(sample_data))
fit!(gm, sample_data)

distr = distribution(gm)
x = range(minimum(sample_data), maximum(sample_data), length=500)
plt = histogram(sample_data, normalized=true, linewidth=0, alpha=.5)
log_lik = log_likelihood(distr, sample_data)
plot!(plt, distr, x, title="Log-likelihood: $(round(log_lik, digits=5))")
savefig(plt, "img/mixture_em.png")

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
