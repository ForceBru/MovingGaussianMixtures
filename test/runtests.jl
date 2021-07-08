using DelimitedFiles
using Plots

using MovingGaussianMixtures

const sample_data = readdlm("sample_data.csv")[:, 1]

const N_COMPONENTS = 6
const STEP_SIZE = 5
const WIN_SIZE = length(sample_data) - STEP_SIZE * 6

@assert WIN_SIZE > 0

# Fit regular k-means
km = KMeans(N_COMPONENTS, WIN_SIZE)
fit!(km, @view sample_data[1:WIN_SIZE])

@show km.μ

# Fit Gaussian mixture model
gm = GaussianMixture(N_COMPONENTS, WIN_SIZE)
fit!(gm, @view sample_data[1:WIN_SIZE])

distr = distribution(gm)
x = range(minimum(sample_data), maximum(sample_data), length=500)
savefig(plot(distr, x), "img/mixture_em.png")

@show rand(distr)

@show predict(distr, sample_data[1:10])

# Fit moving Gaussian mixture
mgm = MovingGaussianMixture(N_COMPONENTS, WIN_SIZE, STEP_SIZE)
fit!(mgm, sample_data)

par = params(mgm)
@show size(par.P)

println("\nWeights:")
display(round.(par.P, digits=4))

println("\n\nStandard deviations:")
display(round.(par.Σ, digits=4))
println()
