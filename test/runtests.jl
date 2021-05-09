using DelimitedFiles
using Test

using Plots, HDF5
using BenchmarkTools

using MovingGaussianMixtures

sample_data = readdlm("sample_data.csv")[:, 1]

# Don't automatically show plots
default(show=false)
ENV["GKSwstype"]="nul"
gr()

const PLOT_SIZE = (800 * 1.5, 800)
const N_COMPONENTS = UInt(7)
const WINDOW_SIZE = UInt(10)
const OUT_FILE = "sample_results.h5"

@testset verbose=true "Basic computations" begin
	@test let
		@info "Estimating with k-means..."
		ret_kmeans = kmeans(sample_data, N_COMPONENTS)
		lik = log_likelihood(sample_data, ret_kmeans)
		plt = plot(ret_kmeans, sample_data, size=PLOT_SIZE, title="Log-likelihood: $lik")
		savefig(plt, "img/mixture_kmeans.png")
		true
	end

	@test let
		@info "Estimating with EM..."
		ret_em = em(sample_data, N_COMPONENTS)
		lik = log_likelihood(sample_data, ret_em)
		plt = plot(ret_em, sample_data, size=PLOT_SIZE, title="Log-likelihood: $lik")
		savefig(plt, "img/mixture_em.png")
		true
	end

	@test let
		@info "Running MGM (kmeans)..."
		ret_mov_kmeans = moving_kmeans(sample_data, N_COMPONENTS, WINDOW_SIZE, step_size=UInt(1))
		plt = scatter(ret_mov_kmeans, markersize=4, alpha=.5, size=PLOT_SIZE)
		savefig(plt, "img/running_kmeans.png")
		true
	end

	@test let
		@info "Running MGM (EM)..."
		ret_mov_em = moving_em(sample_data, N_COMPONENTS, WINDOW_SIZE, step_size=UInt(1))
		plt = scatter(ret_mov_em, markersize=4, alpha=.5, size=PLOT_SIZE)
		savefig(plt, "img/running_em.png")
		true
	end
end

@testset verbose=true "Saving/reading" begin
	@test let
		ret_mov_em = moving_em(sample_data, N_COMPONENTS, WINDOW_SIZE, step_size=UInt(1))
		@info "Saving moving EM data to $OUT_FILE..."
		h5open(OUT_FILE, "w") do fid
			write(fid, ret_mov_em)
		end
		true
	end

	@test begin
		@info "Reading data from $OUT_FILE..."
		ret_read = h5open(OUT_FILE, "r") do fid
			read(fid, MovingGaussianMixtures.MovingGaussianMixture, parse_dates=false)
		end
		true
	end

	@test let
		@info "Plotting data from $OUT_FILE..."
		ret_read = h5open(OUT_FILE, "r") do fid
			read(fid, MovingGaussianMixtures.MovingGaussianMixture, parse_dates=false)
		end
		ret_mov_kmeans = moving_kmeans(sample_data, N_COMPONENTS, WINDOW_SIZE, step_size=UInt(1))
		plt_em = scatter(ret_read, markersize=2, alpha=.8, shade=:P)
		plt_kmeans = scatter(ret_mov_kmeans, markersize=2, alpha=.8, shade=:P)
		savefig(plot(plt_em, plt_kmeans, size=PLOT_SIZE), "img/running_shaded.png")
		true
	end
end

@testset verbose=true "Benchmarks" begin
	@test let
		@info "Benchmarking EM..."
		data = GaussianMixture(sample_data, UInt(20))
		@show Int(em!(data, sample_data).n_iter)
		b_data = @benchmark em!($data, $sample_data)
		show(stdout, "text/plain", b_data)
		println()
		true
	end
end
