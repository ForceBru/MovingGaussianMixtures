using DelimitedFiles
using Plots, HDF5

import MovingGaussianMixtures

sample_data = readdlm("sample_data.csv")[:, 1]

# Don't automatically show plots
default(show=false)
const DISPLAY = false

function do_display(last=false)
	display(plt)

	if ! last
		print("Hit ENTER for next plot: ")
		readline()
	end
end

const N_COMPONENTS = UInt(7)
const WINDOW_SIZE = UInt(10)
const OUT_FILE = "sample_results.h5"

@info "Estimating with k-means..."
ret_kmeans = MovingGaussianMixtures.kmeans(sample_data, N_COMPONENTS)
plt = plot(ret_kmeans, sample_data)
savefig(plt, "img/mixture_kmeans.png")

DISPLAY && do_display()

@info "Estimating with EM..."
ret_em = MovingGaussianMixtures.em(sample_data, N_COMPONENTS)
plt = plot(ret_em, sample_data)
savefig(plt, "img/mixture_em.png")

DISPLAY && do_display()

@info "Running MGM (kmeans)..."
ret_mov_kmeans = MovingGaussianMixtures.moving_kmeans(sample_data, N_COMPONENTS, WINDOW_SIZE, step_size=UInt(1))
plt = scatter(ret_mov_kmeans, markersize=2, alpha=.5)
savefig(plt, "img/running_kmeans.png")

plt = scatter(ret_mov_kmeans, markersize=2, alpha=.8, shade=true)
savefig(plt, "img/running_kmeans_shaded.png")

@info "Running MGM (EM)..."
ret_mov_em = MovingGaussianMixtures.moving_em(sample_data, N_COMPONENTS, WINDOW_SIZE, step_size=UInt(1))
plt = scatter(ret_mov_em, markersize=2, alpha=.5)
savefig(plt, "img/running_em.png")

@info "Saving moving EM data to $OUT_FILE..."
h5open(OUT_FILE, "w") do fid
	write(fid, ret_mov_em)
end
@info "Data saved!"

@info "Reading data from $OUT_FILE..."
ret_read = h5open(OUT_FILE, "r") do fid
	read(fid, MovingGaussianMixtures.MovingGaussianMixture, parse_dates=false)
end
@info "Data read!"

@info "Plotting data from $OUT_FILE..."
plt = scatter(ret_read, markersize=2, alpha=.8, shade=true)
savefig(plt, "img/running_em_shaded.png")

DISPLAY && do_display(true)

@info "ALL DONE!"

