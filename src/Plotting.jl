"Plot kernel density estimate"
function Plots.plot(
	est::GaussianMixtureEstimate{k}, x_orig::Union{Missing, AbstractVector}=missing;
	n_sigmas::Real=3, x_length::Integer=500,
	title::Union{Missing, AbstractString}=missing, bins=50,
	alpha::Real=.5, ylims=(0.0, Inf)
) where k
	@assert 0 <= alpha <= 1

	x_lo, x_hi = if x_orig === missing
		idx_lo, idx_hi = argmin(est.μ), argmax(est.μ)

		est.μ[idx_lo] - n_sigmas * est.σ[idx_lo], est.μ[idx_hi] + n_sigmas * est.σ[idx_hi]
	else
		minimum(x_orig), maximum(x_orig)
	end

	x = range(x_lo, x_hi; length=x_length)
	kde = zeros(size(x))
	est_sorted = sort(est, by=:p)

	if title === missing
		title = "Gaussian mixture with $(Int(k)) components"
	end

	plt = if x_orig === missing
		Plots.plot(title=title)
	else
		Plots.histogram(
			x_orig, normalize=:pdf, bins=bins,
			title=title, label="Original data",
			linewidth=0, alpha=alpha,
			fontfamily="monospace"
		)
	end

	# Plot individual components
	for (i, (p, μ, σ)) ∈ enumerate(zip(est_sorted.p, est_sorted.μ, est_sorted.σ))
		pdf_ = p .* pdf(x, μ, σ)
		kde .+= pdf_

		label = i <= 5 ? @sprintf("p=%.2f μ=% .3f σ=%.3f", p, μ, σ) : ""

		Plots.plot!(plt, x, pdf_, label=label, ylims=ylims, fontfamily="monospace")
	end

	# Plot full KDE
	Plots.plot!(plt, x, kde, label="KDE", ylims=ylims, linewidth=2, legend=true, fontfamily="monospace")
end


function Plots.scatter(
	data::MovingGaussianMixture, what=:M;
	title::Union{Missing, AbstractString}=missing, markersize=.5, seriescolor=:deep,
	shade=false, kwargs...
	)
	@assert what ∈ (:M, :Σ) "Don't know how to plot $what"
	
	mat = (what == :M) ? data.M : data.Σ
	
	if title === missing
		title = "Moving gaussian mixture $(repr(data.key))"
	end
	
	Plots.scatter(
		data.dates, mat';
		label="", title=title,
		markerstrokewidth=0, markerstrokealpha=0, markersize=markersize,
		marker_z=(shade ? data.P' : nothing),
		seriescolor=(shade ? seriescolor : :auto),
		kwargs...
	)
end
