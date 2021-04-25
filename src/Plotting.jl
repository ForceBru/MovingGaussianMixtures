"Plot kernel density estimate"
function Plots.plot(
	est::GaussianMixtureEstimate{k, T}, x_orig::Union{Missing, AbstractVector}=missing;
	n_sigmas::Real=3, x_length=500
) where {k, T <: Real}
	x_lo, x_hi = let
		idx_lo, idx_hi = argmin(est.μ), argmax(est.μ)

		est.μ[idx_lo] - n_sigmas * est.σ[idx_lo], est.μ[idx_hi] + n_sigmas * est.σ[idx_hi]
	end

	x = range(x_lo, x_hi; length=x_length)
	kde = zeros(size(x))
	est_sorted = sort(est, by=:p)

	title = "Gaussian mixture with $(Int(k)) components"
	plt = if x_orig === missing
		Plots.plot(title=title)
	else
		Plots.histogram(
			x_orig, normalize=:pdf,
			title=title, label="Original data",
			linewidth=0, alpha=.5
		)
	end

	# Plot individual components
	for (i, (p, μ, σ)) ∈ enumerate(zip(est_sorted.p, est_sorted.μ, est_sorted.σ))
		pdf_ = p .* pdf(x, μ, σ)
		kde .+= pdf_

		label = i <= 5 ? @sprintf("p=%.2f μ=%+.3f σ=%.3f", p, μ, σ) : ""

		Plots.plot!(plt, x, pdf_, label=label, ylims=(0.0, Inf))
	end

	# Plot full KDE
	Plots.plot!(plt, x, kde, label="KDE", ylims=(0.0, Inf), linewidth=2)
end


function Plots.scatter(data::MovingGaussianMixture, what=:M; markersize=.5, shade=false, kwargs...)
	@assert what ∈ (:M, :Σ) "Don't know how to plot $what"
	
	mat = (what == :M) ? data.M : data.Σ
	
	title = "Moving gaussian mixture $(repr(data.key))"
	
	Plots.scatter(
		data.dates, mat';
		label="", title=title,
		markerstrokewidth=0, markerstrokealpha=0, markersize=markersize,
		marker_z=(shade ? data.P' : nothing),
		seriescolor=(shade ? :deep : :auto),
		kwargs...
	)
end
