using RecipesBase

"""
    plot(est::GaussianMixtureEstimate, x_orig::Union{Missing, AbstractVector}=missing;
	    n_sigmas::Real=3, x_length::Integer=500
    )

Plot kernel density estimate.

- `x_orig` - data used to fit the model; used to plot a histogram in the background
"""
@recipe function f(est::GaussianMixtureEstimate, x_orig::Union{Missing, AbstractVector}=missing;
	n_sigmas::Real=3, x_length::Integer=500
)
	x_lo, x_hi = if x_orig === missing
		idx_lo, idx_hi = argmin(est.μ), argmax(est.μ)

		est.μ[idx_lo] - n_sigmas * est.σ[idx_lo], est.μ[idx_hi] + n_sigmas * est.σ[idx_hi]
	else
		minimum(x_orig), maximum(x_orig)
	end

	x = range(x_lo, x_hi; length=x_length)
	kde = zeros(size(x))
	est_sorted = sort(est, by=:p)

	ylims --> (0.0, Inf)

	if x_orig !== missing
		@series begin
			seriestype := :histogram
			normalize := :pdf
			label --> "Original data"
			linewidth --> 0
			seriesalpha --> 0.5

			x_orig
		end
	end

	# Plot individual components
	for (i, (p, μ, σ)) ∈ enumerate(zip(est_sorted.p, est_sorted.μ, est_sorted.σ))
		pdf_ = p .* pdf(x, μ, σ)
		kde .+= pdf_

		label = i <= 5 ? @sprintf("p=%.2e μ=% .3e σ=%.3e", p, μ, σ) : ""

		@series begin
			label := label
			fontfamily --> "monospace"

			x, pdf_
		end
	end

	# Plot full KDE
	@series begin
		label --> "KDE ($(est.algorithm))"
		fontfamily --> "monospace"
		linewidth := 2

		x, kde
	end
end


# Recepies don't support type signatures
@recipe function f(data::MovingGaussianMixture, what::Symbol=:M; shade=missing)
	@assert typeof(shade) <: Union{Missing, Symbol}
	@assert typeof(what) <: Symbol
	@assert what ∈ (:M, :Σ) "Don't know how to plot $what"
	
	mat = (what == :M) ? data.M : data.Σ
	n_components = size(data.P, 1)

	title --> "Moving Gaussian mixture\n(components: $n_components, algo: $(data.algorithm))"
	markerstrokewidth --> 0
	label --> ""

	if shade !== missing
		if shade == :P
			seriescolor --> :deep
		end

		if what == :M
			@assert shade ∈ (:P, :Σ)
			marker_z := (shade == :P) ? data.P' : data.Σ'
		elseif what == :Σ
			@assert shade ∈ (:P, :M)
			marker_z := (shade == :P) ? data.P' : data.M'
		else
			error("Got `what == $what`, which is a bug")
		end
	end

	data.dates, mat'
end
