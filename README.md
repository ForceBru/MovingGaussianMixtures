# MovingGaussianMixtures.jl

Estimate 1D finite Gaussian mixture models in Julia.

## Functions

The interface is similar to that of SciKit-Learn. To fit, for example, a Gaussian mixture model (GMM) to data, create a `GaussianMixture` and call `fit!` on it and the data.

- `GaussianMixture(K, N; warm_start=false)` represents a Gaussian mixture model with `K` components that will be fit on data of length `N`.
- `MovingGaussianMixture(K, win_size; warm_start=false)` represents a series of Gaussian mixtures with `K` components each fit on rolling windows of length `win_size` of the data.
- `KMeans(K, N; warm_start=false)` represents the state of the `K`-means algorithm used to initialize the means of the GMM.

Each model is fit using the `fit!` method. For example:

```julia
my_data = rand(200)

# Gaussian mixture model with 5 components
# whose `fit!` method expects data of length 100
gm = GaussianMixture(5, 100)

fit!(gm, my_data[1:100])
```

## Results

The results are available after a call like `fit!(model, my_data)`.

- Call the `distribution` method on a fitted `GaussianMixture` to obtain the `Distributions.jl` `UnivariateGMM` object corresponding to the fitted model.
- Call the `params` method on a fitted `MovingGaussianMixture` to obtain a struct with fields:
    - `range` - the indices of `my_data` to which each mixture corresponds
    - `K` - number of components in each mixture
    - `P`, `M` and `Σ` - `K x N` matrices; each _column_ holds the weights `p`, means `μ` and standard deviations `σ` of a mixture model fitted on the corresponding window of `my_data`

### Convergence monitoring

- Call `nconverged(model)` to see how many mixtures converged
    - Returns `0` or `1` for `GaussianMixture`, since there's only one mixture
    - Returns an integer between `0` and the number of windows for `MovingGaussianMixture`
- Call `converged_pct(model)` to see the _percentage_ of converged mixtures
    - Returns `0.0` or `100.0` for `GaussianMixture`, since there's only one mixture
    - Returns a float in range `[0.0, 100.0]` for `MovingGaussianMixture`
- Check `model.converged` to see which windows converged or failed to converge
- Check `model.n_iter` for the number of iterations till convergence

## Plotting

`UnivariateGMM` currently [cannot be plotted by `StatsPlots`](https://github.com/JuliaPlots/StatsPlots.jl/issues/448), but this package provides a simple implementation that can plot the resulting density and its individual components.

## Example

For code and sample data see [tests](test/).

### Estimation with EM

![em](test/img/mixture_em.png)

### Estimation with EM across 7 windows

```
Weights:
6×7 Matrix{Float64}:
 0.0543  0.0733  0.0587  0.0565  0.0691  0.0184  0.0181
 0.0382  0.2393  0.2309  0.181   0.1687  0.185   0.1944
 0.3608  0.2085  0.1822  0.2193  0.2325  0.2499  0.3344
 0.1143  0.086   0.0665  0.0764  0.0981  0.0899  0.0359
 0.368   0.3376  0.3625  0.3618  0.3531  0.401   0.403
 0.0644  0.0553  0.0992  0.105   0.0785  0.0559  0.0142
\______/\______/\______/
 win 1   win 2   win 3

Standard deviations:
6×7 Matrix{Float64}:
 0.006   0.007   0.0068  0.0069  0.007   0.0013  0.0013
 0.0007  0.0032  0.0035  0.0012  0.0012  0.0012  0.0015
 0.0016  0.0013  0.0013  0.0039  0.0038  0.0071  0.0077
 0.0007  0.0006  0.0007  0.0008  0.0007  0.0007  0.0001
 0.0031  0.0033  0.0034  0.0035  0.0038  0.0038  0.0037
 0.0106  0.01    0.011   0.0112  0.0106  0.0108  0.0065
```

# WARNING

Versions `0.2+` are _not_ compatible with [versions `0.1+`](https://github.com/ForceBru/MovingGaussianMixtures.jl/tree/eeac185117ac6c9ab5fbe54c046fa42dc51957fb)!