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
 0.0551  0.0729  0.0633  0.0669  0.0694  0.0593  0.0522
 0.1934  0.2342  0.2349  0.1817  0.1691  0.1628  0.1354
 0.2292  0.2131  0.1802  0.2258  0.2329  0.2592  0.2766
 0.1092  0.084   0.0701  0.0763  0.0978  0.0992  0.0967
 0.3524  0.3392  0.3599  0.3574  0.3531  0.3554  0.3724
 0.0608  0.0566  0.0916  0.0919  0.0776  0.064   0.0668
\______/\______/\______/
 win 1   win 2   win 3

Standard deviations:
6×7 Matrix{Float64}:
 0.0061  0.0061  0.0061  0.0061  0.0061  0.0061  0.0061
 0.0011  0.0011  0.0011  0.0011  0.0011  0.0011  0.0011
 0.0041  0.0041  0.0041  0.0041  0.0041  0.0041  0.0041
 0.0006  0.0006  0.0006  0.0006  0.0006  0.0006  0.0006
 0.0041  0.0041  0.0041  0.0041  0.0041  0.0041  0.0041
 0.0104  0.0104  0.0104  0.0104  0.0104  0.0104  0.0104
```

# WARNING

Versions `0.2+` are _not_ compatible with [versions `0.1+`](https://github.com/ForceBru/MovingGaussianMixtures.jl/tree/eeac185117ac6c9ab5fbe54c046fa42dc51957fb)!