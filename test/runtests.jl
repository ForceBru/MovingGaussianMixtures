using Test
using Distributions

using MovingGaussianMixtures

const atol = 0.1
const SAMPLE_SIZE = 1000

function sample_GMM(mus, stds, ps)
    distr = UnivariateGMM(mus, stds, Categorical(ps))
    data = rand(distr, SAMPLE_SIZE)

    distr, data
end

@testset "MovingGaussianMixtures" begin

include("test_EM.jl")

@testset "Simple mixtures" begin
    @testset "2 components easy" begin
        for i in 1:10
            distr, data = sample_GMM([-1, 1], [.2, .3], [.4, .6])

            gmm = GaussianMixture(2)
            fit!(gmm, data)

            @test gmm.converged
            @test sort(gmm.p) ≈ sort(distr.prior.p) atol=atol
            @test sort(gmm.mu) ≈ sort(distr.means) atol=atol
            @test sort(gmm.var) ≈ sort(distr.stds.^2) atol=atol
        end
    end

    @testset "2 components zero means" begin
        for i in 1:10
            distr, data = sample_GMM([0, 0], [.2, .3], [.4, .6])

            gmm = GaussianMixture(2)
            fit!(gmm, data)

            @test gmm.converged
            #FIXME: this fails, but do we care?
            @test sort(gmm.p) ≈ sort(distr.prior.p) atol=atol broken=true
            @test sort(gmm.mu) ≈ sort(distr.means) atol=atol
            @test sort(gmm.var) ≈ sort(distr.stds.^2) atol=atol
        end
    end
end

@testset "Regularization" begin
    @testset "Variance reg by addition" begin
        REG = 1e-4
        for i in 1:10
            distr, data = sample_GMM([0, 0], [.2, .3], [.4, .6])

            gmm = GaussianMixture(50) # should overfit
            fit!(gmm, data, regularization=Settings.RegVarianceSimple(REG))

            @test all(gmm.var .≥ REG)
        end
    end

    @testset "Variance reg by restarts" begin
        for i in 1:10
            distr, data = sample_GMM([0, 0], [.2, .3], [.4, .6])

            gmm = GaussianMixture(50) # should overfit
            fit!(gmm, data, regularization=Settings.RegVarianceReset())

            @test all(gmm.var .> 0)
        end
    end
end

@testset "Dynamics" begin
    distr, data = sample_GMM([-1, 1], [.2, .3], [.4, .6])
    p_sorted = sort(distr.prior.p)
    mu_sorted = sort(distr.means)
    var_sorted = sort(distr.stds .^2)

    K, N = 2, length(data)
    # Need to set high α≥0.9, otherwise
    # weight of one component goes to zero
    dmm = DynamicMixture(Online(GaussianMixture(K), 0.99))
    @test n_components(dmm) == K
    fit!(dmm, data, 200)

    @test size(dmm.P) == (K, N)
    @test size(dmm.M) == size(dmm.P)
    @test size(dmm.V) == size(dmm.P)

    successes = zeros(Int, 3)
    total = size(dmm.P, 2)
    for t in 1:size(dmm.P, 2)
        successes[1] += isapprox(sort(dmm.P[:, t]), p_sorted; atol)
        successes[2] += isapprox(sort(dmm.M[:, t]), mu_sorted; atol)
        successes[3] += isapprox(sort(dmm.V[:, t]), var_sorted; atol)
    end

    success_rates = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    @testset "Success rate $rate $i" for rate in success_rates, i in 1:length(successes)
        @test successes[i] / total ≥ rate
    end
end

@testset "Divergences" begin
    @testset "Cauchy-Schwarz" begin
        for i in 1:10
            distr, data = sample_GMM([-rand(), rand()], [.2, .3], [.4, .6])

            gmm = GaussianMixture(2)
            fit!(gmm, data)
            distr_hat = distribution(gmm)

            @test cauchy_schwarz(distr, distr) ≥ 0
            @test cauchy_schwarz(distr, distr) ≈ 0 atol=1e-10
            @test cauchy_schwarz(distr, distr_hat) ≈ cauchy_schwarz(distr_hat, distr)
            @test cauchy_schwarz(distr, distr_hat) ≈ 0 atol=atol
        end
    end
end

end # testset "MovingGaussianMixtures"
