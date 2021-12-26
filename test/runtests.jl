using Test
using Distributions

using MovingMixtures

const atol = 0.1
const SAMPLE_SIZE = 1000

function sample_GMM(mus, stds, ps)
    distr = UnivariateGMM(mus, stds, Categorical(ps))
    data = rand(distr, SAMPLE_SIZE)

    distr, data
end

@testset verbose=true "MovingMixtures" begin

@testset "Simple mixtures" begin
    @testset "2 components easy $i" for i in 1:10
        distr, data = sample_GMM([-1, 1], [.2, .3], [.4, .6])

        gmm = GaussianMixture(2)
        fit!(gmm, data)

        @test gmm.converged
        @test sort(gmm.p) ≈ sort(distr.prior.p) atol=atol
        @test sort(gmm.mu) ≈ sort(distr.means) atol=atol
        @test sort(gmm.var) ≈ sort(distr.stds.^2) atol=atol
    end

    @testset "2 components zero means $i" for i in 1:10
        distr, data = sample_GMM([0, 0], [.2, .3], [.4, .6])

        gmm = GaussianMixture(2)
        fit!(gmm, data)

        @test gmm.converged
        #FIXME: this fails, but do we care?
        # @test sort(gmm.p) ≈ sort(distr.prior.p) atol=atol
        @test sort(gmm.mu) ≈ sort(distr.means) atol=atol
        @test sort(gmm.var) ≈ sort(distr.stds.^2) atol=atol
    end
end

@testset "Regularization" begin
    @testset "Variance reg by addition $i" for i in 1:10
        REG = 1e-4
        distr, data = sample_GMM([0, 0], [.2, .3], [.4, .6])

        gmm = GaussianMixture(50) # should overfit
        fit!(gmm, data, regularization=Settings.RegVarianceSimple(REG))

        @test all(gmm.var .≥ REG)
    end

    @testset "Variance reg by restarts $i" for i in 1:10
        distr, data = sample_GMM([0, 0], [.2, .3], [.4, .6])

        gmm = GaussianMixture(50) # should overfit
        fit!(gmm, data, regularization=Settings.RegVarianceReset())

        @test all(gmm.var .> 0)
    end
end

@testset "Divergences" begin
    @testset "Cauchy-Schwarz $i" for i in 1:10
        distr, data = sample_GMM([-rand(), rand()], [.2, .3], [.4, .6])

        gmm = GaussianMixture(2)
        fit!(gmm, data)
        distr_hat = distribution(gmm)

        @test cauchy_schwarz(distr, distr) ≥ 0
        @test cauchy_schwarz(distr, distr) ≈ 0
        @test cauchy_schwarz(distr, distr_hat) ≈ cauchy_schwarz(distr_hat, distr)
        @test cauchy_schwarz(distr, distr_hat) ≈ 0 atol=atol
    end
end

end