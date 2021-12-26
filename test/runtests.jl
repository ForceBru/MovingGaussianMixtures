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

function isapprox_sorted(x::AbstractVector, y::AbstractVector; atol::Real)::Bool
    @assert atol > 0

    isapprox(sort(x), sort(y); atol)
end

@testset "Simple mixtures" begin
    @testset "2 components easy" begin
        distr, data = sample_GMM([-1, 1], [.2, .3], [.4, .6])

        gmm = GaussianMixture(2)
        fit!(gmm, data)

        @test gmm.converged
        @test isapprox_sorted(gmm.p, distr.prior.p; atol)
        @test isapprox_sorted(gmm.mu, distr.means; atol)
        @test isapprox_sorted(gmm.var, distr.stds.^2; atol)
    end

    @testset "2 components zero means" begin
        distr, data = sample_GMM([0, 0], [.2, .3], [.4, .6])

        gmm = GaussianMixture(2)
        fit!(gmm, data)

        @test gmm.converged
        @test isapprox_sorted(gmm.p, distr.prior.p; atol)
        @test isapprox_sorted(gmm.mu, distr.means; atol)
        @test isapprox_sorted(gmm.var, distr.stds.^2; atol)
    end
end

@testset "Regularization" begin
    @testset "Variance reg by addition" begin
        REG = 1e-4
        distr, data = sample_GMM([0, 0], [.2, .3], [.4, .6])

        gmm = GaussianMixture(50) # should overfit
        fit!(gmm, data, regularization=Settings.RegVarianceSimple(REG))

        @test all(gmm.var .≥ REG)
    end

    @testset "Variance reg by restarts" begin
        distr, data = sample_GMM([0, 0], [.2, .3], [.4, .6])

        gmm = GaussianMixture(50) # should overfit
        fit!(gmm, data, regularization=Settings.RegVarianceReset())

        @test all(gmm.var .> 0)
    end
end
