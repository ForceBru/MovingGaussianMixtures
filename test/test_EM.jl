correct_normal_pdf(x, mu, var) = exp(-(x - mu)^2 / (2var)) / sqrt(2π * var)

correct_log_likelihood(G, p, mu, var, x) = sum(
    log(
        sum(
            p[k] * correct_normal_pdf(x[n], mu[k], var[k])
            for k in eachindex(p)
        )
    )
    for n in eachindex(x)
)

"ELBO = E_Z[ log(p(X, Z)) ] - E_Z[ log(q(Z)) ]"
correct_ELBO(G, p, mu, var, x) = sum(
    sum(
        # E_Z[ log(p(X, Z)) ]
        G[k, n] * log(p[k]) + G[k, n] * log(correct_normal_pdf(x[n], mu[k], var[k]))
        # E_Z[ log(q(Z)) ]
        - G[k, n] * log(G[k, n] + 1e-100)
        for k in eachindex(p)
    )
    for n in eachindex(x)
)

function correct_step_E(G, p, mu, var, x)
    G_new = [
        p[k] * correct_normal_pdf(x[n], mu[k], var[k])
        for k in eachindex(p), n in eachindex(x)
    ]
    G_new ./= sum(G_new, dims=1)

    G_new
end

function correct_step_M(G, p, mu, var, x)
    K = size(G, 1)

    evidences = [sum(G, dims=2)...]
    @test size(evidences) == (K, )

    p_new = evidences ./ sum(evidences)
    mu_new = [
        sum(G[k, :] .* x)
        for k in 1:K
    ] ./ evidences
    var_new = [
        sum(G[k, :] .* (x .- mu_new[k]).^2)
        for k in 1:K
    ] ./ evidences

    p_new, mu_new, var_new
end

@testset "EM algorithm ($K components, $N data points)" for K in 1:5, N in 100:100:500
    G = rand(K, N)
    G ./= sum(G, dims=1)
    x = rand(N) .- 0.5

    p = rand(K)
    p ./= sum(p)
    mu = 2(rand(K) .- 0.5)
    var = rand(K) .+ 0.01 # ensure that `var > 0`

    original = (
        G=copy(G), x=copy(x),
        p=copy(p), mu=copy(mu), var=copy(var)
    )

    @test all(sum(G, dims=1) .≈ 1)
    @test sum(p) ≈ 1
    @test all(var .> 0)

    @testset "Log-likelihood computation" begin
        @test MovingGaussianMixtures.log_likelihood(G, p, mu, var, x, nothing) ≈ correct_log_likelihood(G, p, mu, var, x)
        # This shouldn't change input data
        @test G == original[:G]
        @test x == original[:x]
        @test p == original[:p]
        @test mu == original[:mu]
        @test var == original[:var]
    end

    @testset "ELBO computation" begin
        @test MovingGaussianMixtures.ELBO(G, p, mu, var, x, nothing) ≈ correct_ELBO(G, p, mu, var, x)
        # This shouldn't change input data
        @test G == original[:G]
        @test x == original[:x]
        @test p == original[:p]
        @test mu == original[:mu]
        @test var == original[:var]
    end

    @testset "E step no regularization" begin
        G_copy = copy(G)
        G_correct = correct_step_E(G_copy, p, mu, var, x)
        MovingGaussianMixtures.step_E!(G_copy, p, mu, var, x, nothing)

        @test G_copy ≈ G_correct
        @test x == original[:x]
        @test p == original[:p]
        @test mu == original[:mu]
        @test var == original[:var]
    end

    @testset "M step no regularization" begin
        p_copy, mu_copy, var_copy = copy(p), copy(mu), copy(var)
        p_correct, mu_correct, var_correct = correct_step_M(G, p_copy, mu_copy, var_copy, x)
        MovingGaussianMixtures.step_M!(G, p_copy, mu_copy, var_copy, x, nothing)

        @test p_copy ≈ p_correct
        @test mu_copy ≈ mu_correct
        @test var_copy ≈ var_correct
        @test x == original[:x]
        @test G == original[:G]
    end
end
