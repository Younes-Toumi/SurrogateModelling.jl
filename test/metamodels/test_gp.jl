using SurrogateModelling, UncertaintyQuantification
using AbstractGPs, ParameterHandling
using DataFrames, Random, Statistics
using Test

Random.seed!(42)

# testing the one dimensional construction of GP

println("testing GP surrogate: 1D construction...")
@testset "1D Construction" begin
    # deterministic design: no RNG needed for the data
    n_train = 8
    x = collect(range(0, 1; length=n_train))
    data = DataFrame(x = x, y = (6 .* x .- 2).^2 .* sin.(12 .* x .- 4))   # Forrester

    gp = GaussianProcess(data, :y)

    @test size(gp.X) == (n_train, 1)
    @test gp.y == data.y
    @test gp.x_names == [:x]              # every column except :y is an input
    @test gp.y_symbol == :y
    @test gp.kernel_type isa GPMatern52   # the documented default
    @test gp.posterior === nothing        # not fitted yet
    @test length(gp.θ0.lengthscale) == 1  # ARD: one lengthscale per input
    @test_throws ErrorException predict(gp, gp.X)   # predicting before fit! must fail
end

# testing the higher dimension construction of GP
println("testing GP surrogate: 2D construction...")
@testset "2D Construction" begin
    # aleatory design using UQ.jl
    n_train =10
    
    x1 = RandomVariable(Uniform(-1, 1), :x1)
    x2 = RandomVariable(Uniform(-1, 1), :x2)

    model = Model( # himmelblau
        df -> (df.x1 .^ 2 .+ df.x2 .- 11) .^ 2 .+ (df.x1 .+ df.x2 .^ 2 .- 7) .^ 2,
        :y
    )

    data_train = sample([x1, x2], MonteCarlo(n_train))
    evaluate!(model, data_train)

    gp = GaussianProcess(data_train, :y)

    @test size(gp.X) == (n_train, 2)
    @test gp.y == data_train.y
    @test gp.x_names == [:x1, :x2]        # every column except :y is an input
    @test gp.y_symbol == :y
    @test gp.kernel_type isa GPMatern52   # the documented default
    @test gp.posterior === nothing        # not fitted yet
    @test length(gp.θ0.lengthscale) == 2  # ARD: one lengthscale per input
    @test_throws ErrorException predict(gp, gp.X)   # predicting before fit! must fail
end

println("testing GP surrogate: fitting...")
@testset "Fitting" begin
    n_train = 10
    n_test = 1001
    
    x1 = RandomVariable(Uniform(-1, 1), :x1)
    x2 = RandomVariable(Uniform(-1, 1), :x2)

    model = Model( # himmelblau
        df -> (df.x1 .^ 2 .+ df.x2 .- 11) .^ 2 .+ (df.x1 .+ df.x2 .^ 2 .- 7) .^ 2,
        :y
    )

    data_train = sample([x1, x2], MonteCarlo(n_train))
    data_test = sample([x1, x2], MonteCarlo(n_test))

    # adds the output vector :y to the dataframes
    evaluate!(model, data_train)
    evaluate!(model, data_test)

    gp = GaussianProcess(data_train, :y)
    
    θ_pre = ParameterHandling.value(gp.θ)   # unwrap the bounded/positive constraint 

    # fitting the hyperparameters to data
    fit!(gp)

    θ_post = gp.θ

    @testset "pre-fit parameters" begin
        # θ0 holds the same values as θ, without the constraint wrappers.
        @test gp.θ0.lengthscale ≈ θ_pre.lengthscale
        @test gp.θ0.variance ≈ θ_pre.variance
    end

    @testset "optimiser improved the fit" begin
        nlml(gp, θ) = -logpdf(GP(build_kernel(gp.kernel_type, θ))(RowVecs(gp.X), 1e-8), gp.y)

        @test gp.posterior !== nothing
        @test θ_post.lengthscale != θ_pre.lengthscale   # the parameters moved...
        @test nlml(gp, θ_post) < nlml(gp, θ_pre)          # ...and in the right direction
    end

    @testset "hyperparameters stay within their bounds" begin
        l_min = 0.1 * min_pairwise_distance(gp.X)   # same bounds as default_ard_θ
        l_max = max_pairwise_distance(gp.X)
        @test all(l_min .≤ θ_post.lengthscale .≤ l_max)
        @test 1e-9 ≤ θ_post.variance ≤ 100 * var(gp.y)
    end    

    @testset "interpolates the training data" begin
        μ_train, σ2_train = predict(gp, gp.X) # we predict training data
        σ_train = sqrt.(σ2_train)
        @test maximum(abs.(μ_train .- gp.y)) ≤ 1e-6 * std(gp.y) # mean should match the output
        @test maximum(σ2_train) ≈ 1e-8 atol = 1e-10
    end
end

println("testing GP surrogate: predict...")
@testset "Predict" begin
    n_train = 10
    n_test = 1001
    
    x1 = RandomVariable(Uniform(-1, 1), :x1)
    x2 = RandomVariable(Uniform(-1, 1), :x2)

    model = Model( # himmelblau
        df -> (df.x1 .^ 2 .+ df.x2 .- 11) .^ 2 .+ (df.x1 .+ df.x2 .^ 2 .- 7) .^ 2,
        :y
    )

    data_train = sample([x1, x2], MonteCarlo(n_train))

    # adds the output vector :y to the dataframes
    evaluate!(model, data_train)

    gp = GaussianProcess(data_train, :y)
    fit!(gp)

    X_new = [0.1 0.2; -0.7 0.4; 0.9 -0.9]
    m = size(X_new, 1)

    μ, σ² = predict(gp, X_new)

    n_samples = 5
    gp_samples = predict(gp, X_new; mode=:sample, n_samples=n_samples)
    
    @test predict(gp, X_new; mode = :mean) ≈ μ
    @test predict(gp, X_new; mode = :var)  ≈ σ²   # :var gives a variance, the 
    @test size(gp_samples) == (m, n_samples)          # m rows (points) × 5 columns (samples)
end