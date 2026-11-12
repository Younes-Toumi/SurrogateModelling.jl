using SurrogateModelling
using DataFrames
using Test

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

@testset "2D Construction" begin
    # deterministic design: no RNG needed for the data
    n_train = 8
    x1 = collect(range(-3, 3; length=n_train))
    x2 = collect(range(-3, 3; length=n_train))

    data = DataFrame(x1 = x1, x2 = x2, y = (x1 .^ 2 .+ x2 .- 11) .^ 2 .+ (x1 .+ x2 .^ 2 .- 7) .^ 2)   # Himmelblau

    gp = GaussianProcess(data, :y)

    @test size(gp.X) == (n_train, 2)
    @test gp.y == data.y
    @test gp.x_names == [:x1, :x2]        # every column except :y is an input
    @test gp.y_symbol == :y
    @test gp.kernel_type isa GPMatern52   # the documented default
    @test gp.posterior === nothing        # not fitted yet
    @test length(gp.θ0.lengthscale) == 2  # ARD: one lengthscale per input
    @test_throws ErrorException predict(gp, gp.X)   # predicting before fit! must fail
end

@testset "Fitting" begin
    nothing
end

@testset "Interface Contact" begin
    nothing
end

@testset "Kernel" begin
    nothing
end

@testset "Accuracy" begin
    nothing
end