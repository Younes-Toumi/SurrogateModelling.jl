using SurrogateModelling
using SurrogateModelling: fit_scaler, scale, unscale, variance_scale
using Random, Statistics
using Test

Random.seed!(42)

const SCALERS = (IdentityScaler, MinMaxScaler, ZScoreScaler)
const n = 50

# fit, scale, and check the two properties every scaler must have:
# finite output of the same size, and an exact round trip
function check_roundtrip(S, X; atol=0.0)
    s  = fit_scaler(S, X)
    Xs = scale(s, X)
    @test size(Xs) == size(X)
    @test all(isfinite, Xs)
    @test unscale(s, Xs) ≈ X atol=atol
    return s, Xs
end

println("testing scalings: normal and constant data...")
@testset "Normal data" begin
    # columns with very different offsets and spreads
    X = hcat(randn(n), 3 .+ 10 .* rand(n), 1e3 .* randn(n))

    for S in SCALERS
        @testset "$S" begin
            s, Xs = check_roundtrip(S, X)
            if S === IdentityScaler
                @test Xs == X
            elseif S === MinMaxScaler
                @test vec(minimum(Xs, dims=1)) ≈ zeros(3) atol=1e-12
                @test vec(maximum(Xs, dims=1)) ≈ ones(3)
            elseif S === ZScoreScaler
                @test vec(mean(Xs, dims=1)) ≈ zeros(3) atol=1e-12
                @test vec(std(Xs, dims=1))  ≈ ones(3)
            end
        end
    end
end

@testset "Constant columns (incl. zero)" begin
    # three constant columns (positive, negative, zero) next to a normal one
    X = hcat(fill(5.0, n), fill(-3.0, n), zeros(n), randn(n))

    for S in SCALERS
        @testset "$S" begin
            s, Xs = check_roundtrip(S, X)
            if S !== IdentityScaler
                # constant columns map to 0 instead of NaN/Inf ...
                @test all(iszero, Xs[:, 1:3])
                # ... and the guard leaves the normal column untouched
                if S === MinMaxScaler
                    @test extrema(Xs[:, 4]) == (0.0, 1.0)
                else
                    @test std(Xs[:, 4]) ≈ 1.0
                end
            end
        end
    end
end

@testset "Near-constant columns (noise below tolerance)" begin
    # relative noise of ~1e-12: numerically a constant, should NOT be blown up
    X = hcat(5.0 .+ 1e-12 .* randn(n),
             1e-12 .* randn(n),
             1e6 .+ 1e-6 .* randn(n))

    for S in (MinMaxScaler, ZScoreScaler)
        @testset "$S" begin
            s, Xs = check_roundtrip(S, X; atol=1e-9)
            spread(v) = maximum(v) - minimum(v)
            for j in 1:3
                # noise must not be amplified: a real rescaling would blow
                # a spread of ~1e-12 up to ~1 (MinMax) or ~4 (ZScore)
                @test spread(Xs[:, j]) ≤ 1.01 * spread(X[:, j]) + eps()
            end
        end
    end
end

@testset "variance_scale matches unscale" begin
    # unscale is affine (a·x + b), so variances must transform by a²
    for y in (3 .+ 2 .* randn(n), fill(4.0, n), zeros(n))
        Y = reshape(y, :, 1)
        for S in SCALERS
            s  = fit_scaler(S, Y)
            Ys = 0.1 .* randn(n, 1)   # any values in the scaled space
            @test var(unscale(s, Ys)) ≈ only(variance_scale(s, [var(Ys)]))
        end
    end
end


println("testing scalings: AbstractMatrix...")
@testset "AbstractMatrix inputs (views, adjoints)" begin
    X = randn(n, 2)
    for S in SCALERS
        s = fit_scaler(S, X)
        @test scale(s, @view X[1:10, :]) ≈ scale(s, X[1:10, :])
        @test scale(s, permutedims(X)') ≈ scale(s, X)   # Adjoint
    end
end

println("testing scalings: pipeline...")
@testset "ScalingPipeline" begin
    X = hcat(rand(n), 10 .* randn(n))
    for y in (3 .+ 2 .* randn(n), zeros(n))          # normal and constant-zero output
        for (Sin, Sout) in ((IdentityScaler, ZScoreScaler),
                            (MinMaxScaler,   ZScoreScaler),
                            (MinMaxScaler,   MinMaxScaler))
            p  = fit_pipeline(X, y, Sin, Sout)
            ys = scale_output(p, y)
            @test all(isfinite, scale_input(p, X))
            @test all(isfinite, ys)
            @test inverse_mean(p, ys) ≈ y
            @test all(≥(0), inverse_variance(p, fill(0.5, n)))
        end
    end
end

@testset "Single sample" begin
    # e.g. one training point; std of one value is NaN
    X = [1.0 2.0]
    for S in SCALERS
        @testset "$S" begin
            check_roundtrip(S, X)
        end
    end
end
