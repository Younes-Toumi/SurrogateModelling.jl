struct ZScoreScaler <: AbstractScaler
    μ::Vector{Float64}
    σ::Vector{Float64}
end

function fit_scaler(::Type{ZScoreScaler}, X::AbstractMatrix)
    μ = vec(mean(X, dims=1))
    σ = vec(std(X, dims=1))
    # constant / near-constant columns, or a single sample (std = NaN)
    tol = 1e-9 .* max.(abs.(μ), 1)
    σ[(σ .<= tol) .| isnan.(σ)] .= 1.0
    return ZScoreScaler(μ, σ)
end

function scale(s::ZScoreScaler, X::AbstractMatrix)
    return (X .- s.μ') ./ s.σ'
end

function unscale(s::ZScoreScaler, X::AbstractMatrix)
    return X .* s.σ' .+ s.μ'
end

function variance_scale(s::ZScoreScaler, V::AbstractArray)
    return V .* (s.σ .^ 2)
end