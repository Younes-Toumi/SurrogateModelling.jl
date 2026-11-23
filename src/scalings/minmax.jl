struct MinMaxScaler <: AbstractScaler
    min::Vector{Float64}
    max::Vector{Float64}
end

function fit_scaler(::Type{MinMaxScaler}, X::Matrix)

    minv = vec(minimum(X, dims=1))
    maxv = vec(maximum(X, dims=1))

    return MinMaxScaler(minv, maxv)
end

function scale(s::MinMaxScaler, X::Matrix)
    return (X .- s.min') ./ (s.max' .- s.min')
end

function unscale(s::MinMaxScaler, X::Matrix)
    return X .* (s.max' .- s.min') .+ s.min'
end

function variance_scale(s::MinMaxScaler, V::AbstractArray)
    scale = (s.max .- s.min)
    return V .* (scale .^ 2)
end