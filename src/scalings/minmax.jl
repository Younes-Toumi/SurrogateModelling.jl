struct MinMaxScaler <: AbstractScaler
    min::Vector{Float64}
    max::Vector{Float64}
end

function fit_scaler(::Type{MinMaxScaler}, X::AbstractMatrix)

    minv = vec(minimum(X, dims=1))
    maxv = vec(maximum(X, dims=1))

    # we avoid 0 division if they are equal by removing the min
    # if they are 0, we set it to one.
    rng = maxv .- minv
    rng[rng .<= 1e-9 .* max.(abs.(maxv), 1)] .= 1.0
    maxv = minv .+ rng
    return MinMaxScaler(minv, maxv)
end

function scale(s::MinMaxScaler, X::AbstractMatrix)
    return (X .- s.min') ./ (s.max' .- s.min')
end

function unscale(s::MinMaxScaler, X::AbstractMatrix)
    return X .* (s.max' .- s.min') .+ s.min'
end

function variance_scale(s::MinMaxScaler, V::AbstractArray)
    scale = (s.max .- s.min)
    return V .* (scale .^ 2)
end