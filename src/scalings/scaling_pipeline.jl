struct ScalingPipeline
    input_scaler
    output_scaler
end

# ============================================================
# 1. matrix aware
# ============================================================

function fit_pipeline(X::AbstractMatrix, y::AbstractVector, input_type, output_type)
    input_scaler  = fit_scaler(input_type, X)
    output_scaler = fit_scaler(output_type, reshape(y, :, 1))
    return ScalingPipeline(input_scaler, output_scaler)
end

function scale_input(p::ScalingPipeline, X::AbstractMatrix)
    return scale(p.input_scaler, X)
end

function scale_output(p::ScalingPipeline, y::AbstractVector)
    return vec(scale(p.output_scaler, reshape(y, :, 1)))
end

# ============================================================
# Dataframe aware
# ============================================================

function fit_pipeline(data::DataFrame, output::Symbol, input_type, output_type)
    x_names = propertynames(data[:, Not(output)])
    X = Matrix(data[:, x_names])
    y = Vector(data[:, output])
    return fit_pipeline(X, y, input_type, output_type)
end

function scale_data(p::ScalingPipeline, data::DataFrame, output::Symbol)
    x_names  = propertynames(data[:, Not(output)])
    X_scaled = scale_input(p, Matrix(data[:, x_names]))
    y_scaled = scale_output(p, Vector(data[:, output]))

    df = DataFrame(X_scaled, collect(x_names))
    df[!, output] = y_scaled
    return df
end

function unscale_data(p::ScalingPipeline, data::DataFrame, output::Symbol)
    x_names  = propertynames(data[:, Not(output)])
    X        = unscale(p.input_scaler, Matrix(data[:, x_names]))
    y        = inverse_mean(p, Vector(data[:, output]))

    df = DataFrame(X, collect(x_names))
    df[!, output] = y
    return df
end


# ============================================================
# scale mena and variance
# ============================================================
 
function inverse_mean(p::ScalingPipeline, μ::AbstractVector)
    return vec(unscale(p.output_scaler, reshape(μ, :, 1)))
end

function inverse_variance(p::ScalingPipeline, σ2::AbstractVector)
    return variance_scale(p.output_scaler, σ2)
end

export 
    IdentityScaler, MinMaxScaler, ZScoreScaler, 
    fit_pipeline, scale_input, scale_output, scale,
    scale_data, unscale_data,
    inverse_mean, inverse_variance