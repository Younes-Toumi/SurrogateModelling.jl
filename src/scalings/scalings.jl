abstract type AbstractScaler end

function fit_scaler(::Type{T}, X) where T
    error("fit_scaler not implemented for type $T")
end

function scale(s::AbstractScaler, X)
    error("scale not implemented for $(typeof(s))")
end

function unscale(s::AbstractScaler, X)
    error("unscale not implemented for $(typeof(s))")
end

function variance_scale(s::AbstractScaler, V)
    error("variance_scale not implemented for $(typeof(s))")
end
