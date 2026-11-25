struct IdentityScaler <: AbstractScaler end

function fit_scaler(::Type{IdentityScaler}, ::AbstractMatrix)
    return IdentityScaler()
end

function scale(::IdentityScaler, X::AbstractMatrix)
    return X
end

function unscale(::IdentityScaler, X::AbstractMatrix)
    return X
end

function variance_scale(::IdentityScaler, V::AbstractArray)
    return V
end