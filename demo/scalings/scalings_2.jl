using SurrogateModelling
using UncertaintyQuantification
using Random
using DataFrames
Random.seed!(42)

# ── Variables & Model ─────────────────────────────────────────────
x1 = ones(5) .* 0
x2 = ones(5) .* 2
X  = [x1 x2]

model = Model(
    rv -> rv.x1.^2 .+ rv.x2.^2,
    :y
)

data_train   = DataFrame(X, :auto)
evaluate!(model, data_train)

x_names = propertynames(data_train[:, Not(:y)])

# ── Scaling ───────────────────────────────────────────────────────
X_train = Matrix(data_train[:, x_names])
y_train = Vector(data_train[:, :y])

pipeline = fit_pipeline(
    X_train,
    y_train,
    MinMaxScaler,
    MinMaxScaler
)

X_train_scaled = scale_input(pipeline, X_train)
y_train_scaled = scale_output(pipeline, y_train)

println("input scaled min: $(round.(pipeline.input_scaler.min, digits=3))")
println("input scaled max: $(round.(pipeline.input_scaler.max, digits=3))")

println("output scaled min: $(round.(pipeline.output_scaler.min, digits=3))")
println("output scaled max: $(round.(pipeline.output_scaler.max, digits=3))")