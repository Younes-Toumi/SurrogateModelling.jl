using SurrogateModelling
using Test

@testset "SurrogateModelling.jl" begin
    @testset "GP Surrogate" include("metamodels/test_gp.jl")
end