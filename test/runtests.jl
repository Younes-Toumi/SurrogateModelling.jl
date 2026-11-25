using SurrogateModelling
using Test

@testset "SurrogateModelling.jl" begin
    
    println("testing scalings...")
    @testset "Scalings"     include("test_scalings.jl")

    println("testing GP surrogate...")
    @testset "GP Surrogate" include("metamodels/test_gp.jl")

end