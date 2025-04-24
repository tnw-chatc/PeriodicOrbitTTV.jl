using PeriodicOrbit
using NbodyGradient
using Test

@testset "PeriodicOrbit" begin
    println("Basic testing...")
    @testset "Basic Testing" begin
        include("test_basic.jl")
    end
    println("Finished.")
end