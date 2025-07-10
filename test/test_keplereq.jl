using PeriodicOrbit
using PeriodicOrbit: solve_kepler_equation

using Test

epsilon = eps(Float64)

@testset "Kepler Equation Solver" begin
    M_cands = [0., π/2, 1π, 3π/2,
    0. + 1e-4, π/2 + 1e-4, 1π + 1e-4, 3π/2 + 1e-4,
    0. - 1e-4, π/2 - 1e-4, 1π - 1e-4, 3π/2 - 1e-4,
    0., -π/2, -1π, -3π/2,
    ]

    e_cands = [0.0, 0.0001, 0.1, 0.5, 0.9, 0.99999]

    for M in M_cands, e in e_cands
        println("Testing: $M, $e")
        E = solve_kepler_equation(M, e)
        
        println("$M, $(E - e * sin(E))")
        @test isapprox(M, E - e * sin(E); rtol=epsilon)
    end
end