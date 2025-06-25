using PeriodicOrbit
using NbodyGradient

using Test

@testset "Combined Jacobians" begin
    optvec_0 = ([0.1, 0.2, 0.3, 0.4,
    2π, 1e-4, π/2,
    0., 1., 2π,
    1e-4, 1e-4,
    365.242])

    orbparams = OrbitParameters(([1e-5, 1e-5, 1e-4, 1e-6]), 
                                    ([0.5, 0.5]), 
                                    (2.000), 
                                    (8*365.242), 
                                    ([1., 1., 5., 3., 2.]))

    nplanet = 4
    epsilon = (eps(Float64))

    # ForwardDiff
    orbit_0 = Orbit(nplanet, OptimParameters(nplanet, deepcopy(optvec_0)), orbparams)

    # Finite Difference for the entire things
    optvec_derivatives = []
    for j in 1:length(optvec_0)

        optvec_pos = copy(optvec_0)
        optvec_neg = copy(optvec_0)

        optvec_pos[j] += epsilon
        optvec_neg[j] -= epsilon

        # Finite difference
        jac_pos = Orbit(nplanet, OptimParameters(nplanet, optvec_pos), orbparams).final_elem
        jac_neg = Orbit(nplanet, OptimParameters(nplanet, optvec_neg), orbparams).final_elem

        optvec_derivative = reduce(vcat, eachcol((jac_pos - jac_neg) / (2 * epsilon)))
        push!(optvec_derivatives, optvec_derivative)
    end

    optvec_derivatives = reduce(hcat, optvec_derivatives)

    println(size(orbit_0.jac_combined))
    println(orbit_0.jac_combined)

    println()

    println(size(optvec_derivatives))
    println(optvec_derivatives)
    
    # Test for Jacobian 1 -> Check if each element relative differs only up to the Float64 machine precision
    test_matrix = isapprox.(orbit_0.jac_combined, optvec_derivatives; rtol=epsilon)

    for i in eachindex(test_matrix)
            @test test_matrix[i]
    end
end