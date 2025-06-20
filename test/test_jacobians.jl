using PeriodicOrbit
using NbodyGradient

using Test

# Test if Jacobian 1 from AutoDiff yields the same result as one from Finite Difference
@testset "Jacobian 1" begin
    optvec_0 = BigFloat.([0.1, 0.2, 0.3,
    2π, 1e-4,
    0., 1.,
    1e-4,
    365.242])

    orbparams = OrbitParameters(BigFloat.([1e-5, 1e-5, 1e-4]), 
                                    BigFloat.([0.5]), 
                                    BigFloat(2.000), 
                                    BigFloat(4*365.242), 
                                    BigFloat.([1., 1., 5., 3., 2.]))

    nplanet = 3
    epsilon = BigFloat(eps(Float64))

    # ForwardDiff
    orbit_0 = Orbit(nplanet, OptimParameters(nplanet, deepcopy(optvec_0)), orbparams)

    # Finite Difference
    optvec_derivatives = []
    for j in 1:length(optvec_0)

        optvec_pos = copy(optvec_0)
        optvec_neg = copy(optvec_0)

        optvec_pos[j] += epsilon
        optvec_neg[j] -= epsilon

        # Finite difference
        jac_pos = Orbit(nplanet, OptimParameters(nplanet, optvec_pos), orbparams).elem_matrix[2:end,:]
        jac_neg = Orbit(nplanet, OptimParameters(nplanet, optvec_neg), orbparams).elem_matrix[2:end,:]

        optvec_derivative = reduce(vcat, eachcol((jac_pos - jac_neg) / (2 * epsilon)))
        push!(optvec_derivatives, optvec_derivative)
    end

    optvec_derivatives = reduce(hcat, optvec_derivatives)

    # Test for Jacobian 1 -> Check if each element relative differs only up to the Float64 machine precision
    test_matrix = isapprox.(orbit_0.jac_1, optvec_derivatives; rtol=epsilon)

    for i in eachindex(test_matrix)
            @test test_matrix[i]
    end
end

# TODO: Implement a test to check if the COMBINED jacobians work as expected
# This test set does not use BigFloat precision
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
        jac_pos = Orbit(nplanet, OptimParameters(nplanet, optvec_pos), orbparams).final_elem_mat
        jac_neg = Orbit(nplanet, OptimParameters(nplanet, optvec_neg), orbparams).final_elem_mat

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
