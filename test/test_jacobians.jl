using PeriodicOrbit
using NbodyGradient

using Test

@testset "Jacobian 1" begin
    optvec_0 = BigFloat.([0.1, 0.2, 0.3, 0.4,
    2π, 1e-4, π/2,
    0., 1., 2π,
    1e-4, 1e-4,
    365.242])

    orbparams = OrbitParameters(BigFloat.([1e-5, 1e-5, 1e-4, 1e-6]), 
                                    BigFloat.([0.5, 0.5]), 
                                    BigFloat(2.000), 
                                    BigFloat(8*365.242), 
                                    BigFloat.([1., 1., 5., 3., 2.]))

    nplanet = 4
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
        jac_pos = Orbit(nplanet, OptimParameters(nplanet, optvec_pos), orbparams).elem_matrix
        jac_neg = Orbit(nplanet, OptimParameters(nplanet, optvec_neg), orbparams).elem_matrix

        optvec_derivative = reduce(vcat, eachcol((jac_pos - jac_neg) / (2 * epsilon)))
        push!(optvec_derivatives, optvec_derivative)
    end

    optvec_derivatives = reduce(hcat, optvec_derivatives)

    test_matrix = isapprox.(orbit_0.jac, optvec_derivatives; rtol=epsilon)

    for i in eachindex(test_matrix)
            @test test_matrix[i]
    end
end