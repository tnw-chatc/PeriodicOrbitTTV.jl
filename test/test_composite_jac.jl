using PeriodicOrbit
using PeriodicOrbit: extract_elements, compute_diff_squared, compute_diff_squared_jacobian
using NbodyGradient
using LinearAlgebra: normalize, cross

using LsqFit
using Optim
using FiniteDifferences

using Test

optvec_0 = BigFloat.([0.1, 0.07, 0.05,
0., 0., 0.,
0., 0.,
1e-4,
365.242,
3e-6, 5e-6, 7e-5,
2.000,
0.00
])

orbparams = OrbitParameters(3,
                                BigFloat.([0.5, 0.5]), 
                                BigFloat(8*365.242), 
                                BigFloat.([1., 1., 5., 3., 2.]))

nplanet = 3
epsilon = BigFloat(eps(Float64))

# ForwardDiff
optparams = OptimParameters(nplanet, deepcopy(optvec_0))

orbit_0 = Orbit(nplanet, optparams, orbparams)

# optres = find_periodic_orbit(OptimParameters(nplanet, deepcopy(optvec_0)), orbparams; use_jac=false, weighted=false)

function fff(x)
    optparams = OptimParameters(nplanet, x)

    diff_squared = compute_diff_squared(optparams, orbparams, nplanet; weighted=false)

    return diff_squared
end

function ggg(x)
    optparams = OptimParameters(nplanet, x)

    return Orbit(nplanet, optparams, orbparams).final_elem
end

@testset "Composite & Combined Jacobian" begin
    @testset "Composite Jacobian" begin   
        # Composite Jacobians
        global fd_jac_composite = jacobian(central_fdm(2,1), fff, optvec_0)[1]
        global an_jac_composite = compute_diff_squared_jacobian(optparams, orbparams, nplanet)

        for i in 1:1:size(an_jac_composite, 1), j in 1:1:size(an_jac_composite, 2)
            @test isapprox(an_jac_composite[i,j], fd_jac_composite[i,j]; atol=epsilon)
        end
    end

    @testset "Combined Jacobian" begin   
        # Combined Jacobians
        global fd_jac_0 = jacobian(central_fdm(2,1), ggg, optvec_0)[1]
        global an_jac_0 = orbit_0.jac_combined

        for i in 1:1:size(an_jac_0, 1), j in 1:1:size(an_jac_0, 2)
            @test isapprox(an_jac_0[i,j], fd_jac_0[i,j]; atol=epsilon)
        end
    end
end