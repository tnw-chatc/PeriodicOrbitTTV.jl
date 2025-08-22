using PeriodicOrbit
using PeriodicOrbit: extract_elements, compute_diff_squared, compute_diff_squared_jacobian
using NbodyGradient
using LinearAlgebra: normalize, cross

using LsqFit
using Optim
using FiniteDifferences

using Test

optvec_0 = BigFloat.([0.004927252383216913, 0.009566802745561853, 0.0007783008365053432, 0.13419851562243246, 0.0673251134472058, 0.017117806838380507, 3.1414933041899062, -3.1248525292536713, 0.0015818461857177034, 366.5768575418027, 4.6999999999998574e-5, 2.5599999999991813e-5, 7.840000000000029e-5, 2.006952082012, -1.0601743894425463e-30, 1460.968])

orbparams = OrbitParameters(3,
                                BigFloat.([0.5, 0.5]), BigFloat(2000.))

nplanet = 3
epsilon = BigFloat.(eps(Float64))

# ForwardDiff
optparams = OptimParameters(nplanet, deepcopy(optvec_0))

orbit_0 = Orbit(nplanet, optparams, orbparams)

using DelimitedFiles
using PeriodicOrbit: compute_tt_jacobians
elemIC = create_elem_ic(orbit_0)

tt_data = convert(Matrix{BigFloat}, readdlm("test/ttv_test_data.in",',',comments=true,comment_char='#'));

tt = compute_tt(orbit_0, elemIC, orbparams.obstmax)

tmod, ip, jp = match_transits(tt_data, elemIC.elements, tt.tt, tt.count, nothing)

function fff(x)
    optparams = OptimParameters(nplanet, x)

    orbit = Orbit(nplanet, optparams, orbparams)
    elemIC = create_elem_ic(orbit)
    tt = compute_tt(orbit, elemIC, orbparams.obstmax) # TODO: Get rid of the hardcode here

    # Append the TT information
    tmod, ip, jp = match_transits(tt_data, elemIC.elements, tt.tt, tt.count, nothing)

    diff_squared = compute_diff_squared(optparams, orbparams, nplanet, tmod)

    return diff_squared
end

@testset "Composite Jacobian" begin
    @testset "Composite Jacobian" begin   
        println("Starting...")
        # Composite Jacobians
        global fd_jac_composite = jacobian(central_fdm(2,1), fff, optvec_0)[1]
        global an_jac_composite = compute_diff_squared_jacobian(optparams, orbparams, nplanet, tt_data)

        for i in 1:1:size(an_jac_composite, 1), j in 1:1:size(an_jac_composite, 2)
            @test isapprox(an_jac_composite[i,j], fd_jac_composite[i,j]; atol=1e-6)
        end
    end
end