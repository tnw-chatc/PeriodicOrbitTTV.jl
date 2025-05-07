using PeriodicOrbit
using NbodyGradient
using LinearAlgebra: dot, cross
using Test

function Base.isapprox(a::Elements,b::Elements;tol=1e-10)
    fields = setdiff(fieldnames(Elements),[:a,:e,:ϖ])
    for i in fields
        af = getfield(a,i)
        bf = getfield(b,i)
        if abs(af - bf) > tol
            return false
        end
    end
    return true
end

@testset "Cartesian to Elements" begin
    ωs = [0., π/2, π, -π/2, 1e-4, -1e-4, 
    π/2 + 1e-4, π/2 - 1e-4, 
    -π/2 + 1e-4, -π/2 - 1e-4, 
    π + 1e-4, π - 1e-4, 
    -π + 1e-4, -π - 1e-4]

    es = [0.0001, 0.1, 0.5, 0.9]

    # ωs = [0., π/2, π, -π/2, 1e-4, -1e-4]
    # es = [0.0001, 0.1, 0.5, 0.9]
    # Is = [0., 1., π/2]
    # Ωs = [0., 1., π/2, -π/2]

    for ω in ωs, e in es
        p1 = Elements(m=1)
        p2 = Elements(m=1e-4, P=365.242, e=e, ω=ω, I=π/2, Ω=0)
        p3 = Elements(m=1e-4, P=2*365.242, e=e, ω=ω, I=π/2, Ω=0)


        ic = ElementsIC(0., 3, p1, p2, p3)
        s = State(ic)

        bodies = [p1, p2, p3]
        
        println("Testing: ω = $ω, e = $e")
        for i=1:ic.nbody
            pre_elems = ic.elements[i,:]
            elems = get_orbital_elements(s, ic)[i]
            post_elems = [elems.m, elems.P, elems.t0, elems.e * cos(elems.ω), elems.e * sin(elems.ω), elems.I, elems.Ω]
            @test isapprox(pre_elems, post_elems; atol=1e-8)
        end
    end
end

# @testset "Orbit initialization" begin
#     ωs = [0., π/2, π, -π/2, 1e-4, -1e-4]
#     es = [0.0001, 0.1, 0.5, 0.9]
#     Is = [0., 1., π/2]
#     Ωs = [0., 1., π/2, -π/2]

#     for ω in ωs, e in es, I in Is, Ω in Ωs
#         p1 = Elements(m=1)
#         p2 = Elements(m=1e-4, P=365.242, e=e, ω=ω, I=I, Ω=Ω)
#         p3 = Elements(m=1e-4, P=2*365.242, e=e, ω=ω, I=I, Ω=Ω)

#         ic = ElementsIC(0., 3, p1, p2, p3)
#         s = State(ic)

#         o = Orbit(s, ic, 2.001)

#         init_from_M!(o.s, o.ic, 0., 2)
#         init_from_M!(o.s, o.ic, 0., 3)

#         anoms = get_anomalies(o.s, o.ic)

#         println("Testing : ω = $ω, e = $e, I = $I, Ω= $Ω")
#         for i=1:o.nplanet
#             test_anom = anoms[i][2] - π
#             if test_anom >= 0
#                 test_anom = abs(test_anom - π)
#             else
#                 test_anom = abs(test_anom + π)
#             end
#             @test isapprox(test_anom, 0.; atol=1e-3)
#         end
#     end
# end
