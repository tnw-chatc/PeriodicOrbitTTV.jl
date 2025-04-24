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

    for ω in ωs, e in es
        p1 = Elements(m=1)
        p2 = Elements(m=1e-6, P=365.242, e=e, ω=ω)
        p3 = Elements(m=1e-6, P=2*365.242, e=e, ω=ω)


        ic = ElementsIC(0., 3, p1, p2, p3)
        s = State(ic)

        bodies = [p1, p2, p3]
        
        println("Testing: ω = $ω, e = $e")
        for i=1:ic.nbody
            pre_elems = ic.elements[i,:]
            elems = get_orbital_elements(s, ic)[i]
            post_elems = [elems.m, elems.P, elems.t0, elems.e * cos(elems.ω), elems.e * sin(elems.ω), elems.I, elems.Ω]
            @test isapprox(pre_elems, post_elems; atol=1e-10)
        end
    end
end
