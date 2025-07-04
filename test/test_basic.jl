using PeriodicOrbit
using NbodyGradient
using LinearAlgebra: dot, cross
using Test

@testset "Orbit Initialization" begin

    # For n = 4 planets
    optvecs = []

    push!(optvecs, [0.1, 0.2, 0.3, 0.4,
    0.00001, 1.3, 5.0,
    0.2, 0.3, π,
    1e-4, 1e-4,
    365.242])

    push!(optvecs, [0.3, 0.2, 0.3, 0.3,
    4., π, -3.,
    0.2, 0.3, π,
    1e-4, 1e-4,
    365.242])

    push!(optvecs, [0.7, 0.7, 0.7, 0.7,
    2π, 1e-4, π/2,
    0.00001, -0.0001, 2π,
    1e-4, 1e-4,
    365.242])

    push!(optvecs, [1e-3, 1e-3, 1e-4, 1e-4,
    π/2, -π/2, π/2,
    π/2, -π/2, -π/2,
    0., 0.,
    365.242])

    push!(optvecs, [0.9, 0.9, 0.9, 0.9,
    2π, -2π, 2π,
    -2π, 2π, -2π,
    1e-4, 1e-4,
    366.242])

    for j in eachindex(optvecs)
        vec = optvecs[j]

        optparams = OptimParameters(4, vec)
        orbparams = OrbitParameters([1e-4, 1e-4, 1e-4, 1e-4], [0.5, 0.5], 2.000, 8*365.242, [1., 1., 5., 3., 2.])

        orbit = Orbit(4, optparams, orbparams)

        elems = get_orbital_elements(orbit.s, orbit.ic)
        anoms = get_anomalies(orbit.s, orbit.ic)

        omega = Vector{Float64}(undef, 4)
        omega[1] = 0.0
        for i=2:orbit.nplanet
            # Test for eccentricities
            @test isapprox(elems[i].e, vec[i-1]; rtol=sqrt(eps(Float64)))
            # Test for longitudes of periastron
            omega[i] = vec[6+i] + omega[i-1]
            @test isapprox(rem2pi(elems[i].ω, RoundNearest), rem2pi(omega[i-1], RoundNearest); rtol=sqrt(eps(Float64)))
            # Test for mean anomalies
            @test isapprox(rem2pi(anoms[i][2], RoundNearest), rem2pi(vec[3+i], RoundNearest); rtol=sqrt(eps(Float64)))
        end
    end
end