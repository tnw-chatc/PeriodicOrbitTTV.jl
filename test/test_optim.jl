using PeriodicOrbit
using NbodyGradient
using LinearAlgebra: normalize, cross
using Test

@testset "Preliminary Optimization" begin
    # Optimization vector (like what we did in Python)
    # The order is the same: eccentricities, mean anomalies, ω differences, and period ratios
    vec = [0.05, 0.07, 0.05, 0.07,
        0., 0., 0.,
        0., 0., 0.,
        1e-4, 1e-4,
        365.242,
    ]

    # Number of planets, optimization vectors
    optparams = OptimParameters(4, vec)
    # Three arguements: planet masses vector, C values (in this case a vector consist of 0.5s, and kappa)
    orbparams = OrbitParameters([3e-6, 5e-6, 7e-5, 3e-5], [0.5, 0.5], 2.000, 8*365.2422, [1., 1., 5., 3., 2.])

    # OPTIMIZATION
    optres = find_periodic_orbit(optparams, orbparams; verbose=false)

    # Orbit object takes three arguments: number of planets, opt params, and orbit params
    o = Orbit(4, optres, orbparams)

    # Extract initial parameters
    initial_elements = get_orbital_elements(o.s, o.ic)
    initial_anomalies = get_anomalies(o.s, o.ic)

    # Make sure we are using a new Orbit object
    o = Orbit(4, optres, orbparams)

    # Integration
    step_size = 0.1 * optres.inner_period
    ctime = 0.0
    intr = Integrator(ahl21!, 10., orbparams.tsys)
    intr(o.s)

    @test o.s.t[1] == orbparams.tsys # Integration time must equal to tsys

    # Extract final parameters
    final_elements = get_orbital_elements(o.s, o.ic)
    final_anomalies = get_anomalies(o.s, o.ic)

    println("\nDifferences (final - initial):")
    for i in 1:(o.nplanet) 
        println("  Planet $(i):")
        println("    Semi-major axis diff: ", final_elements[i+1].a - initial_elements[i+1].a)
        println("    Eccentricity diff: ", final_elements[i+1].e - initial_elements[i+1].e)
        println("    Argument of perihelion diff: ", rem2pi(final_elements[i+1].ω - initial_elements[i+1].ω, RoundNearest))
        println("    Mean anomaly diff: ", rem2pi(final_anomalies[i][2] - initial_anomalies[i][2], RoundNearest))
        
        if i >= 2
            rel_ω_initial = initial_elements[i+1].ω - initial_elements[i].ω
            rel_ω_final = final_elements[i+1].ω - final_elements[i].ω
            println("    Relative orientation diff: ", rem2pi(rel_ω_final - rel_ω_initial, RoundNearest))
        end
    end

end