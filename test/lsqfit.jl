using PeriodicOrbit
using NbodyGradient
using Test


include("../src/Lsqfit.jl")

function verify_periodic_orbit()
    vec = [0.05, 0.07, 0.05, 0.07, 0., 0., 0., 0., 0., 0., 1e-4, 1e-4, 365.242]
    optparams = OptimParameters(4, vec)
    orbparams = OrbitParameters([3e-6, 5e-6, 7e-5, 3e-5], [0.5, 0.5], 2.000, 8*365.2422, [1., 1., 5., 3., 2.])
    
    optimized_params, opt_result = find_periodic_orbit_practical_fixed(optparams, orbparams; verbose=false)
    

    println("  Final objective: $(Optim.minimum(opt_result))")
    println("  Converged: $(Optim.converged(opt_result))")
    
   
    println("Testing periodicity")
    println("="^60)
    

    o = Orbit(4, optimized_params, orbparams)
    
    # Extract initial orbital elements and anomalies
    initial_elements = get_orbital_elements(o.s, o.ic)
    initial_anomalies = get_anomalies(o.s, o.ic)
    
    println("Initial orbital elements (from optimized parameters):")
    for i in 1:o.nplanet
        elem = initial_elements[i+1]
        anom = initial_anomalies[i]
        println("  Planet $i:")
        println("    a = $(elem.a) AU")
        println("    e = $(elem.e)")
        println("    P = $(elem.P) days")
        println("    ω = $(elem.ω) rad")
        println("    M = $(anom[2]) rad")
    end
    
    # Create a FRESH orbit for integration 
    o_test = Orbit(4, optimized_params, orbparams)
    
    # Integrate for the full system period
    println("\nIntegrating for system period: $(orbparams.tsys) days...")
    step_size = 0.1 * optimized_params.inner_period
    intr = Integrator(ahl21!, 10., orbparams.tsys)
    intr(o_test.s)
    
    println("  Final time: $(o_test.s.t[1]) days")
    
    # Extract final orbital elements and anomalies
    final_elements = get_orbital_elements(o_test.s, o_test.ic)
    final_anomalies = get_anomalies(o_test.s, o_test.ic)
    
    println("\nFinal orbital elements (after integration):")
    for i in 1:o.nplanet
        elem = final_elements[i+1]
        anom = final_anomalies[i]
        println("  Planet $i:")
        println("    a = $(elem.a) AU")
        println("    e = $(elem.e)")
        println("    P = $(elem.P) days")
        println("    ω = $(elem.ω) rad")
        println("    M = $(anom[2]) rad")
    end
    
  
    println("PERIODICITY CHECK: DIFFERENCES (final - initial)")
   
    max_diff = 0.0
    
    for i in 1:(o.nplanet) 
        println("  Planet $(i):")
        
        a_diff = final_elements[i+1].a - initial_elements[i+1].a
        println("    Semi-major axis diff: $(a_diff)")
        max_diff = max(max_diff, abs(a_diff))
        
        e_diff = final_elements[i+1].e - initial_elements[i+1].e
        println("    Eccentricity diff: $(e_diff)")
        max_diff = max(max_diff, abs(e_diff))
       
        ω_diff = rem2pi(final_elements[i+1].ω - initial_elements[i+1].ω, RoundNearest)
        println("    Argument of perihelion diff: $(ω_diff)")
        max_diff = max(max_diff, abs(ω_diff))
        
        M_diff = rem2pi(final_anomalies[i][2] - initial_anomalies[i][2], RoundNearest)
        println("    Mean anomaly diff: $(M_diff)")
        max_diff = max(max_diff, abs(M_diff))
        
        if i >= 2
            rel_ω_initial = initial_elements[i+1].ω - initial_elements[i].ω
            rel_ω_final = final_elements[i+1].ω - final_elements[i].ω
            rel_ω_diff = rem2pi(rel_ω_final - rel_ω_initial, RoundNearest)
            println("    Relative orientation diff: $(rel_ω_diff)")
            max_diff = max(max_diff, abs(rel_ω_diff))
        end
        
        println()
    end
    
   
    println("="^60)
    println("Maximum absolute difference: $(max_diff)")
   
    
    return optimized_params, max_diff
end


params, max_diff = verify_periodic_orbit()
