using PeriodicOrbit
using PeriodicOrbit: extract_elements
using NbodyGradient
using LinearAlgebra: normalize, cross

using LsqFit
using Optim

optvec_0 = ([0.1, 0.07, 0.05, 0.07,
0., 0., 0., 0.,
0., 0., 0.,
1e-4, 1e-4,
365.242,
3e-6, 5e-6, 7e-5, 3e-5,
2.001,
0.00,
])

orbparams = OrbitParameters((4), 
                                ([0.5, 0.5]),  
                                (8*365.242), 
                                ([1., 1., 5., 3., 2.]))

# optvec_0 = ([0.01, 0.07, 0.05,
# 0., 0., 0.,
# 0., 0.,
# 1e-4,
# 365.242,
# 3e-6, 5e-6, 7e-5,
# 2.001,
# 0.00
# ])

# orbparams = OrbitParameters(3,
#                                 ([0.5, 0.5]), 
#                                 (4*365.242), 
#                                 ([1., 1., 5., 3., 2.]))

nplanet = 4
epsilon = (eps(Float64))

# ForwardDiff
orbit_0 = Orbit(nplanet, OptimParameters(nplanet, deepcopy(optvec_0)), orbparams)

fit = find_periodic_orbit(OptimParameters(nplanet, deepcopy(optvec_0)), orbparams)

# Get the optimized parameters
optres = fit.param


orbit_1 = Orbit(nplanet, OptimParameters(nplanet, deepcopy(optres)), orbparams)

# Extract initial orbital elements and anomalies
initial_elements = get_orbital_elements(orbit_1.s, orbit_1.ic)
initial_anomalies = get_anomalies(orbit_1.s, orbit_1.ic)

println("Initial orbital elements (from optimized parameters):")
for i in 1:orbit_1.nplanet
    elem = initial_elements[i+1]
    anom = initial_anomalies[i]
    println("  Planet $i:")
    println("    a = $(elem.a) AU")
    println("    e = $(elem.e)")
    println("    P = $(elem.P) days")
    println("    ω = $(elem.ω) rad")
    println("    M = $(anom[2]) rad")
end

# Extract final orbital elements and anomalies
final_elements = get_orbital_elements(orbit_1.state_final, orbit_1.ic)
final_anomalies = get_anomalies(orbit_1.state_final, orbit_1.ic)

println("Final orbital elements (from optimized parameters):")
for i in 1:orbit_1.nplanet
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

for i in 1:(orbit_1.nplanet) 
    println("  Planet $(i):")
    
    a_diff = final_elements[i+1].a - initial_elements[i+1].a
    println("    Semi-major axis diff: $(a_diff)")
    global max_diff = max(max_diff, abs(a_diff))
    
    e_diff = final_elements[i+1].e - initial_elements[i+1].e
    println("    Eccentricity diff: $(e_diff)")
    global max_diff = max(max_diff, abs(e_diff))
    
    ω_diff = rem2pi(final_elements[i+1].ω - initial_elements[i+1].ω, RoundNearest)
    println("    Argument of perihelion diff: $(ω_diff)")
    global max_diff = max(max_diff, abs(ω_diff))
    
    M_diff = rem2pi(final_anomalies[i][2] - initial_anomalies[i][2], RoundNearest)
    println("    Mean anomaly diff: $(M_diff)")
    global max_diff = max(max_diff, abs(M_diff))
    
    if i >= 2
        rel_ω_initial = initial_elements[i+1].ω - initial_elements[i].ω
        rel_ω_final = final_elements[i+1].ω - final_elements[i].ω
        rel_ω_diff = rem2pi(rel_ω_final - rel_ω_initial, RoundNearest)
        println("    Relative orientation diff: $(rel_ω_diff)")
        global max_diff = max(max_diff, abs(rel_ω_diff))
    end
    
    println()
end

println("="^60)
println("Maximum absolute difference: $(max_diff)")