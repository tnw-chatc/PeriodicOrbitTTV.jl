using PeriodicOrbit
using NbodyGradient
using LinearAlgebra: normalize, cross
using Test
using PyCall, PyPlot

rebound = pyimport("rebound")

# Unit testing for 2-planet system which is specially handled due to not having period ratio deviations
# TODO: Automate this

# Optimization vector (like what we did in Python)
# The order is the same: eccentricities, mean anomalies, ω differences, and period ratios
vec = [0.2, 0.1,
    0.,
    0.,
    365.242,
]

# Number of planets, optimization vectors
optparams = OptimParameters(2, vec)
# Three arguements: planet masses vector, C values (in this case a vector consist of 0.5s, and kappa)
orbparams = OrbitParameters([3e-6, 5e-6], [0.], 2.000, 4*365.2422, [1., 1., 5., 3., 2.])

# OPTIMIZATION
optres = find_periodic_orbit(optparams, orbparams)

# Orbit object takes three arguments: number of planets, opt params, and orbit params
o = Orbit(2, optres, orbparams)

initial_elements = get_orbital_elements(o.s, o.ic)
initial_anomalies = get_anomalies(o.s, o.ic)

# Rebound visualization
sim = rebound.Simulation()
sim.add(m=1)
sim.G = 4π^2 / 365.242^2
for i=1:o.nplanet
    sim.add(m=orbparams.mass[i], x=o.s.x[1,i+1], y=o.s.x[2,i+1], z=o.s.x[3,i+1], vx=o.s.v[1,i+1], vy=o.s.v[2,i+1], vz=o.s.v[3,i+1])
end



rebound.OrbitPlot(sim, projection="xz")
plt.title("Before")

# Make sure we are using a new Orbit object
o = Orbit(2, optres, orbparams)

# Integration
step_size = 0.1 * optres.inner_period
ctime = 0.0
intr = Integrator(ahl21!, 10., orbparams.tsys)
intr(o.s)

final_elements = get_orbital_elements(o.s, o.ic)
final_anomalies = get_anomalies(o.s, o.ic)

# Rebound visualization
sim = rebound.Simulation()
sim.add(m=1)
sim.G = 4π^2 / 365.242^2
for i=1:o.nplanet
    println("$i")
    sim.add(m=orbparams.mass[i], x=o.s.x[1,i+1], y=o.s.x[2,i+1], z=o.s.x[3,i+1], vx=o.s.v[1,i+1], vy=o.s.v[2,i+1], vz=o.s.v[3,i+1])
end

rebound.OrbitPlot(sim, projection="xz")
plt.title("After")

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