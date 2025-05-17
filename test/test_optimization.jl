using PeriodicOrbit
using NbodyGradient
using LinearAlgebra: normalize, cross
using Test
using Optimization
using PyPlot, PyCall
using OptimizationOptimJL, Zygote

rebound = pyimport("rebound")

# TODO: Automate this when everything is ready

# Optimization vector (like what we did in Python)
# The order is the same: eccentricities, mean anomalies, ω differences, and period ratios
vec = [0.1, 0.1, 0.1, 0.1,
    0., 0., 0.,
    0., 0., 0.,
    1e-4, 1e-4,
    365.2422]

# Number of planets, optimization vectors
optparams = OptimParameters(4, vec)
# Three arguements: planet masses vector, C values (in this case a vector consist of 0.5s, and kappa)
orbparams = OrbitParameters([1e-4, 3e-4, 5e-4, 9e-4], [0.5, 0.5], 2.000, 8*365.2422, [1., 1., 5., 3., 2.])

# OPTIMIZATION
optres = find_periodic_orbit(optparams, orbparams)

# Orbit object takes three arguments: number of planets, opt params, and orbit params
o = Orbit(4, optres, orbparams)

# Rebound visualization
sim = rebound.Simulation()
sim.add(m=1)
sim.G = 4π^2 / 365.242^2
for i=1:4
    sim.add(m=orbparams.mass[i], x=o.s.x[1,i+1], y=o.s.x[2,i+1], z=o.s.x[3,i+1], vx=o.s.v[1,i+1], vy=o.s.v[2,i+1], vz=o.s.v[3,i+1])
end

rebound.OrbitPlot(sim, projection="xz")
plt.title("Before")

step_size = 0.1 * optres.inner_period
ctime = 0.0

intr = Integrator(ahl21!, 10., orbparams.tsys)
intr(o.s)

# Rebound visualization
sim = rebound.Simulation()
sim.add(m=1)
sim.G = 4π^2 / 365.242^2
for i=1:4
    sim.add(m=orbparams.mass[i], x=o.s.x[1,i+1], y=o.s.x[2,i+1], z=o.s.x[3,i+1], vx=o.s.v[1,i+1], vy=o.s.v[2,i+1], vz=o.s.v[3,i+1])
end

rebound.OrbitPlot(sim, projection="xz")
plt.title("After")