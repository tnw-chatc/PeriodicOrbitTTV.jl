module PeriodicOrbit
__precompile__(false)

using Rotations
using LinearAlgebra
using NbodyGradient

export Orbit, OptimParameters, OrbitParameters

export convert_to_elements, get_orbital_elements, get_anomalies, find_periodic_orbit
export get_Jacobi_orbital_elements, get_Jacobi_anomalies
export integrate_to_M!
export optimize!

export orbital_to_cartesian, compute_derivatives, match_transits, compute_tt_jacobians

# For debugging
export mag
export get_relative_positions, get_relative_masses, M2t0, compute_tt

include("orbits.jl")
include("elements.jl")
include("optimizer.jl")
include("orbital_converter.jl")

end # module PeriodicOrbit