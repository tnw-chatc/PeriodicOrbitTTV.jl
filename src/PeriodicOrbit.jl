module PeriodicOrbit
__precompile__(false)

using Rotations
using LinearAlgebra
using NbodyGradient

export Orbit, OptimParameters, OrbitParameters

export convert_to_elements, get_orbital_elements, get_anomalies, find_periodic_orbit
export integrate_to_M!
export optimize!

export orbital_to_cartesian, compute_derivatives

# For debugging
export normvec
export mag
export get_relative_positions, M2t0

include("orbits.jl")
include("elements.jl")
include("optimizer.jl")
include("orbital_converter.jl")

end # module PeriodicOrbit