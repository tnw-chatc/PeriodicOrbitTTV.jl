module PeriodicOrbit
__precompile__(false)

using Rotations
using LinearAlgebra
using NbodyGradient

export Orbit, OptimParameters, OrbitParameters

export convert_to_elements, get_orbital_elements, get_anomalies, find_periodic_orbit
export integrate_to_M!
export optimize!

# For debugging
export normvec
export mag
export get_relative_positions, M2t0
export extract_nbody_jacobians, compute_cartesian_to_elements_jacobian, convert_sic_to_elemmat

include("orbits.jl")
include("elements.jl")
include("optimizer.jl")
include("jacobians.jl")

end # module PeriodicOrbit