using NbodyGradient: State, Elements, ElementsIC, InitialConditions
using NbodyGradient: kepler, ekepler
using Rotations
using LinearAlgebra: dot

"""
    Orbit{T<:AbstractFloat}

Orbit object encapsulates the information about `State` and `InittialConditions` of the system

# Fields
- `s::State` : `State` object of the system
- `ic::InitialConditions` : `InitialConditions` object of the system.
- `nplanet::Int` : The number of the planets in the system
"""
mutable struct Orbit{T<:AbstractFloat}
    s::State
    ic::InitialConditions
    κ::T
    nplanet::Int

    function Orbit(s::State, ic::InitialConditions, κ::T) where T <: AbstractFloat

        # Gets the number of planets
        nplanet = ic.nbody - 1
    
        new{T}(s, ic, κ, nplanet)
    end
end

"""Optimization parameters."""
@kwdef struct OptimParameters{T<:AbstractFloat}
    e::Vector{T}
    M::Vector{T}
    Δω::Vector{T}
    Pratio::Vector{T}
end

"""
    OptimParameters(N::Int, vec::Vector{T}) 

Convert a plain, non-keyworded optimization paramenter vector into OptimParameters object. Used as an argument for Orbit constructor.

# Arguments
- `N:Int` : The number of planets (N >= 2)
- `vec::Vector{T}` : The optimization vector as a plain, non-keyworded vector

`vec::Vector{T}` has a specific order: `N` eccentricities, `N - 1` mean anomalies, `N - 1` omega differences, and `N - 2` period ratios as defined in Gozdziewski and Migaszewski (2020). 
One example for a four-planet system:
```
vec = [0.1, 0.2, 0.3, 0.4,  # Eccentricities 
    π, -π/2, 0,             # Mean anomalies
    0., π/2, π,             # Omega differences
    1e-4, 1e-4,]            # Period ratios
```
Note that `vec::Vector{T}` must be consistent with the given the number of planets.
"""
function OptimParameters(N::Int, vec::Vector{T}) where T <: AbstractFloat
    if N == 2
        OptimParameters(vec[1:2], vec[3], vec[4], nothing)
    elseif N == 1
        error("N must be greater than 1!")
    end

    @assert length(vec) == 4*N - 4 "The vector is inconsistent with N!"

    e = vec[1:N]
    M = vec[N+1:2N-1]
    Δω = vec[2N:3N-2]
    Pratio = vec[3N-1:end]

    OptimParameters(e, M, Δω, Pratio)
end

# Converts OptimParameters to a vector
tovector(x::OptimParameters) = [getfield(x, field) for field in fieldnames(typeof(x))]

"""
    OrbitParameters{T<:AbstractFloat}

Orbital parameters that will not be affected by the optimization

# Fields
- `mass::Vector{T}` : Mass of each planet
- `cfactor::Vector{T}` : Constants C_i defined in G&M (2020)
- `κ::T` : Constant κ defined in G&M (2020)

One example for a four-planet system:
```
OrbitParameters([1e-4, 1e-4, 1e-4, 1e-4],   # Masses of the planets
                [0.5, 0.5],                 # C_i factors
                2.000)                      # κ
```

Note that the length of `cfactor::Vector{T}` must be 2 elements shorter than `mass:Vector{T}`

"""
@kwdef struct OrbitParameters{T<:AbstractFloat}
    mass::Vector{T}
    cfactor::Vector{T}
    κ::T
end

"""
    Orbit(n::Int, optparams::OptimParameters{T}, orbparams::OrbitParameters{T}) where T <: AbstractFloat

Main constructor for Orbit object. Access states and initial conditions of the system via `s` and `ic` attributes.

# Arguments
- `n::Int` : The number of planets
- `optparams::OptimParameters{T}` : Optimization parameters
- `orbparams::OrbitParameters{T}` : Orbit parameters

# Examples

The following example is to initialize a four-planet system
```
# The order is the same: eccentricities, mean anomalies, ω differences, and period ratios
vec = [0.1, 0.1, 0.1, 0.1,
    1., 1., 2.,
    0., 0., 0.,
    1e-4, 1e-4,]

# Number of planets, optimization vectors
optparams = OptimParameters(4, vec)
# Three arguements: planet masses vector, C values (in this case a vector consist of 0.5s, and kappa)
orbparams = OrbitParameters([1e-4, 1e-4, 1e-4, 1e-4], [0.5, 0.5], 2.000)

# Orbit object takes three arguments: number of planets, opt params, and orbit params
orbit = Orbit(4, optparams, orbparams)
```

Access `State` and `InitialConditions` using `orbit.s` and `orbit.ic`, respectively.
"""
Orbit(n::Int, optparams::OptimParameters{T}, orbparams::OrbitParameters{T}) where T <: AbstractFloat = begin
    ONE_YEAR = 365.242

    # Initializes arrays
    t0_init = Vector{T}(undef, n)
    periods = Vector{T}(undef, n)
    omegas = Vector{T}(undef, n)
    planets = Vector{Elements}(undef, n)

    pratio_nom = Vector{T}(undef, n-1)
    pratio_nom[1] = orbparams.κ
    
    for i = 2:n-1
        pratio_nom[i] = 1/(1 + orbparams.cfactor[i-1]*(1 - pratio_nom[i-1]))
    end 

    # Fills in missing M
    mean_anoms = vcat(0.0, optparams.M)

    # Calculates the actual ω's from Δω and periods
    omegas[1] = 0.
    periods[1] = ONE_YEAR
    for i = 2:n
        periods[i] = pratio_nom[i-1] * periods[i-1]
        omegas[i] = optparams.Δω[i-1] + omegas[i-1]
    end

    # Calculates t0 for initialization
    for i = 1:n
        t0_init[i] = M2t0(mean_anoms[i], optparams.e[i], periods[i], omegas[i])
    end

    # Primary object
    star = Elements(m=1.)

    for i = 1:n 
        planets[i] = Elements(m=orbparams.mass[i], P=periods[i], e=optparams.e[i], ω=omegas[i], I=π/2, t0=t0_init[i])
    end

    ic = ElementsIC(0., n+1, star, planets...)
    s = State(ic)

    Orbit(s, ic, orbparams.κ)
end

"""Will remove soon"""
function optimize!(optparams::OptimParameters{T}) where T <: AbstractFloat
    # Target mean anomaly for two planet system is 4π
    orbit = Orbit(2, optparams, 2.00)

    target_M = 4π
    target_time, times, Ms = integrate_to_M!(orbit.s, orbit.ic, target_M, 1);

    orbit.s = State(orbit.ic)
    intr = Integrator(0.5, 0., target_time)
    intr(orbit.s)

    # TODO: Implement the actual optimization for N-body system (currently only for 2 planets)
    # TODO: Make sure the angle wrapping works as intended
    final_e1 = get_orbital_elements(orbit.s, orbit.ic)[2].e
    final_e2 = get_orbital_elements(orbit.s, orbit.ic)[3].e
    final_M = get_anomalies(orbit.s, orbit.ic)[1][2]
    final_Δω = get_orbital_elements(orbit.s, orbit.ic)[3].ω - get_orbital_elements(orbit.s, orbit.ic)[2].ω

    final_optparams = [final_e1, final_e2, rem2pi(final_M, RoundNearest), rem2pi(final_Δω, RoundNearest)]

    # Init params
    init_optparams = tovector(optparams)
    
    # Wrap angles
    init_optparams[3] = rem2pi(init_optparams[3], RoundNearest)
    init_optparams[4] = rem2pi(init_optparams[4], RoundNearest)

    diff = final_optparams .- init_optparams

    return sum(diff.^2)
end

Base.show(io::IO,::MIME"text/plain",o::Orbit{T}) where {T} = begin
    println("Orbit")
    println("Number of planets: $(o.ic.nbody-1)")
    println("Current time: $(o.s.t[1])")
    println("Planet masses: $(o.ic.m[2:end])")
    return
end

"""Calculates t0 offset to correct initialization on the x-axis"""
function M2t0(target_M::T, e::T, P::T, ω::T) where T <: AbstractFloat
    # Offsetting ω
    ω += π/2

    # Eccentric anomaly
    E = 2 * atan( sqrt((1 - e) / (1 + e)) * tan(ω / 2) )

    # Mean anomaly
    M = E - e * sin(E)

    # Time since periastron passage
    tp = P * (M + target_M) / 2π

    # Negates the time since we want to reverse it
    return -tp
end
