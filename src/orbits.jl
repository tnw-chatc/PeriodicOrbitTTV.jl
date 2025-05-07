using NbodyGradient: State, Elements, ElementsIC, InitialConditions
using NbodyGradient: kepler, ekepler
using Rotations
using LinearAlgebra: dot

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

"""Vectorized optimization parameters."""
@kwdef struct OptimParameters{T<:AbstractFloat}
    e::Vector{T}
    M::Vector{T}
    Δω::Vector{T}
    Pratio::Vector{T}
end

"""Constructor for optimization parameters."""
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

Base.Broadcast.broadcastable(x::OptimParameters) = Ref(x)

@kwdef struct OrbitParameters{T<:AbstractFloat}
    mass::Vector{T}
    cfactor::Vector{T}
    κ::T
end

"""
Alternative constructor for `Orbit` object

Takes the number of planets, `OptimParameters` object and (for now) `κ`
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
    println("Current time: $(o.s.t)")
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
