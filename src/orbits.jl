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

@kwdef struct OptimParameters{T<:AbstractFloat}
    e1::T
    e2::T
    M::T
    Δω::T
end

# Converts OptimParameters to a vector
function tovector(x::OptimParameters)
    return [getfield(x, field) for field in fieldnames(typeof(x))]
end

Base.Broadcast.broadcastable(x::OptimParameters) = Ref(x)

@kwdef struct OrbitParameters{T<:AbstractFloat}
    # TODO: Implement this in a better way somehow
    m1::T
    m2::T 
    κ::T 
end

"""
Alternative constructor for `Orbit` object

Takes the number of planets, `OptimParameters` object and (for now) `κ`
"""
Orbit(N::Int, optparams::OptimParameters{T}, κ::T) where T <: AbstractFloat = begin
    ONE_YEAR = 365.242

    # t0 corrections for initialization
    t0_init_1 = M2t0(0., optparams.e1, ONE_YEAR, 0.)
    t0_init_2 = M2t0(optparams.M, optparams.e2, κ*ONE_YEAR, optparams.Δω)

    p1 = Elements(m=1)
    p2 = Elements(m=1e-4, P=ONE_YEAR, e=optparams.e1, ω=0, I=π/2, t0=t0_init_1)
    p3 = Elements(m=1e-4, P=κ*ONE_YEAR, e=optparams.e2, ω=optparams.Δω, I=π/2, t0=t0_init_2)

    ic = ElementsIC(0., N+1, p1, p2, p3)
    s = State(ic)

    Orbit(s, ic, κ)
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
