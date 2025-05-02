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
        # Rotation object to make initialization starts on x-axis
        # TODO: rotate about normal vector instead of the z-axis
        rotmat = RotXYZ(0,0,π/2)
        s.x .= rotmat * s.x
        s.v .= rotmat * s.v

        nplanet = ic.nbody - 1
    
        new{T}(s, ic, κ, nplanet)
    end
end

@kwdef mutable struct OptimParameters{T<:AbstractFloat}
    e1::T
    e2::T
    M::T
    Δω::T
end

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
    p1 = Elements(m=1)
    p2 = Elements(m=1e-4, P=365.242, e=optparams.e1, ω=0,)
    p3 = Elements(m=1e-4, P=κ*365.242, e=optparams.e2, ω=optparams.Δω,)

    ic = ElementsIC(0., N+1, p1, p2, p3)
    s = State(ic)

    orbit = Orbit(s, ic, κ)

    # TODO: Implement mean anomaly initialization
    # init_from_M!(o.s, o.ic, optparams.M, 2)
    # init_from_M!(o.s, o.ic, optparams.M, 3)

    orbit
end

function optimize!(orbit::Orbit)
    # Target mean anomaly for two planet system is 4π
    target_M = 4π
    target_time, times, Ms = integrate_to_M!(orbit.s, orbit.ic, target_M, 1);

    orbit.s = State(orbit.ic)
    intr = Integrator(0.5, 0., target_time)
    intr(orbit.s)

    # TODO: Implement the actual optimization
end

Base.show(io::IO,::MIME"text/plain",o::Orbit{T}) where {T} = begin
    println("Orbit")
    println("Current time: $(o.s.t)")
    return
end

"""Initializes a planet using mean anomaly"""
function init_from_M!(s::State, ic::InitialConditions, M::T, index::Int) where T <: AbstractFloat
    # TODO: Also implement a version taking an array of M instead of an individual index
    # TODO: Fix the accuracy issue

    elems = get_orbital_elements(s, ic)[index]
    e = elems.e
    a = elems.a
    Ω = elems.Ω
    I = elems.I
    ω = elems.ω
    n = 2π / elems.P

    # Get true and eccentric anomalies.
    f = kepler(M, e)
    E = ekepler(M, e)

    # Rotates x and v on the plane to the true frame.
    # TODO: Handle the relative positive correctly. Might use t0 instead when initializing
    rmag = a * (1 - e^2) / (1 + e * cos(f))
    rplane = [rmag * cos(f), rmag * sin(f), 0]
    vplane = [-n * a^2 * sin(E) / rmag, n * a^2 * sqrt(1-e^2) * cos(E) / rmag, 0]

    # Rotation matrix in M&D
    P1 = [cos(ω) -sin(ω) 0.0; sin(ω) cos(ω) 0.0; 0.0 0.0 1.0]
    P2 = [1.0 0.0 0.0; 0.0 cos(I) -sin(I); 0.0 sin(I) cos(I)]
    P3 = [cos(Ω) -sin(Ω) 0.0; sin(Ω) cos(Ω) 0.0; 0.0 0.0 1.0]
    P321 = P3*P2*P1

    r_new = P321 * rplane
    v_new = P321 * vplane

    s.x[:,index] .= r_new
    s.v[:,index] .= v_new
end