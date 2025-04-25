using NbodyGradient: State, Elements, ElementsIC, InitialConditions
using Rotations

mutable struct Orbit{T<:AbstractFloat}
    s::State
    ic::InitialConditions
    κ::T
    nplanet::Int

    function Orbit(s::State, ic::InitialConditions, κ::T) where T <: AbstractFloat
        # Rotation object to make initialization starts on x-axis
        rotmat = RotXYZ(0,0,π/2)
        s.x .= rotmat * s.x
        s.v .= rotmat * s.v

        nplanet = ic.nbody - 1
    
        new{T}(s, ic, κ, nplanet)
    end
end

function optimize!(orbit::Orbit, target::T) where T <: AbstractFloat
    target_time, times, Ms = integrate_to_M!(orbit.s, orbit.ic, target, 1);

    orbit.s = State(orbit.ic)
    intr = Integrator(0.5, 0., target_time)
    intr(orbit.s)
end

Base.show(io::IO,::MIME"text/plain",o::Orbit{T}) where {T} = begin
    println("Orbit")
    println("Current time: $(o.s.t)")
    return
end