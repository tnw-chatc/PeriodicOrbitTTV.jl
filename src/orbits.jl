using NbodyGradient: State, Elements, ElementsIC, InitialConditions
using Rotations

struct Orbit{T<:AbstractFloat}
    s::State
    ic::InitialConditions
    κ::T

    function Orbit(s::State, ic::InitialConditions, κ::T) where T <: AbstractFloat
        # Rotation object to make initialization starts on x-axis
        rotmat = RotXYZ(0,0,π/2)
        s.x .= rotmat * s.x
        s.v .= rotmat * s.v
    
        new{T}(s, ic, κ)
    end
end
