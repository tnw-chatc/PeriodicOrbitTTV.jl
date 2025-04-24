using LinearAlgebra
using NbodyGradient
using NbodyGradient: get_relative_masses, get_relative_positions
using NbodyGradient: Point

mag(x) = sqrt(dot(x,x))
mag(x,v) = sqrt(dot(x,v))

Rdotmag(R::T,V::T,h::T) where T <: AbstractFloat = V^2 - (h/R)^2 

function point2vector(x::NbodyGradient.Point)
    return [x.x, x.y, x.z]
end

function convert_to_elements(x::Vector{T}, v::Vector{T}, Gm::T, t::T) where T <: AbstractFloat
    # (2.126) R^2 = x^2 + y^2 + z^2
    r = mag(x)
   # (2.127) V^2 = vx^2 + vy^2 + vz^2
    v2 = dot(v, v)
    vmag = mag(v) # Magnitude of velocity
    # (2.129) h = r × v
    hvec = cross(x, v)
    h = mag(hvec)
    # Specific orbital energy (2.28): ε = v^2/2 - Gm/r
    ε = 0.5 * v2 - Gm / r
    # (2.134) a = [2/r - v^2 / Gm]^-1
    if abs(ε) < 1e-10
        # Parabolic
        a = Inf
        e = 1.0
    else
        a = -Gm / (2ε)
        # (2.135) e = sqrt(1 - h^2 / (Gm * a))
        e = sqrt(1.0 + 2ε * h^2 / Gm^2)
    end
    # (2.131) inclination I = arccos(hz / h)
    # hx, hy, hz = unpack(hvec)
    hx, hy, hz = hvec
    I = acos(clamp(hz / h, -1, 1))
    # (2.132-2.133) Ω from hvec
    Ω = I > 1e-8 ? atan(hx, -hy) : 0.0
    evec = (cross(v, hvec) / Gm) .- (x / r)
    ω = 0.0
    if I > 1e-8
        nvec = [-hy, hx, 0.0]
        n = mag(nvec)
        if n > 0
            ω = acos(clamp(dot(nvec, evec) / (n * e), -1, 1))
            if evec[3] < 0
                ω = 2π - ω
            end
        end
    else
        ω = atan(evec[2], evec[1])
        if evec[3] < 0
            ω = 2π - ω
        end
    end
    # (2.20) true anomaly f = angle between evec and r
    cosf = clamp(dot(evec, x) / (e * r), -1, 1)
    sinf = dot(x, v) / (e * sqrt(Gm * a))
    f = atan(sinf, cosf)
    # (2.42) r = a(1 - e cosE)
    # (2.50, 2.51) Eccentric anomaly and mean anomaly
    E = 0.0
    M = 0.0
    if e < 1.0
        E = 2 * atan( sqrt((1 - e) / (1 + e)) * tan(f / 2) )
        M = E - e * sin(E)
    elseif e ≈ 1.0
        M = 0.0
    else
        # Hyperbolic orbit
        F = asinh( sqrt((e - 1) / (1 + e)) * tanh(f / 2) )
        M = e * sinh(F) - F
    end
    # (2.39) and (2.140)  mean anomaly , time of periapsis
    n = sqrt(Gm / abs(a^3))
    τ = t - M / n
    return (
        a=a,
        e=e,
        I=I,
        Ω=mod2pi(Ω),
        ω=mod2pi(ω),
        f=mod2pi(f),
        M=mod2pi(M),
        E=mod2pi(E),
        τ=τ,
        n=n,
        h=h
    )
end

function get_orbital_elements(s::State{T}, ic::InitialConditions{T}) where T <: AbstractFloat
    elems = Elements{T}[]
    μs = get_relative_masses(ic)
    X, V = get_relative_positions(s.x, s.v, ic)
    X = point2vector.(X)
    V = point2vector.(V)
    push!(elems, Elements(m=ic.m[1]))  # Central body
    i = 1; b = 0
    while i < ic.nbody
        if first(ic.ϵ[i, :]) == zero(T)
            b += 1
        end
        a, e, I, Ω, ω, f, M, E, τ, n, h = convert_to_elements(X[i+b], V[i+b], μs[i+b], s.t[1])
        push!(elems, Elements(ic.m[i+1], 2π / n, 0.0, e*cos(ω), e*sin(ω), I, Ω, a, e, ω, τ))
        if b > 0
            b -= 2
        elseif b < 0
            i += 1
        end
        i += 1
    end
    return elems
end

function get_anomalies(s::State{T}, ic::InitialConditions{T}) where T <: AbstractFloat
    anoms = Vector{T}[]
    μs = get_relative_masses(ic)
    X, V = get_relative_positions(s.x, s.v, ic)
    X = point2vector.(X)
    V = point2vector.(V)
    i = 1; b = 0
    while i < ic.nbody
        if first(ic.ϵ[i, :]) == zero(T)
            b += 1
        end
        a, e, I, Ω, ω, f, M, E, τ, n, h = convert_to_elements(X[i+b], V[i+b], μs[i+b], s.t[1])
        push!(anoms, [f, M, E])
        if b > 0
            b -= 2
        elseif b < 0
            i += 1
        end
        i += 1
    end
    return anoms
end