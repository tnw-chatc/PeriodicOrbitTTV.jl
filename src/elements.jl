using LinearAlgebra
using NbodyGradient
using NbodyGradient: hierarchy, amatrix
using NbodyGradient: Point

mag(x) = sqrt(dot(x,x))
mag(x,v) = sqrt(dot(x,v))

Rdotmag(R::T,V::T,h::T) where T <: AbstractFloat = V^2 - (h/R)^2 

function point2vector(x::NbodyGradient.Point)
    return [x.x, x.y, x.z]
end

using LinearAlgebra

function convert_to_elements(x::Vector{T}, v::Vector{T}, Gm::T, t::T) where T <: Real
    R = norm(x)
    v2 = dot(v, v)
    hvec = cross(x, v)
    h = norm(hvec)

    # (2.134)
    a = 1. / (2/R - v2/Gm)

    # (2.135)
    e = sqrt(clamp(1 - h^2 / (Gm * a),0.0,1.0))

    # (2.136)
    I = acos(clamp(hvec[3] / h, -1, 1))

    # (2.137) -check sign convention with eric
    if abs(I) >= 1e-8
        sinΩ = hvec[1] / (h * sin(I))
        cosΩ = -hvec[2] / (h * sin(I))
        Ω = atan(sinΩ, cosΩ)
    else
        Ω = 0.0
    end

    # (2.138)
    if abs(I) >= 1e-8
        sin_ω_plus_f = x[3] / (R * sin(I))
        cos_ω_plus_f = (1 / cos(Ω)) * (x[1] / R + sin(Ω) * sin_ω_plus_f * cos(I))
        ω_plus_f = atan(sin_ω_plus_f, cos_ω_plus_f)
    else
        ω_plus_f = atan(x[2], x[1])
    end

    # ω_plus_f = atan(x[2], x[1])

    # (2.139) assuming Rdot = (x.v)/R which is projection of velocity in radial direction
    sinf = a * (1 - e^2) * dot(x, v) / (h * e * R)
    cosf = (1 / e) * (a * (1 - e^2) / R - 1)
    f = atan(sinf, cosf)

    ω = ω_plus_f - f

    # (2.140)
    # Step 1: Solve for E explicitly from Eq. (2.42)
    cosE = (a - R) / (a * e)
    E = acos(clamp(cosE, -1, 1))

    # Adjust E based on radial velocity
    Rdot = dot(x, v) / R
    if Rdot < 0
        E = 2π - E
    end

    #  Mean anomaly M from Eq. (2.51)
    M = E - e * sin(E)

    #  Mean motion n from Eq. (2.26)
    n_mean = sqrt(Gm / a^3)

    #  τ from M and n (Eq. 2.51 rearranged)
    τ = t - M / n_mean


    return (
        a = a,
        e = e,
        I = I,
        Ω = mod2pi(Ω),
        ω = mod2pi(ω),
        f = mod2pi(f),
        M = mod2pi(M),
        E = mod2pi(E),
        τ = τ,
        n = n_mean,
        h = h
    )
end

# Return Jacobi relative masses if passed with an IC instead of mass vector
function get_relative_Jacobi_masses(masses::Vector{T}) where T <: Real
    N = length(masses)
    M = zeros(N-1)
    G = 39.4845/(365.242 * 365.242) # AU^3 Msol^-1 Day^-2
    Hmat = hierarchy([N, ones(Int64,N-1)...])
    for i in 1:N-1
        for j in 1:N
            M[i] += abs(Hmat[i,j])*masses[j]
        end
    end

    return G .* M
end

"""Calculate the Heliocentric relative mass of each planet."""
function get_relative_masses(masses::Vector{T}) where T <: Real
    G = 39.4845/(365.242 * 365.242) # AU^3 Msol^-1 Day^-2

    return [1. + masses[i] for i in eachindex(masses)[2:end]] .* G
end

"""Calculate the Heliocentrc relative positions of each planet."""
function get_relative_positions(x,v)
    X = x[:,2:end] .- x[:,1]
    V = v[:,2:end] .- v[:,1]

    return X, V
end

""" Calculate the relative Jacobi positions from the A-Matrix. """
function get_relative_Jacobi_positions(x,v,nbody,amat)
    n = nbody
    X = zeros(3,n)
    V = zeros(3,n)

    X .= permutedims(amat*x')
    V .= permutedims(amat*v')

    return X, V
end

"""
A structure to store orbital elements as `NbodyGradient.Elements` cannot handle `Real` types.
"""
struct RealElements{T <: Real}
    m::T
    P::T
    t0::T
    ecosω::T
    esinω::T
    I::T
    Ω::T
    a::T
    e::T
    ω::T
    tp::T
end

# TODO: Quality of life improvement: Implement fancy showing of the elements

# Special function for the primary body
RealElements(m::T) where T <: Real = RealElements(m, zeros(T, 10)...)

function get_Jacobi_orbital_elements(x::Matrix{T}, v::Matrix{T}, masses::Vector{T}; time=0.) where T <: Real
    elems = RealElements[]
    μs = get_relative_Jacobi_masses(masses)
    n = length(masses)
    Hmat = hierarchy([n, ones(Int64,n-1)...])
    amat = NbodyGradient.amatrix(Hmat, masses)
    X, V = get_relative_Jacobi_positions(x, v, n, amat)

    push!(elems, RealElements(masses[1]))  # Central body
    i = 1; b = 0
    while i < length(masses)
        if first(Hmat[i, :]) == zero(T)
            b += 1
        end
        a, e, I, Ω, ω, f, M, E, τ, n, h = convert_to_elements(X[:,i+b], V[:,i+b], μs[i+b], convert(T, time))
        # TODO: Make sure if these conversions are a good thing to do...?
        push!(elems, RealElements(masses[i+1], 2π / n, convert(T, 0.0), e*cos(ω), e*sin(ω), I, convert(T, Ω), a, e, ω, τ))
        if b > 0
            b -= 2
        elseif b < 0
            i += 1
        end
        i += 1
    end
    return elems
end

# Allow calling the function using `State` and `ic` instead of Cartesians
get_Jacobi_orbital_elements(s::State{T}, ic::InitialConditions{T}) where T <: Real = get_Jacobi_orbital_elements(s.x, s.v, ic.m; time=s.t[1])

function get_orbital_elements(x::Matrix{T}, v::Matrix{T}, masses::Vector{T}; time=0.) where T <: Real
    elems = RealElements[]
    μs = get_relative_masses(masses)
    X, V = get_relative_positions(x, v)
    n = length(masses)
    Hmat = hierarchy([n, ones(Int64,n-1)...])

    push!(elems, RealElements(masses[1]))  # Central body
    i = 1; b = 0
    while i < length(masses)
        if first(Hmat[i, :]) == zero(T)
            b += 1
        end
        a, e, I, Ω, ω, f, M, E, τ, n, h = convert_to_elements(X[:,i+b], V[:,i+b], μs[i+b], convert(T, time))
        # TODO: Make sure if these conversions are a good thing to do...?
        push!(elems, RealElements(masses[i+1], 2π / n, convert(T, 0.0), e*cos(ω), e*sin(ω), I, convert(T, Ω), a, e, ω, τ))
        if b > 0
            b -= 2
        elseif b < 0
            i += 1
        end
        i += 1
    end
    return elems
end

# Allow calling the function using `State` and `ic` instead of Cartesians
get_orbital_elements(s::State{T}, ic::InitialConditions{T}) where T <: Real = get_orbital_elements(s.x, s.v, ic.m; time=s.t[1])

function get_anomalies(x::Matrix{T}, v::Matrix{T}, masses::Vector{T}; time=0.) where T <: Real
    anoms = Vector[]
    μs = get_relative_masses(masses)
    X, V = get_relative_positions(x, v)
    n = length(masses)
    Hmat = hierarchy([n, ones(Int64,n-1)...])

    i = 1; b = 0
    while i < length(masses)
        if first(Hmat[i, :]) == zero(T)
            b += 1
        end
        a, e, I, Ω, ω, f, M, E, τ, n, h = convert_to_elements(X[:,i+b], V[:,i+b], μs[i+b], convert(T, time))
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

# Allow calling the function using `State` and `ic` instead of Cartesians
get_anomalies(s::State{T}, ic::InitialConditions{T}) where T <: Real = get_anomalies(s.x, s.v, ic.m; time=s.t[1])

function get_Jacobi_anomalies(x::Matrix{T}, v::Matrix{T}, masses::Vector{T}; time=0.) where T <: Real
    anoms = Vector[]
    μs = get_relative_Jacobi_masses(masses)
    n = length(masses)
    Hmat = hierarchy([n, ones(Int64,n-1)...])
    amat = NbodyGradient.amatrix(Hmat, masses)
    X, V = get_relative_Jacobi_positions(x, v, n, amat)

    i = 1; b = 0
    while i < length(masses)
        if first(Hmat[i, :]) == zero(T)
            b += 1
        end
        a, e, I, Ω, ω, f, M, E, τ, n, h = convert_to_elements(X[:,i+b], V[:,i+b], μs[i+b], convert(T, time))
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

# Allow calling the function using `State` and `ic` instead of Cartesians
get_Jacobi_anomalies(s::State{T}, ic::InitialConditions{T}) where T <: Real = get_Jacobi_anomalies(s.x, s.v, ic.m; time=s.t[1])
