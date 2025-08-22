using NbodyGradient: State, Elements, ElementsIC, InitialConditions, CartesianIC, Derivatives, TransitTiming
using NbodyGradient: kepler, ekepler, zero_out!
using Rotations
using LinearAlgebra: dot

using ForwardDiff

"""
    Orbit{T<:Real}

Orbit object encapsulates the information about `State` and `InittialConditions` of the system

# Fields
- `s::State` : `State` object of the system.
- `ic::InitialConditions` : `InitialConditions` object of the system.
- `κ::T` : The κ constant for a family of the periodic orbit.
- `nplanet::Int` : The number of the planets in the system.

- `jac_1::Matrix{T}` : The Jacobian for Orbital elements to Cartesian conversion
- `jac_2::Matrix{T}` : The Jacobian for Time Evolution (imported from `NbodyGradient.jl`)
- `jac_3::Matrix{T}` : The Jacobian for Cartesian to Orbital elements conversion
- `jac_combined::Matrix{T}` : The combined Jacobian of the final elements w.r.t. initial elements

- `final_elem::Vector{T}` : The final orbital elements after `tsys`
- `state_final::State` : The `State` structure corresponding to `final_elem`
"""
mutable struct Orbit{T<:Real}
    s::State
    ic::InitialConditions
    κ::T
    nplanet::Int

    jac_1::Matrix{T} # Orbital elements to Cartesian
    jac_2::Matrix{T} # Time evolution
    jac_3::Matrix{T} # Cartesian back to final orbital elements
    jac_combined::Matrix{T} # Combined jacobian J3 * J2 * J1

    final_elem::Vector{T}
    state_final::State # State after integration for testing only

    function Orbit(s::State, ic::InitialConditions, κ::T, jac_1::Matrix{T}, jac_2::Matrix{T}, jac_3::Matrix{T}, final_elem::Vector{T}, state_final::State) where T <: Real

        # Gets the number of planets
        nplanet = ic.nbody - 1

        jac_combined = jac_3 * jac_2 * jac_1
        # jac_combined = Matrix{T}(undef, 0, 0)
    
        new{T}(s, ic, κ, nplanet, jac_1, jac_2, jac_3, jac_combined, final_elem, state_final)
    end
end

"""Optimization parameters."""
@kwdef struct OptimParameters{T<:Real}
    e::Vector{T}        # nplanet
    M::Vector{T}        # nplanet
    Δω::Vector{T}       # nplanet - 1
    Pratio::Vector{T}   # nplanet - 2
    inner_period::T     # 1
    masses::Vector{T}   # nplanet
    kappa::T            # 1
    ω1::T               # 1
    tsys::T             # 1
end

"""
    OptimParameters(N::Int, vec::Vector{T}) 

Convert a plain, non-keyworded optimization paramenter vector into OptimParameters object. Used as an argument for Orbit constructor.

# Arguments
- `N:Int` : The number of planets (N >= 2)
- `vec::Vector{T}` : The optimization vector as a plain, non-keyworded vector

`vec::Vector{T}` has a specific order: `N` eccentricities, `N` mean anomalies, `N - 1` omega differences, and `N - 2` period ratios as defined in Gozdziewski and Migaszewski (2020), `1` innermost planet period, `N` masses, `1` Kappa, `1` innermost longitude of periastron, and `1` PO system period. `5N+1` elements in total.

One example for a four-planet system:
```
optvec_0 = ([0.1, 0.07, 0.05, 0.07, # Eccentricities
    0., 0., 0., 0.,                 # Mean anomalies
    0., 0., 0.,                     # Omega differences
    1e-4, 1e-4,                     # Period ratio deviations
    365.242,                        # Innermost planet period
    3e-6, 5e-6, 7e-5, 3e-5,         # Masses
    2.000,                          # Kappa
    0.00                            # Innermost longitude of periastron
    8*365.242                       # Periodic orbit system period
])
```
Note that `vec::Vector{T}` must be consistent with the given the number of planets, which is `5N+1`.
"""
function OptimParameters(N::Int, vec::Vector{T}) where T <: Real
    # TODO: Have to do this some time later
    if N == 2
        OptimParameters(vec[1:2], vec[3:4], vec[5:5], T[], vec[6], vec[7:8], vec[9], vec[10], vec[11])
    elseif N == 1
        error("N must be greater than 1!")
    end

    if length(vec) != 5 * N + 1
        error("The vector is inconsistent with N! Expected $(5 * N + 1), got $(length(vec)) instead")
    end

    e = vec[1:N]
    M = vec[N+1:2N]
    Δω = vec[2N+1:3N-1]
    Pratio = vec[3N:4N-3]
    inner_period = vec[4N-2]
    masses = vec[4N-1:5N-2]
    kappa = vec[end-2]
    ω1 = vec[end-1]
    tsys = vec[end]

    OptimParameters(e, M, Δω, Pratio, inner_period, masses, kappa, ω1, tsys)
end

# Converts OptimParameters to a vector
tovector(x::OptimParameters) = reduce(vcat, [getfield(x, field) for field in fieldnames(typeof(x))])

"""
    OrbitParameters{T<:Real}

Orbital parameters that will not be affected by the optimization

# Fields
- `nplanet::Int` : The number of the planets
- `cfactor::Vector{T}` : Constants C_i defined in G&M (2020)

One example for a four-planet system:
```
OrbitParameters(4,                          # The number of the planets
                [0.5, 0.5])                # C_i factors 
```

Note that the length of `cfactor::Vector{T}` must be 2 elements shorter than `mass:Vector{T}`

"""
@kwdef struct OrbitParameters{T<:Real}
    nplanet::Int
    cfactor::Vector{T}
    obstmax::T
end

"""
    Orbit(n::Int, optparams::OptimParameters{T}, orbparams::OrbitParameters{T}) where T <: Real

Main constructor for Orbit object. Access states and initial conditions of the system via `s` and `ic` attributes.

# Arguments
- `n::Int` : The number of planets
- `optparams::OptimParameters{T}` : Optimization parameters
- `orbparams::OrbitParameters{T}` : Orbit parameters)

# Orbit object takes three arguments: number of planets, opt params, and orbit params
orbit = Orbit(4, optparams, orbparams)
```

Access `State` and `InitialConditions` using `orbit.s` and `orbit.ic`, respectively. 
The Jacobians of the optimization can be called from fields `jac_1`, `jac_2`, `jac_3`, and `jac_combined`.
The final orbital elements and its `State` can also be called from `Orbit` structure.
"""
Orbit(n::Int, optparams::OptimParameters{T}, orbparams::OrbitParameters{U}) where {T <: Real, U <: Real} = begin

    optvec = tovector(optparams)
    periods, mean_anoms, omegas = compute_system_init(optvec, orbparams)

    pos, vel, pos_star, vel_star = orbital_to_cartesian(optparams.masses, periods, mean_anoms, omegas, optparams.e)

    positions = hcat(pos_star, pos) 
    velocities = hcat(vel_star, vel)

    ic_mat = vcat(vcat(1., optparams.masses)', vcat(positions, velocities))

    ic = CartesianIC(convert(T, 0.), n+1, permutedims(ic_mat))
    s = State(ic)

    # # Compute derivatives (Jac 1)
    jac_1 = compute_derivative_system_init(optvec, orbparams)

    # # Compute time evolution Jacobian (Jac 2)
    jac_2, s_final = calculate_jac_time_evolution(deepcopy(s), optparams.tsys, optparams.inner_period)

    # # Export the elements for testing later
    final_elem = extract_elements(deepcopy(s_final), ic, orbparams)

    # # Compute derivatives (Jac 3)
    jac_3 = compute_jac_final(s_final, ic, orbparams)

    Orbit(s, ic, optparams.kappa, jac_1, jac_2, jac_3, final_elem, s_final)
end

"""Calculate periods, mean anomalies, and longitudes of periastron based on optvec (a plain, vectorized version of OptimParameters object). These quantities will be used to initialize the `Orbit` structure."""
function compute_system_init(optvec::Vector{T}, orbparams::OrbitParameters{U}) where {T <: Real, U <: Real}

    n = orbparams.nplanet

    optparams = OptimParameters(n, optvec)

    # Initializes arrays
    periods = Vector{T}(undef, n)
    omegas = Vector{T}(undef, n)

    pratio_nom = Vector{T}(undef, n-1)
    pratio_nom[1] = optparams.kappa
    
    for i = 2:n-1
        pratio_nom[i] = 1/(1 + orbparams.cfactor[i-1]*(1 - pratio_nom[i-1]))
    end 

    # Fills in missing M
    mean_anoms = optparams.M

    # Fill in Period ratio deviation for later use
    period_dev = vcat(0.0, optparams.Pratio)

    # Calculates the actual ω's from Δω and periods
    omegas[1] = optparams.ω1
    periods[1] = optparams.inner_period
    for i = 2:n
        periods[i] = (pratio_nom[i-1] + period_dev[i-1]) * periods[i-1]
        omegas[i] = optparams.Δω[i-1] + omegas[i-1]
    end

    return periods, mean_anoms, omegas

end

"""Compute Jacobian 1 (orbital elements to Cartesians)"""
function compute_derivative_system_init(optvec, orbparams)

    # Function for AutoDiff
    function f(x)
        optparams = OptimParameters(orbparams.nplanet, x)

        periods, mean_anoms, omegas = compute_system_init(x, orbparams)
        
        pos, vel, pos_star, vel_star = orbital_to_cartesian(optparams.masses, periods, mean_anoms, omegas, optparams.e)

        positions = hcat(pos_star, pos) 
        velocities = hcat(vel_star, vel)

        mat = vcat(vcat(positions, velocities), vcat(1., optparams.masses)')

        return mat
    end

    J = ForwardDiff.jacobian(f, optvec)

    # Append time dependence gradient to jacobian
    J = vcat(J, fill(zero(eltype(J)), size(J, 2))')

    # Ensure that derivative w.r.t to itself is 1
    J[end, end] = one(eltype(J))

    return J
end

"""Compute Jacobian 2 (Cartesian time evolution)

Use a deepcopied `State` only, as the integrator mutates the `State` object."""
function calculate_jac_time_evolution(state::State{T}, tsys::T, inn_period::T) where T <: Real
    d = Derivatives(T, state.n)

    # TODO: Do we really need to loop through every step?
    step_size = 0.01 * inn_period
    nsteps = ceil(Int, tsys / step_size)
    h = tsys / nsteps
    t_initial = state.t[1]
    
    zero_out!(d)
    for i in 1:nsteps
        ahl21!(state, d, h)
        state.t[1] = t_initial + (i * h)
    end        

    # Integrator(ahl21!, convert(T, 1.), convert(T, 0.), tsys)(state)

    # Return the time evolution jacobian (Jacobian 2)
    return hcat(copy(state.jac_step), state.dqdt), state
end


"""Extract optimization state parameters from `State` and `InitialConditions`"""
function extract_elements(x::Matrix{T}, v::Matrix{T}, masses::Vector{T}, orbparams) where T <: Real
    nplanet = length(masses) - 1

    # Extract orbital elements and anomalies
    elems = get_orbital_elements(x, v, masses)
    anoms = get_anomalies(x, v, masses)

    e = [elems[i].e for i in eachindex(elems)[2:end]]
    M = [anoms[i][2] for i in eachindex(anoms)[1:end]] # Now include the first mean anomaly
    ωdiff = [elems[i].ω - elems[i-1].ω for i in eachindex(elems)[3:end]]

    kappa = elems[3].P / elems[2].P

    # Period ratio deviation
    pratio_nom = Vector{T}(undef, nplanet-1)
    pratio_nom[1] = kappa

    for i = 2:nplanet-1
        pratio_nom[i] = 1/(1 + orbparams.cfactor[i-1]*(1 - pratio_nom[i-1]))
    end 

    pratiodev = [(elems[i].P / elems[i-1].P) - pratio_nom[i-2] for i in eachindex(elems)[4:end]]
    inner_period = elems[2].P

    if nplanet == 2
        return vcat(e, M, ωdiff, inner_period, masses[2:end], kappa, elems[2].ω)
    else
        return vcat(e, M, ωdiff, pratiodev, inner_period, masses[2:end], kappa, elems[2].ω)
    end
end

# Allow calling the function using `State` and `ic` instead of Cartesians
extract_elements(s::State{T}, ic::InitialConditions{T}, orbparams::OrbitParameters{T}) where T <: Real = extract_elements(s.x, s.v, ic.m, orbparams)

"""Compute Jacobian 3 (Cartesians back to orbital elements)"""
function compute_jac_final(s::State{T}, ic::InitialConditions{T}, orbparams::OrbitParameters{T}) where T <: Real
    input_mat = vcat(s.x, s.v, ic.m')

    # Function for Jacobian 3
    function f(input)
        xx = input[1:3,:]
        vv = input[4:6,:]
        masses = input[7,:]

        return extract_elements(xx, vv, masses, orbparams)
    end

    J = ForwardDiff.jacobian(f, input_mat)

    return J
end


Base.show(io::IO,::MIME"text/plain",o::Orbit{T}) where {T} = begin
    println("Orbit")
    println("Number of planets: $(o.ic.nbody-1)")
    println("Current time: $(o.s.t[1])")
    println("Planet masses: $(o.ic.m[2:end])")
    return
end

"""Calculates t0 offset to correct initialization on the x-axis"""
function M2t0(target_M::T, e::T, P::T, ω::T) where T <: Real
    # Offsetting true anomaly
    f = ω + pi/2

    # Eccentric anomaly
    E = 2 * atan( sqrt((1 - e) / (1 + e)) * tan(f / 2) )

    # Mean anomaly
    M = E - e * sin(E)

    # Time since periastron passage
    tp = P * (M + target_M) / 2π

    # Negates the time since we want to reverse it
    return -tp
end

# ========
# Transit Timing Variation
# ========

# """Deprecated: Create ElementsIC based on `Orbit` structure."""
# function create_elem_ic(orbit::Orbit{T}) where T <: Real
#     ic = orbit.ic
#     state = orbit.s

#     elems = get_orbital_elements(state, ic)
#     anoms = get_anomalies(state, ic)

#     # Constructing vector
#     elem_mat = reduce(vcat, 
#         [[orbit.s.m[i], elems[i].P, 0., elems[i].e*cos(elems[i].ω), elems[i].e*sin(elems[i].ω), deg2rad(90.), 0.]' 
#             for i = 1:orbit.nplanet+1])

#     elem_mat[:,3] = vcat(0., M2t0.([anoms[i][2] for i in 1:orbit.nplanet], [elems[i+1].e for i in 1:orbit.nplanet], [elems[i+1].P for i in 1:orbit.nplanet], [elems[i+1].ω for i in 1:orbit.nplanet]))
    
#     return ElementsIC(convert(T, 0.), orbit.nplanet+1, elem_mat)
# end

"""Compute transit timing"""
function compute_tt(orbit::Orbit{T}, cartIC::CartesianIC{T}, tmax::T) where T <: Real
    tt = TransitTiming(tmax, cartIC)

    h = 0.01 * get_orbital_elements(orbit.s, orbit.ic)[2].P
    intr = Integrator(h, convert(T, 0.0), tmax)

    ss = deepcopy(orbit.s)
    ss.x .= RotXYZ(pi/2,0,0) * ss.x
    ss.v .= RotXYZ(pi/2,0,0) * ss.v

    intr(ss, tt, grad=true)

    return tt
end

function match_transits(data::Matrix{T}, orbit::Orbit{T}, tt::Matrix{T}, count::Vector{Int64}, ntt) where T <: Real
    ntransit = ntt === nothing ? size(data)[1] : ntt
    ip = zeros(Int64,ntransit)
    jp = zeros(Int64,ntransit)
    
    i = 1; ip0 = 0;
    tmod = zeros(T,ntransit); j = 1;

    # elements = create_elem_ic(orbit).elements
    periods = [get_Jacobi_orbital_elements(orbit.s, orbit.ic)[i].P for i in 1:orbit.ic.nbody]

    while i <= ntransit
        ip[i] = data[i,1]+1
        if ip[i] != ip0
            j = 1
            ip0 = ip[i]
        end
        tdiff = Inf
        while j <= count[ip[i]] && tdiff > 0.1*periods[ip[i]]
            tdiff = abs(tt[ip[i],j] - data[i,3])
            j += 1
        end
        
        jp[i] = j-1
        tmod[i] = tt[ip[i],jp[i]]
        i += 1
    end
    return tmod, ip, jp
end

"""Compute the combined Jacobian for TransitTiming"""
function compute_tt_jacobians(orbit::Orbit{T}, orbparams::OrbitParameters{T}, cartIC::CartesianIC, tt_data::Matrix{T}) where T <: Real
    tt = compute_tt(orbit, cartIC, orbparams.obstmax) # TODO: Get rid of the hardcode here

    ntransit = size(tt_data)[1]

    tmod, ip, jp = match_transits(tt_data, orbit, tt.tt, tt.count, nothing)

    # Helper function for rotating the orbit from xz- to xy-planes
    function rotate_orbit(mat, angle=-pi/2)
        new_mat = copy(mat)
        rot_mat = RotXYZ(angle,0,0)
        
        new_mat[1:3,:] = rot_mat * mat[1:3,:]
        new_mat[4:6,:] = rot_mat * mat[4:6,:]

        return new_mat
    end

    raw_tt = [rotate_orbit(tt.dtdq0[ip[i],jp[i],:,:]) for i = 1:ntransit]

    return permutedims(reduce(hcat, [reshape(raw_tt[i], :) for i = 1:ntransit]))
end

"""Temporary TransitTiming structure"""
function TransitTiming(tmax::T,ic::CartesianIC{T},ti::Int64=1) where T<:AbstractFloat
    n = ic.nbody
    temp_state = State(ic)
    periods = [get_Jacobi_orbital_elements(temp_state, ic)[i].P for i in 1:ic.nbody]
    ind = isfinite.(tmax./periods)
    ntt = maximum(ceil.(Int64,abs.(tmax./periods[ind])).+3)
    tt = zeros(T,n,ntt)
    dtdq0 = zeros(T,n,ntt,7,n)
    dtdelements = zeros(T,n,ntt,7,n)
    count = zeros(Int64,n)
    occs = setdiff(collect(1:n),ti)
    dtdq = zeros(T,1,7,n)
    gsave = zeros(T,n)
    s_prior = State(ic)
    s_transit = State(ic)
    return TransitTiming(tt,dtdq0,dtdelements,count,ntt,ti,occs,dtdq,gsave,s_prior,s_transit)
end