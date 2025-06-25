using NbodyGradient: State, Elements, ElementsIC, InitialConditions, CartesianIC, Derivatives
using NbodyGradient: kepler, ekepler, zero_out!
using Rotations
using LinearAlgebra: dot

using ForwardDiff

"""
    Orbit{T<:Real}

Orbit object encapsulates the information about `State` and `InittialConditions` of the system

# Fields
- `s::State` : `State` object of the system
- `ic::InitialConditions` : `InitialConditions` object of the system.
- `nplanet::Int` : The number of the planets in the system
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

    function Orbit(s::State, ic::InitialConditions, κ::T, jac_1::Matrix{T}, jac_2::Matrix{T}, jac_3::Matrix{T}, final_elem::Vector{T}) where T <: Real

        # Gets the number of planets
        nplanet = ic.nbody - 1

        jac_combined = jac_3 * jac_2 * jac_1
    
        new{T}(s, ic, κ, nplanet, jac_1, jac_2, jac_3, jac_combined, final_elem)
    end
end

"""Optimization parameters."""
@kwdef struct OptimParameters{T<:Real}
    e::Vector{T}
    M::Vector{T}
    Δω::Vector{T}
    Pratio::Vector{T}
    inner_period::T
end

"""
    OptimParameters(N::Int, vec::Vector{T}) 

Convert a plain, non-keyworded optimization paramenter vector into OptimParameters object. Used as an argument for Orbit constructor.

# Arguments
- `N:Int` : The number of planets (N >= 2)
- `vec::Vector{T}` : The optimization vector as a plain, non-keyworded vector

`vec::Vector{T}` has a specific order: `N` eccentricities, `N - 1` mean anomalies, `N - 1` omega differences, and `N - 2` period ratios as defined in Gozdziewski and Migaszewski (2020), and `1` innermost planet period. 
One example for a four-planet system:
```
vec = [0.1, 0.2, 0.3, 0.4,  # Eccentricities 
    π, -π/2, 0,             # Mean anomalies
    0., π/2, π,             # Omega differences
    1e-4, 1e-4,             # Period ratios
    365.242]                # Innermost planet period
```
Note that `vec::Vector{T}` must be consistent with the given the number of planets.
"""
function OptimParameters(N::Int, vec::Vector{T}) where T <: Real
    if N == 2
        OptimParameters(vec[1:2], vec[3:3], vec[4:4], T[], vec[5])
    elseif N == 1
        error("N must be greater than 1!")
    end

    if length(vec) != 4 * N - 3
        error("The vector is inconsistent with N! Expected $(4 * N - 3), got $(length(vec)) instead")
    end

    e = vec[1:N]
    M = vec[N+1:2N-1]
    Δω = vec[2N:3N-2]
    Pratio = vec[3N-1:end-1]
    inner_period = vec[end]

    OptimParameters(e, M, Δω, Pratio, inner_period)
end

# Converts OptimParameters to a vector
tovector(x::OptimParameters) = reduce(vcat, [getfield(x, field) for field in fieldnames(typeof(x))])

"""
    OrbitParameters{T<:Real}

Orbital parameters that will not be affected by the optimization

# Fields
- `mass::Vector{T}` : Mass of each planet
- `cfactor::Vector{T}` : Constants C_i defined in G&M (2020)
- `κ::T` : Constant κ defined in G&M (2020)
- `tsys::T` : Periodic orbit system period (i.e., integration time)
- `weights::Vector{T}` : The weights for calculating differences during optimization. The order follows the parameters of `OptimParameters`.

One example for a four-planet system:
```
OrbitParameters([1e-4, 1e-4, 1e-4, 1e-4],   # Masses of the planets
                [0.5, 0.5],                 # C_i factors
                2.000,                      # κ
                8*365.242,                  # Periodic orbit system period
                [1., 1., 5., 3., 2.])       # Optimization weightss         
```

Note that the length of `cfactor::Vector{T}` must be 2 elements shorter than `mass:Vector{T}`

"""
@kwdef struct OrbitParameters{T<:Real}
    mass::Vector{T}
    cfactor::Vector{T}
    κ::T
    tsys::T
    weights::Vector{T}
end

"""
    Orbit(n::Int, optparams::OptimParameters{T}, orbparams::OrbitParameters{T}) where T <: Real

Main constructor for Orbit object. Access states and initial conditions of the system via `s` and `ic` attributes.

# Arguments
- `n::Int` : The number of planets
- `optparams::OptimParameters{T}` : Optimization parameters
- `orbparams::OrbitParameters{T}` : Orbit parameters

# Examples

The following example is to initialize a four-planet system
```
# The order is the same: eccentricities, mean anomalies, ω differences, and period ratios
vec = [0.05, 0.07, 0.05, 0.07,
    0., 0., 0.,
    0., 0., 0.,
    1e-4, 1e-4,
    365.242,
]

# Number of planets, optimization vectors
optparams = OptimParameters(4, vec)
# Three arguements: planet masses vector, C values (in this case a vector consist of 0.5s, and kappa)
orbparams = OrbitParameters([3e-6, 5e-6, 7e-5, 3e-5], [0.5, 0.5], 2.000, 8*365.2422, [1., 1., 5., 3., 2.])

# Orbit object takes three arguments: number of planets, opt params, and orbit params
orbit = Orbit(4, optparams, orbparams)
```

Access `State` and `InitialConditions` using `orbit.s` and `orbit.ic`, respectively.
"""
Orbit(n::Int, optparams::OptimParameters{T}, orbparams::OrbitParameters{U}) where {T <: Real, U <: Real} = begin

    optvec = tovector(optparams)
    periods, mean_anoms, omegas = compute_system_init(optvec, orbparams)

    pos, vel, pos_star, vel_star = orbital_to_cartesian(orbparams.mass, periods, mean_anoms, omegas, optparams.e)

    positions = hcat(pos_star, pos) 
    velocities = hcat(vel_star, vel)

    ic_mat = vcat(vcat(1., orbparams.mass)', vcat(positions, velocities))

    ic = CartesianIC(convert(T, 0.), n+1, permutedims(ic_mat))
    s = State(ic)

    # Compute derivatives (Jac 1)
    jac_1 = compute_derivative_system_init(optvec, orbparams)

    # Compute time evolution Jacobian (Jac 2)
    jac_2, s_final = calculate_jac_time_evolution(deepcopy(s), orbparams.tsys, optparams.inner_period)

    # Export the elements for testing later
    final_elem = extract_elements(deepcopy(s), ic, orbparams)

    # Compute derivatives (Jac 3)
    jac_3 = compute_jac_final(s_final, ic, orbparams)

    Orbit(s, ic, orbparams.κ, jac_1, jac_2, jac_3, final_elem)
end

"""Calculate the system initialization based on optvec (a plain, vectorized version of OptimParameters object)"""
function compute_system_init(optvec::Vector{T}, orbparams::OrbitParameters{U}) where {T <: Real, U <: Real}

    n = length(orbparams.mass)

    optparams = OptimParameters(n, optvec)

    # Initializes arrays
    periods = Vector{T}(undef, n)
    omegas = Vector{T}(undef, n)

    pratio_nom = Vector{T}(undef, n-1)
    pratio_nom[1] = orbparams.κ
    
    for i = 2:n-1
        pratio_nom[i] = 1/(1 + orbparams.cfactor[i-1]*(1 - pratio_nom[i-1]))
    end 

    # Fills in missing M
    mean_anoms = vcat(0.0, optparams.M)

    # Fill in Period ratio deviation for later use
    period_dev = vcat(0.0, optparams.Pratio)

    # Calculates the actual ω's from Δω and periods
    omegas[1] = 0.
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
        optparams = OptimParameters(length(orbparams.mass), x)

        periods, mean_anoms, omegas = compute_system_init(x, orbparams)
        
        pos, vel, pos_star, vel_star = orbital_to_cartesian(orbparams.mass, periods, mean_anoms, omegas, optparams.e)

        positions = hcat(pos_star, pos) 
        velocities = hcat(vel_star, vel)

        mat = vcat(vcat(positions, velocities), vcat(1., orbparams.mass)')

        return mat
    end

    J = ForwardDiff.jacobian(f, optvec)

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
    return copy(state.jac_step), state
end


"""Extract optimization state parameters from `State` and `InitialConditions`"""
function extract_elements(x::Matrix{T}, v::Matrix{T}, masses::Vector{T}, orbparams) where T <: Real
    nplanet = length(masses) - 1

    # Extract orbital elements and anomalies
    elems = get_orbital_elements(x, v, masses)
    anoms = get_anomalies(x, v, masses)

    e = [elems[i].e for i in eachindex(elems)[2:end]]
    M = [anoms[i][2] for i in eachindex(anoms)[2:end]]
    ωdiff = [elems[i].ω - elems[i-1].ω for i in eachindex(elems)[3:end]]

    # Period ratio deviation
    pratio_nom = Vector{T}(undef, nplanet-1)
    pratio_nom[1] = orbparams.κ

    for i = 2:nplanet-1
        pratio_nom[i] = 1/(1 + orbparams.cfactor[i-1]*(1 - pratio_nom[i-1]))
    end 

    pratiodev = [(elems[i].P / elems[i-1].P) - pratio_nom[i-2] for i in eachindex(elems)[4:end]]
    inner_period = elems[2].P

    return vcat(e, M, ωdiff, pratiodev, inner_period)
end

extract_elements(s::State{T}, ic::InitialConditions{T}, orbparams::OrbitParameters{T}) where T <: Real = extract_elements(s.x, s.v, ic.m, orbparams)

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
