using NbodyGradient: State, Elements, ElementsIC, InitialConditions, Derivatives
using NbodyGradient: kepler, ekepler, zero_out!, hierarchy
using Rotations
using LinearAlgebra: dot
using Zygote

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
    elem_matrix::Matrix{T}
    jac_1::Matrix{T} # Orbital elements to ElementsIC
    jac_2::Matrix{T} # ElementsIC to Cartesians
    jac_3::Matrix{T} # Time Evolution
    jac_4::Matrix{T} # Final Cartesians to Final ElementsIC matrix
    jac_5::Matrix{T} # Final ElementsIC matrix to final orbital elements

    function Orbit(s::State, ic::InitialConditions, κ::T, elems::Matrix{T}, 
        jac_1::Matrix{T}, jac_2::Matrix{T}, jac_3::Matrix{T}, jac_4::Matrix{T}, jac_5::Matrix{T}) where T <: Real

        # Gets the number of planets
        nplanet = ic.nbody - 1
    
        new{T}(s, ic, κ, nplanet, elems, jac_1, jac_2, jac_3, jac_4, jac_5)
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
Orbit(n::Int, optparams::OptimParameters{T}, orbparams::OrbitParameters{T}) where T <: Real = begin

    function optparams_to_elementsIC(optvec::Vector{T}) where T <: Real
        nplanet = length(orbparams.mass)
        optparams = OptimParameters(nplanet, optvec)

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

        # Fill in Period ratio deviation for later use
        period_dev = vcat(0.0, optparams.Pratio)

        # Calculates the actual ω's from Δω and periods
        omegas[1] = 0.
        periods[1] = optparams.inner_period
        for i = 2:n
            periods[i] = (pratio_nom[i-1] + period_dev[i-1]) * periods[i-1]
            omegas[i] = optparams.Δω[i-1] + omegas[i-1]
        end

        # Calculates t0 for initialization
        for i = 1:n
            t0_init[i] = M2t0(mean_anoms[i], optparams.e[i], periods[i], omegas[i])
        end

        star_elem = [[1., 0., 0., 0., 0., 0., 0.]]
        planet_elems = vcat([
            [orbparams.mass[i], periods[i], t0_init[i], optparams.e[i]*cos(omegas[i]), optparams.e[i]*sin(omegas[i]), π/2, 0.] for i in 1:nplanet
        ])

        # Combine star and planets and convert to matrix
        elem_matrix = reduce(vcat, vcat(star_elem, planet_elems)')

        return elem_matrix
    end 
    
    # An internal function to calculate time evolution jacobian (Jacobian 3)
    # Use a deepcopied `State` only, as the integrator mutates the `State` object.
    function calculate_jac_time_evolution(state::State{T}, tsys::T, inn_period::T) where T <: Real
        d = Derivatives(T, state.n)

        step_size = 0.01 * inn_period
        nsteps = ceil(Int, tsys / step_size)
        h = tsys / nsteps
        t_initial = s.t[1]
        
        zero_out!(d)
        for i in 1:nsteps
            ahl21!(state, d, h)
            s.t[1] = t_initial + (i * h)
        end        

        # Return the time evolution jacobian (Jacobian 3)
        return copy(state.jac_step)
    end

    # NOTE: I basically copied this snippet from `optimize.jl`. May have to refactor to take care of redundancy.
    """Compute the optimization parameter based on `ElementsIC` matrixs, which will later be used in Finite Difference"""
    function extract_elements(elems::Matrix{T}) where T <: Real
        ic = ElementsIC(convert(T, 0.), nplanet+1, convert(Matrix{T}, elems))
        s = State(ic)

        # Extract orbital elements and anomalies
        # Note that the index of `elems` has the star entries while `anoms` does not.
        elems = get_orbital_elements(s, ic)
        anoms = get_anomalies(s, ic)

        e = [elems[i].e for i in eachindex(elems)[2:end]]
        M = [anoms[i][2] for i in eachindex(anoms)[2:end]] # We only need 2nd to nth mean anomalies
        ωdiff = [elems[i].ω - elems[i-1].ω for i in eachindex(elems)[3:end]]

        # Period ratio deviation
        pratio_nom = Vector{T}(undef, nplanet-1)
        pratio_nom[1] = orbparams.κ
    
        for i = 2:nplanet-1
            pratio_nom[i] = 1/(1 + orbparams.cfactor[i-1]*(1 - pratio_nom[i-1]))
        end 

        pratiodev = [(elems[i].P / elems[i-1].P) - pratio_nom[i-2] for i in eachindex(elems)[4:end]]
        inner_period = elems[2].P

        # Return the results as a vector
        return vcat(e, M, ωdiff, pratiodev, inner_period)
    end


    """Calculate Jacobian 5 (final `ElementsIC` matrix to final orbital elements) using Finite Difference"""
    function calculate_jac_fic_to_felems(elems_0::Matrix{T}) where T <: Real
        # Using machine precision in Float64
        # NOTE: Should we ust `T` instead of just Float64 for BigFloat testing compatability?
        epsilon = eps(Float64)

        derivatives = []
        # Slice out the first row and column (star and masses)
        for i in 2:size(elems_0, 1), j in 2:size(elems_0, 2)

            elems_pos = copy(elems_0)
            elems_neg = copy(elems_0)

            elems_pos[i,j] += epsilon
            elems_neg[i,j] -= epsilon

            jac_pos = extract_elements(elems_pos)
            jac_neg = extract_elements(elems_neg)

            derivative = reduce(vcat, eachcol((jac_pos - jac_neg) / (2 * epsilon)))
            push!(derivatives, derivative)
        end

        derivatives = reduce(hcat, derivatives)
        return derivatives
    end

    # Number of planets    
    nplanet = length(orbparams.mass)

    # Hierarchy matrix
    H = hierarchy([nplanet+1, ones(Int64,nplanet)...])

    # Deepcopy to preserve the original type as ForwardDiff mutates the function (somehow?)
    elem_mat = deepcopy(optparams_to_elementsIC(tovector(optparams)))

    # Define first Jacobian (orbital elements to ElementsIC)
    jac_elems_to_ic = (p -> ForwardDiff.jacobian(optparams_to_elementsIC, p))(tovector(optparams))

    ic = ElementsIC(convert(T, 0.), nplanet+1, convert(Matrix{T}, elem_mat))
    s = State(ic)

    # Drop the star entries
    elem_mat_wo_star = elem_mat[2:end,:]
    jac_elems_to_ic = jac_elems_to_ic[setdiff(1:7*(nplanet+1), 1:nplanet+1:7*(nplanet+1)),:]

    # Define second Jacobian (ElementsIC to Cartesian)
    # Drop the star entries
    jac_ic_to_cart = deepcopy(s.jac_init)[8:end, 8:end]

    # Define third Jacobian (Time evolution)
    # Drop the star entries
    jac_time_evolution = calculate_jac_time_evolution(deepcopy(s), convert(T, orbparams.tsys), get_orbital_elements(s, ic)[2].P)[8:end, 8:end]

    # Define fouth Jacobian (Final Cartesian to Final ElementsIC)
    # Drop the star columns and mass rows
    # TODO: Make sure this is the right thing to do
    jac_fcart_to_fic = compute_cartesian_to_elements_jacobian(deepcopy(s), ic)[1][1:nplanet*6,8:end]

    # Define the fifth Jacobian (Final ElementsIC to orbital elements)
    jac_fic_to_felems = calculate_jac_fic_to_felems(deepcopy(elem_mat))

    Orbit(s, ic, orbparams.κ, elem_mat, jac_elems_to_ic, jac_ic_to_cart, jac_time_evolution, jac_fcart_to_fic, jac_fic_to_felems)
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
