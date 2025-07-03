using NbodyGradient
using LinearAlgebra
using Optim

using LsqFit

"""
    find_periodic_orbit(optparams::OptimParameters{T}, orbparams::OrbitParameters{T}; use_jac=false, verbose=false) where T <: AbstractFloat

Find a periodic configuration that is periodic. Return the final parameter vector as `OptimParameters`.

# Arguments
- `optparams::OptimParameters{T}` : The optimization parameters. See its constructor's docstring.
- `orbparams::OrbitParameters{T}` : The orbit parameters. These parameters are static during the entire optimizations. See its constructor's docstring.

# Optionals
- `use_jac::Bool` : Calculate and use Jacobians for optimization if true. False by default.
- `verbose::Bool` : Print optimization log if true. False by default.
"""
function find_periodic_orbit(optparams::OptimParameters{T}, orbparams::OrbitParameters{T}; 
    use_jac::Bool=true, weighted::Bool=true) where T <: Real

    function objective_function(_, p)
        optparams = OptimParameters(nplanet, p)

        diff_squared = compute_diff_squared(optparams, orbparams, nplanet; weighted=weighted)

        return diff_squared
    end

    function jacobian(_, p)
        optparams = OptimParameters(nplanet, p)

        jac = compute_diff_squared_jacobian(optparams, orbparams, nplanet; weighted=weighted)

        return jac
    end 

    optvec = tovector(optparams)
    nplanet = length(orbparams.mass)
    weights = orbparams.weights

    # Dummy data
    xdata = zeros(T, length(optvec))
    # Target is all zero
    ydata = zeros(T, length(optvec))

    # TODO: May have to make it easier to configure
    lower_bounds = Float64[]
    upper_bounds = Float64[]

    # Bounds for eccentricities
    append!(lower_bounds, fill(0.001, nplanet))
    append!(upper_bounds, fill(0.9, nplanet))
        
    # Bounds for mean anomalies
    append!(lower_bounds, fill(-π, nplanet-1))
    append!(upper_bounds, fill(π, nplanet-1))

    # Bounds for omegas
    append!(lower_bounds, fill(-π, nplanet-1))
    append!(upper_bounds, fill(π, nplanet-1))

    # Bounds for period ratio deviations
    append!(lower_bounds, fill(-0.1, nplanet-2))
    append!(upper_bounds, fill(0.1, nplanet-2))

    # Bounds for innermost period
    push!(lower_bounds, 0.5*365.242)
    push!(upper_bounds, 2.0*365.242)

    # Use Autodiffed Jacobian if parsed
    if use_jac
        fit = curve_fit(objective_function, jacobian, xdata, ydata, optvec, lower=lower_bounds, upper=upper_bounds; maxIter=1000)
    else
        fit = curve_fit(objective_function, xdata, ydata, optvec, lower=lower_bounds, upper=upper_bounds; maxIter=1000)
    end

    # Parse the LsqFit object 
    # TODO: May wanna change this later
    return fit
end

function compute_diff_squared(optparams::OptimParameters{T}, orbparams::OrbitParameters{T}, nplanet::Int; weighted::Bool=true) where T <: Real
    orbit = Orbit(nplanet, optparams, orbparams)

    init_elems = extract_elements(orbit.s, orbit.ic, orbparams)
    init_optparams = OptimParameters(nplanet, init_elems)

    final_elems = orbit.final_elem
    final_optparams = OptimParameters(nplanet, final_elems)

    # Calculate the differences for each elements
    diff_e = final_optparams.e - init_optparams.e
    diff_M = rem2pi.(final_optparams.M - init_optparams.M, RoundNearest)
    diff_ωdiff = rem2pi.(final_optparams.Δω - init_optparams.Δω, RoundNearest)
    diff_pratiodev = final_optparams.Pratio - init_optparams.Pratio
    diff_inner_period = final_optparams.inner_period - init_optparams.inner_period

    # Calculated weighted differences if parsed
    if weighted
        weights = orbparams.weights
        
        diff_e *= weights[1]
        diff_M *= weights[2]
        diff_ωdiff *= weights[3]
        diff_pratiodev *= weights[4]
        diff_inner_period *= weights[5]
    end

    # Create a vector
    diff = vcat(diff_e, diff_M, diff_ωdiff, diff_pratiodev, diff_inner_period)

    return diff
end

function compute_diff_squared_jacobian(optparams::OptimParameters{T}, orbparams::OrbitParameters{T}, nplanet::Int; weighted::Bool=true) where T <: Real
    orbit = Orbit(nplanet, optparams, orbparams)

    init_elems = extract_elements(orbit.s, orbit.ic, orbparams)
    init_optparams = OptimParameters(nplanet, init_elems)

    final_elems = orbit.final_elem
    final_optparams = OptimParameters(nplanet, final_elems)

    # Calculate the differences for each elements
    diff_e = final_optparams.e - init_optparams.e
    diff_M = rem2pi.(final_optparams.M - init_optparams.M, RoundNearest)
    diff_ωdiff = rem2pi.(final_optparams.Δω - init_optparams.Δω, RoundNearest)
    diff_pratiodev = final_optparams.Pratio - init_optparams.Pratio
    diff_inner_period = final_optparams.inner_period - init_optparams.inner_period

    # Calculated weighted differences if parsed
    if weighted
        weights = orbparams.weights
        
        diff_e *= weights[1]
        diff_M *= weights[2]
        diff_ωdiff *= weights[3]
        diff_pratiodev *= weights[4]
        diff_inner_period *= weights[5]
    end

    diff_vec = vcat(diff_e, diff_M, diff_ωdiff, diff_pratiodev, diff_inner_period)

    weights = orbparams.weights
    weights_vec = reduce(vcat, [fill(weights[1], length(optparams.e)),
                                fill(weights[2], length(optparams.M)),
                                fill(weights[3], length(optparams.Δω )),
                                fill(weights[4], length(optparams.Pratio)),
                                fill(weights[5], length(optparams.inner_period))])

    jac = weights_vec .* (orbit.jac_combined - I)

    return jac
end
