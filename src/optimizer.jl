using NbodyGradient
using LinearAlgebra
using Optim

using LsqFit

"""
    find_periodic_orbit(optparams::OptimParameters{T}, orbparams::OrbitParameters{T}; use_jac=false, verbose=false) where T <: Real

Find a periodic configuration that is periodic. Return the final parameter vector as `OptimParameters`.

# Arguments
- `optparams::OptimParameters{T}` : The optimization parameters. See its constructor's docstring.
- `orbparams::OrbitParameters{T}` : The orbit parameters. These parameters are static during the entire optimizations. See its constructor's docstring.

# Optionals
- `use_jac::Bool` : Calculate and use Jacobians for optimization. True by Default.
- `trace::Bool` : Show optimizer trace. False by default.
- `eccmin::T` : The lower bounds for eccentricities
- `maxit::Int64` : Maximum optimizer iterations
- `optim_weights::Float64` : The weights of the optimization parameters. The length must equal `4*nplanet-2`
- `prior_weights::Float64` : The weight of priors. Default to 1e8 for masses, kappa, and ω1. The lenght must equal `5*nplanet+1`
- `lower_bounds::Float64` : The lower bounds of the optimization. The lenght must equal `5*nplanet+1`
- `upper_bounds::Float64` : The upper bounds of the optimization. The lenght must equal `5*nplanet+1`
""" 
function find_periodic_orbit(optparams::OptimParameters{T}, orbparams::OrbitParameters{T}; 
    use_jac::Bool=true, trace::Bool=false,eccmin::T=1e-3,maxit::Int64=1000, optim_weights=nothing,
    prior_weights=nothing, lower_bounds=nothing, upper_bounds=nothing) where T <: Real

    function objective_function(_, p)
        optparams = OptimParameters(nplanet, p)

        diff_squared = compute_diff_squared(optparams, orbparams, nplanet)

        return diff_squared
    end

    function jacobian(_, p)
        optparams = OptimParameters(nplanet, p)

        jac = compute_diff_squared_jacobian(optparams, orbparams, nplanet)

        return jac
    end 

    optvec = tovector(optparams)
    nplanet = orbparams.nplanet

    # Dummy data
    xdata = zeros(T, length(optvec))
    # Target is all zero with the prior appended
    ydata = vcat(zeros(T, 4*nplanet-2), deepcopy(optvec))

    # Check and initialize default optimization weights 
    if optim_weights !== nothing && length(optim_weights) != 4 * nplanet - 2
        error("Inconsistent optimization weights. Expected $(4 * nplanet - 2), got $(length(optim_weights)) instead.")
    end

    if optim_weights === nothing
        optim_weights = fill(convert(T, 1.), 4*nplanet-2)
    end

    # Check and initialize default prior weights
    if prior_weights !== nothing && length(prior_weights) != 5 * nplanet + 1
        error("Inconsistent prior weights. Expected $(5 * nplanet + 1), got $(length(prior_weights)) instead.")
    end

    if prior_weights === nothing
        prior_weights = vcat(fill(zero(T), 4*nplanet-2), fill(1e8, nplanet+3))
    end

    fit_weight = vcat(optim_weights, prior_weights)

    # Check and initialize default lower bounds
    if lower_bounds !== nothing && length(lower_bounds) != 5 * nplanet + 1
        error("Inconsistent lower bounds. Expected $(5 * nplanet + 1), got $(length(lower_bounds)) instead.")
    end

    if lower_bounds === nothing
        lower_bounds = vcat(
            fill(eccmin, nplanet),
            fill(-2π, nplanet),
            fill(-2π, nplanet-1),
            fill(-0.5, nplanet-2),
            0.5*365.242,
            fill(1e-8, nplanet),
            1.9,
            -2π,
            7*365.242,
        )
    end

    # Check and initialize default upper bounds
    if upper_bounds !== nothing && length(upper_bounds) != 5 * nplanet + 1
        error("Inconsistent upper bounds. Expected $(5 * nplanet + 1), got $(length(upper_bounds)) instead.")
    end

    if upper_bounds === nothing
        upper_bounds = vcat(
            fill(0.9, nplanet),
            fill(2π, nplanet),
            fill(2π, nplanet-1),
            fill(100.0, nplanet-2),
            2.0*365.242,
            fill(1e-2, nplanet),
            2.1,
            2π,
            9*365.242,
        )
    end
  
    # Use Autodiffed Jacobian if parsed
    if use_jac
        fit = curve_fit(objective_function, jacobian, xdata, ydata, fit_weight, optvec, lower=lower_bounds, upper=upper_bounds; maxIter=maxit, show_trace=trace)
    else
        fit = curve_fit(objective_function, xdata, ydata, fit_weight, optvec, lower=lower_bounds, upper=upper_bounds; maxIter=maxit, show_trace=trace)
    end

    # Parse the LsqFit object 
    return fit
end

"""Routine for finding PO with a TT constraint"""
function find_periodic_orbit(optparams::OptimParameters{T}, orbparams::OrbitParameters{T}, tt_data::Matrix{T}; 
    use_jac::Bool=true, trace::Bool=false,eccmin::T=1e-3,maxit::Int64=1000, optim_weights=nothing,
    prior_weights=nothing, tt_weights=nothing, lower_bounds=nothing, upper_bounds=nothing) where T <: Real

    function objective_function(_, p)
        optparams = OptimParameters(nplanet, p)

        orbit = Orbit(nplanet, optparams, orbparams)
        elemIC = create_elem_ic(orbit)
        tt = compute_tt(orbit, elemIC, orbparams.obstmax) # TODO: Get rid of the hardcode here

        # Append the TT information
        tmod, ip, jp = match_transits(tt_data, elemIC.elements, tt.tt, tt.count, nothing)

        diff_squared = compute_diff_squared(optparams, orbparams, nplanet, tmod)

        return diff_squared
    end

    function jacobian(_, p)
        optparams = OptimParameters(nplanet, p)

        jac = compute_diff_squared_jacobian(optparams, orbparams, nplanet, tt_data)

        return jac
    end 

    nplanet = orbparams.nplanet

    orbit = Orbit(nplanet, optparams, orbparams)
    elemIC = create_elem_ic(orbit)
    tt = compute_tt(orbit, elemIC, orbparams.obstmax)

    # Append the TT information
    tmod, ip, jp = match_transits(tt_data, elemIC.elements, tt.tt, tt.count, nothing)
    ttobs = tt_data[:,3]

    optvec = tovector(optparams)

    # Target is all zero with the prior and TTV information appended
    ydata = vcat(zeros(T, 4*nplanet-2), deepcopy(optvec), deepcopy(ttobs))
    # Dummy data
    xdata = zeros(T, length(ydata))

    # Check and initialize default optimization weights 
    if optim_weights !== nothing && length(optim_weights) != 4 * nplanet - 2
        error("Inconsistent optimization weights. Expected $(4 * nplanet - 2), got $(length(optim_weights)) instead.")
    end

    if optim_weights === nothing
        optim_weights = fill(convert(T, 1.), 4*nplanet-2)
    end

    # Check and initialize default prior weights
    if prior_weights !== nothing && length(prior_weights) != 5 * nplanet + 1
        error("Inconsistent prior weights. Expected $(5 * nplanet + 1), got $(length(prior_weights)) instead.")
    end

    if prior_weights === nothing
        prior_weights = vcat(fill(zero(T), 4*nplanet-2), fill(1e8, nplanet+3))
    end

    # Check and initialize default TT weights
    if tt_weights !== nothing && length(tt_weights) != length(tmod)
        error("Inconsistent prior weights. Expected $(length(tmod)), got $(length(tt_weights)) instead.")
    end

    if tt_weights === nothing
        tt_weights = ones(T, length(tmod))
    end

    fit_weight = vcat(optim_weights, prior_weights, tt_weights)

    # Check and initialize default lower bounds
    if lower_bounds !== nothing && length(lower_bounds) != 5 * nplanet + 1
        error("Inconsistent lower bounds. Expected $(5 * nplanet + 1), got $(length(lower_bounds)) instead.")
    end

    if lower_bounds === nothing
        lower_bounds = vcat(
            fill(eccmin, nplanet),
            fill(-2π, nplanet),
            fill(-2π, nplanet-1),
            fill(-0.5, nplanet-2),
            0.5*365.242,
            fill(1e-8, nplanet),
            1.9,
            -2π,
            7*365.242,
        )
    end

    # Check and initialize default upper bounds
    if upper_bounds !== nothing && length(upper_bounds) != 5 * nplanet + 1
        error("Inconsistent upper bounds. Expected $(5 * nplanet + 1), got $(length(upper_bounds)) instead.")
    end

    if upper_bounds === nothing
        upper_bounds = vcat(
            fill(0.9, nplanet),
            fill(2π, nplanet),
            fill(2π, nplanet-1),
            fill(100.0, nplanet-2),
            2.0*365.242,
            fill(1e-2, nplanet),
            2.1,
            2π,
            9*365.242,
        )
    end
  
    # Use Autodiffed Jacobian if parsed
    if use_jac
        fit = curve_fit(objective_function, jacobian, xdata, ydata, fit_weight, optvec, lower=lower_bounds, upper=upper_bounds; maxIter=maxit, show_trace=trace)
    else
        fit = curve_fit(objective_function, xdata, ydata, fit_weight, optvec, lower=lower_bounds, upper=upper_bounds; maxIter=maxit, show_trace=trace)
    end

    # Parse the LsqFit object 
    return fit
end

function compute_diff_squared(optparams::OptimParameters{T}, orbparams::OrbitParameters{T}, nplanet::Int) where T <: Real
    orbit = Orbit(nplanet, optparams, orbparams)

    init_optparams = optparams

    final_elems = orbit.final_elem
    final_optparams = OptimParameters(nplanet, vcat(final_elems, zero(T)))

    # Calculate the differences for each elements
    diff_e = final_optparams.e - init_optparams.e
    diff_M = rem2pi.(final_optparams.M - init_optparams.M, RoundNearest)
    diff_ωdiff = rem2pi.(final_optparams.Δω - init_optparams.Δω, RoundNearest)
    diff_pratiodev = final_optparams.Pratio - init_optparams.Pratio
    diff_inner_period = final_optparams.inner_period - init_optparams.inner_period

    # Create a vector, and appended with constant quantities
    diff = vcat(diff_e, diff_M, diff_ωdiff, diff_pratiodev, diff_inner_period)
    
    priors = tovector(optparams)

    return [diff; priors]
end

function compute_diff_squared(optparams::OptimParameters{T}, orbparams::OrbitParameters{T}, nplanet::Int, ttmod::Vector{T}) where T <: Real
    obj_vec = compute_diff_squared(optparams, orbparams, nplanet)

    return [obj_vec; ttmod]
end

function compute_diff_squared_jacobian(optparams::OptimParameters{T}, orbparams::OrbitParameters{T}, nplanet::Int) where T <: Real
    orbit = Orbit(nplanet, optparams, orbparams)

    # Helper function for subtracting the matrix with identity
    function subtract_iden(mat)
        N = size(mat, 1)
        mat[1:N,1:N] = mat[1:N,1:N] - I
        return mat
    end

    # Jacobian of the difference
    diff_jac = subtract_iden(orbit.jac_combined[begin:4*nplanet-2,:])

    return [diff_jac; Matrix{T}(I, 5*nplanet+1, 5*nplanet+1)]
end

function compute_diff_squared_jacobian(optparams::OptimParameters{T}, orbparams::OrbitParameters{T}, nplanet::Int, tt_data::Matrix{T}) where T <: Real
    obj_jac = compute_diff_squared_jacobian(optparams, orbparams, nplanet)

    orbit = Orbit(nplanet, optparams, orbparams)
    elemIC = create_elem_ic(orbit)

    jac_tt_1 = compute_tt_jacobians(orbit, orbparams, elemIC, tt_data)

    tt_jac = jac_tt_1 * orbit.jac_1[1:end-1,:]

    the_jac = [obj_jac; tt_jac]

    return the_jac
end
