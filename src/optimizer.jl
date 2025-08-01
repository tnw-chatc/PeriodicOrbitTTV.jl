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
- `prior_weight::Float64` : The weight of priors. Default to 1e8 for masses, kappa, and ω1. The lenght must equal `5*nplanet`
"""
function find_periodic_orbit(optparams::OptimParameters{T}, orbparams::OrbitParameters{T}; 
    use_jac::Bool=true, trace::Bool=false,eccmin::T=1e-3,maxit::Int64=1000, 
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

    # Check and initialize default prior weights
    if prior_weights !== nothing && length(prior_weights) != 5 * nplanet + 1
        error("Inconsistent prior weights. Expected $(5 * nplanet + 1), got $(length(prior_weights)) instead.")
    end

    if prior_weights === nothing
        fit_weight = vcat(fill(1e1, 4*nplanet-2), fill(0., 4*nplanet-2), fill(1e8, nplanet+3))
    else
        fit_weight = vcat(fill(1e1, 4*nplanet-2), prior_weights)
    end

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
            fill(0.5, nplanet-2),
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
    final_optparams = OptimParameters(nplanet, vcat(final_elems, 0.))

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
