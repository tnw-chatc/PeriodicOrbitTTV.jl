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
- `scale_factor::Float64` : The scale factor to help the optimization perform better. The lenght must equal `5*nplanet+1`. Each element should have the same order of magnitude as the corresponding element of the optimization vector.
""" 
function find_periodic_orbit(optparams::OptimParameters{T}, orbparams::OrbitParameters{T}; 
    use_jac::Bool=true, trace::Bool=false,eccmin::T=1e-3,maxit::Int64=1000, optim_weights=nothing,
    prior_weights=nothing, lower_bounds=nothing, upper_bounds=nothing, scale_factor=nothing) where T <: Real

    function objective_function(_, θ)
        p = scale_factor .* θ

        optparams = OptimParameters(nplanet, p)

        diff_squared = compute_diff_squared(optparams, orbparams, nplanet)

        return diff_squared
    end

    function jacobian(_, θ)
        p = scale_factor .* θ

        optparams = OptimParameters(nplanet, p)

        jac = compute_diff_squared_jacobian(optparams, orbparams, nplanet)

        return jac .* scale_factor'
    end 

    optvec = tovector(optparams)
    nplanet = orbparams.nplanet

    # Check and initialize default scale factor
    if scale_factor !== nothing && length(scale_factor) != 5 * nplanet + 1
        error("Inconsistent optimization weights. Expected $(5 * nplanet + 1), got $(length(scale_factor)) instead.")
    end

    if scale_factor === nothing
        scale_factor = reduce(vcat, [
            fill(1e-3, nplanet),
            fill(1., nplanet),
            fill(1., nplanet-1),
            fill(1e-3, nplanet-2),
            fill(1e3, 1),
            fill(1e-5, nplanet),
            fill(1, 1),
            fill(1, 1),
            fill(1e4, 1),
        ])
    end

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
        fit = curve_fit(objective_function, jacobian, xdata, ydata, fit_weight, optvec ./ scale_factor, lower=lower_bounds ./ scale_factor, upper=upper_bounds ./ scale_factor; maxIter=maxit, show_trace=trace)
    else
        fit = curve_fit(objective_function, xdata, ydata, fit_weight, optvec ./ scale_factor, lower=lower_bounds ./ scale_factor, upper=upper_bounds ./ scale_factor; maxIter=maxit, show_trace=trace)
    end

    # Parse the LsqFit object 
    return fit, scale_factor
end

"""Routine for finding PO with a TT constraint"""
function find_periodic_orbit(optparams::OptimParameters{T}, orbparams::OrbitParameters{T}, tt_data::Matrix{T}; 
    use_jac::Bool=true, trace::Bool=false,eccmin::T=1e-3,maxit::Int64=1000, optim_weights=nothing,
    prior_weights=nothing, tt_weights=nothing, lower_bounds=nothing, upper_bounds=nothing, scale_factor=nothing) where T <: Real

    function objective_function(_, θ)
        p = scale_factor .* θ

        optparams = OptimParameters(nplanet, p)

        orbit = Orbit(nplanet, optparams, orbparams)
        tt = compute_tt(orbit.ic, orbparams.obstmax) # TODO: Get rid of the hardcode here

        # Append the TT information
        tmod, ip, jp = match_transits(tt_data, orbit, tt.tt, tt.count, nothing)

        diff_squared = compute_diff_squared(optparams, orbparams, nplanet, tmod)

        return diff_squared
    end

    function jacobian(_, θ)
        p = scale_factor .* θ

        optparams = OptimParameters(nplanet, p)

        jac = compute_diff_squared_jacobian(optparams, orbparams, nplanet, tt_data)

        return jac .* scale_factor'
    end 

    nplanet = orbparams.nplanet

    # Check and initialize default scale factor
    if scale_factor !== nothing && length(scale_factor) != 5 * nplanet + 1
        error("Inconsistent optimization weights. Expected $(5 * nplanet + 1), got $(length(scale_factor)) instead.")
    end

    if scale_factor === nothing
        scale_factor = reduce(vcat, [
            fill(1e-3, nplanet),
            fill(1., nplanet),
            fill(1., nplanet-1),
            fill(1e-3, nplanet-2),
            fill(1e3, 1),
            fill(1e-5, nplanet),
            fill(1, 1),
            fill(1, 1),
            fill(1e4, 1),
        ])
    end

    orbit = Orbit(nplanet, optparams, orbparams)
    tt = compute_tt(orbit.ic, orbparams.obstmax)

    # Append the TT information
    tmod, ip, jp = match_transits(tt_data, orbit, tt.tt, tt.count, nothing)
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
        fit = curve_fit(objective_function, jacobian, xdata, ydata, fit_weight, optvec ./ scale_factor, lower=lower_bounds ./ scale_factor, upper=upper_bounds ./ scale_factor; maxIter=maxit, show_trace=trace)
    else
        fit = curve_fit(objective_function, xdata, ydata, fit_weight, optvec ./ scale_factor, lower=lower_bounds ./ scale_factor, upper=upper_bounds ./ scale_factor; maxIter=maxit, show_trace=trace)
    end

    # Parse the LsqFit object 
    return fit, scale_factor
end

"""Routine for finding PO with a TT constraint with variable PO weights"""
function find_periodic_orbit(optparams::OptimParameters{T}, orbparams::OrbitParameters{T}, tt_data::Matrix{T}, optim_sigmas::Vector{T}; 
    use_jac::Bool=true, trace::Bool=false,eccmin::T=1e-3,maxit::Int64=1000,
    prior_weights=nothing, tt_weights=nothing, lower_bounds=nothing, upper_bounds=nothing, scale_factor=nothing, sigma_weights=nothing) where T <: Real

    function objective_function(_, θ)
        p = scale_factor .* θ

        optparams = OptimParameters(nplanet, p[1:5nplanet+1])
        po_sigmas = p[5nplanet+1+1:end]

        orbit = Orbit(nplanet, optparams, orbparams)
        tt = compute_tt(orbit.ic, orbparams.obstmax) # TODO: Get rid of the hardcode here

        # Append the TT information
        tmod, ip, jp = match_transits(tt_data, orbit, tt.tt, tt.count, nothing)

        diff_squared = compute_diff_squared(optparams, orbparams, nplanet, tmod, po_sigmas)

        return diff_squared
    end

    function jacobian(_, θ)
        p = scale_factor .* θ

        optparams = OptimParameters(nplanet, p[1:5nplanet+1])
        po_sigmas = p[5nplanet+1+1:end]

        jac = compute_diff_squared_jacobian_var_weights(optparams, orbparams, nplanet, tt_data, po_sigmas)

        return jac .* scale_factor'
    end 

    nplanet = orbparams.nplanet

    # Check and initialize default scale factor
    if scale_factor !== nothing && length(scale_factor) != 9 * nplanet - 1
        error("Inconsistent optimization weights. Expected $(9 * nplanet - 1), got $(length(scale_factor)) instead.")
    end

    if scale_factor === nothing
        scale_factor = reduce(vcat, [
            fill(1e-3, nplanet),
            fill(1., nplanet),
            fill(1., nplanet-1),
            fill(1e-3, nplanet-2),
            fill(1e3, 1),
            fill(1e-5, nplanet),
            fill(1, 1),
            fill(1, 1),
            fill(1e4, 1),
        ])
    end

    orbit = Orbit(nplanet, optparams, orbparams)
    tt = compute_tt(orbit.ic, orbparams.obstmax)

    # Append the TT information
    tmod, ip, jp = match_transits(tt_data, orbit, tt.tt, tt.count, nothing)
    ttobs = tt_data[:,3]

    optvec = vcat(tovector(optparams), optim_sigmas)

    # Target is all zero with the prior and TTV information appended
    ydata = vcat(zeros(T, 4*nplanet-2), deepcopy(optvec[1:5*nplanet+1]), zeros(T, 4*nplanet-2), deepcopy(ttobs))
    # Dummy data
    xdata = zeros(T, length(ydata))

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

    # Initialize the weights of sigmas for internal use by the optimizer.
    if sigma_weights === nothing
        sigma_weights = fill(1., 4nplanet-2)
    end

    fit_weight = vcat(fill(1., 4nplanet-2), prior_weights, sigma_weights, tt_weights)

    # Check and initialize default lower bounds
    if lower_bounds !== nothing && length(lower_bounds) != 9 * nplanet - 1
        error("Inconsistent lower bounds. Expected $(9 * nplanet - 1), got $(length(lower_bounds)) instead.")
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
    if upper_bounds !== nothing && length(upper_bounds) != 9 * nplanet - 1
        error("Inconsistent upper bounds. Expected $(9 * nplanet - 1), got $(length(upper_bounds)) instead.")
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
        fit = curve_fit(objective_function, jacobian, xdata, ydata, fit_weight, optvec ./ scale_factor, lower=lower_bounds ./ scale_factor, upper=upper_bounds ./ scale_factor; maxIter=maxit, show_trace=trace)
    else
        fit = curve_fit(objective_function, xdata, ydata, fit_weight, optvec ./ scale_factor, lower=lower_bounds ./ scale_factor, upper=upper_bounds ./ scale_factor; maxIter=maxit, show_trace=trace)
    end

    # Parse the LsqFit object 
    return fit, scale_factor
end

function compute_diff_squared(optparams::OptimParameters{T}, orbparams::OrbitParameters{T}, nplanet::Int) where T <: Real
    orbit = Orbit(nplanet, optparams, orbparams)
    
    init_elems = orbit.init_elem
    final_elems = orbit.final_elem

    # Compute diff using util function, excluding the last two elements (kappa and ω1)
    diff = param_diff(nplanet, final_elems, init_elems)[1:end-2]
    
    priors = tovector(optparams)

    return [diff; priors]
end

function compute_diff_squared(optparams::OptimParameters{T}, orbparams::OrbitParameters{T}, nplanet::Int, ttmod::Vector{T}) where T <: Real
    obj_vec = compute_diff_squared(optparams, orbparams, nplanet)

    return [obj_vec; ttmod]
end

function compute_diff_squared(optparams::OptimParameters{T}, orbparams::OrbitParameters{T}, nplanet::Int, ttmod::Vector{T}, po_sigmas::Vector{T}) where T <: Real
    obj_vec = compute_diff_squared(optparams, orbparams, nplanet)

    sigma_0 = 1e-16 # NOTE: Somewhat arbitrary here
    ln_sigmas_sq = sqrt.(2 .* log.(po_sigmas ./ sigma_0))

    return [obj_vec ./ vcat(po_sigmas, fill(1., 5nplanet+1)); ln_sigmas_sq; ttmod]
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

    jac_tt_1 = compute_tt_jacobians(orbit, orbparams, orbit.ic, tt_data)

    tt_jac = jac_tt_1 * orbit.jac_1[1:end-1,:]

    the_jac = [obj_jac; tt_jac]

    return the_jac
end

function compute_diff_squared_jacobian_var_weights(optparams::OptimParameters{T}, orbparams::OrbitParameters{T}, nplanet::Int, tt_data::Matrix{T}, po_sigmas::Vector{T}) where T <: Real
    orbit = Orbit(nplanet, optparams, orbparams)

    # Helper function for subtracting the matrix with identity
    function subtract_iden(mat)
        N = size(mat, 1)
        mat[1:N,1:N] = mat[1:N,1:N] - I
        return mat
    end

    # Jacobian of the difference 4N-2 * 5N+1
    diff_jac = subtract_iden(orbit.jac_combined[begin:4*nplanet-2,:])

    # With identity matrix 9N-1 * 5N+1 (priors)
    diff_jac = [diff_jac; Matrix{T}(I, 5*nplanet+1, 5*nplanet+1)]

    # With identity matrix from variable weights 13N-3 * 9N-1
    sigma_0 = 1e-16
    ln_sigmas_vs_sigmas(sm) = 1 / (sm * sqrt(2*log(sm/sigma_0)))
    matrix_sigma_vs_sigma = diagm(ln_sigmas_vs_sigmas.(po_sigmas))
    diff_jac = cat(diff_jac, matrix_sigma_vs_sigma, dims=(1,2))

    # Correct derivatives ΔX_i vs w_i
    obj_vec = compute_diff_squared(optparams, orbparams, nplanet)[1:4nplanet-2]
    for i in 1:4nplanet-2
        diff_jac[i, 5nplanet+1 + i] =  -1 * obj_vec[i] / (po_sigmas[i]^2)
    end

    jac_tt_1 = compute_tt_jacobians(orbit, orbparams, orbit.ic, tt_data)
    tt_jac = jac_tt_1 * orbit.jac_1[1:end-1,:]

    dims, _ = size(tt_jac)
    # Vertically append the jacobian from TT with zero padding for weights
    # Dimension: (13N-3 + n.tt) * 9N-1
    the_jac = [diff_jac; hcat(tt_jac, zeros(dims, 4nplanet-2))]

    return the_jac
end
