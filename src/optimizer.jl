using NbodyGradient
using LinearAlgebra
using Optim

using ForwardDiff

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
function find_periodic_orbit(optparams::OptimParameters{T}, orbparams::OrbitParameters{T}; use_jac::Bool=false, verbose::Bool=false) where T <: AbstractFloat
    # TODO: Implement `use_jac` and `verbose`

    """Extract optimization state parameters from `State` and `InitialConditions`"""
    function _extract_elements(s::State, ic::InitialConditions)
        # Extract orbital elements and anomalies
        elems = get_orbital_elements(s, ic)
        anoms = get_anomalies(s, ic)

        e = [elems[i].e for i in eachindex(elems)[2:end]]
        M = [anoms[i][2] for i in eachindex(anoms)]
        ωdiff = [elems[i].ω - elems[i-1].ω for i in eachindex(elems)[3:end]]

        # Period ratio deviation
        pratio_nom = Vector{T}(undef, nplanet-1)
        pratio_nom[1] = orbparams.κ
    
        for i = 2:nplanet-1
            pratio_nom[i] = 1/(1 + orbparams.cfactor[i-1]*(1 - pratio_nom[i-1]))
        end 

        pratiodev = [(elems[i].P / elems[i-1].P) - pratio_nom[i-2] for i in eachindex(elems)[4:end]]
        inner_period = elems[2].P

        return e, M, ωdiff, pratiodev, inner_period
    end

    function objective_function(optvec)
        optparams = OptimParameters(nplanet, optvec)

        # Initialize the orbit system
        orbit = Orbit(nplanet, optparams, orbparams)      

        # Extract initial state
        init_e, init_M, init_ωdiff, init_pratiodev, init_inner_period = _extract_elements(orbit.s, orbit.ic)

        # Integrate the system
        step_size = 0.1 * init_inner_period
        current_time = 0.0
        
        while current_time < integration_time
            next_step = min(step_size, integration_time - current_time)
            intr = Integrator(ahl21!, next_step, next_step)
            intr(orbit.s)
            current_time = orbit.s.t[1]
        end

        # Extract final state
        final_e, final_M, final_ωdiff, final_pratiodev, final_inner_period = _extract_elements(orbit.s, orbit.ic)

        # Calculate differences
        diff_e = final_e - init_e
        diff_M = rem2pi.(final_M - init_M, RoundNearest)
        diff_ωdiff = rem2pi.(final_ωdiff - init_ωdiff, RoundNearest)
        diff_pratiodev = final_pratiodev - init_pratiodev
        diff_inner_period = final_inner_period - init_inner_period

        # Multiply the weights
        diff_e *= weights[1]
        diff_M *= weights[2]
        diff_ωdiff *= weights[3]
        diff_pratiodev *= weights[4]
        diff_inner_period *= weights[5]

        # Sum of the squares
        diff = vcat(diff_e, diff_M, diff_ωdiff, diff_pratiodev, diff_inner_period)
        # println("DIFF NOW: $diff")
        if verbose
            println("DIFF SQUARED NOW: $(sum(diff.^2))")
        end

        return sum(diff.^2)
    end

    nplanet = length(orbparams.mass)
    integration_time = orbparams.tsys
    weights = orbparams.weights

    # Set up parameter bounds
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
    
    println("Starting optimization with parameters: $optparams")
    println("  Inner planet period: $(optparams.inner_period)")
    println("  Eccentricities: $(optparams.e)")
    println("  Arguments of perihelion: $(optparams.Δω)")
    println("  Period ratio deviations: $(optparams.Pratio)")
    println("  Mean anomalies: $(optparams.M)")
  
    optvec = tovector(optparams)

    try
        test_result = objective_function(optvec)
        println("Test objective function evaluation: ", test_result)
    catch e
        println("Error testing objective function:")
        println(e)
        rethrow(e)
    end
    
    # Optimize using Optim.jl
    println("Starting optimization...")
    max_iterations = 200
    result = optimize(
        objective_function,
        lower_bounds,
        upper_bounds,
        optvec,
        Fminbox(NelderMead()),
        # Optim.Options(iterations=max_iterations, show_trace=true)
        Optim.Options(iterations=max_iterations)
    )
    
    # Extract optimized parameters
    final_optvec = Optim.minimizer(result)

    final_optparams = OptimParameters(nplanet, final_optvec)

    println("Starting optimization with parameters: $final_optparams")
    println("  Inner planet period: $(final_optparams.inner_period)")
    println("  Eccentricities: $(final_optparams.e)")
    println("  Arguments of perihelion: $(final_optparams.Δω)")
    println("  Period ratio deviations: $(final_optparams.Pratio)")
    println("  Mean anomalies: $(final_optparams.M)")

    return final_optparams
end
