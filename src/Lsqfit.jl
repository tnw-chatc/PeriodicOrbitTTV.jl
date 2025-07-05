using NbodyGradient
using LinearAlgebra
using PeriodicOrbit
using Optim


function find_periodic_orbit_practical_fixed(optparams::OptimParameters{T}, orbparams::OrbitParameters{T}; 
                                             use_jac::Bool=true, verbose::Bool=false) where T <: AbstractFloat
    
    nplanet = length(orbparams.mass)
    weights = orbparams.weights

    function optparams_to_vector(opt::OptimParameters)
        return vcat(opt.e, opt.M, opt.Δω, opt.Pratio, opt.inner_period)
    end

    function residual_function(optvec)
        try
            optparams_current = OptimParameters(nplanet, optvec)
            orbit = Orbit(nplanet, optparams_current, orbparams)

            # Extract initial state
            init_elems = get_orbital_elements(orbit.s, orbit.ic)
            init_anoms = get_anomalies(orbit.s, orbit.ic)

            init_e = [init_elems[i].e for i in eachindex(init_elems)[2:end]]
            init_M = [init_anoms[i][2] for i in eachindex(init_anoms)[2:end]]  # Fixed!
            init_ωdiff = [init_elems[i].ω - init_elems[i-1].ω for i in eachindex(init_elems)[3:end]]

            # Period ratio deviation
            pratio_nom = Vector{T}(undef, nplanet-1)
            pratio_nom[1] = orbparams.κ
            for i = 2:nplanet-1
                pratio_nom[i] = 1/(1 + orbparams.cfactor[i-1]*(1 - pratio_nom[i-1]))
            end 
            init_pratiodev = [(init_elems[i].P / init_elems[i-1].P) - pratio_nom[i-2] for i in eachindex(init_elems)[4:end]]
            init_inner_period = init_elems[2].P

            # Get final state from orbit
            final_elems = get_orbital_elements(orbit.state_final, orbit.ic)
            final_anoms = get_anomalies(orbit.state_final, orbit.ic)

            final_e = [final_elems[i].e for i in eachindex(final_elems)[2:end]]
            final_M = [final_anoms[i][2] for i in eachindex(final_anoms)[2:end]]  # Fixed!
            final_ωdiff = [final_elems[i].ω - final_elems[i-1].ω for i in eachindex(final_elems)[3:end]]

            pratio_nom = Vector{T}(undef, nplanet-1)
            pratio_nom[1] = orbparams.κ
            for i = 2:nplanet-1
                pratio_nom[i] = 1/(1 + orbparams.cfactor[i-1]*(1 - pratio_nom[i-1]))
            end 
            final_pratiodev = [(final_elems[i].P / final_elems[i-1].P) - pratio_nom[i-2] for i in eachindex(final_elems)[4:end]]
            final_inner_period = final_elems[2].P

            # Calculate differences
            diff_e = final_e - init_e
            diff_M = rem2pi.(final_M - init_M, RoundNearest)
            diff_ωdiff = rem2pi.(final_ωdiff - init_ωdiff, RoundNearest)
            diff_pratiodev = final_pratiodev - init_pratiodev
            diff_inner_period = final_inner_period - init_inner_period

            # Apply weights
            diff_e .*= weights[1]
            diff_M .*= weights[2]
            diff_ωdiff .*= weights[3]
            diff_pratiodev .*= weights[4]
            diff_inner_period *= weights[5]

            residuals = vcat(diff_e, diff_M, diff_ωdiff, diff_pratiodev, diff_inner_period)
            return residuals
        catch e
            if verbose
                println(" Error in residual function: ", e)
            end
            return fill(1e3, length(optvec))
        end
    end

    function objective_function(p)
        residuals = residual_function(p)
        return sum(residuals.^2)
    end

    optvec = optparams_to_vector(optparams)
    
    if verbose
        println("Starting optimization with $(length(optvec)) parameters")
        initial_obj = objective_function(optvec)
        println("Initial objective value: $initial_obj")
    end

   
    result = optimize(
        objective_function, 
        optvec, 
        LBFGS(), 
        Optim.Options(
            iterations=100,      
            f_tol=1e-10,        
            g_tol=1e-8,         
            show_trace=verbose,
            show_every=10
        )
    )
    
    final_optvec = Optim.minimizer(result)
    final_optparams = OptimParameters(nplanet, final_optvec)
    
    if verbose
        println("\nFinal Results:")
        println("  Converged: $(Optim.converged(result))")
        println("  Iterations: $(Optim.iterations(result))")
        println("  Final objective value: $(Optim.minimum(result))")
        println("  Final residual norm: $(norm(residual_function(final_optvec)))")
        println("  Total function evaluations: $(result.f_calls)")
    end
    
    return final_optparams, result
end
