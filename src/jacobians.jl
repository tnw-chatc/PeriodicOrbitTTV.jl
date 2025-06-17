import NbodyGradient: Derivatives, zero_out!
using NbodyGradient: Integrator, ahl21!


function extract_nbody_jacobians(orbit::Orbit{T}, integration_time::T) where T<:AbstractFloat
   
    s = State(orbit.ic)
    d = Derivatives(T, s.n)
    
    println("Extracting Jacobians from NbodyGradient...")
    println("System has $(s.n) bodies")
    
    jac_elements_to_cartesian = copy(s.jac_init)
    println("Jacobian 2 (∂{x,v}/∂elements) size: $(size(jac_elements_to_cartesian))")
    
    
    x_initial = copy(s.x)
    v_initial = copy(s.v)
    t_initial = s.t[1]
  
    inner_period = 365.242  
    
    try
        initial_elements = get_orbital_elements(s, orbit.ic)
        if length(initial_elements) >= 2
            inner_period = initial_elements[2].P
        end
    catch e
        println("Warning: Could not extract initial period, using default: $e")
    end
    
    step_size = 0.01 * inner_period
    nsteps = ceil(Int, integration_time / step_size)
    h = integration_time / nsteps
    
    println("Integrating for $(integration_time) days with $(nsteps) steps (h=$(round(h, digits=4)))")
    

    zero_out!(d)
    for i in 1:nsteps
        ahl21!(s, d, h)
        s.t[1] = t_initial + (i * h)
    end
    
    # JACOBIAN 3: ∂{x,v}(T)/∂{x,v}(0) (time evolution)
    jac_time_evolution = copy(s.jac_step)
    println("Jacobian 3 (∂{x,v}(T)/∂{x,v}(0)) size: $(size(jac_time_evolution))")
    
    # Store final state
    x_final = copy(s.x)
    v_final = copy(s.v)
    t_final = s.t[1]
    
    # JACOBIAN 4: ∂elements/∂{x,v} (final Cartesian → final orbital elements)
    jac_cartesian_to_elements, final_elements = compute_cartesian_to_elements_jacobian(s, orbit.ic)
    println("Jacobian 4 (∂elements/∂{x,v}) size: $(size(jac_cartesian_to_elements))")
    
  
    println("\n=== Jacobian Analysis ===")
    cond2 = cond(jac_elements_to_cartesian)
    cond3 = cond(jac_time_evolution) 
    cond4 = cond(jac_cartesian_to_elements)
    
    println("Condition number of Jacobian 2: $(round(cond2, digits=2))")
    println("Condition number of Jacobian 3: $(round(cond3, digits=2))")
    println("Condition number of Jacobian 4: $(round(cond4, digits=2))")
    
    # Combined Jacobian 4 * 3 * 2 (elements -elements after time T)
    jac_combined = jac_cartesian_to_elements * jac_time_evolution * jac_elements_to_cartesian
    cond_combined = cond(jac_combined)
    println("Condition number of combined Jacobian: $(round(cond_combined, digits=2))")
    
    return (
        jac_elements_to_cartesian = jac_elements_to_cartesian,  # Jacobian 2
        jac_time_evolution = jac_time_evolution,                # Jacobian 3  
        jac_cartesian_to_elements = jac_cartesian_to_elements,  # Jacobian 4
        jac_combined = jac_combined,                            # Combined 4*3*2
        initial_state = (x=x_initial, v=v_initial, t=t_initial),
        final_state = (x=x_final, v=v_final, t=t_final),
        final_elements = final_elements
    )
end

"""
Compute ∂elements/∂{x,v} using finite differences
"""
function compute_cartesian_to_elements_jacobian(s::State{T}, ic, δ=1e-8) where T<:AbstractFloat
    n = s.n
    
    # Get current orbital elements
    nominal_elements = get_orbital_elements(s, ic)
    
    # Each non-central body has: P, t0, ecosω, esinω, I, Ω (6 parameters)
    # Plus all bodies have mass (n parameters)
    # Total elements: 6*(n-1) + n
    n_orbital_params = 6 * (n-1)
    n_masses = n  
    n_elements_total = n_orbital_params + n_masses
    
    # Cartesian state has 7 parameters per body: x, y, z, vx, vy, vz, m
    n_cartesian = 7 * n
    
    # Initialize Jacobian matrix
    jac = zeros(T, n_elements_total, n_cartesian)
    
    # Finite difference over each Cartesian coordinate
    for body in 2:n
        for coord in 1:7  # x, y, z, vx, vy, vz, m
            # Create perturbed state
            s_pert = deepcopy(s)
            ic_pert = deepcopy(ic)
            
            # Apply perturbation
            # TODO: Implement central difference for better precision
            if coord <= 3  # Position
                s_pert.x[coord, body] += δ
            elseif coord <= 6  # Velocity
                s_pert.v[coord-3, body] += δ
            else  # Mass (coord == 7)
                s_pert.m[body] += δ
                ic_pert.m[body] += δ
            end
            
            # Get perturbed elements
            try
                elements_pert = get_orbital_elements(s_pert, ic_pert)
                
                # Compute finite differences
                cartesian_idx = (body-1)*7 + coord
                
                for b in 2:n
                    orbital_base_idx = (b-2)*6 + 1
                    
                    param_names = [:P, :t0, :ecosω, :esinω, :I, :Ω]
                    for (p, param) in enumerate(param_names)
                        element_idx = orbital_base_idx + p - 1
                        if hasfield(typeof(elements_pert[b]), param) && hasfield(typeof(nominal_elements[b]), param)
                            jac[element_idx, cartesian_idx] = (getfield(elements_pert[b], param) - getfield(nominal_elements[b], param)) / δ
                        end
                    end
                end
                
        
                for b in 1:n
                    mass_idx = n_orbital_params + b
                    jac[mass_idx, cartesian_idx] = (elements_pert[b].m - nominal_elements[b].m) / δ
                end
            catch e
                println("Warning: Error computing finite difference for body $body, coord $coord: $e")
       
            end
        end
    end
    
    return jac, nominal_elements
end


function find_periodic_orbit_with_jacobians(optparams::OptimParameters{T}, orbparams::OrbitParameters{T}; verbose::Bool=false) where T <: AbstractFloat
    
    nplanet = length(orbparams.mass)
    integration_time = orbparams.tsys
    weights = orbparams.weights
    
    function objective_function_with_jacobians(optvec)
        try
            optparams_local = OptimParameters(nplanet, optvec)

            orbit = Orbit(nplanet, optparams_local, orbparams)


            init_elems = get_orbital_elements(orbit.s, orbit.ic)
            init_anoms = get_anomalies(orbit.s, orbit.ic)

            init_e = [init_elems[i].e for i in eachindex(init_elems)[2:end]]
            init_M = [init_anoms[i][2] for i in eachindex(init_anoms)]
            init_ωdiff = [init_elems[i].ω - init_elems[i-1].ω for i in eachindex(init_elems)[3:end]]


            pratio_nom = Vector{T}(undef, nplanet-1)
            pratio_nom[1] = orbparams.κ
        
            for i = 2:nplanet-1
                pratio_nom[i] = 1/(1 + orbparams.cfactor[i-1]*(1 - pratio_nom[i-1]))
            end 

            init_pratiodev = [(init_elems[i].P / init_elems[i-1].P) - pratio_nom[i-2] for i in eachindex(init_elems)[4:end]]
            init_inner_period = init_elems[2].P

    
            jacobians = extract_nbody_jacobians(orbit, integration_time)
            
            if verbose
                println("Extracted Jacobians successfully")
                println("  Jacobian 2 condition: $(round(cond(jacobians.jac_elements_to_cartesian), digits=2))")
                println("  Jacobian 3 condition: $(round(cond(jacobians.jac_time_evolution), digits=2))")
                println("  Jacobian 4 condition: $(round(cond(jacobians.jac_cartesian_to_elements), digits=2))")
            end


            final_elems = jacobians.final_elements
            
           n
            s_final = State(orbit.ic)
            s_final.x .= jacobians.final_state.x
            s_final.v .= jacobians.final_state.v
            s_final.t[1] = jacobians.final_state.t
            final_anoms = get_anomalies(s_final, orbit.ic)

            final_e = [final_elems[i].e for i in eachindex(final_elems)[2:end]]
            final_M = [final_anoms[i][2] for i in eachindex(final_anoms)]
            final_ωdiff = [final_elems[i].ω - final_elems[i-1].ω for i in eachindex(final_elems)[3:end]]

            # Period ratio deviation
            final_pratiodev = [(final_elems[i].P / final_elems[i-1].P) - pratio_nom[i-2] for i in eachindex(final_elems)[4:end]]
            final_inner_period = final_elems[2].P

            # Calculate differences (same as original)
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

          
            diff = vcat(diff_e, diff_M, diff_ωdiff, diff_pratiodev, diff_inner_period)
            
            if verbose
                println("DIFF SQUARED NOW: $(sum(diff.^2))")
                println("Individual diffs: e=$(norm(diff_e)), M=$(norm(diff_M)), ω=$(norm(diff_ωdiff)), P_ratio=$(norm(diff_pratiodev)), P_inner=$(abs(diff_inner_period))")
            end

            return sum(diff.^2)
            
        catch e
            println("Error in objective function: $e")
            return 1e10  
        end
    end

    lower_bounds = Float64[]
    upper_bounds = Float64[]
    
   
    append!(lower_bounds, fill(0.001, nplanet))
    append!(upper_bounds, fill(0.9, nplanet))
        

    append!(lower_bounds, fill(-π, nplanet-1))
    append!(upper_bounds, fill(π, nplanet-1))

    append!(lower_bounds, fill(-π, nplanet-1))
    append!(upper_bounds, fill(π, nplanet-1))
    
   
    append!(lower_bounds, fill(-0.1, nplanet-2))
    append!(upper_bounds, fill(0.1, nplanet-2))

    push!(lower_bounds, 0.5*365.242)
    push!(upper_bounds, 2.0*365.242)
    
    println("Starting optimization with Jacobian extraction...")
    println("Parameters: $optparams")
  
    optvec = reduce(vcat, tovector(optparams))

 
    try
        test_result = objective_function_with_jacobians(optvec)
        println("Test objective function evaluation: ", test_result)
    catch e
        println("Error testing objective function:")
        println(e)
        rethrow(e)
    end
    
   
    println("Starting optimization...")
    max_iterations = 50  # Reduced for testing
    result = optimize(
        objective_function_with_jacobians,
        lower_bounds,
        upper_bounds,
        optvec,
        Fminbox(NelderMead()),
        Optim.Options(iterations=max_iterations, show_trace=verbose)
    )
    

    final_optvec = Optim.minimizer(result)
    final_optparams = OptimParameters(nplanet, final_optvec)

    println("Optimization complete!")
    println("Final parameters: $final_optparams")

    return final_optparams, result
end


function find_periodic_orbit_enhanced(optparams::OptimParameters{T}, orbparams::OrbitParameters{T}; use_jac::Bool=false, verbose::Bool=false) where T <: AbstractFloat
    if use_jac
        println("Using Jacobian-enhanced optimization...")
        return find_periodic_orbit_with_jacobians(optparams, orbparams; verbose=verbose)
    else
        println("Using original optimization method...")
        return find_periodic_orbit(optparams, orbparams; use_jac=use_jac, verbose=verbose)
    end
end