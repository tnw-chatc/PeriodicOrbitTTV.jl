include("PeriodicOrbit.jl")
using .PeriodicOrbit


using NbodyGradient
using LinearAlgebra
using Optim


function find_periodic_orbit_jacobian(optparams::OptimParameters{T}, orbparams::OrbitParameters{T}; 
                                     verbose=false, max_iterations=200) where T <: AbstractFloat
    
    nplanet = length(orbparams.mass)
    integration_time = orbparams.tsys
    

    expected_length = nplanet + nplanet + (nplanet-1) + max(0, nplanet-2) + 1
    
    if length(orbparams.weights) == expected_length
        weights = orbparams.weights
    else
        weights = T[]
        append!(weights, ones(T, nplanet))
        append!(weights, ones(T, nplanet) * 5.0)
        append!(weights, ones(T, nplanet-1) * 3.0)
        if nplanet >= 3
            append!(weights, ones(T, nplanet-2) * 2.0)
        end
        
        
        push!(weights, 2.0)
        
        if verbose
            println("Warning: weights vector has incorrect length ($(length(orbparams.weights)) vs expected $expected_length). Using default weights.")
        end
    end
    

    function compute_differences(orbit::Orbit, init_elems, init_anoms, integration_time)
        
        s_copy = deepcopy(orbit.s)
        ic_copy = deepcopy(orbit.ic)
        
       
        step_size = 0.1 * init_elems[2].P
        current_time = 0.0
        
        while current_time < integration_time
            next_step = min(step_size, integration_time - current_time)
            intr = Integrator(ahl21!, next_step, next_step)
            intr(s_copy)
            current_time = s_copy.t[1]
        end
        
  
        final_elems = get_orbital_elements(s_copy, ic_copy)
        final_anoms = get_anomalies(s_copy, ic_copy)
        
        
        diff_e = [final_elems[i].e - init_elems[i].e for i in 2:nplanet+1]
        diff_M = [rem2pi(final_anoms[i][2] - init_anoms[i][2], RoundNearest) for i in 1:nplanet]
        diff_ωdiff = [rem2pi((final_elems[i].ω - final_elems[i-1].ω) - (init_elems[i].ω - init_elems[i-1].ω), RoundNearest) for i in 3:nplanet+1]
        
       
        if nplanet >= 3
            pratio_nom = Vector{T}(undef, nplanet-1)
            pratio_nom[1] = orbparams.κ
            for i = 2:nplanet-1
                pratio_nom[i] = 1/(1 + orbparams.cfactor[i-1]*(1 - pratio_nom[i-1]))
            end
            
            init_pratiodev = [(init_elems[i].P / init_elems[i-1].P) - pratio_nom[i-2] for i in 4:nplanet+1]
            final_pratiodev = [(final_elems[i].P / final_elems[i-1].P) - pratio_nom[i-2] for i in 4:nplanet+1]
            diff_pratiodev = final_pratiodev - init_pratiodev
        else
            diff_pratiodev = T[]
        end
        
        diff_inner_period = final_elems[2].P - init_elems[2].P
        
        differences = vcat(diff_e, diff_M, diff_ωdiff, diff_pratiodev, [diff_inner_period])
        
        return differences
    end
    
 
    function objective_differences(optvec)
        optparams_local = OptimParameters(nplanet, optvec)
        orbit = Orbit(nplanet, optparams_local, orbparams)
        
        init_elems = get_orbital_elements(orbit.s, orbit.ic)
        init_anoms = get_anomalies(orbit.s, orbit.ic)
        
        differences = compute_differences(orbit, init_elems, init_anoms, integration_time)
     
        weighted_diff = differences .* sqrt.(weights)
        
        return weighted_diff
    end
    
   
    function objective_scalar(optvec)
        diff = objective_differences(optvec)
        return sum(diff.^2)
    end
    
  
    function gradient_finitediff!(g, optvec)
        eps_value =1e-8
        eps = sqrt(eps_value)  
        n = length(optvec)
        
        
        for i in 1:n
            optvec_plus = copy(optvec)
            optvec_minus = copy(optvec)
            optvec_plus[i] += eps
            optvec_minus[i] -= eps
            
            f_plus = objective_scalar(optvec_plus)
            f_minus = objective_scalar(optvec_minus)
            
            g[i] = (f_plus - f_minus) / (2 * eps)
        end
        
        if verbose
            println("Gradient norm: ", norm(g))
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
    
    
    if nplanet >= 3
        append!(lower_bounds, fill(-0.1, nplanet-2))
        append!(upper_bounds, fill(0.1, nplanet-2))
    end
    
    
    push!(lower_bounds, 0.5*365.242)
    push!(upper_bounds, 2.0*365.242)
    
   
    optvec = vcat(
        optparams.e,
        optparams.M,
        optparams.Δω,
        optparams.Pratio,
        [optparams.inner_period]
    )
    
    if verbose
        println("Starting LBFGS optimization...")
        println("Number of planets: ", nplanet)
        println("Number of parameters: ", length(optvec))
        println("Initial objective value: ", objective_scalar(optvec))
    end
    
    
    result = optimize(
        objective_scalar,
        gradient_finitediff!,
        lower_bounds,
        upper_bounds,
        optvec,
        Fminbox(LBFGS()),
        Optim.Options(
            iterations=max_iterations,
            show_trace=verbose,
            f_abstol=1e-10,
            g_tol=1e-8
        )
    )
    
 
    final_optvec = Optim.minimizer(result)
    final_optparams = OptimParameters(nplanet, final_optvec)
    
    if verbose
        println("\nOptimization complete!")
        println("Final objective value: ", Optim.minimum(result))
        println("Converged: ", Optim.converged(result))
        println("Iterations: ", Optim.iterations(result))
        println("\nFinal parameters:")
        println("  Inner planet period: $(final_optparams.inner_period)")
        println("  Eccentricities: $(final_optparams.e)")
        println("  Arguments of perihelion: $(final_optparams.Δω)")
        println("  Period ratio deviations: $(final_optparams.Pratio)")
        println("  Mean anomalies: $(final_optparams.M)")
    end
    
    return final_optparams, result
end


function main()

    nplanet = 3

    vec = [
        0.05, 0.07, 0.09, 
        0.0, π/4,      
        0.0, π/3,        
        0.001,           
        365.242      
    ]
    
    optparams = OptimParameters(nplanet, vec)
   
    weights_correct = [
        1.0, 1.0, 1.0,   
        5.0, 5.0, 5.0,  
        3.0, 3.0,      
        2.0,             
        2.0             
    ]
    
   
    orbparams = OrbitParameters(
        mass = [3e-6, 5e-6, 7e-6],
        cfactor = [0.5],
        κ = 2.0,
        tsys = 4 * 365.242,  
        weights = weights_correct
    )
    

    println("Starting LBFGS optimization...")
    final_optparams, result = find_periodic_orbit_jacobian(
        optparams, 
        orbparams, 
        verbose=true,
        max_iterations=500
    )

    println("\n=== OPTIMIZATION RESULTS ===")
    println("Final objective value: ", Optim.minimum(result))
    println("Converged: ", Optim.converged(result))
    println("Iterations: ", Optim.iterations(result))
    
    
    println("\n=== VERIFYING PERIODICITY ===")
    orbit = Orbit(nplanet, final_optparams, orbparams)
    
    # Get initial state
    init_elems = get_orbital_elements(orbit.s, orbit.ic)
    init_anoms = get_anomalies(orbit.s, orbit.ic)
    

    init_e = [init_elems[i].e for i in 2:nplanet+1]
    init_M = [init_anoms[i][2] for i in 1:nplanet]
    init_ω = [init_elems[i].ω for i in 2:nplanet+1]
    init_P = [init_elems[i].P for i in 2:nplanet+1]
    
    
    integration_time = orbparams.tsys
    step_size = 0.1 * init_elems[2].P
    current_time = 0.0
    
    while current_time < integration_time
        next_step = min(step_size, integration_time - current_time)
        intr = Integrator(ahl21!, next_step, next_step)
        intr(orbit.s)
        current_time = orbit.s.t[1]
    end
    

    final_elems = get_orbital_elements(orbit.s, orbit.ic)
    final_anoms = get_anomalies(orbit.s, orbit.ic)
    
 
    final_e = [final_elems[i].e for i in 2:nplanet+1]
    final_M = [final_anoms[i][2] for i in 1:nplanet]
    final_ω = [final_elems[i].ω for i in 2:nplanet+1]
    final_P = [final_elems[i].P for i in 2:nplanet+1]
    
 
    println("\nDifferences after integration (final - initial):")
    

    println("\nEccentricities:")
    for i in 1:nplanet
        println("  Planet $i: e_diff = ", final_e[i] - init_e[i])
    end
    

    println("\nMean anomalies:")
    for i in 1:nplanet
        diff_M = rem2pi(final_M[i] - init_M[i], RoundNearest)
        println("  Planet $i: M_diff = ", diff_M)
    end
    
  
    println("\nArguments of perihelion differences:")
    for i in 2:nplanet
        init_Δω = init_ω[i] - init_ω[i-1]
        final_Δω = final_ω[i] - final_ω[i-1]
        diff_Δω = rem2pi(final_Δω - init_Δω, RoundNearest)
        println("  Δω_$(i-1),$(i) diff = ", diff_Δω)
    end
    

    if nplanet >= 3
        println("\nPeriod ratio deviations:")
        pratio_nom = Vector{Float64}(undef, nplanet-1)
        pratio_nom[1] = orbparams.κ
        for i = 2:nplanet-1
            pratio_nom[i] = 1/(1 + orbparams.cfactor[i-1]*(1 - pratio_nom[i-1]))
        end
        
        for i in 3:nplanet
            init_pratio_dev = (init_P[i] / init_P[i-1]) - pratio_nom[i-2]
            final_pratio_dev = (final_P[i] / final_P[i-1]) - pratio_nom[i-2]
            diff_pratio = final_pratio_dev - init_pratio_dev
            println("  P_ratio_$(i-1),$(i) deviation diff = ", diff_pratio)
        end
    end
    

    println("\nInner planet period:")
    println("  P_inner diff = ", final_P[1] - init_P[1])
    

    println("\nActual periods:")
    for i in 1:nplanet
        println("  Planet $i: P = ", final_P[i])
    end
    
    println("\nActual period ratios:")
    for i in 2:nplanet
        println("  P_$i/P_$(i-1) = ", final_P[i]/final_P[i-1])
    end

    println("\n=== OPTIMIZATION OBJECTIVE VALUE ===")
    println("Sum of squared weighted differences: ", Optim.minimum(result))
    println("This should be close to zero for a truly periodic orbit.")
    
    return final_optparams, result
end


final_optparams, result = main()