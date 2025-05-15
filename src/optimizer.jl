using NbodyGradient
using LinearAlgebra
using Optimization

function find_periodic_orbit(optparams::OptimParameters{T}, orbparams::OrbitParameters{T}) where T <: AbstractFloat

    function objective_func()
        # Initialize the orbit system
        nplanet = length(orbparams.mass)
        integration_time = orbparams.tsys
        weights = orbparams.weights
        orbit = Orbit(nplanet, optparams, orbparams)

        # Extract initial state
        init_elems = get_orbital_elements(orbit.s, orbit.ic)
        init_anoms = get_anomalies(orbit.s, orbit.ic)

        init_e = [init_elems[i].e for i in eachindex(init_elems)[2:end]]
        init_M = [init_anoms[i][2] for i in eachindex(init_anoms)]
        init_ωdiff = [init_elems[i].ω - init_elems[i-1].ω for i in eachindex(init_elems)[3:end]]
        # TODO: Properly implement this as defined in G&M
        init_pratiodev = [init_elems[i].P / init_elems[i-1].P for i in eachindex(init_elems)[4:end]]
        init_inner_period = init_elems[2].P

        # Integrate the system
        step_size = 0.01 * actual_period
        current_time = 0.0
        
        while current_time < integration_time
            next_step = min(step_size, integration_time - current_time)
            intr = Integrator(ahl21!, next_step, next_step)
            intr(orbit.s)
            current_time = orbit.s.t[1]
        end

        # Extract final state
        final_elems = get_orbital_elements(orbit.s, orbit.ic)
        final_anoms = get_anomalies(orbit.s, orbit.ic)

        final_e = [final_elems[i].e for i in eachindex(final_elems)[2:end]]
        final_M = [final_anoms[i][2] for i in eachindex(final_anoms)]
        final_ωdiff = [final_elems[i].ω - final_elems[i-1].ω for i in eachindex(final_elems)[3:end]]
        # TODO: Properly implement this as defined in G&M
        final_pratiodev = [final_elems[i].P / final_elems[i-1].P for i in eachindex(final_elems)[4:end]]
        final_inner_period = final_elems[2].P

        # Calculate differences
        diff_e = final_e - init_e
        diff_M = rem2pi.(final_M - init_M, RoundNearest)
        diff_ωdiff = rem2pi.(final_e - init_e, RoundNearest)
        diff_pratiodev = final_e - init_e
        diff_inner_period = final_inner_period - init_inner_period

        # Multiply the weights
        diff_e *= weights[1]
        diff_M *= weights[2]
        diff_ωdiff *= weights[3]
        diff_pratiodev *= weights[4]
        diff_inner_period *= weights[5]

        # Sum of the squares
        diff = vcat(diff_e, diff_M, diff_ωdiff, diff_pratiodev, diff_inner_period)

        return sum(diff.^2)
    end

    

end

"""
# Arguments
- `n_planets::Int=2`: Number of planets in the system
- `planet_masses::Vector{Float64}=nothing`: Masses of the planets. If not provided, defaults are used.
- `kappa::Float64=2.0`: Base period ratio between adjacent planets
- `max_iterations::Int=200`: Maximum number of optimization iterations
- `initial_eccentricities::Vector{Float64}=nothing`: Initial eccentricity values
- `initial_omegas::Vector{Float64}=nothing`: Initial argument of pericentre values
- `initial_mean_anomalies::Vector{Float64}=nothing`: Initial mean anomaly values
- `period_ratio_deviations::Vector{Float64}=nothing`: Initial X5, X6, etc. values

# Returns
- `s`: Final state
- `ic`: Initial conditions
- `result`: Optimization result
"""
function find_periodic_orbit(n_planets::Int=2; 
                            planet_masses=nothing,
                            kappa::Float64=2.0,
                            max_iterations::Int=200,
                            initial_eccentricities=nothing,
                            initial_omegas=nothing,
                            initial_mean_anomalies=nothing,
                            period_ratio_deviations=nothing)
    println("Finding periodic orbit configuration for $n_planets planets...")
    
    # Set default planet masses if not provided
    if isnothing(planet_masses)
        planet_masses = [3e-4 * (1 + 0.5*i) for i in 0:(n_planets-1)]
    elseif length(planet_masses) != n_planets
        error("Length of planet_masses ($(length(planet_masses))) must match n_planets ($n_planets)")
    end
    
    # Set default eccentricities if not provided
    if isnothing(initial_eccentricities)
        initial_eccentricities = [0.05 + 0.05*i for i in 0:(n_planets-1)]
    elseif length(initial_eccentricities) != n_planets
        error("Length of initial_eccentricities must match n_planets")
    end
    
    # Set default omegas if not provided
    if isnothing(initial_omegas)
        initial_omegas = zeros(n_planets)
        for i in 2:n_planets
            initial_omegas[i] = (i-1) * π/n_planets
        end
    elseif length(initial_omegas) != n_planets
        error("Length of initial_omegas must match n_planets")
    end
    
    # Set default mean anomalies if not provided
    if isnothing(initial_mean_anomalies)
        initial_mean_anomalies = zeros(n_planets)
    elseif length(initial_mean_anomalies) != n_planets
        error("Length of initial_mean_anomalies must match n_planets")
    end
    
    # Set default period ratio deviations if not provided
    if isnothing(period_ratio_deviations)
        period_ratio_deviations = zeros(n_planets-1)
    elseif length(period_ratio_deviations) != n_planets-1
        error("Length of period_ratio_deviations must be n_planets-1")
    end
    
    # Calculate the number of parameters
    # - 1 for the innermost period
    # - n_planets for eccentricities
    # - n_planets-1 for omegas (first one is fixed at 0)
    # - n_planets-1 for period ratio deviations
    # - n_planets-1 for mean anomalies (first one is fixed at 0)
    n_params = 1 + n_planets + (n_planets-1) + (n_planets-1) + (n_planets-1)
    
    function objective_function(params)
        try
            # Extract parameters
            period_inner = params[1]
            
            # Extract eccentricities (1 for each planet)
            eccentricities = params[2:(2+n_planets-1)]
            
            # Extract omegas (n_planets-1, first one is fixed at 0)
            omegas = [0.0; params[(2+n_planets):(2+n_planets+(n_planets-1)-1)]]
            
            # Extract period ratio deviations (n_planets-1)
            period_ratio_deviations = params[(2+n_planets+(n_planets-1)):(2+n_planets+(n_planets-1)+(n_planets-1)-1)]
            
            # Extract mean anomalies (n_planets-1, first one is fixed at 0)
            mean_anomalies = [0.0; params[(2+n_planets+(n_planets-1)+(n_planets-1)):(2+n_planets+(n_planets-1)+(n_planets-1)+(n_planets-1)-1)]]
            
            # Calculate period ratios
            period_ratios = kappa .+ period_ratio_deviations
            
            # Calculate actual periods for each planet
            periods = [period_inner]
            for i in 2:n_planets
                push!(periods, periods[i-1] * period_ratios[i-1])
            end
            
            # Create star and planets
            star = Elements(m=1.0)
            planets = Elements[]
            
            for i in 1:n_planets
                planet = Elements(
                    m=planet_masses[i],
                    P=periods[i],
                    e=eccentricities[i],
                    ω=omegas[i],
                    I=π/2,  # Fixed inclination
                    Ω=0.0   # Fixed longitude of ascending node
                )
                push!(planets, planet)
            end
            
            # Initialize the system
            ic = ElementsIC(0.0, n_planets+1, star, planets...)
            s_initial = State(ic)
            
            # Extract initial state
            initial_elements = get_orbital_elements(s_initial, ic)
            initial_anomalies = get_anomalies(s_initial, ic)
            
            # Create arrays to store initial parameters
            ini_param_e = Float64[]  # Eccentricities
            ini_param_rel_ω = Float64[]  # Relative omegas
            ini_param_period_ratio = Float64[]  # Period ratios
            ini_param_M = Float64[]  # Mean anomalies
            
            # Fill initial parameters
            for i in 2:(n_planets+1)  # Starting from 2 because 1 is the star
                push!(ini_param_e, initial_elements[i].e)
                push!(ini_param_M, initial_anomalies[i-1][2])  # Mean anomaly
                
                if i > 2
                    # Relative orientation with previous planet
                    push!(ini_param_rel_ω, initial_elements[i].ω - initial_elements[i-1].ω)
                    # Period ratio with previous planet
                    push!(ini_param_period_ratio, initial_elements[i].P / initial_elements[i-1].P)
                end
            end
            
            # Actual period of inner planet from orbital elements
            actual_period = initial_elements[2].P
            
        
            integration_time = 2^(n_planets-1) * actual_period
            
            # Create state for integration
            s = State(ic)
            
            # Integrate the system
            step_size = 0.01 * actual_period
            current_time = 0.0
            
            while current_time < integration_time
                next_step = min(step_size, integration_time - current_time)
                intr = Integrator(ahl21!, next_step, next_step)
                intr(s)
                current_time = s.t[1]
            end
            
            # Extract final state
            final_elements = get_orbital_elements(s, ic)
            final_anomalies = get_anomalies(s, ic)
            
            # Create arrays to store final parameters
            end_param_e = Float64[]  
            end_param_rel_ω = Float64[]  
            end_param_period_ratio = Float64[]  
            end_param_M = Float64[]  
            
            
            for i in 2:(n_planets+1) 
                push!(end_param_e, final_elements[i].e)
                push!(end_param_M, final_anomalies[i-1][2])  
                
                if i > 2
                    # Relative orientation with previous planet
                    push!(end_param_rel_ω, final_elements[i].ω - final_elements[i-1].ω)
                    # Period ratio with previous planet
                    push!(end_param_period_ratio, final_elements[i].P / final_elements[i-1].P)
                end
            end
            
          
            differences = Float64[]
            
     
            for i in 1:n_planets
                push!(differences, ini_param_e[i] - end_param_e[i])
            end
            
            # Relative omega differences (need wrapping)
            for i in 1:(n_planets-1)
                diff = ini_param_rel_ω[i] - end_param_rel_ω[i]
                # Wrap to [-π, π]
                while diff > π
                    diff -= 2π
                end
                while diff < -π
                    diff += 2π
                end
                push!(differences, diff)
            end
            
            # Period ratio differences
            for i in 1:(n_planets-1)
                push!(differences, ini_param_period_ratio[i] - end_param_period_ratio[i])
            end
            
            
            for i in 1:n_planets
                diff = ini_param_M[i] - end_param_M[i]
                
                while diff > π
                    diff -= 2π
                end
                while diff < -π
                    diff += 2π
                end
                push!(differences, diff)
            end
            
            # Calculate weighted sum of squared differences
            
            weights = ones(length(differences))
            
            # Higher weight on relative orientations
            rel_ω_indices = (n_planets+1):(2*n_planets-1)
            weights[rel_ω_indices] .= 3.0
            
            # Higher weight on period ratios
            period_ratio_indices = (2*n_planets):(3*n_planets-2)
            weights[period_ratio_indices] .= 2.0
            # Highest weight on mean anomalies
            mean_anomaly_indices = (3*n_planets-1):(4*n_planets-3)
            weights[mean_anomaly_indices] .= 5.0
            
            return sum(weights .* differences.^2)
        catch e
            println("Error in objective function: ", e)
            return 1e10
        end
    end
    
  
    initial_params = Float64[]
    
    #innermost period
    push!(initial_params, 1.0)
    
    #eccentricities
    append!(initial_params, initial_eccentricities)
    
    #megas (excluding the first which is fixed at 0)
    append!(initial_params, initial_omegas[2:end])
    
    #period ratio deviations
    append!(initial_params, period_ratio_deviations)
    
    #mean anomalies (excluding the first which is fixed at 0)
    append!(initial_params, initial_mean_anomalies[2:end])
    
    # Set up parameter bounds
    lower_bounds = Float64[]
    upper_bounds = Float64[]
    
    # Bounds for innermost period
    push!(lower_bounds, 0.5)
    push!(upper_bounds, 2.0)
    
    # Bounds for eccentricities
    append!(lower_bounds, fill(0.001, n_planets))
    append!(upper_bounds, fill(0.9, n_planets))
    
    # Bounds for omegas
    append!(lower_bounds, fill(-π, n_planets-1))
    append!(upper_bounds, fill(π, n_planets-1))
    
    # Bounds for period ratio deviations
    append!(lower_bounds, fill(-0.1, n_planets-1))
    append!(upper_bounds, fill(0.1, n_planets-1))
    
    # Bounds for mean anomalies
    append!(lower_bounds, fill(-π, n_planets-1))
    append!(upper_bounds, fill(π, n_planets-1))
    
  
    println("Starting optimization with parameters:")
    println("  Inner planet period: $(initial_params[1])")
    println("  Eccentricities: $(initial_eccentricities)")
    println("  Arguments of perihelion: $(initial_omegas)")
    println("  Period ratio deviations: $(period_ratio_deviations)")
    println("  Mean anomalies: $(initial_mean_anomalies)")
    
  
    try
        test_result = objective_function(initial_params)
        println("Test objective function evaluation: ", test_result)
    catch e
        println("Error testing objective function:")
        println(e)
        rethrow(e)
    end
    
    # Optimize using Optim.jl
    println("Starting optimization...")
    result = optimize(
        objective_function,
        lower_bounds,
        upper_bounds,
        initial_params,
        Fminbox(NelderMead()),
        Optim.Options(iterations=max_iterations, show_trace=true)
    )
    
    # Extract optimized parameters
    opt_params = Optim.minimizer(result)
    
    opt_period_inner = opt_params[1]
    opt_eccentricities = opt_params[2:(2+n_planets-1)]
    opt_omegas = [0.0; opt_params[(2+n_planets):(2+n_planets+(n_planets-1)-1)]]
    opt_period_ratio_deviations = opt_params[(2+n_planets+(n_planets-1)):(2+n_planets+(n_planets-1)+(n_planets-1)-1)]
    opt_mean_anomalies = [0.0; opt_params[(2+n_planets+(n_planets-1)+(n_planets-1)):(2+n_planets+(n_planets-1)+(n_planets-1)+(n_planets-1)-1)]]
    
    opt_period_ratios = kappa .+ opt_period_ratio_deviations
    

    opt_periods = [opt_period_inner]
    for i in 2:n_planets
        push!(opt_periods, opt_periods[i-1] * opt_period_ratios[i-1])
    end
    
    println("\nOptimization complete!")
    println("Optimized parameters:")
    println("  Inner planet period: $(opt_period_inner)")
    println("  Eccentricities: $(opt_eccentricities)")
    println("  Arguments of perihelion: $(opt_omegas)")
    println("  Period ratio deviations: $(opt_period_ratio_deviations)")
    println("  Actual period ratios: $(opt_period_ratios)")
    println("  Mean anomalies: $(opt_mean_anomalies)")
    println("Final objective value: ", Optim.minimum(result))
    
    # Create optimized system
    star = Elements(m=1.0)
    planets = Elements[]
    
    for i in 1:n_planets
        planet = Elements(
            m=planet_masses[i],
            P=opt_periods[i],
            e=opt_eccentricities[i],
            ω=opt_omegas[i],
            I=π/2,
            Ω=0.0
        )
        push!(planets, planet)
    end
    
    ic = ElementsIC(0.0, n_planets+1, star, planets...)
    s = State(ic)
    
    optimized_elements = get_orbital_elements(s, ic)
    optimized_anomalies = get_anomalies(s, ic)
    
    println("\nOptimized system characteristics:")
    for i in 2:(n_planets+1)  
        println("  Planet $(i-1):")
        println("    Period: $(optimized_elements[i].P)")
        println("    Semi-major axis: $(optimized_elements[i].a)")
        println("    Eccentricity: $(optimized_elements[i].e)")
        println("    Argument of perihelion: $(optimized_elements[i].ω)")
        println("    Mean anomaly: $(optimized_anomalies[i-1][2])")
        
        if i > 2
            println("    Period ratio with previous: $((optimized_elements[i].P / optimized_elements[i-1].P))")
        end
    end
    
    println("\n===== VERIFICATION OF PERIODICITY =====")
    
  
    star = Elements(m=1.0)
    planets = Elements[]
    
    for i in 1:n_planets
        planet = Elements(
            m=planet_masses[i],
            P=opt_periods[i],
            e=opt_eccentricities[i],
            ω=opt_omegas[i],
            I=π/2,
            Ω=0.0
        )
        push!(planets, planet)
    end
    
    ic_test = ElementsIC(0.0, n_planets+1, star, planets...)
    s_test = State(ic_test)
    
   
    initial_elements = get_orbital_elements(s_test, ic_test)
    initial_anomalies = get_anomalies(s_test, ic_test)
    

    integration_time = 2^(n_planets -1) * initial_elements[2].P
    step_size = 0.01 * initial_elements[2].P
    current_time = 0.0
    
    while current_time < integration_time
        next_step = min(step_size, integration_time - current_time)
        intr = Integrator(ahl21!, next_step, next_step)
        intr(s_test)
        current_time = s_test.t[1]
    end
    
    # Extract final state
    final_elements = get_orbital_elements(s_test, ic_test)
    final_anomalies = get_anomalies(s_test, ic_test)
    
    # Helper function for angle difference
    function angle_diff(angle1, angle2)
        diff = angle1 - angle2
        # Wrap to [-π, π]
        while diff > π
            diff -= 2π
        end
        while diff < -π
            diff += 2π
        end
        return diff
    end
    

    println("\nDifferences (final - initial):")
    for i in 2:(n_planets+1) 
        println("  Planet $(i-1):")
        println("    Semi-major axis diff: ", final_elements[i].a - initial_elements[i].a)
        println("    Eccentricity diff: ", final_elements[i].e - initial_elements[i].e)
        println("    Argument of perihelion diff: ", angle_diff(final_elements[i].ω, initial_elements[i].ω))
        println("    Mean anomaly diff: ", angle_diff(final_anomalies[i-1][2], initial_anomalies[i-1][2]))
        
        if i > 2
            rel_ω_initial = initial_elements[i].ω - initial_elements[i-1].ω
            rel_ω_final = final_elements[i].ω - final_elements[i-1].ω
            println("    Relative orientation diff: ", angle_diff(rel_ω_final, rel_ω_initial))
        end
    end
    
    return s, ic, result
end


# function run_examples()
#     try
#         println("\n=========== 2-PLANET SYSTEM ===========")
#         s2, ic2, result2 = find_periodic_orbit(2)
#         println("Optimization of 2-planet system completed successfully!")
        
#         println("\n=========== 3-PLANET SYSTEM ===========")
#         s3, ic3, result3 = find_periodic_orbit(3)
#         println("Optimization of 3-planet system completed successfully!")
        
#         println("\n=========== 4-PLANET SYSTEM ===========")
#         s4, ic4, result4 = find_periodic_orbit(4)
#         println("Optimization of 4-planet system completed successfully!")
        
#         return (s2, ic2, result2), (s3, ic3, result3), (s4, ic4, result4)
#     catch e
#         println("Error during optimization:")
#         println(e)
#         println("\nBacktrace:")
#         for (exc, bt) in Base.catch_stack()
#             showerror(stdout, exc, bt)
#             println()
#         end
#     end
# end


# function run_example(n_planets::Int=2; kwargs...)
#     try
#         println("\n=========== $n_planets-PLANET SYSTEM ===========")
#         s, ic, result = find_periodic_orbit(n_planets; kwargs...)
#         println("Optimization of $n_planets-planet system completed successfully!")
#         return s, ic, result
#     catch e
#         println("Error during optimization:")
#         println(e)
#         println("\nBacktrace:")
#         for (exc, bt) in Base.catch_stack()
#             showerror(stdout, exc, bt)
#             println()
#         end
#     end
# end


# # results = run_examples()  # Run for 2, 3, and 4 planets
# # s, ic, result = run_example(2)  # Run just for 2 planets
# # s, ic, result = run_example(3)  # Run just for 3 planets
# #s, ic, result = run_example(4)  # Run just for 4 planets
# s, ic, result = run_example(4, 
#   planet_masses=[3e-6, 5e-6,  7e-5, 3e-5],
#   initial_eccentricities=[0.05, 0.07,0.05, 0.07],
#   kappa=2.0,  
#   max_iterations=600
# )