using NbodyGradient
using LinearAlgebra
using Optim
include("PeriodicOrbit.jl")
using .PeriodicOrbit

function find_periodic_orbit()
    println("Finding periodic orbit configuration...")
    kappa = 2.0  
    
    function objective_function(params)
        period_inner = params[1] 
        initial_e_1 = params[2]
        initial_e_2 = params[3]
        pomega_2 = params[4]
        
        # Fixed parameters
        pomega_1 = 0.0
        mean_anomaly_1 = 0.0
        mean_anomaly_2 = 0.0 

        G = 1.0 
        m_star = 1.0
        

        a_1 = (period_inner^2 * G * m_star / (4 * π^2))^(1/3)
        a_2 = a_1 * kappa^(2/3) 
        
        try
            star = Elements(m=1.0)
  
            planet1 = Elements(m=3e-4, P=period_inner, e=initial_e_1, ω=pomega_1, I=π/2, Ω=0.0)
            planet2 = Elements(m=5e-4, P=period_inner*kappa, e=initial_e_2, ω=pomega_2, I=π/2, Ω=0.0)
            
            # Initialize the system
            ic = ElementsIC(0.0, 3, star, planet1, planet2)
            s_initial = State(ic)
            
            initial_elements = get_orbital_elements(s_initial, ic)

            ini_param = [
                initial_elements[2].e,
                initial_elements[3].e,
                initial_elements[3].ω - initial_elements[2].ω  # Relative pomega
            ]
            

            actual_period = initial_elements[2].P
            integration_time = 2 * actual_period
            s = State(ic)
            
 
            step_size = 0.01 * actual_period
            current_time = 0.0
            
            while current_time < integration_time
                next_step = min(step_size, integration_time - current_time)
                intr = Integrator(ahl21!, next_step, next_step)
                intr(s)
                current_time = s.t[1]
            end
            
            final_elements = get_orbital_elements(s, ic)
            end_param = [
                final_elements[2].e,
                final_elements[3].e,
                final_elements[3].ω - final_elements[2].ω  # Relative pomega
            ]
            
            differences = zeros(3)
            differences[1] = ini_param[1] - end_param[1]
            differences[2] = ini_param[2] - end_param[2]
            diff = ini_param[3] - end_param[3]

            # Wrap to [-π, π]
            while diff > π
                diff -= 2π
            end
            while diff < -π
                diff += 2π
            end
            differences[3] = diff
            
            return sum(differences.^2)
        catch e
            println("Error in objective function: ", e)
            return 1e10
        end
    end
    
    # Initial parameter values
    period_inner = 1.0  
    initial_e_1 = 0.05
    initial_e_2 = 0.1
    pomega_2 = 1.0
    
    initial_params = [
        period_inner,
        initial_e_1, initial_e_2,
        pomega_2
    ]
    

    lower_bounds = [
        0.5,          
        0.001, 0.001,  
        -π           
    ]
    
    upper_bounds = [
        2.0,        
        0.9, 0.9,      
        π              
    ]
    
    println("Starting optimization with parameters:")
    println("  Inner planet period: $(period_inner)")
    println("  Eccentricities: e₁=$(initial_e_1), e₂=$(initial_e_2)")
    println("  Argument of perihelion of planet 2: ω₂=$(pomega_2)")
    
    # Test objective function
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
        Optim.Options(iterations=200, show_trace=true)
    )
    
    
    opt_params = Optim.minimizer(result)
    opt_period = opt_params[1]
    opt_e_1, opt_e_2 = opt_params[2:3]
    opt_pomega_2 = opt_params[4]
    
    println("\nOptimization complete!")
    println("Optimized parameters:")
    println("  Inner planet period: $(opt_period)")
    println("  Eccentricities: e₁=$(opt_e_1), e₂=$(opt_e_2)")
    println("  Argument of perihelion of planet 2: ω₂=$(opt_pomega_2)")
    println("Final objective value: ", Optim.minimum(result))
    
    
    star = Elements(m=1.0)
    planet1 = Elements(m=3e-6, P=opt_period, e=opt_e_1, ω=0.0, I=π/2, Ω=0.0)
    planet2 = Elements(m=5e-6, P=opt_period*kappa, e=opt_e_2, ω=opt_pomega_2, I=π/2, Ω=0.0)
    
    ic = ElementsIC(0.0, 3, star, planet1, planet2)
    s = State(ic)
    
    optimized_elements = get_orbital_elements(s, ic)
    
    println("\nOptimized system characteristics:")
    println("  Inner planet period: $(optimized_elements[2].P)")
    println("  Outer planet period: $(optimized_elements[3].P)")
    println("  a₁ = $(optimized_elements[2].a)")
    println("  a₂ = $(optimized_elements[3].a)")
    println("  P₂/P₁ = $((optimized_elements[3].P / optimized_elements[2].P))")

    println("\n===== VERIFICATION OF PERIODICITY =====")


    # Create fresh system with optimized parameters
    star = Elements(m=1.0)
    planet1 = Elements(m=3e-6, P=opt_period, e=opt_e_1, ω=0.0, I=π/2, Ω=0.0)
    planet2 = Elements(m=5e-6, P=opt_period*kappa, e=opt_e_2, ω=opt_pomega_2, I=π/2, Ω=0.0)

    ic_test = ElementsIC(0.0, 3, star, planet1, planet2)
    s_test = State(ic_test)


    initial_elements = get_orbital_elements(s_test, ic_test)
    initial_anomalies = get_anomalies(s_test, ic_test)

    integration_time = 2 * initial_elements[2].P
    step_size = 0.01 * initial_elements[2].P
    current_time = 0.0

    while current_time < integration_time
        next_step = min(step_size, integration_time - current_time)
        intr = Integrator(ahl21!, next_step, next_step)
        intr(s_test)
        current_time = s_test.t[1]
    end

    # Get final state
    final_elements = get_orbital_elements(s_test, ic_test)
    final_anomalies = get_anomalies(s_test, ic_test)
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
    println("  Inner planet:")
    println("    Semi-major axis diff: ", final_elements[2].a - initial_elements[2].a)
    println("    Eccentricity diff: ", final_elements[2].e - initial_elements[2].e)
    println("    Argument of perihelion diff: ", angle_diff(final_elements[2].ω, initial_elements[2].ω))
    println("    Mean anomaly diff: ", angle_diff(final_anomalies[1][2], initial_anomalies[1][2]))
    println("  Outer planet:")
    println("    Semi-major axis diff: ", final_elements[3].a - initial_elements[3].a)
    println("    Eccentricity diff: ", final_elements[3].e - initial_elements[3].e)
    println("    Argument of perihelion diff: ", angle_diff(final_elements[3].ω, initial_elements[3].ω))
    println("    Mean anomaly diff: ", angle_diff(final_anomalies[2][2], initial_anomalies[2][2]))
    println("  Relative orientation diff: ", 
        angle_diff(final_elements[3].ω - final_elements[2].ω, initial_elements[3].ω - initial_elements[2].ω))

    return s, ic, result
end

# Main execution
try
    optimized_system, ic, result = find_periodic_orbit()
    println("Optimization completed successfully!")
catch e
    println("Error during optimization:")
    println(e)
    println("\nBacktrace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end
