using NbodyGradient
using LinearAlgebra
using Optim

function find_periodic_orbit()
    println("Finding periodic orbit configuration...")
    kappa = 2.0  # Target period ratio
    
    function objective_function(params)
        period_inner = params[1] 
        initial_e_1 = params[2]
        initial_e_2 = params[3]
        pomega_2 = params[4]
        
        #X5 - deviation from nominal period ratio
        X5 = params[5]

        actual_period_ratio = kappa + X5
        
        # Fixed parameters
        pomega_1 = 0.0
        mean_anomaly_1 = 0.0
        mean_anomaly_2 = 0.0 

        G = 1.0 
        m_star = 1.0
        
        try
            star = Elements(m=1.0)
  
            planet1 = Elements(
                m=3e-4,             
                P=period_inner,     
                t0=0.0,              
                ecosω=initial_e_1 * cos(pomega_1),  
                esinω=initial_e_1 * sin(pomega_1),
                I=π/2,               
                Ω=0.0               
            )
            
            
            planet2 = Elements(
                m=5e-4,              
                P=period_inner * actual_period_ratio, 
                t0=0.0,              
                ecosω=initial_e_2 * cos(pomega_2),  
                esinω=initial_e_2 * sin(pomega_2),  
                I=π/2,               
                Ω=0.0               
            )
            
            
            ic = ElementsIC(0.0, 3, star, planet1, planet2)
            s_initial = State(ic)
            
            initial_elements = NbodyGradient.get_orbital_elements(s_initial, ic)

          
            ini_param = [
                initial_elements[2].e,               
                initial_elements[3].e,                
                initial_elements[3].ω - initial_elements[2].ω,  
                initial_elements[3].P / initial_elements[2].P   
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
            
         
            final_elements = NbodyGradient.get_orbital_elements(s, ic)
            
            end_param = [
                final_elements[2].e,              
                final_elements[3].e,               
                final_elements[3].ω - final_elements[2].ω, 
                final_elements[3].P / final_elements[2].P   
            ]
            
       
            differences = zeros(4)
            
           
            differences[1] = ini_param[1] - end_param[1]
            differences[2] = ini_param[2] - end_param[2]
            
            # Angle difference (need wrapping)
            diff = ini_param[3] - end_param[3]
            # Wrap to [-π, π]
            while diff > π
                diff -= 2π
            end
            while diff < -π
                diff += 2π
            end
            differences[3] = diff
            
            # Period ratio difference
            differences[4] = ini_param[4] - end_param[4]
            
            
            weights = [1.0, 1.0, 3.0, 2.0]
            return sum(weights .* differences.^2)
        catch e
            println("Error in objective function: ", e)
            return 1e10
        end
    end
    
    # Initial parameter values
    initial_params = [
        1.0,    # period_inner
        0.05,   # initial_e_1
        0.1,    # initial_e_2
        1.0,    # pomega_2
        0.0     # X5 (start with no deviation)
    ]
    
    # Parameter bounds
    lower_bounds = [
        0.5,     # period_inner
        0.001,   # initial_e_1
        0.001,   # initial_e_2
        -π,      # pomega_2
        -0.1     # X5 (allow small negative deviation)
    ]
    
    upper_bounds = [
        2.0,     # period_inner
        0.9,     # initial_e_1
        0.9,     # initial_e_2
        π,       # pomega_2
        0.1      # X5 (allow small positive deviation)
    ]
    
    println("Starting optimization with parameters:")
    println("  Inner planet period: $(initial_params[1])")
    println("  Eccentricities: e₁=$(initial_params[2]), e₂=$(initial_params[3])")
    println("  Argument of perihelion of planet 2: ω₂=$(initial_params[4])")
    println("  Period ratio deviation (X5): $(initial_params[5])")
    
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
    
    # Get optimized parameters
    opt_params = Optim.minimizer(result)
    opt_period = opt_params[1]
    opt_e_1, opt_e_2 = opt_params[2:3]
    opt_pomega_2 = opt_params[4]
    opt_X5 = opt_params[5]
    opt_period_ratio = kappa + opt_X5
    
    println("\nOptimization complete!")
    println("Optimized parameters:")
    println("  Inner planet period: $(opt_period)")
    println("  Eccentricities: e₁=$(opt_e_1), e₂=$(opt_e_2)")
    println("  Argument of perihelion of planet 2: ω₂=$(opt_pomega_2)")
    println("  Period ratio deviation (X5): $(opt_X5)")
    println("  Actual period ratio: $(opt_period_ratio)")
    println("Final objective value: ", Optim.minimum(result))
    
    # Create optimized system for verification
    star = Elements(m=1.0)
    planet1 = Elements(
        m=3e-4,
        P=opt_period,
        t0=0.0,
        ecosω=opt_e_1 * cos(0.0),
        esinω=opt_e_1 * sin(0.0),
        I=π/2,
        Ω=0.0
    )
    
    planet2 = Elements(
        m=5e-4,
        P=opt_period * opt_period_ratio,
        t0=0.0,
        ecosω=opt_e_2 * cos(opt_pomega_2),
        esinω=opt_e_2 * sin(opt_pomega_2),
        I=π/2,
        Ω=0.0
    )
    
    ic = ElementsIC(0.0, 3, star, planet1, planet2)
    s = State(ic)
    
    optimized_elements = NbodyGradient.get_orbital_elements(s, ic)
    
    println("\nOptimized system characteristics:")
    println("  Inner planet period: $(optimized_elements[2].P)")
    println("  Outer planet period: $(optimized_elements[3].P)")
    println("  a₁ = $(optimized_elements[2].a)")
    println("  a₂ = $(optimized_elements[3].a)")
    println("  P₂/P₁ = $((optimized_elements[3].P / optimized_elements[2].P))")

    # Verify periodicity by integration
    println("\n===== VERIFICATION OF PERIODICITY =====")

    # Create fresh system with optimized parameters
    star = Elements(m=1.0)
    planet1 = Elements(
        m=3e-4,
        P=opt_period,
        t0=0.0,
        ecosω=opt_e_1 * cos(0.0),
        esinω=opt_e_1 * sin(0.0),
        I=π/2,
        Ω=0.0
    )
    
    planet2 = Elements(
        m=5e-4,
        P=opt_period * opt_period_ratio,
        t0=0.0,
        ecosω=opt_e_2 * cos(opt_pomega_2),
        esinω=opt_e_2 * sin(opt_pomega_2),
        I=π/2,
        Ω=0.0
    )

    ic_test = ElementsIC(0.0, 3, star, planet1, planet2)
    s_test = State(ic_test)

    # Get initial state
    initial_elements = NbodyGradient.get_orbital_elements(s_test, ic_test)

    # Integrate for two inner planet periods
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
    final_elements = NbodyGradient.get_orbital_elements(s_test, ic_test)
    
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
    
    # Print differences
    println("\nDifferences (final - initial):")
    println("  Inner planet:")
    println("    Semi-major axis diff: ", final_elements[2].a - initial_elements[2].a)
    println("    Eccentricity diff: ", final_elements[2].e - initial_elements[2].e)
    println("    Argument of perihelion diff: ", angle_diff(final_elements[2].ω, initial_elements[2].ω))
    println("  Outer planet:")
    println("    Semi-major axis diff: ", final_elements[3].a - initial_elements[3].a)
    println("    Eccentricity diff: ", final_elements[3].e - initial_elements[3].e)
    println("    Argument of perihelion diff: ", angle_diff(final_elements[3].ω, initial_elements[3].ω))
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