using NbodyGradient
using LinearAlgebra
using Optim
include("PeriodicOrbit.jl")
using .PeriodicOrbit

function find_periodic_orbit()
    println("Finding periodic orbit configuration...")
    kappa = 2.0  # Target period ratio
    
    function objective_function(params)
        period_inner = params[1] 
        initial_e_1 = params[2]
        initial_e_2 = params[3]
        pomega_2 = params[4]
        
        # New parameter: X5 - deviation from nominal period ratio
        X5 = params[5]
        
        # Calculate the actual period ratio based on X5
        actual_period_ratio = kappa + X5
        
        # Fixed parameters
        pomega_1 = 0.0
        mean_anomaly_1 = 0.0
        mean_anomaly_2 = params[6]  # Mean anomaly of outer planet

        G = 1.0 
        m_star = 1.0
        
        try
            star = Elements(m=1.0)
  
            # Create the planets using the correct constructors
            planet1 = Elements(m=3e-4, P=period_inner, e=initial_e_1, ω=pomega_1, I=π/2, Ω=0.0)
            planet2 = Elements(m=5e-4, P=period_inner*actual_period_ratio, e=initial_e_2, ω=pomega_2, I=π/2, Ω=0.0)
            
            # Initialize the system
            ic = ElementsIC(0.0, 3, star, planet1, planet2)
            s_initial = State(ic)
            
            initial_elements = get_orbital_elements(s_initial, ic)
            initial_anomalies = get_anomalies(s_initial, ic)

            # Calculate initial parameters to measure periodicity
            ini_param = [
                initial_elements[2].e,                  # Inner planet eccentricity
                initial_elements[3].e,                  # Outer planet eccentricity
                initial_elements[3].ω - initial_elements[2].ω,  # Relative orientation
                initial_elements[3].P / initial_elements[2].P,  # Period ratio
                initial_anomalies[1][2],                # Inner planet mean anomaly
                initial_anomalies[2][2]                 # Outer planet mean anomaly
            ]
            
            # Actual period of inner planet from orbital elements
            actual_period = initial_elements[2].P
            
            # Integration time (2 orbital periods)
            integration_time = 2 * actual_period
            
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
            
            # Get final orbital elements
            final_elements = get_orbital_elements(s, ic)
            final_anomalies = get_anomalies(s, ic)
            
            # Calculate final parameters
            end_param = [
                final_elements[2].e,                  # Inner planet eccentricity
                final_elements[3].e,                  # Outer planet eccentricity
                final_elements[3].ω - final_elements[2].ω,  # Relative orientation
                final_elements[3].P / final_elements[2].P,  # Period ratio
                final_anomalies[1][2],                # Inner planet mean anomaly  
                final_anomalies[2][2]                 # Outer planet mean anomaly
            ]
            
            # Calculate differences
            differences = zeros(6)
            
            # Eccentricity differences (no wrapping)
            differences[1] = ini_param[1] - end_param[1]
            differences[2] = ini_param[2] - end_param[2]
            
            # Angle differences (need wrapping)
            # Relative orientation
            diff = ini_param[3] - end_param[3]
            while diff > π
                diff -= 2π
            end
            while diff < -π
                diff += 2π
            end
            differences[3] = diff
            
            # Period ratio difference
            differences[4] = ini_param[4] - end_param[4]
            
            # Mean anomaly differences (need wrapping)
            # Inner planet
            diff = ini_param[5] - end_param[5]
            while diff > π
                diff -= 2π
            end
            while diff < -π
                diff += 2π
            end
            differences[5] = diff
            
            # Outer planet
            diff = ini_param[6] - end_param[6]
            while diff > π
                diff -= 2π
            end
            while diff < -π
                diff += 2π
            end
            differences[6] = diff
            
            # Calculate weighted sum of squared differences
            # Higher weight on maintaining the relative orientation and period ratio
            weights = [1.0, 1.0, 3.0, 2.0, 1.0, 1.0]
            return sum(weights .* differences.^2)
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
    X5 = 0.0
    mean_anomaly_2 = 0.0
    
    initial_params = [
        period_inner,
        initial_e_1, initial_e_2,
        pomega_2,
        X5,
        mean_anomaly_2
    ]
    
    # Parameter bounds
    lower_bounds = [
        0.5,          # period_inner
        0.001, 0.001,  # eccentricities
        -π,           # pomega_2
        -0.1,         # X5
        -π            # mean_anomaly_2
    ]
    
    upper_bounds = [
        2.0,          # period_inner
        0.9, 0.9,      # eccentricities
        π,            # pomega_2
        0.1,          # X5
        π             # mean_anomaly_2
    ]
    
    println("Starting optimization with parameters:")
    println("  Inner planet period: $(period_inner)")
    println("  Eccentricities: e₁=$(initial_e_1), e₂=$(initial_e_2)")
    println("  Argument of perihelion of planet 2: ω₂=$(pomega_2)")
    println("  Period ratio deviation (X5): $(X5)")
    println("  Mean anomaly of outer planet: $(mean_anomaly_2)")
    
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
    opt_mean_anomaly_2 = opt_params[6]
    
    println("\nOptimization complete!")
    println("Optimized parameters:")
    println("  Inner planet period: $(opt_period)")
    println("  Eccentricities: e₁=$(opt_e_1), e₂=$(opt_e_2)")
    println("  Argument of perihelion of planet 2: ω₂=$(opt_pomega_2)")
    println("  Period ratio deviation (X5): $(opt_X5)")
    println("  Actual period ratio: $(opt_period_ratio)")
    println("  Mean anomaly of outer planet: $(opt_mean_anomaly_2)")
    println("Final objective value: ", Optim.minimum(result))
    
    # Create optimized system for verification
    star = Elements(m=1.0)
    
    # Now create the planets using optimized parameters
    # We need to adjust the t0 parameter to set the mean anomaly correctly
    planet1 = Elements(m=3e-4, P=opt_period, e=opt_e_1, ω=0.0, I=π/2, Ω=0.0)
    planet2 = Elements(m=5e-4, P=opt_period*opt_period_ratio, e=opt_e_2, ω=opt_pomega_2, I=π/2, Ω=0.0)
    
    ic = ElementsIC(0.0, 3, star, planet1, planet2)
    s = State(ic)
    
    optimized_elements = get_orbital_elements(s, ic)
    optimized_anomalies = get_anomalies(s, ic)
    
    println("\nOptimized system characteristics:")
    println("  Inner planet period: $(optimized_elements[2].P)")
    println("  Outer planet period: $(optimized_elements[3].P)")
    println("  a₁ = $(optimized_elements[2].a)")
    println("  a₂ = $(optimized_elements[3].a)")
    println("  P₂/P₁ = $((optimized_elements[3].P / optimized_elements[2].P))")
    println("  Inner planet mean anomaly: $(optimized_anomalies[1][2])")
    println("  Outer planet mean anomaly: $(optimized_anomalies[2][2])")

    println("\n===== VERIFICATION OF PERIODICITY =====")

    # Create fresh system with optimized parameters
    star = Elements(m=1.0)
    planet1 = Elements(m=3e-4, P=opt_period, e=opt_e_1, ω=0.0, I=π/2, Ω=0.0)
    planet2 = Elements(m=5e-4, P=opt_period*opt_period_ratio, e=opt_e_2, ω=opt_pomega_2, I=π/2, Ω=0.0)

    ic_test = ElementsIC(0.0, 3, star, planet1, planet2)
    s_test = State(ic_test)

    # Get initial state
    initial_elements = get_orbital_elements(s_test, ic_test)
    initial_anomalies = get_anomalies(s_test, ic_test)

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
    
    # Print differences
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