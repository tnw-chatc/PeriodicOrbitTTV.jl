using NbodyGradient
using LinearAlgebra
include("PeriodicOrbit.jl")
using .PeriodicOrbit

function test_integration_accuracy()
    println("Testing integration accuracy...")
    
    # Create a two-planet system
    star = Elements(m=1.0)
    planet1 = Elements(m=3e-6, P=1.0, e=0.1, ω=0.0, I=π/2, t0=0.0)  # Inner planet
    planet2 = Elements(m=5e-6, P=2.0, e=0.05, ω=π/4, I=π/2, t0=0.0)  # Outer planet
    
    # Initialize the system
    ic = ElementsIC(0.0, 3, star, planet1, planet2)
    s_initial = State(ic)
    
    # Record initial state
    println("Initial state:")
    initial_elements = get_orbital_elements(s_initial, ic)
    println("Inner planet semi-major axis: ", initial_elements[2].a)
    println("Inner planet eccentricity: ", initial_elements[2].e)
    println("Inner planet period: ", initial_elements[2].P)
    
    # Set up integration parameters
    step_sizes = [0.1, 0.05, 0.01]
    
    # Store final states for each step size
    final_states = []
    
    for step_size in step_sizes
        println("\nTesting step size: ", step_size)
        
        # Create a fresh copy of the initial state
        s = State(ic)
        
        # Set up integration time (1 orbit of the inner planet)
        integration_time = initial_elements[2].P
        
        # Manually step through integration to reach target time
        elapsed_time = 0.0
        num_steps = 0
        
        while elapsed_time < integration_time
            # For the last step, adjust step size to hit exactly integration_time
            if elapsed_time + step_size > integration_time
                final_step = integration_time - elapsed_time
                intr = Integrator(ahl21!, final_step, final_step)
            else
                intr = Integrator(ahl21!, step_size, step_size)
            end
            
            # Perform integration step
            intr(s)
            elapsed_time = s.t[1]
            num_steps += 1
            
            # Print progress occasionally
            if num_steps % 10 == 0
                println("  Step $(num_steps): time = $(elapsed_time)")
            end
        end
        
        # Get final state
        final_elements = get_orbital_elements(s, ic)
        println("\nFinal state after one orbital period:")
        println("Inner planet semi-major axis: ", final_elements[2].a)
        println("Inner planet eccentricity: ", final_elements[2].e)
        
        # Calculate relative changes in orbital elements
        a_change = abs(final_elements[2].a - initial_elements[2].a) / initial_elements[2].a
        e_change = abs(final_elements[2].e - initial_elements[2].e) / max(initial_elements[2].e, 1e-10)
        
        println("Relative changes in orbital elements:")
        println("  Semi-major axis change: ", a_change)
        println("  Eccentricity change: ", e_change)
        
        # Check conservation properties
        if a_change < 0.01
            println("Semi-major axis well conserved (< 1% change)")
        else
            println("WARNING: Significant semi-major axis change!")
        end
        
        if e_change < 0.05
            println("Eccentricity well conserved (< 5% change)")
        else
            println("WARNING: Significant eccentricity change!")
        end
        
        # Store the final state for this step size
        push!(final_states, (s, a_change, e_change))
    end
    
    # Return the final state from the smallest step size (most accurate)
    return final_states[end][1], ic, final_states
end

# Run the test and explicitly capture the return values
final_state, ic, all_results = test_integration_accuracy()

# Print a summary comparison of all step sizes
println("\n===== SUMMARY =====")
println("Step Size | Semi-major Axis Change | Eccentricity Change")
println("---------------------------------------------------")
for (i, step_size) in enumerate([0.1, 0.05, 0.01])
    _, a_change, e_change = all_results[i]
    println("$(step_size) | $(a_change) | $(e_change)")
end
