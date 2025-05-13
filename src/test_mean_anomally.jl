using NbodyGradient
using LinearAlgebra
using NbodyGradient: get_relative_masses, get_relative_positions
include("PeriodicOrbit.jl")
using .PeriodicOrbit

function test_mean_anomaly()
    println("Testing mean anomaly calculations...")
    
    star = Elements(m=1.0)
    planet = Elements(m=0.001, P=1.0, e=0.1, ω=0.0, I=π/2, Ω=0.0)
    ic = ElementsIC(0.0, 2, star, planet)
    s_initial = State(ic)

    initial_elements = get_orbital_elements(s_initial, ic)
    initial_anomalies = get_anomalies(s_initial, ic)
    

    println("Initial state:")
    println("  Semi-major axis: $(initial_elements[2].a)")
    println("  Eccentricity: $(initial_elements[2].e)")
    println("  True anomaly: $(initial_anomalies[1][1])")
    println("  Mean anomaly: $(initial_anomalies[1][2])")
    println("  Eccentric anomaly: $(initial_anomalies[1][3])")

    half_period = 0.5 * initial_elements[2].P
    

    s = State(ic)
    step_size = 0.01
    current_time = 0.0
    
    while current_time < half_period
        next_step = min(step_size, half_period - current_time)
        intr = Integrator(ahl21!, next_step, next_step)
        intr(s)
        current_time = s.t[1]
    end

    final_elements = get_orbital_elements(s, ic)
    final_anomalies = get_anomalies(s, ic)
    

    println("\nFinal state after half period (time = $current_time):")
    println("  Semi-major axis: $(final_elements[2].a)")
    println("  Eccentricity: $(final_elements[2].e)")
    println("  True anomaly: $(final_anomalies[1][1])")
    println("  Mean anomaly: $(final_anomalies[1][2])")
    println("  Eccentric anomaly: $(final_anomalies[1][3])")
    

    expected_ma_change = π  
    actual_ma_change = final_anomalies[1][2] - initial_anomalies[1][2]
    

    while actual_ma_change > π
        actual_ma_change -= 2π
    end
    while actual_ma_change < -π
        actual_ma_change += 2π
    end
    
    actual_ma_change = abs(actual_ma_change)
    
    println("\nMean anomaly change check:")
    println("  Expected change after half period: π")
    println("  Actual change: $actual_ma_change")
    println("  Difference: $(abs(actual_ma_change - π))")
    
    if abs(actual_ma_change - π) < 0.1
        println("k")
    else
        println("oh no!")
    end
    

    s = State(ic)
    full_period = initial_elements[2].P
    current_time = 0.0
    
    while current_time < full_period
        next_step = min(step_size, full_period - current_time)
        intr = Integrator(ahl21!, next_step, next_step)
        intr(s)
        current_time = s.t[1]
    end
    

    full_final_elements = get_orbital_elements(s, ic)
    full_final_anomalies = get_anomalies(s, ic)
    

    println("\nFinal state after full period (time = $current_time):")
    println("  Semi-major axis: $(full_final_elements[2].a)")
    println("  Eccentricity: $(full_final_elements[2].e)")
    println("  True anomaly: $(full_final_anomalies[1][1])")
    println("  Mean anomaly: $(full_final_anomalies[1][2])")
    println("  Eccentric anomaly: $(full_final_anomalies[1][3])")
    

    expected_ma_change_full = 2π  # After full period, should be back to original (mod 2π)
    actual_ma_change_full = full_final_anomalies[1][2] - initial_anomalies[1][2]
    
  
    while actual_ma_change_full < 0
        actual_ma_change_full += 2π
    end
    while actual_ma_change_full >= 2π
        actual_ma_change_full -= 2π
    end
    
    println("\nMean anomaly change check (full orbit):")
    println("  Expected change after full period: 0 (mod 2π)")
    println("  Actual change: $actual_ma_change_full")
    
    if abs(actual_ma_change_full) < 0.1
        println("k")
    else
        println("oh no")
    end
    
    println("\n\nTesting with a 4-planet system...")
    
    planet1 = Elements(m=0.0003, P=1.0, e=0.05, ω=0.0, I=π/2, Ω=0.0)
    planet2 = Elements(m=0.0004, P=2.0, e=0.07, ω=π/4, I=π/2, Ω=0.0)
    planet3 = Elements(m=0.0005, P=4.0, e=0.09, ω=π/2, I=π/2, Ω=0.0)
    planet4 = Elements(m=0.0006, P=8.0, e=0.11, ω=3π/4, I=π/2, Ω=0.0)
    

    ic_multi = ElementsIC(0.0, 5, star, planet1, planet2, planet3, planet4)
    s_multi = State(ic_multi)
    
 
    initial_elements_multi = get_orbital_elements(s_multi, ic_multi)
    initial_anomalies_multi = get_anomalies(s_multi, ic_multi)
    
 
    println("\nInitial state of 4-planet system:")
    for i in 1:4
        println("Planet $i:")
        println("  Period: $(initial_elements_multi[i+1].P)")
        println("  Semi-major axis: $(initial_elements_multi[i+1].a)")
        println("  Eccentricity: $(initial_elements_multi[i+1].e)")
        println("  Mean anomaly: $(initial_anomalies_multi[i][2])")
    end
    
    innermost_period = initial_elements_multi[2].P
    
    s_multi_integrated = State(ic_multi)
    current_time = 0.0
    
    while current_time < innermost_period
        next_step = min(step_size, innermost_period - current_time)
        intr = Integrator(ahl21!, next_step, next_step)
        intr(s_multi_integrated)
        current_time = s_multi_integrated.t[1]
    end
    
    final_elements_multi = get_orbital_elements(s_multi_integrated, ic_multi)
    final_anomalies_multi = get_anomalies(s_multi_integrated, ic_multi)
    
    println("\nFinal state after one innermost period (time = $current_time):")
    for i in 1:4
        println("Planet $i:")
        println("  Period: $(final_elements_multi[i+1].P)")
        println("  Semi-major axis: $(final_elements_multi[i+1].a)")
        println("  Eccentricity: $(final_elements_multi[i+1].e)")
        println("  Initial mean anomaly: $(initial_anomalies_multi[i][2])")
        println("  Final mean anomaly: $(final_anomalies_multi[i][2])")
        
        expected_change = 2π * (innermost_period / initial_elements_multi[i+1].P)
        actual_change = final_anomalies_multi[i][2] - initial_anomalies_multi[i][2]
        
     
        while actual_change < 0
            actual_change += 2π
        end
        while actual_change >= 2π
            actual_change -= 2π
        end
        
        println("  Expected MA change: $(expected_change)")
        println("  Actual MA change: $(actual_change)")
        println("  Difference: $(abs(actual_change - mod(expected_change, 2π)))")
        
        if abs(actual_change - mod(expected_change, 2π)) < 0.1
            println("  k")
        else
            println("oh no")
        end
    end
    
    return true
end


test_mean_anomaly()