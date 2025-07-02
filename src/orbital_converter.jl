using ForwardDiff
using LinearAlgebra

"""
    orbital_to_cartesian(masses, periods, mean_anomalies, longitudes_periastron, eccentricities; M_star=1.0)

Convert orbital elements to Cartesian coordinates for a planetary system.

# Arguments
- `masses`: Vector of planet masses in solar masses
- `periods`: Vector of orbital periods in days
- `mean_anomalies`: Vector of mean anomalies in radians
- `longitudes_periastron`: Vector of longitudes of periastron in radians
- `eccentricities`: Vector of eccentricities
- `M_star`: Mass of central star in solar masses (default: 1.0)

# Returns
- `positions`: 3×N matrix of planet positions in AU
- `velocities`: 3×N matrix of planet velocities in AU/day
- `star_pos`: 3-element vector of star position in center-of-mass frame (AU)
- `star_vel`: 3-element vector of star velocity in center-of-mass frame (AU/day)
"""
function orbital_to_cartesian(masses::Vector{U}, periods::Vector{T}, mean_anomalies::Vector{T}, longitudes_periastron::Vector{T}, eccentricities::Vector{T}; M_star=1.0) where {T <: Real, U <: Real}
    n_planets = length(masses)
    
    # Initialize output arrays with the same type, T, as the input vectors:
    positions = zeros(T,3, n_planets)
    velocities = zeros(T,3, n_planets)

    # rel_mass = get_relative_masses(masses)

    rel_mass = [1. + masses[i] for i in 1:n_planets] .* 39.4845/(365.242 * 365.242)
    
    # Convert orbital elements to Cartesian for each planet
    for i in 1:n_planets
        # Semi-major axis from Kepler's third law (including planet mass)
        # P² = (4π²/G(M_star + m_planet)) * a³
        # With units: P in days, M in solar masses, a in AU
        # G = 4π²/(365.25²) in AU³/(solar mass × day²)
        G_units = 39.4845/(365.242 * 365.242)  # AU³/(M_sun × day²)
        a = ((periods[i]^2 * rel_mass[i]) / (4π^2))^(1/3)
        
        # Solve Kepler's equation for eccentric anomaly
        E = solve_kepler_equation(mean_anomalies[i], eccentricities[i])
        
        # True anomaly
        ν = 2 * atan(sqrt((1 + eccentricities[i])/(1 - eccentricities[i])) * tan(E/2))
        
        # Distance from focus
        r = a * (1 - eccentricities[i]^2) / (1 + eccentricities[i] * cos(ν))
        
        # Position in orbital plane
        x_orb = r * cos(ν)
        y_orb = r * sin(ν)
        
        # Velocity in orbital plane
        h = sqrt(rel_mass[i] * a * (1 - eccentricities[i]^2))  # specific angular momentum
        vx_orb = -h * sin(E) / (r * sqrt(1 - eccentricities[i]^2))
        vy_orb = h * cos(E) / r

        # For plane-parallel system, we only need rotation by longitude of periastron
        ω = longitudes_periastron[i]
        cos_ω = cos(ω)
        sin_ω = sin(ω)
        
        # Transform to heliocentric coordinates
        positions[1, i] = x_orb * cos_ω - y_orb * sin_ω
        positions[2, i] = x_orb * sin_ω + y_orb * cos_ω
        positions[3, i] = 0.0  # plane-parallel
        
        velocities[1, i] = vx_orb * cos_ω - vy_orb * sin_ω
        velocities[2, i] = vx_orb * sin_ω + vy_orb * cos_ω
        velocities[3, i] = 0.0  # plane-parallel
    end
    
    # Compute center of mass correction for star
    total_mass = M_star + sum(masses)
    
    # Star position and velocity in center-of-mass frame
    star_pos = -sum(masses[i] * positions[:, i] for i in 1:n_planets) / total_mass
    star_vel = -sum(masses[i] * velocities[:, i] for i in 1:n_planets) / total_mass
    # Now, convert the positions and velocities to the COM frame:
    for i in 1:n_planets
      positions[:,i] .+= star_pos
      velocities[:,i] .+= star_vel
    end
    return positions, velocities, star_pos, star_vel
end

"""
    solve_kepler_equation(M, e; tol=1e-12, max_iter=100)

Solve Kepler's equation M = E - e*sin(E) for eccentric anomaly E using Newton-Raphson method.
"""
function solve_kepler_equation(M::T, ecc::T; max_iter=100) where {T <: Real}
    E = M + sign(sin(M))*0.85*ecc # Initial guess
    E_old = 2E
    for i in 1:max_iter
        f = E - ecc * sin(E) - M
        fp = 1 - ecc * cos(E)
        
        E_new = E - f / fp
        
        if E_new == E || E_new == E_old
            # println("Check Kepler: ",E_new - ecc * sin(E_new) - M)
            return E_new
        end
        E_old = E
        E = E_new
    end
    error("Kepler equation failed to converge after $max_iter iterations")
end

"""
    compute_derivatives(masses, periods, mean_anomalies, longitudes_periastron, eccentricities; M_star=1.0)

Compute derivatives of Cartesian coordinates with respect to masses and orbital elements using automatic differentiation.

# Returns
- `pos_derivs`: Dictionary with derivative matrices for positions
- `vel_derivs`: Dictionary with derivative matrices for velocities  
- `star_pos_derivs`: Dictionary with derivative vectors for star position
- `star_vel_derivs`: Dictionary with derivative vectors for star velocity

Each dictionary contains keys: :masses, :periods, :mean_anomalies, :longitudes_periastron, :eccentricities
"""
function compute_derivatives(masses, periods, mean_anomalies, longitudes_periastron, eccentricities; M_star=1.0)
    n_planets = length(masses)
    
    # Pack parameters into a single vector for ForwardDiff
    params = vcat(masses, periods, mean_anomalies, longitudes_periastron, eccentricities)
    
    # Define function that unpacks parameters and returns flattened output
    function f(p)
        m = p[1:n_planets]
        P = p[n_planets+1:2*n_planets]
        M_anom = p[2*n_planets+1:3*n_planets]
        ω = p[3*n_planets+1:4*n_planets]
        e = p[4*n_planets+1:5*n_planets]
        
        pos, vel, star_pos, star_vel = orbital_to_cartesian(m, P, M_anom, ω, e; M_star=M_star)
        
        # Flatten output for ForwardDiff
        return vcat(vec(pos), vec(vel), star_pos, star_vel)
    end
    
    # Compute Jacobian
    J = ForwardDiff.jacobian(f, params)
    
    # Extract derivatives and reshape
    n_out = 3 * n_planets * 2 + 6  # positions + velocities + star pos + star vel
    
    pos_derivs = Dict()
    vel_derivs = Dict()
    star_pos_derivs = Dict()
    star_vel_derivs = Dict()
    
    # Indices for output components
    pos_idx = 1:3*n_planets
    vel_idx = 3*n_planets+1:6*n_planets
    star_pos_idx = 6*n_planets+1:6*n_planets+3
    star_vel_idx = 6*n_planets+4:6*n_planets+6
    
    # Extract derivatives with respect to each parameter type
    param_names = [:masses, :periods, :mean_anomalies, :longitudes_periastron, :eccentricities]
    
    for (i, name) in enumerate(param_names)
        param_idx = (i-1)*n_planets+1:i*n_planets
        
        pos_derivs[name] = reshape(J[pos_idx, param_idx], 3, n_planets, n_planets)
        vel_derivs[name] = reshape(J[vel_idx, param_idx], 3, n_planets, n_planets)
        star_pos_derivs[name] = J[star_pos_idx, param_idx]
        star_vel_derivs[name] = J[star_vel_idx, param_idx]
    end
    
    return pos_derivs, vel_derivs, star_pos_derivs, star_vel_derivs
end

"""
    test_derivatives(masses, periods, mean_anomalies, longitudes_periastron, eccentricities; 
                    M_star=1.0, h=1e-8, verbose=true)

Test derivatives computed by automatic differentiation against finite differences using BigFloat precision.

# Arguments
- `h`: Step size for finite differences (default: 1e-8)
- `verbose`: Print detailed comparison results (default: true)

# Returns
- `max_error`: Maximum relative error found across all derivatives
- `passed`: Boolean indicating if test passed (max error < 1e-6)
"""
function test_derivatives(masses, periods, mean_anomalies, longitudes_periastron, eccentricities; 
                         M_star=1.0, h=1e-8, verbose=true)
    
    # Convert to BigFloat for high precision finite differences
    masses_bf = BigFloat.(masses)
    periods_bf = BigFloat.(periods)
    mean_anomalies_bf = BigFloat.(mean_anomalies)
    longitudes_periastron_bf = BigFloat.(longitudes_periastron)
    eccentricities_bf = BigFloat.(eccentricities)
    M_star_bf = BigFloat(M_star)
    h_bf = BigFloat(h)
    
    n_planets = length(masses)
    
    # Get analytical derivatives
    pos_derivs, vel_derivs, star_pos_derivs, star_vel_derivs = 
        compute_derivatives(masses, periods, mean_anomalies, longitudes_periastron, eccentricities; M_star=M_star)
    
    # Helper function for high precision orbital conversion
    function orbital_to_cartesian_bf(m, P, M_anom, ω, e, M_s)
        n = length(m)
        pos = zeros(BigFloat, 3, n)
        vel = zeros(BigFloat, 3, n)
        
        for i in 1:n
            G_units = 4 * BigFloat(π)^2 / BigFloat(365.25)^2
            a = ((P[i]^2 * G_units * (M_s + m[i])) / (4 * BigFloat(π)^2))^(1//3)
            
            # Solve Kepler equation with BigFloat precision
            E = solve_kepler_equation_bf(M_anom[i], e[i])
#            E = solve_kepler_equation(M_anom[i], e[i])
            ν = 2 * atan(sqrt((1 + e[i])/(1 - e[i])) * tan(E/2))
            
            r = a * (1 - e[i]^2) / (1 + e[i] * cos(ν))
            x_orb = r * cos(ν)
            y_orb = r * sin(ν)
            
            h_ang = sqrt(G_units * (M_s + m[i]) * a * (1 - e[i]^2))
            vx_orb = -h_ang * sin(ν) / r
            vy_orb = h_ang * (e[i] + cos(ν)) / r
            
            cos_ω = cos(ω[i])
            sin_ω = sin(ω[i])
            
            pos[1, i] = x_orb * cos_ω - y_orb * sin_ω
            pos[2, i] = x_orb * sin_ω + y_orb * cos_ω
            pos[3, i] = BigFloat(0)
            
            vel[1, i] = vx_orb * cos_ω - vy_orb * sin_ω
            vel[2, i] = vx_orb * sin_ω + vy_orb * cos_ω
            vel[3, i] = BigFloat(0)
        end
        
        total_mass = M_s + sum(m)
        star_pos = -sum(m[i] * pos[:, i] for i in 1:n) / total_mass
        star_vel = -sum(m[i] * vel[:, i] for i in 1:n) / total_mass
        # Now, convert the positions and velocities to the COM frame:
        for i in 1:n_planets
          pos[:,i] .+= star_pos
          vel[:,i] .+= star_vel
        end
        
        return pos, vel, star_pos, star_vel
    end
    
    function solve_kepler_equation_bf(M, e; tol=BigFloat(1e-15), max_iter=100)
        E = M
        for i in 1:max_iter
            f = E - e * sin(E) - M
            fp = 1 - e * cos(E)
            E_new = E - f / fp
            if abs(E_new - E) < tol
                return E_new
            end
            E = E_new
        end
        error("Kepler equation failed to converge")
    end
    
    max_error = 0.0
    param_names = [:masses, :periods, :mean_anomalies, :longitudes_periastron, :eccentricities]
    param_arrays = [masses_bf, periods_bf, mean_anomalies_bf, longitudes_periastron_bf, eccentricities_bf]
    
    if verbose
        println("Testing derivatives with finite differences (h = $h)")
        println("="^60)
    end
    
    for (param_idx, (name, param_array)) in enumerate(zip(param_names, param_arrays))
        if verbose
            println("Testing derivatives w.r.t. $name:")
        end
        
        for i in 1:n_planets
            # Compute finite difference
            param_plus = copy(param_array)
            param_minus = copy(param_array)
            param_plus[i] += h_bf
            param_minus[i] -= h_bf
            
            arrays_plus = [masses_bf, periods_bf, mean_anomalies_bf, longitudes_periastron_bf, eccentricities_bf]
            arrays_minus = [masses_bf, periods_bf, mean_anomalies_bf, longitudes_periastron_bf, eccentricities_bf]
            arrays_plus[param_idx] = param_plus
            arrays_minus[param_idx] = param_minus
            
            pos_plus, vel_plus, star_pos_plus, star_vel_plus = 
                orbital_to_cartesian_bf(arrays_plus[1], arrays_plus[2], arrays_plus[3], arrays_plus[4], arrays_plus[5], M_star_bf)
            pos_minus, vel_minus, star_pos_minus, star_vel_minus = 
                orbital_to_cartesian_bf(arrays_minus[1], arrays_minus[2], arrays_minus[3], arrays_minus[4], arrays_minus[5], M_star_bf)
            
            # Finite difference derivatives
            pos_fd = (pos_plus - pos_minus) / (2 * h_bf)
            vel_fd = (vel_plus - vel_minus) / (2 * h_bf)
            star_pos_fd = (star_pos_plus - star_pos_minus) / (2 * h_bf)
            star_vel_fd = (star_vel_plus - star_vel_minus) / (2 * h_bf)
            
            # Compare with analytical derivatives
            for j in 1:n_planets
                for k in 1:3
                    # Position derivatives
                    analytical = pos_derivs[name][k, j, i]
                    numerical = Float64(pos_fd[k, j])
                    rel_error = abs(analytical - numerical) / (abs(numerical) + 1e-15)
                    max_error = max(max_error, rel_error)
                    
                    # Velocity derivatives
                    analytical = vel_derivs[name][k, j, i]
                    numerical = Float64(vel_fd[k, j])
                    rel_error = abs(analytical - numerical) / (abs(numerical) + 1e-15)
                    max_error = max(max_error, rel_error)
                end
            end
            
            # Star derivatives
            for k in 1:3
                analytical = star_pos_derivs[name][k, i]
                numerical = Float64(star_pos_fd[k])
                rel_error = abs(analytical - numerical) / (abs(numerical) + 1e-15)
                max_error = max(max_error, rel_error)
                
                analytical = star_vel_derivs[name][k, i]
                numerical = Float64(star_vel_fd[k])
                rel_error = abs(analytical - numerical) / (abs(numerical) + 1e-15)
                max_error = max(max_error, rel_error)
            end
            
            if verbose
                println("  Planet $i: max relative error = $(max_error)")
            end
        end
    end
    
    passed = max_error < 1e-6
    
    if verbose
        println("="^60)
        println("Overall maximum relative error: $max_error")
        println("Test $(passed ? "PASSED" : "FAILED") (threshold: 1e-6)")
    end
    
    return max_error, passed
end

# Example usage and test
function example_usage()
    # Example system: 2 planets
    masses = [1e-3, 5e-4]  # Jupiter-like and smaller planet in solar masses
    periods = [365.25, 2*365.25]  # 1 and 2 year periods in days
    mean_anomalies = [0.0, π/2]  # radians
    longitudes_periastron = [0.0, π/4]  # radians
    eccentricities = [0.05, 0.1]  # small eccentricities
    
    println("Example planetary system conversion:")
    println("Masses: $masses M_sun")
    println("Periods: $periods days")
    println("Mean anomalies: $mean_anomalies rad")
    println("Longitudes of periastron: $longitudes_periastron rad")
    println("Eccentricities: $eccentricities")
    println()
    
    # Convert to Cartesian coordinates
    pos, vel, star_pos, star_vel = orbital_to_cartesian(masses, periods, mean_anomalies, 
                                                       longitudes_periastron, eccentricities)
    
    println("Planet positions (AU):")
    for i in 1:length(masses)
        println("  Planet $i: [$(pos[1,i]), $(pos[2,i]), $(pos[3,i])]")
    end
    println()
    
    println("Planet velocities (AU/day):")
    for i in 1:length(masses)
        println("  Planet $i: [$(vel[1,i]), $(vel[2,i]), $(vel[3,i])]")
    end
    println()
   
   # Calculate COM position
    M_sun = 1.0
    total_mass = M_sun + sum(masses)
    com_check = (M_sun * star_pos + sum(masses[i] * pos[:, i] for i in 1:length(masses))) / total_mass
    #println("COM position: x=$(round(com_check[1], digits=16)), y=$(round(com_check[2], digits=16)), z=$(round(com_check[3], digits=16))")
    println("COM position: ",com_check)

    # Calculate COM velocity
    com_vel_check = (M_sun * star_vel + sum(masses[i] * vel[:, i] for i in 1:length(masses))) / total_mass
    #println("COM velocity: vx=$(round(com_vel_check[1], digits=16)), vy=$(round(com_vel_check[2], digits=16)), vz=$(round(com_vel_check[3], digits=16))")
    println("COM velocity: ",com_vel_check)
 
    println("Star position in CM frame (AU): [$star_pos[1], $star_pos[2], $star_pos[3]]")
    println("Star velocity in CM frame (AU/day): [$star_vel[1], $star_vel[2], $star_vel[3]]")
    println()
    
    # Test derivatives
    max_error, passed = test_derivatives(masses, periods, mean_anomalies, 
                                       longitudes_periastron, eccentricities)
    
    return pos, vel, star_pos, star_vel
end
